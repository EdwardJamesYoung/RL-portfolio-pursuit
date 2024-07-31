import numpy as np
import scipy.optimize
import pandas as pd
import random
import time
import os
import uuid
import mlflow
from pprint import pformat
import json
import itertools
from typing import List, Tuple, Any, Dict, Callable
from collections import deque
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    RationalQuadratic,
    WhiteKernel,
    Matern,
)
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_random_state

from config import InsurerConfig, MarketConfig, ExperimentConfig, CustomerConfig

from tqdm.notebook import tqdm

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import matplotlib.pyplot as plt

import pandera as pa
from pandera import Column, DataFrameSchema
from pandera.typing import DataFrame

from einops import repeat

from copy import deepcopy

########################################

# Ensure that we have read-write access to the directory
current_dir = os.getcwd()
assert os.access(
    current_dir, os.R_OK
), "You do not have read permissions for the current working directory"
assert os.access(
    current_dir, os.W_OK
), "You do not have write permissions for the current working directory"

# Generate a unique ID which is used to prevent interferance when logging images in parallel
unique_id = uuid.uuid4().hex

########################################


def set_seeds(seed: int):
    """
    Sets the seeds for various packages to ensure repoducability.

    Args:
        seed (int): The random seed to set for all packages
    """
    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for scikit-learn
    check_random_state(seed)

    # Additional steps to ensure reproducibility in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_large_list(list_to_log: List, list_name: str):
    """
    Log a large list using MLflow by converting to a json file.

    Args:
        list_to_log (list): The list to be logged.
        list_name (str): The name to give to the logged list.
    """
    # Convert the list to a JSON string
    json_str = json.dumps(list_to_log)

    # Log the JSON string as a text file
    mlflow.log_text(json_str, f"{list_name}.json")


#########################################


# Define various schemas that we will use to validate dataframes using pandera.
customer_columns = [
    "Car valuation",
    "Number of previous claims",
    "Years driving",
    "Age",
    "Marital status",
    "Child indicator",
]
customer_columns += [f"Occupation_{ii}" for ii in range(CustomerConfig.num_occ)]
customer_columns += [f"Location_{ii}" for ii in range(CustomerConfig.num_loc)]
# Columns for market datasets
market_input_columns = customer_columns
market_output_columns = ["Top 1", "Top 3", "Top 5"]
market_complete_columns = market_input_columns + market_output_columns
# Columns for incoming and saved datasets
complete_columns = (
    customer_columns + market_output_columns + ["expected_cost", "action", "response"]
)
new_customer_columns = customer_columns + ["expected_cost"]
# Columns for conversion datasets
# conversion_input_columns = customer_columns + market_output_columns + ["action"]
conversion_input_columns = market_output_columns + ["action"]
conversion_output_columns = ["response"]
conversion_complete_columns = conversion_input_columns + conversion_output_columns
# Columns for bidding datasets
# bidding_input_columns = customer_columns + market_output_columns + ["k"]
bidding_input_columns = market_output_columns + ["k"]
bidding_output_columns = ["action"]
bidding_complete_columns = bidding_input_columns + bidding_output_columns


# Map from columns lists into dictionaries
def lst_to_col(columns: List[str]) -> Dict[str, Column]:
    """
    Helper function which converts a list of column names into the right format for pandera.

    Args:
        columns (List[str]): list of column names.

    Returns:
        Dict[str, Column]: dictionary containing keys given my column names and values which are pandera Columns.
    """
    return {column_name: Column() for column_name in columns}


# Create pandera schemas that can be used to perform typechecking on dataframes.

# Customer schemas
CompleteSchema = DataFrameSchema(lst_to_col(complete_columns), strict=True)
NewCustomerSchema = DataFrameSchema(lst_to_col(new_customer_columns), strict=True)
# Conversion schemas
ConversionInputSchema = DataFrameSchema(
    lst_to_col(conversion_input_columns), strict=True
)
ConversionCompleteSchema = DataFrameSchema(
    lst_to_col(conversion_complete_columns), strict=True
)
# Market schemas
MarketInputSchema = DataFrameSchema(lst_to_col(market_input_columns), strict=True)
MarketCompleteSchema = DataFrameSchema(lst_to_col(market_complete_columns), strict=True)
# Bidding schemas
BiddingInputSchema = DataFrameSchema(lst_to_col(bidding_input_columns), strict=True)
BiddingCompleteSchema = DataFrameSchema(
    lst_to_col(bidding_complete_columns), strict=True
)
# Value schema
ValueCompleteSchema = DataFrameSchema(lst_to_col(["rho", "U", "tau"]), strict=True)


# A series of helper functions which extract relevant information from a dataframe.


@pa.check_io(complete_df=CompleteSchema, out=ConversionCompleteSchema)
def create_conversion_training_data(
    complete_df: DataFrame[CompleteSchema],
) -> DataFrame[ConversionCompleteSchema]:
    return complete_df[conversion_complete_columns]


@pa.check_io(complete_df=CompleteSchema, out=MarketCompleteSchema)
def create_market_training_data(
    complete_df: DataFrame[CompleteSchema],
) -> DataFrame[MarketCompleteSchema]:
    return complete_df[market_complete_columns]


@pa.check_io(complete_df=CompleteSchema, out=BiddingInputSchema)
def create_bidding_input_data(
    complete_df: DataFrame[CompleteSchema], k_values: List[float]
) -> DataFrame[BiddingInputSchema]:
    complete_with_k = complete_df.assign(k=k_values)
    return complete_with_k[bidding_input_columns]


@pa.check_io(bidding_input_df=BiddingInputSchema, out=BiddingCompleteSchema)
def create_bidding_training_data(
    bidding_input_df: DataFrame[BiddingInputSchema], actions: List[float]
) -> DataFrame[BiddingCompleteSchema]:
    return bidding_input_df.assign(action=actions)


####################################################


class InsurerBase(ABC):
    def __init__(self, idx):
        # Market index
        self.idx = idx
        # Store the amount of profit the firm has made
        self.profit = 0
        # Total reward
        self.total_reward = 0
        # Store the number of customers the firm has seen
        self.total_customers = 0
        # Store the number of successful sales the firm has had
        self.successful_sales = 0
        # Count the number of time steps the insurer has been in the market
        self.time_steps = 0

    @abstractmethod
    def make_offers(
        self, customer_profiles: List[Dict], expected_costs: np.ndarray
    ) -> np.ndarray:
        """
        This function has the insurer take in data from a set of customers and generate offers for the customer

        Args:
            customer_profiles (List[dict]): A list of length num_customers with each entry a dictionary containing customer features.
            expected_costs (np.ndarray): A numpy array of shape num_customers which contains the expected cost of serving each customer.

        Returns:
            np.ndarray: A numpy array of shape num_customers containing the offer made to each customer.
        """

        pass

    @abstractmethod
    def store_customers(
        self,
        customer_profiles: List[Dict],
        expected_costs: np.ndarray,
        offers: np.ndarray,
        responses: List[bool],
    ):
        """
        This function stores the information from the customer, and calls any operations the insurers may wish to perform.

        Args:
            customer_profiles (List[dict]): A list of length num_customers with each entry a dictionary containing customer features.
            expected_costs (np.ndarray): A numpy array of shape num_customers which contains the expected cost of serving each customer.
            offers_list (np.ndarray): This is a numpy array with shape (num_customers x num_insurers) which gives the offers made by each insurer for each customer.
            responses (List[bool]): This is a binary array with shape num_customers which says whether each customer accepted your offer.
        """
        pass

    def conversion_rate(self):
        """
        Computes the conversion rate for the insurer, i.e., the number of successful sales divided by the total number of customer interactions

        Returns:
            float: The conversion rate
        """
        if self.total_customers == 0:
            return 0
        else:
            return self.successful_sales / self.total_customers

    def profit_per_conversion(self):
        """
        Computes the average profit made by the customer per successful sale

        Returns:
            float: The profit per conversion
        """
        if self.successful_sales == 0:
            return 0
        else:
            return self.profit / self.successful_sales

    def clear_counters(self):
        """
        Clears the various counters the insurer uses such that other metrics
        (such as conversion rate and profit per conversion) are computed as if
        the company entered the market just after clear_counters was called
        """
        self.profit = 0
        self.total_customers = 0
        self.successful_sales = 0
        self.time_steps = 0


class Insurer(InsurerBase):
    def __init__(
        self,
        idx: int,
        insurer_type: str = "Null",
        parameters=None,
        market_model_type: str = "Random Forest",
        bidding_strategy: str = "epsilon-greedy",
    ):
        super().__init__(idx)

        # There are three possible insurer types:
        #       "Null", which indicates that the insurer does not pursuit a portfolio.
        #       "Baseline", which indicates that the insurer uses the baseline method for portfolio pursuit.
        #       "RL", which indicates that the insurer uses the RL algorithm for portfolio pursuit.
        self.insurer_type = insurer_type

        # If the insurer is performing portfolio pursuit we set the random seed and the categorise in the portfolio.
        if self.insurer_type != "Null":
            if "num_categories" in parameters:
                self.num_categories = parameters["num_categories"]
            else:
                self.num_categories = 5

            if "trial_number" in parameters:
                seed = 100 * (parameters["trial_number"] + 1)
            else:
                seed = 100

            set_seeds(seed)
            mlflow.log_param("random_seed", seed)

            self.portfolio = generate_random_portfolio(self.num_categories)

            mlflow.log_text(
                self.portfolio.get_text_description(),
                f"insurer_{self.idx}_type_{self.insurer_type}_portfolio_categories.txt",
            )

        self.mode = "burn-in"
        self.epoch = 0

        # The market model takes in customer features and predicts the rest of the market behaviour
        self.market_model_type = market_model_type
        self.market_model = MarketModel(
            model_type=market_model_type, parameters=parameters
        )

        # The conversion model takes in a price and a customer feature and outputs the probability of acceptance
        if "conversion_model_type" in parameters:
            self.conversion_model_type = parameters["conversion_model_type"]
        else:
            self.conversion_model_type = "MLP"

        self.conversion_model = ConversionModel(parameters=parameters)

        # The bidding model takes in a customer feature vector and an expected_costs for that customer and outputs an offer
        if "bidding_model_type" in parameters:
            self.bidding_model_type = parameters["bidding_model_type"]
        else:
            self.bidding_model_type = "GP-Mat"

        self.bidding_model = BiddingModel(parameters=parameters)

        mlflow.log_param(f"insurer_{idx}_type", self.insurer_type)
        mlflow.log_param(f"insurer_{idx}_market_model_type", self.market_model_type)
        mlflow.log_param(
            f"insurer_{idx}_conversion_model_type", self.conversion_model_type
        )
        mlflow.log_param(f"insurer_{idx}_bidding_model_type", self.bidding_model_type)

        if self.insurer_type == "RL":
            # First, set any hyperparameters that have been passed in
            if "num_next_customer_samples" in parameters:
                self.num_next_customer_samples = parameters["num_next_customer_samples"]
            else:
                self.num_next_customer_samples = 500

            if "num_portfolio_samples" in parameters:
                self.portfolio_samples_per_step = parameters["num_portfolio_samples"]
            else:
                self.portfolio_samples_per_step = 24

            if "portfolio_augment_samples" in parameters:
                self.portfolio_augment_samples = parameters["portfolio_augment_samples"]
            else:
                self.portfolio_augment_samples = 120

            self.value_function = ValueFunction(deepcopy(self.portfolio), parameters)

        elif self.insurer_type == "Baseline":
            if "beta" in parameters:
                beta = parameters["beta"]
            else:
                beta = 0.02

            if "hill_coefficient" in parameters:
                hc = parameters["hill_coefficient"]
            else:
                hc = 1

            fn = lambda x, y: 1 + beta * (x**hc - y**hc) / (x**hc + y**hc + 1e-6)
            self.modulation_function = ModulationFunction(fn)

            self.value_function = ValueFunction(deepcopy(self.portfolio), parameters)

        elif self.insurer_type == "Null":
            pass
        else:
            raise TypeError(
                f"Insurer type should be either 'RL', 'Baseline', or 'Null'. Got type {self.insurer_type}."
            )

        # Create lists to store losses and profits
        self.losses = []
        self.profits = []
        self.profit_minus_losses = []

        # Maintain a list of the terminal portfolios we end up with for each burn-in epoch.
        self.terminal_rhos = []

        # Set the number of time steps, and the step size
        self.T = ExperimentConfig.epoch_customers
        self.dt = 1 / self.T
        self.t = 0

        # Set the bidding strategy
        self.bidding_strategy = bidding_strategy

        # We store customers in a deque of dataframes.
        self.customer_deque = deque(
            maxlen=InsurerConfig.retrain_epoch_num * ExperimentConfig.epoch_customers
        )

        # "num_bidding_points" is the number of datapoints on which the bidding model is trained.
        if "num_bidding_points" in parameters:
            self.num_bidding_points = parameters["num_bidding_points"]
        else:
            self.num_bidding_points = 500

        # Set training flag for market model
        self.bidding_model_trained = False
        self.market_model_trained = False
        self.conversion_model_trained = False

    def train_auxiliary_models(self):
        """
        Trains all the auxiliary models (i.e., the conversion, market, and bidding models) on available historic data.
        """
        designator = f"insurer_{self.idx}_mode_{self.mode}_epoch_{self.epoch}"

        complete_df = pd.concat(self.customer_deque, axis=0, ignore_index=True)
        complete_df.reset_index(drop=True, inplace=True)
        complete_df.fillna(0, inplace=True)

        conversion_cv_scores, conversion_accuracy = self.train_conversion_model(
            complete_df
        )
        mlflow.log_metric(
            f"conversion_cv_mean_accuracy_{designator}", conversion_cv_scores.mean()
        )
        mlflow.log_metric(f"conversion_accuracy_{designator}", conversion_accuracy)

        market_score = self.train_market_model(complete_df)
        mlflow.log_metric(f"market_r_squared_{designator}", market_score)

        bidding_cv_scores, bidding_r_squared = self.train_bidding_model(
            complete_df.sample(n=self.num_bidding_points, replace=True)
        )
        mlflow.log_metric(
            f"bidding_cv_mean_r_squared_{designator}", bidding_cv_scores.mean()
        )
        mlflow.log_metric(f"bidding_r_squared_{designator}", bidding_r_squared)

    def record_variables(self):
        """
        Log various variables related to model performance.
        """
        # Define the designator for recording.
        designator = f"insurer_{self.idx}_mode_{self.mode}_epoch_{self.epoch}"

        mlflow.log_metric(f"profit_{designator}", self.profit)
        mlflow.log_metric(f"conversion_rate_{designator}", self.conversion_rate())

        if self.insurer_type == "RL":
            mlflow.log_metric(
                f"k_perturbation_mean_{designator}",
                sum(self.value_function.k_perturbations)
                / len(self.value_function.k_perturbations),
            )

            mlflow.log_metric(
                f"k_perturbation_max_{designator}",
                max(self.value_function.k_perturbations),
            )

            non_zero = [x for x in self.value_function.k_perturbations if x != 0]
            mean_k_perturbation = sum(non_zero) / len(non_zero) if non_zero else 0

            mlflow.log_metric(
                f"k_perturbation_mean_non_zero_{designator}", mean_k_perturbation
            )

            mlflow.log_metric(
                f"k_perturbation_prop_non_zero_{designator}",
                len(non_zero) / len(self.value_function.k_perturbations),
            )

        if self.insurer_type != "Null":
            # Add the terminal portfolio to the list
            self.terminal_rhos.append(self.portfolio.get_representation())

            final_loss = self.value_function.loss(
                self.portfolio.get_representation(), self.value_function.target_rho
            )

            log_large_list(self.profits, f"profits_list_{designator}")
            log_large_list(self.losses, f"loss_list_{designator}")
            log_large_list(
                self.profit_minus_losses, f"profit_minus_losses_list_{designator}"
            )

            reward_for_epoch = self.profit - final_loss
            mlflow.log_metric(f"final_loss_{designator}", final_loss)
            mlflow.log_metric(f"reward_for_epoch_{designator}", reward_for_epoch)
            self.total_reward += reward_for_epoch
            mlflow.log_metric(f"total_reward_{designator}", self.total_reward)

            # Create plots

            plt.figure()
            plt.plot(range(len(self.losses)), self.losses)
            plt.xlabel("Time step")
            plt.ylabel("Loss value")
            plt.title("Epoch losses as a function of time")

            file_name = f"portfolio_losses_{designator}_{unique_id}.png"
            plot_path = os.path.join(current_dir, file_name)
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            plt.figure()
            plt.plot(range(len(self.profit_minus_losses)), self.profit_minus_losses)
            plt.xlabel("Time step")
            plt.ylabel("Profit minus loss")
            plt.title("Total profit minus loss over epoch")

            file_name = f"profit_minus_losses_{designator}_{unique_id}.png"
            plot_path = os.path.join(current_dir, file_name)
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

    def epoch_reset(self):
        """
        Resets the insurer at the end of each epoch.
        This includes emptying the profits, losses, and profits-minus-losses lists.
        We also empty the portfolio, reset various counters, and increment the epoch counter.
        """

        self.profits = []
        if self.insurer_type != "Null":

            self.terminal_rhos.append(self.portfolio.get_representation())
            self.losses = []
            self.profit_minus_losses = []
            self.portfolio.vector = np.zeros_like(self.portfolio.vector)
            self.initial_loss = self.value_function.loss(
                self.portfolio.get_representation(), self.value_function.target_rho
            )

        # Reset the profit to zero.
        self.profit = 0
        self.total_customers = 0
        self.successful_sales = 0
        self.t = 0

        self.epoch += 1

    @pa.check_io(out=CompleteSchema)
    def get_complete_dataset(self) -> DataFrame[CompleteSchema]:
        return pd.concat(self.customer_deque, axis=0, ignore_index=True)

    @pa.check_io(complete_df=CompleteSchema)
    def train_market_model(self, complete_df: DataFrame[CompleteSchema]) -> float:
        """
        Trains the market model, returning the r-squared score achieved by the model.

        Args:
            complete_df (DataFrame[CompleteSchema]): A dataframe from which we generate training data for the model.

        Returns:
            float: the r-squared score achieved by the market model.
        """
        market_complete_df = create_market_training_data(complete_df)
        score = self.market_model.train(market_complete_df)
        self.market_model_trained = True
        return score

    @pa.check_io(complete_df=CompleteSchema)
    def train_conversion_model(
        self, complete_df: DataFrame[CompleteSchema]
    ) -> Tuple[np.ndarray, float]:
        """
        Trains the conversion model, returning the cross-validation accuracy of the model and the overall accuracy.

        Args:
            complete_df (DataFrame[CompleteSchema]): A dataframe from which we generate training data for the model.

        Returns:
            Tuple[np.ndarray, float]: The cross-validation scores of the model, and the accuracy of the model on the training data.
        """
        conversion_complete_df = create_conversion_training_data(complete_df)
        cv_scores, accuracy = self.conversion_model.train(conversion_complete_df)
        self.conversion_model_trained = True
        return cv_scores, accuracy

    @pa.check_io(complete_df=CompleteSchema)
    def train_bidding_model(
        self, complete_df: DataFrame[CompleteSchema]
    ) -> Tuple[np.ndarray, float]:
        """
        Trains the bidding model, returning the cross-validation r-squred score and the overall r-squared score

        Args:
            complete_df (DataFrame[CompleteSchema]): A dataframe from which we generate training data for the model.

        Returns:
            Tuple[np.ndarray, float]: The cross-validation r-squared scores of the model, and the r-squared score of the model over its training data.
        """
        num_samples = len(complete_df)

        # Sample k-values with which to augment the dataset.
        k_values = np.random.laplace(loc=1, scale=0.1, size=num_samples).tolist()
        bidding_input_df = create_bidding_input_data(complete_df, k_values)

        # Use the conversion model to compute optimal actions for training the bidding model.
        actions = self.conversion_model.optimal_actions(bidding_input_df)
        bidding_complete_df = create_bidding_training_data(bidding_input_df, actions)

        # Train the model on the optimal actions (amortised optimisation), and set the trained flag to True.
        cv_scores, r_squared = self.bidding_model.train(bidding_complete_df)
        self.bidding_model_trained = True
        return cv_scores, r_squared

    def testing_mode(self):
        """
        Switch the insurer from burn-in mode to testing model.
            1. Reset the epoch counter
            2. Choose a portfolio (on the basis of historic portfolios) to pursue
        """

        # Set the mode of the insurer from "burn in" to "testing".
        self.mode = "testing"
        self.epoch = 0

        if self.insurer_type != "Null":
            # Find the average (terminal) portfolio vector during the burn-in period
            self.average_rho = np.mean(np.array(self.terminal_rhos), axis=0)
            self.target_rho = np.zeros_like(self.average_rho)

            for ii in range(len(self.average_rho)):
                if self.average_rho[ii] <= 10:
                    self.target_rho[ii] = np.floor(2 * self.average_rho[ii]) + 1
                else:
                    if np.random.random() < 0.5:
                        self.target_rho[ii] = np.floor(2 * self.average_rho[ii]) + 1
                    else:
                        self.target_rho[ii] = np.ceil(self.average_rho[ii] / 2)

            log_large_list(self.average_rho.tolist(), "average_portfolio")
            log_large_list(self.target_rho.tolist(), "target_portfolio")
            mlflow.log_metric(
                "avg_tar_loss",
                self.value_function.loss(self.average_rho, self.target_rho),
            )
            # Set the average portfolio:
            self.value_function.average_rho = self.average_rho
            # Set the target portfolio:
            self.value_function.target_rho = self.target_rho

        if self.insurer_type == "RL":
            complete_df = pd.concat(self.customer_deque, axis=0, ignore_index=True)
            complete_df.reset_index(drop=True, inplace=True)
            complete_df.fillna(0, inplace=True)

            # Trains the value function
            self.train_value_function(complete_df)

    @pa.check_io(complete_df=CompleteSchema)
    def train_value_function(self, complete_df: DataFrame[CompleteSchema]):
        """
        This function trains the value function for the RL agents.

        The training algorithm makes use of a customer reply buffer and a next-step value function U.

        The basic steps of the algorithm are as follows:
            1. Iterate backwards in time, from the final time-step to the first
            2. Use the Bellman recursion and the next-step value function to form value estimates for sampled portfolios at the current time step
            3. Fit U to the value estimates
            4. Use U to estimate the values of an additional set of portfolios
            5. Store all the value estimates for the current time step in a dataset, after recentering the values
            6. Once the backwards iteration has finished, train the value function V on the entire dataset


        Args:
            complete_df (DataFrame[CompleteSchema]): The dataframe used to train the value function
        """
        if self.insurer_type == "Null":
            return
        if self.insurer_type == "Baseline":
            return

        # Move the value function to train mode
        self.value_function.mode = "train"

        # Start the training run.
        start_time = time.time()

        value_df_list = []
        U_abs_avg_list = []
        U_avg_list = []
        U_var_list = []

        rho_var = []

        # Iterate backwards through time, starting at the final time step and proceeding backwards.
        for t in range(self.T - 1, 0, -1):

            # Sample a collection from rho-samples
            portfolio_samples, rho = self.value_function.sample_portfolios(
                self.portfolio_samples_per_step, t
            )

            rho_var.append(rho.std(axis=0).mean())

            tau = t / self.T
            next_tau = tau + self.dt

            # If the next step is the terminal time step, use the (negative) loss function as the value function.
            # Otherwise, use the next-step-value-function.
            if next_tau >= 1:
                next_values = (
                    -self.value_function.loss(rho, self.value_function.target_rho)
                    / self.T
                )
            else:
                next_values = self.value_function.next_step_model.predict(rho)

            rho_V = []

            # For each portfolio, form a value estimate.
            for portfolio_idx in range(len(portfolio_samples)):
                portfolio = portfolio_samples[portfolio_idx]

                sample_df = complete_df.sample(
                    n=min(len(complete_df), self.num_next_customer_samples)
                )
                C = sample_df["expected_cost"].values
                new_customer_df = sample_df[new_customer_columns]
                NewCustomerSchema.validate(new_customer_df)

                k_values = self.value_function.compute_k_values(
                    new_customer_df, portfolio, t
                )

                bidding_input_df = sample_df.assign(k=k_values)[bidding_input_columns]
                BiddingInputSchema.validate(bidding_input_df)

                actions = self.bidding_model.predict(bidding_input_df)

                conversion_input_df = sample_df.assign(action=actions)[
                    conversion_input_columns
                ]
                ConversionInputSchema.validate(conversion_input_df)

                probs = self.conversion_model.predict(conversion_input_df)

                rewards = C * probs * (actions - k_values)
                avg_reward = rewards.mean()

                # Apply the Bellman recursion relationship.
                U = next_values[portfolio_idx] + self.dt * avg_reward

                # Store the values in a dataset.
                rho_V.append({"rho": rho[portfolio_idx], "U": U.item(), "tau": tau})
            # Loop over portfolios ends.

            value_df = pd.DataFrame(rho_V)
            ValueCompleteSchema.validate(value_df)
            self.value_function.next_step_train(value_df)

            # Sample additional portfolios for the dataset.
            augment_rho = self.value_function.sample_portfolios(
                self.portfolio_augment_samples, t, return_portfolios=False
            )

            augment_values = self.value_function.next_step_model.predict(augment_rho)

            augment_rho_V = [
                {
                    "rho": augment_rho[portfolio_idx],
                    "U": augment_values[portfolio_idx],
                    "tau": tau,
                }
                for portfolio_idx in range(len(augment_values))
            ]

            rho_V.extend(augment_rho_V)
            value_df = pd.DataFrame(rho_V)
            ValueCompleteSchema.validate(value_df)

            # Having trained on the value_df (uncentered), we now center the values for later training.
            value_df["U"] = value_df["U"] - value_df["U"].mean()
            U_var_list.append(value_df["U"].var())
            value_df_list.append(value_df)
        # Loop over time steps ends.

        complete_value_df = pd.concat(value_df_list, ignore_index=True)
        ValueCompleteSchema.validate(complete_value_df)
        dataset_obtained_time = time.time()

        # Train the value function on the created dataset.
        r2, losses = self.value_function.train(complete_value_df)
        window_size = 10
        smoothed_losses = np.convolve(
            losses, np.ones(window_size) / window_size, mode="valid"
        )

        end_time = time.time()
        dataset_generation_duration = dataset_obtained_time - start_time
        total_train_duration = end_time - start_time
        value_function_train_duration = end_time - dataset_obtained_time

        # Define a designator for this set of metrics
        designator = f"epoch_{self.epoch}"

        mlflow.log_metric(f"value_function_r_squared_{designator}", r2)
        mlflow.log_metric(
            f"dataset_generation_duration_{designator}", dataset_generation_duration
        )
        mlflow.log_metric(
            f"value_function_train_duration_{designator}", value_function_train_duration
        )
        mlflow.log_metric(f"total_train_duration_{designator}", total_train_duration)

        # Plot the variance of the sampled portfolios
        plt.figure()
        plt.plot(range(1, len(rho_var) + 1), rho_var)
        plt.title("Standard deviation of portfolio samples")
        plt.xlabel("Reverse time step")
        plt.ylabel("Mean s.d.")

        file_name = f"portfolio_sample_std_{designator}_{unique_id}.png"
        plot_path = os.path.join(current_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        # Plot the loss curve of the value function
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label="Losses")
        plt.plot(
            range(window_size, len(losses) + 1),
            smoothed_losses,
            label="Smoothed losses",
        )
        plt.legend()
        plt.xlim([1, len(losses)])
        plt.ylim([0, 1.1 * max(losses)])
        plt.title("Losses for training value function")
        plt.xlabel("Gradient step")
        plt.ylabel("Loss")

        file_name = f"vf_loss_curve_{designator}_{unique_id}.png"
        plot_path = os.path.join(current_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        # Plot the average value.
        plt.figure()
        plt.plot(U_avg_list)
        plt.title("Average value across time")
        plt.xlabel("Reverse time step")
        plt.ylabel("Average value")

        file_name = f"average_value_{designator}_{unique_id}.png"
        plot_path = os.path.join(current_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        # Plot the distribution of normalised values as a function of time.
        plt.figure()
        plt.scatter(complete_value_df["tau"], complete_value_df["U"])
        plt.title("Distribution of training points")
        plt.xlabel("Time step")
        plt.ylabel("Recentred value")

        file_name = f"training_points_{designator}_{unique_id}.png"
        plot_path = os.path.join(current_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        # Plot the variance of U as a function of reverse time step.
        plt.figure()
        plt.plot(U_var_list)
        plt.title("Variance of value across time")
        plt.xlabel("Reverse time step")
        plt.ylabel("Variance")

        file_name = f"value_variance_{designator}_{unique_id}.png"
        plot_path = os.path.join(current_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        plt.figure()
        plt.errorbar(
            range(len(self.value_function.mean_cv_scores)),
            self.value_function.mean_cv_scores,
            yerr=self.value_function.std_cv_scores,
        )
        plt.xlabel("Reverse time step")
        plt.ylabel("R squared score")
        plt.title("CV scores for LR model")

        file_name = f"cv_scores_{designator}_{unique_id}.png"
        plot_path = os.path.join(current_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        mlflow.log_metric(
            f"mean_nsm_cv_{designator}",
            sum(self.value_function.mean_cv_scores)
            / len(self.value_function.mean_cv_scores),
        )

        self.value_function.mode = "inference"

    def store_customers(
        self,
        customer_profiles: List[Dict],
        expected_costs: np.ndarray,
        offers: np.ndarray,
        responses: np.ndarray,
    ):
        """
        Stores customer(s) in the container of the insurer

        Args:
            customer_profile (List[dict]): The features for the customer(s)
            expected_costs (np.ndarray): A numpy array of shape (num_customers) which gives the expected cost of serving each customer
            offers (np.ndarray): A numpy array of shape (num_customers x num_insurers) which gives the offers made to the customer by each insurer in the market
            responses (np.ndarray): A numpy array of shape (num_customers) which gives whether the customer accepted the offer made by the insurer or not.
        """
        # Update the number of customers the insurer has seen
        self.time_steps += 1
        self.total_customers += len(customer_profiles)
        self.t = self.total_customers
        # Update the successful sales counter
        self.successful_sales += sum(responses)

        # Create a copy of the customer profile that includes the price and response
        to_be_added = pd.get_dummies(
            pd.DataFrame(customer_profiles), columns=["Occupation", "Location"]
        )

        # Save the expected cost of serving each customer
        to_be_added["expected_cost"] = expected_costs
        # Save the offers we made to the customer
        to_be_added["action"] = offers[:, self.idx] / expected_costs
        # Save the response of the customer
        to_be_added["response"] = responses

        # Add additional columns for the other one-hot encoded variables, if they are missing
        for column in complete_columns:
            if column not in to_be_added:
                to_be_added[column] = 0  # Add missing column with 0s
        # Reorder the columns to be the same as in the training data
        to_be_added = to_be_added[complete_columns]

        # Store the profit made from the customer(s)
        self.profit += (
            to_be_added["response"]
            * (to_be_added["action"] - 1)
            * to_be_added["expected_cost"]
        ).sum()

        # Remove our offer and sort the other offers
        other_offers = np.delete(offers, self.idx, axis=1)
        sorted_offers = np.sort(other_offers, axis=-1)

        # Now save the market variables
        if "Top 1" in market_output_columns:
            to_be_added["Top 1"] = sorted_offers[:, 0] / to_be_added["expected_cost"]
        if "Top 3" in market_output_columns:
            to_be_added["Top 3"] = (
                np.mean(sorted_offers[:, :3], axis=1) / to_be_added["expected_cost"]
            )
        if "Top 5" in market_output_columns:
            to_be_added["Top 5"] = (
                np.mean(sorted_offers[:, :5], axis=1) / to_be_added["expected_cost"]
            )
        if "Top 10" in market_output_columns:
            to_be_added["Top 10"] = (
                np.mean(sorted_offers[:, :10], axis=1) / to_be_added["expected_cost"]
            )

        # Store this in the customer list
        self.customer_deque.append(to_be_added)

        self.profits.append(self.profit)

        if self.insurer_type != "Null":
            # Store the current portfolio in the value function:
            self.portfolio.add_customers(
                to_be_added[to_be_added["response"] == 1].copy()
            )

            self.loss = self.value_function.loss(
                self.portfolio.get_representation(), self.value_function.target_rho
            )
            self.losses.append(self.loss)

            self.profit_minus_loss = self.profit - self.loss
            self.profit_minus_losses.append(self.profit_minus_loss)

    def make_offers(
        self, customer_profiles: List[Dict], expected_costs: np.ndarray
    ) -> np.ndarray:
        """
        Makes offers to a list of customers

        Args:
            customer_profiles (List[dict]): A list of dictionaries, where each dictionary represents a customer.
            expected_costs (np.ndarray): A numpy array, where each element represents the expected cost of serving a customer.

        Returns:
            np.ndarray: A numpy array, where each element represents the offer made to a customer.
        """
        # If the bidding model has not been trained then return an offer at random
        if not (self.bidding_model_trained and self.market_model_trained):
            # A multiplier is selected uniformly and at random between 1 and 1.2
            multipliers = 1 + 0.2 * np.random.rand(*expected_costs.shape)
            return multipliers * expected_costs

        market_input_df = pd.get_dummies(
            pd.DataFrame(customer_profiles), columns=["Occupation", "Location"]
        )

        # Add additional columns for the other one-hot encoded variables, if they are missing
        for column in market_input_columns:
            if column not in market_input_df:
                market_input_df[column] = 0  # Add missing column with 0s
        # Reorder the columns to be the same as in the training data
        market_input_df = market_input_df[market_input_columns]

        MarketInputSchema.validate(market_input_df)
        market_predictions = self.market_model.predict(market_input_df)

        # Compute the k_values
        new_customer_df = market_input_df.assign(expected_cost=expected_costs)
        NewCustomerSchema.validate(new_customer_df)

        if self.insurer_type == "RL":
            k_values = self.value_function.compute_k_values(
                new_customer_df, self.portfolio, t=self.t, store=True
            )
        else:
            k_values = [1] * len(new_customer_df)

        bidding_input_df = market_input_df.assign(k=k_values)
        for ii in range(len(market_output_columns)):
            col_name = market_output_columns[ii]
            bidding_input_df[col_name] = market_predictions[:, ii]
        bidding_input_df = bidding_input_df[bidding_input_columns]

        BiddingInputSchema.validate(bidding_input_df)

        actions = self.bidding_model.predict(bidding_input_df)

        bids = actions * expected_costs

        # Now we implement the baseline bidding strategy
        if self.insurer_type == "Baseline" and self.mode == "testing":
            inclusion = (
                self.portfolio.represent_hypothetical(new_customer_df)
                - self.portfolio.get_representation()
            )
            modulation = self.modulation_function(
                inclusion, self.average_rho, self.target_rho
            )

            bids = bids * modulation

        if self.bidding_strategy == "epsilon-greedy" and (
            self.mode == "burn-in" or self.insurer_type != "RL"
        ):
            # Implement epsilon greedy exploration
            multipliers = np.random.choice(
                np.array([0.93, 0.95, 0.97, 1, 1.03, 1.05, 1.07]),
                size=bids.shape,
                p=np.array([0.02, 0.02, 0.02, 0.88, 0.02, 0.02, 0.02]),
            )
            # multiply the bids by the multiplier
            bids = multipliers * bids

        # Return the bids
        return bids


################################################################################################


class MarketModelBase(ABC):
    @pa.check_io(market_input_df=MarketInputSchema)
    @abstractmethod
    def predict(
        self, market_input_df: DataFrame[MarketInputSchema]
    ) -> List[List[float]]:
        pass

    @abstractmethod
    @pa.check_io(market_complete_df=MarketCompleteSchema)
    def train(self, market_complete_df: DataFrame[MarketCompleteSchema]):
        pass


class MarketModel(MarketModelBase):
    def __init__(self, model_type: str = "Random Forest", parameters=None):
        """
        Initialises the market model class.
        The market model takes in descriptions of customers and predicts the average Top 5 price offered to the customer by the market.

        Args:
            model_type (str, optional): The type of ML model used for the market model. Defaults to "Random Forest".
        """
        if model_type == "Random Forest":
            self.model = RandomForestRegressor()

        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

    def predict(
        self, market_input_df: DataFrame[MarketInputSchema]
    ) -> List[List[float]]:
        """
        Makes predictions using the market model

        Args:
            market_input_df (DataFrame[MarketInputSchema]): The customer features from which predictions are to be made

        Returns:
            List[List[float]]: For each customer, a prediction of the market variables for that customer.
        """
        return self.model.predict(market_input_df)

    def train(self, market_complete_df: DataFrame[MarketCompleteSchema]) -> float:
        """
        Trains the market model

        Args:
            market_complete_df (DataFrame[MarketCompleteSchema]): A dataset containing both the customer features and market variables.

        Returns:
            float: The r-squared score of the market model.
        """
        # Get the feature vectors
        self.X = market_complete_df[market_input_columns]
        # Get the target variables
        self.Y = market_complete_df[market_output_columns]

        # Fit the model
        self.model.fit(self.X, self.Y)

        return self.model.score(self.X, self.Y)


################################################################################################


class ConversionModelBase(ABC):
    @pa.check_io(conversion_input_df=ConversionInputSchema)
    @abstractmethod
    def predict(
        self, conversion_input_df: DataFrame[ConversionInputSchema]
    ) -> List[float]:
        pass

    @pa.check_io(conversion_complete_df=ConversionCompleteSchema)
    @abstractmethod
    def train(self, conversion_complete_df: DataFrame[ConversionCompleteSchema]):
        pass

    @pa.check_io(bidding_input_df=BiddingInputSchema)
    @abstractmethod
    def optimal_actions(
        self, bidding_input_df: DataFrame[BiddingInputSchema]
    ) -> List[float]:
        pass


class ConversionModel(ConversionModelBase):
    def __init__(self, parameters=None):

        if "conversion_model_type" in parameters:
            self.model_type = parameters["conversion_model_type"]
        else:
            self.model_type = "MLP"

        if self.model_type == "Random Forest":
            self.model = RandomForestClassifier(
                criterion="log_loss", class_weight="balanced"
            )
        elif self.model_type == "Decision Tree":
            self.model = DecisionTreeClassifier(
                criterion="log_loss", class_weight="balanced"
            )
        elif self.model_type == "Boosted Decision Tree":
            base_classifier = DecisionTreeClassifier(
                criterion="log_loss", class_weight="balanced", max_depth=3
            )
            self.model = AdaBoostClassifier(
                estimator=base_classifier, n_estimators=100, algorithm="SAMME.R"
            )
        elif self.model_type == "Logistic Regression":
            self.model = LogisticRegression(
                class_weight="balanced",
                C=2.5,
                solver="newton-cholesky",
                max_iter=2500,
            )
        elif self.model_type == "Gaussian Process":
            self.model = GaussianProcessClassifier()

        elif self.model_type == "MLP":

            if "conversion_depth" in parameters:
                self.conversion_depth = parameters["conversion_depth"]
            else:
                self.conversion_depth = 3

            if "conversion_width" in parameters:
                self.conversion_width = parameters["conversion_width"]
            else:
                self.conversion_width = 512

            if "conversion_lr" in parameters:
                self.conversion_lr = parameters["conversion_lr"]
            else:
                self.conversion_lr = 0.0003

            self.model = MLPClassifier(
                hidden_layer_sizes=[self.conversion_width] * self.conversion_depth,
                max_iter=4000,
                learning_rate_init=self.conversion_lr,
            )

        # Create dataframes that will store the data used to train our model.
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

    def predict(
        self, conversion_input_df: DataFrame[ConversionInputSchema]
    ) -> List[float]:
        return self.model.predict_proba(conversion_input_df)[:, 1]

    def train(
        self, conversion_complete_df: DataFrame[ConversionCompleteSchema]
    ) -> Tuple[float, np.ndarray]:
        """
        Trains the conversion model.

        Args:
            conversion_complete_df (DataFrame[ConversionCompleteSchema]): A dataset containing both market variables and the insurer action as inputs, and whether the customer accepted the offer as an output.

        Returns:
            Tuple[float, np.ndarray]: The cross-validation accuracy of the conversion model, and the accuracy of the model over the training data.
        """
        # Get the input variables
        self.X = conversion_complete_df[conversion_input_columns]

        # Get the response variable
        self.Y = (
            conversion_complete_df[conversion_output_columns]
            .astype(int)
            .to_numpy()
            .ravel()
        )

        # Get the cross validation scores for the model
        cv_scores = cross_val_score(self.model, self.X, self.Y)

        # Fit the model
        self.model.fit(self.X, self.Y)

        return cv_scores, self.model.score(self.X, self.Y)

    def optimal_actions(
        self, bidding_input_df: DataFrame[BiddingInputSchema]
    ) -> List[float]:
        """
        Given a trained conversion model, this function computes the optimal actions.

        Args:
            bidding_input_df (DataFrame[BiddingInputSchema]): The input values to the bidding model.

        Returns:
            List[float]: The optimal actions, according to the conversion model.
        """

        actions = []

        # For each set of input variables, compute the optimal action
        for index, row in bidding_input_df.iterrows():
            k = row["k"]
            inputs = row.drop("k").reset_index().T
            inputs.columns = inputs.iloc[0]
            inputs = inputs.drop(inputs.index[0])

            # Use scipy's function for finding the optimal action
            res = scipy.optimize.minimize_scalar(
                lambda a: (k - a) * self.predict(inputs.assign(action=a)),
                bounds=[1, 2],
            )
            if res.success:
                actions.append(float(res.x))
            else:
                actions.append(1.1)

        return actions


################################################################################################


class BiddingModelBase(ABC):
    @pa.check_io(bidding_input_df=BiddingInputSchema)
    @abstractmethod
    def predict(self, bidding_input_df: DataFrame[BiddingInputSchema]) -> List[float]:
        pass

    @pa.check_io(bidding_complete_df=BiddingCompleteSchema)
    @abstractmethod
    def train(self, bidding_complete_df: DataFrame[BiddingCompleteSchema]):
        pass


class BiddingModel(BiddingModelBase):
    def __init__(self, parameters=None):

        if "bidding_model_type" in parameters:
            self.model_type = parameters["bidding_model_type"]
        else:
            self.model_type = "GP-Mat"

        mlflow.log_param("bidding_model_type", self.model_type)

        if self.model_type == "Linear Regression":
            self.model = LinearRegression()
        elif self.model_type == "GP-RBF":
            self.model = Pipeline(
                [
                    ("normalise", StandardScaler()),
                    (
                        "gpr",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel() * RBF(),
                            n_restarts_optimizer=50,
                            normalize_y=True,
                        ),
                    ),
                ]
            )
        elif self.model_type == "GP-Mat":
            self.model = Pipeline(
                [
                    ("normalise", StandardScaler()),
                    (
                        "gpr",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel() * Matern(),
                            n_restarts_optimizer=50,
                            normalize_y=True,
                        ),
                    ),
                ]
            )
        elif self.model_type == "GP-RQ":
            self.model = Pipeline(
                [
                    ("normalise", StandardScaler()),
                    (
                        "gpr",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel() * RationalQuadratic(),
                            n_restarts_optimizer=50,
                            normalize_y=True,
                        ),
                    ),
                ]
            )
        elif self.model_type == "SVR":
            self.model = Pipeline(
                [
                    ("normalise", StandardScaler()),
                    (
                        "svr",
                        SVR(),
                    ),
                ]
            )
        elif self.model_type == "Kernel Ridge":
            self.model = Pipeline(
                [
                    ("normalise", StandardScaler()),
                    ("kernel_ridge", KernelRidge(kernel="rbf")),
                ]
            )
        elif self.model_type == "Ridge":
            self.model = Pipeline(
                [("normalise", StandardScaler()), ("ridge_regressor", Ridge())]
            )
        elif self.model_type == "MLP":
            self.model = Pipeline(
                [("normalise", StandardScaler()), ("mlp", MLPRegressor())]
            )

        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

    def predict(self, bidding_input_df: DataFrame[BiddingInputSchema]) -> List[float]:
        """
        Generates actions given input variables

        Args:
            bidding_input_df (DataFrame[BiddingInputSchema]): The market variables and k-values of the actions to predict

        Returns:
            List[float]: The actions returned by the bidding model
        """
        return np.clip(self.model.predict(bidding_input_df), a_min=1, a_max=2)

    def train(
        self, bidding_complete_df: DataFrame[BiddingCompleteSchema]
    ) -> Tuple[np.ndarray, float]:
        """
        Trains the bidding model.

        Args:
            bidding_complete_df (DataFrame[BiddingCompleteSchema]): Input variables to the bidding model, along with the optimal actions.

        Returns:
            Tuple[np.ndarray, float]: The cross-validation r-squared scores of the bidding model, along with the r-squared score over the training data.
        """

        # Get the input variables
        self.X = bidding_complete_df[bidding_input_columns]

        # Get the response variable
        self.Y = bidding_complete_df[bidding_output_columns].to_numpy().ravel()

        # Find the cross-validation score
        cv_scores = cross_val_score(self.model, self.X, self.Y)

        # Fit the model
        self.model.fit(self.X, self.Y)

        # Compute the r-squared value of the bidding model across the entire dataset
        r_squared = self.model.score(self.X, self.Y)

        return cv_scores, r_squared


##########################################################
class PortfolioBase(ABC):
    @abstractmethod
    def add_customers(self, customers: pd.DataFrame):
        pass

    @abstractmethod
    def get_representation(self) -> np.ndarray:
        pass

    @pa.check_io(customers=NewCustomerSchema)
    @abstractmethod
    def represent_hypothetical(
        self, customers: DataFrame[NewCustomerSchema]
    ) -> np.ndarray:
        pass


class ValueFunctionBase(ABC):
    @abstractmethod
    def sample_k_values(self, num_samples: int) -> List[float]:
        pass

    @abstractmethod
    def sample_portfolios(
        self, num_samples: int, time_step: int, return_portfolios: bool = True
    ) -> List[PortfolioBase]:
        pass

    @pa.check_io(new_customer_df=NewCustomerSchema)
    @abstractmethod
    def compute_k_values(
        self,
        new_customer_df: DataFrame[NewCustomerSchema],
        portfolio: np.ndarray,
        t: float,
        store: bool = False,
    ) -> List[float]:
        pass

    @pa.check_io(value_df=ValueCompleteSchema)
    @abstractmethod
    def train(self, value_df: DataFrame[ValueCompleteSchema]):
        pass


class ValueFunction(ValueFunctionBase):
    def __init__(self, target_portfolio, parameters=None):
        self.loss = L1Loss()

        if "value_lr" in parameters:
            self.value_lr = parameters["value_lr"]
        else:
            self.value_lr = 1e-5

        if "value_width" in parameters:
            self.value_width = parameters["value_width"]
        else:
            self.value_width = 1024

        if "value_depth" in parameters:
            self.value_depth = parameters["value_depth"]
        else:
            self.value_depth = 4

        if "value_train_epochs" in parameters:
            self.value_train_epochs = parameters["value_train_epochs"]
        else:
            self.value_train_epochs = 6

        if "value_weight_decay" in parameters:
            self.value_weight_decay = parameters["value_weight_decay"]
        else:
            self.value_weight_decay = 1e-7

        if "slope" in parameters:
            self.up_slope = 1 + parameters["slope"]
            self.down_slope = 1 - parameters["slope"]
        else:
            self.up_slope = 1.9
            self.down_slope = 0.1

        if "gpu_id" in parameters:
            self.gpu_id = parameters["gpu_id"]
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.mode = "inference"  # Mode should always be either inference or train

        # Define the model
        layers = []

        self.target_portfolio = target_portfolio
        self.target_rho = target_portfolio.vector
        self.num_categories = len(self.target_rho)

        # Input layer
        layers.append(nn.Linear(self.num_categories + 1, self.value_width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(self.value_depth - 1):
            layers.append(nn.Linear(self.value_width, self.value_width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(self.value_width, 1))

        # Build the model
        self.model = nn.Sequential(*layers)
        # Place the model on the correct device
        self.model.to(self.device)

        # Use a linear model for the next-step-value function
        self.next_step_model = LinearRegression()

        # Use the Adam optimiser
        self.optimiser = optim.Adam(
            params=self.model.parameters(),
            lr=self.value_lr,
            weight_decay=self.value_weight_decay,
        )

        self.T = ExperimentConfig.epoch_customers
        self.dt = 1 / self.T

        self.k_perturbations = deque(maxlen=self.T)
        self.k_perturbations.append(0)

        self.k_samples = deque(maxlen=self.T)

        self.portfolio_samples = deque(maxlen=self.T)
        self.rep_samples = set()

        self.mean_cv_scores = deque(maxlen=self.T)
        self.std_cv_scores = deque(maxlen=self.T)

        self.rho_norm = ExperimentConfig.epoch_customers * MarketConfig.success_rate

        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()

        self.model_trained = False

    def sample_portfolios(
        self, num_samples: int, time_step: int, return_portfolios: bool = True
    ) -> Tuple[List[PortfolioBase], np.ndarray]:
        """
        This method samples a collection of portfolios (and their frequency representations) to train the value function on

        Args:
            num_samples (int): The number of portfolio samples to generate.
            time_step (int): Which time-step the portfolios are generated for.
            return_portfolios (bool, optional): Whether or not to return the portfolios in addition to their frequency representations. Defaults to True.

        Returns:
            Tuple[List[PortfolioBase], np.ndarray]: The list of portfolios, and their frequency representations
        """
        samples = set()

        # First sample all possible portfolios if the time step is low enough.
        if (time_step + 1) ** self.num_categories < num_samples:
            samples.update(
                itertools.product(range(time_step + 1), repeat=self.num_categories)
            )
        else:
            # Otherwise, sample portfolios by interpolating between the min and max rate portfolios
            self.target_rate = (
                self.target_rho + np.ones_like(self.target_rho)
            ) / self.T
            self.average_rate = (
                self.average_rho + np.ones_like(self.average_rho)
            ) / self.T
            self.max_rate = (
                self.up_slope * np.maximum(self.average_rho, self.target_rho) / self.T
            )
            self.min_rate = (
                self.down_slope * np.minimum(self.average_rho, self.target_rho) / self.T
            )

            for ii in range(int(num_samples / 3)):
                u = ii / (int(num_samples / 2) - 1)
                rate = u * self.min_rate + (1 - u) * self.max_rate
                rho = normalise_rho(rate * time_step)
                samples.add(tuple(rho))

            seg_samples = len(samples)

            # Compute the number of remaining samples we have to take
            remaining_samples = num_samples - seg_samples

            # Now allocate the remaining samples to binomial draws around the average and target portfolios
            while len(samples) <= seg_samples + remaining_samples / 2:
                samples.add(tuple(np.random.binomial(n=time_step, p=self.average_rate)))

            while len(samples) < num_samples:
                samples.add(tuple(np.random.binomial(n=time_step, p=self.target_rate)))

        # Convert the set back to a numpy array
        sample_rho = np.array(list(samples))

        if not return_portfolios:
            return sample_rho
        else:
            output_list = []
            for ii in range(len(samples)):
                new_portfolio = deepcopy(self.target_portfolio)
                new_portfolio.vector = sample_rho[ii, :]
                output_list.append(new_portfolio)
            return output_list, sample_rho

    def sample_k_values(self, num_samples: int) -> List[float]:
        return random.sample(list(self.k_samples), num_samples)

    def compute_k_values(
        self,
        new_customer_df: DataFrame[NewCustomerSchema],
        portfolio: PortfolioBase,
        t: int,
        store: bool = False,
    ) -> np.ndarray:
        """
        Computes k-values for a set of customers using the value function or next-step value function

        Args:
            new_customer_df (DataFrame[NewCustomerSchema]): A dataframe containing customer features
            portfolio (PortfolioBase): The current portfolio
            t (int): The time step at which we are computing the k-value. Note that this returns k(t+1)
            store (bool, optional): Whether to store the k-value in an internal deque object. Defaults to False.

        Returns:
            np.ndarray: The k-values for these customers.
        """

        C = new_customer_df["expected_cost"].values
        rho_plus = portfolio.represent_hypothetical(new_customer_df)
        rho = portfolio.get_representation().reshape(1, -1)

        num_customers = len(new_customer_df)

        # First, if the next time step is greater than one, apply the boundary condition.
        next_tau = self.dt * (t + 1)
        if next_tau >= 1:
            dV = self.loss(rho_plus, self.target_rho) - self.loss(rho, self.target_rho)
        else:
            # Otherwise, separate out depending on whether we're in inference mode or train mode
            if self.mode == "inference":
                # Provided the model is trained, use that.
                if self.model_trained:
                    with torch.no_grad():
                        dU = self.model(
                            torch.tensor(
                                np.concatenate(
                                    [
                                        rho_plus / self.rho_norm,
                                        next_tau
                                        * np.ones(num_customers).reshape(-1, 1),
                                    ],
                                    axis=1,
                                ),
                                dtype=torch.float32,
                            ).to(self.device)
                        ) - self.model(
                            torch.tensor(
                                np.concatenate(
                                    [
                                        repeat(
                                            rho.squeeze() / self.rho_norm,
                                            "num_cat -> num_customers num_cat",
                                            num_customers=num_customers,
                                        ),
                                        next_tau
                                        * np.ones(num_customers).reshape(-1, 1),
                                    ],
                                    axis=1,
                                ),
                                dtype=torch.float32,
                            ).to(self.device)
                        )
                    dU = dU.cpu().numpy()
                    dU = dU.squeeze()
                # If the model is not trained, just set dU equal to zero
                else:
                    dU = 0
            # If we're in train mode, use the next-step-model
            elif self.mode == "train":
                dU = self.next_step_model.predict(
                    rho_plus
                ) - self.next_step_model.predict(rho.reshape(1, -1))
            else:
                print(
                    "Error! Value function mode should be either 'train' or 'inference'"
                )

            # Compute dV
            dV = self.T * dU

        k_perturbation = -dV / C

        k_values = 1 - dV / C

        # Manually set all the k_values for which d_rho = 0 to 1.
        rows_all_zeros = np.all(rho == rho_plus, axis=1)
        k_values[rows_all_zeros] = 1

        if store:
            self.k_samples.extend(k_values)
            self.k_perturbations.extend(abs(k_perturbation))

        return k_values

    def next_step_train(self, value_df: DataFrame[ValueCompleteSchema]):
        """
        Trains the next-step value function

        Args:
            value_df (DataFrame[ValueCompleteSchema]): A dataset of portfolio description, value pairs.
        """
        X = np.array(value_df["rho"].tolist())
        Y = np.array(value_df["U"].values)

        cvs = cross_val_score(self.next_step_model, X, Y.ravel())

        # Store the mean and standard deviation of the cross-validation scores
        self.mean_cv_scores.append(cvs.mean())
        self.std_cv_scores.append(cvs.std())
        self.next_step_model.fit(X, Y.ravel())

    def train(
        self, value_df: DataFrame[ValueCompleteSchema]
    ) -> Tuple[float, List[float]]:
        """
        Trains the value function.

        Args:
            value_df (DataFrame[ValueCompleteSchema]): The data set used to train the value function

        Returns:
            Tuple[float, List[float]]: The r-squared score of the value function, along with the losses that we achieved overtraining.
        """
        value_dataset = ValueDataset(value_df)
        dataloader = DataLoader(value_dataset, shuffle=True, batch_size=128)

        # Switch the base network into train mode
        self.model.train()

        # Ensure that the model is on the correct device
        self.model.to(self.device)

        losses = []

        for ii in range(self.value_train_epochs):
            for X, Y in dataloader:

                # Move X and Y to the gpu
                X = X.to(self.device)
                Y = Y.to(self.device)

                output = self.model(X).squeeze()
                loss = F.mse_loss(output, Y)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                losses.append(loss.detach().item())

        self.model_trained = True

        # Now we evaluate the model.

        all_true = []
        all_pred = []

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for X, Y in dataloader:
                # Move X and Y to the GPU
                X = X.to(self.device)
                Y = Y.to(self.device)

                # Get model predictions
                output = self.model(X).squeeze()

                # Store the true values and predictions
                all_true.append(Y.cpu().numpy())
                all_pred.append(output.cpu().numpy())

        # Convert lists to numpy arrays
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)

        # Compute R^2 score
        r2 = r2_score(all_true, all_pred)

        return r2, losses

    def train_mse(self):
        err = self.model.predict(self.X) - self.Y
        return (err * err).mean()


class ValueDataset(Dataset):
    """
    The value dataset is used for training the value function.
    """

    @pa.check_io(value_df=ValueCompleteSchema)
    def __init__(self, value_df: DataFrame[ValueCompleteSchema]):
        # Store the number of datapoints
        self.length = len(value_df)

        # Create the torch tensor for the target values
        self.targets = torch.tensor(value_df["U"].values, dtype=torch.float32)

        # Create the normalisation constant for the portfolios
        self.rho_norm = ExperimentConfig.epoch_customers * MarketConfig.success_rate
        # Create the torch tensor for the input features
        self.features = torch.tensor(
            np.concatenate(
                [
                    np.array(value_df["rho"].tolist()) / self.rho_norm,
                    np.array(value_df["tau"].tolist()).reshape(-1, 1),
                ],
                axis=1,
            ),
            dtype=torch.float32,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx])


class Category:
    """
    Categories correspond to sets that define our portfolio.
    Each category has a number of features, and constraints on those features.
    These constraints are expressed as lower and upper bounds on the feature values.
    """

    def __init__(self, constraints: Dict[str, Tuple]):
        # Constraints is a dictionary.
        # The keys are names of customer features
        # The values are tuples containing (inclusive) lower and upper values that the feature must lie between
        self.constraints = constraints

    def get_text_description(self):
        return pformat(self.constraints)

    def check_inclusion(self, customers: pd.DataFrame) -> pd.Series:
        """
        Given a set of customers, checks whether each customer is in the category or not.

        Args:
            customers (pd.DataFrame): The set of customers whose inclusion we wish to check.

        Returns:
            pd.Series: A Series object with length equal to the number of customers, whose entires are True whenever the customer is in the category and false otherwise.
        """
        # Start with a mask where all rows are initially considered
        mask = pd.Series(True, index=customers.index)

        # Update the mask to reflect criteria for each column
        for column, (lower, upper) in self.constraints.items():
            if column not in customers:
                customers[column] = 0
            mask &= (customers[column] >= lower) & (customers[column] <= upper)

        # Return a mask which is True whenever the customer satisfies the criteria, and zero otherwise.
        return mask


def generate_random_portfolio(num_categories: int) -> PortfolioBase:
    """
    This function randomly generates a portfolio.
    It gives the categories for the portfolio, but not their frequency representation.
    The target frequencies are set by the insurer.

    Args:
        num_categories (int): The number of categories in the portfolio.

    Returns:
        PortfolioBase: The randomly generated portfolio
    """
    # We will first choose columns uniformly and at random from the list of customer columns
    selected_columns = random.sample(customer_columns, num_categories)

    # Separate out binary and continuous variables
    binary_columns = ["Marital status", "Child indicator"]
    binary_columns += [f"Occupation_{ii}" for ii in range(CustomerConfig.num_occ)]
    binary_columns += [f"Location_{ii}" for ii in range(CustomerConfig.num_loc)]

    # Initialise an empty list to keep the categories in
    categories = []
    for col in selected_columns:
        # Initialise an empty dictionary in which to store the constraints
        constraints = {}

        # First consider the binary column case
        if col in binary_columns:
            upper = random.randint(0, 1)
            lower = upper

        # Now consider each of the continuous column cases separately
        elif col == "Car valuation":
            # Bulk of the mass is between 4000 and 24000, with a long tail

            samples = np.random.exponential(scale=10000, size=2)
            # Set lower to the smaller of the two samples and upper to the larger
            lower = 100 * int(min(samples) / 100 + 30)
            upper = 100 * int(max(samples) / 100 + 30)

        elif col == "Number of previous claims":
            # This is mostly between 0 and 4
            samples = np.random.poisson(lam=1.4, size=2)
            # Set lower to the smaller of the two samples and upper to the larger
            lower = min(samples)
            upper = max(samples)

        elif col == "Years driving":
            # This is between 0 and 84
            lower = random.randint(0, 40)
            upper = lower + random.randint(10, 30)

        elif col == "Age":
            # This is between 16 and 100
            lower = random.randint(16, 80)
            upper = lower + random.randint(5, 20)

        constraints[col] = (lower, upper)
        categories.append(Category(constraints))

    return QuantityPortfolio(categories)


def normalise_rho(unnormalised_rho: np.ndarray) -> np.ndarray:
    """
    Helper function.
    This takes in a frequency represention which has non-integer entries and returns one with integer entries.
    It does this by taking the integer part of each entry, and then adding on a Bernoulli random variable which is one with probability equal to the remainder.

    Args:
        unnormalised_rho (np.ndarray): The frequency representation that we wish to normalise.

    Returns:
        np.ndarray: The (randomly generated) normalised frequency representation.
    """
    integer_part = np.floor(unnormalised_rho)
    remainder = unnormalised_rho - integer_part
    bernoulli_samples = np.random.random(unnormalised_rho.shape) < remainder
    rho = integer_part + bernoulli_samples
    return rho


class QuantityPortfolio(PortfolioBase):
    def __init__(self, categories: List[Category]):
        self.num_categories = len(categories)
        self.categories = categories
        self.vector = np.zeros(self.num_categories, dtype=np.float64)

    def get_text_description(self):
        representations = []
        for category in self.categories:
            desc = category.get_text_description()
            representations.append(desc)
        description = "\n".join(representations)
        return description

    def add_customers(self, customers: pd.DataFrame):
        """
        Adds customers to the portfolio.

        Args:
            customers (pd.DataFrame): The customers to add.
        """
        for ii in range(self.num_categories):
            self.vector[ii] += self.categories[ii].check_inclusion(customers).sum()

    def get_representation(self) -> np.ndarray:
        """
        Gives the frequency representation for the portfolio.

        Returns:
            np.ndarray: The frequency representation of the portfolio.
        """
        return self.vector

    def represent_hypothetical(self, customers: pd.DataFrame) -> np.ndarray:
        """
        Takes in a collection of customers.
        For each customer, it computes the frequency representation that we would have, if that customer were added.
        Returns these hypothetical frequency representations.

        Args:
            customers (pd.DataFrame): A set of customers, each of whom we wish to find the hypothetical frequency representation for.

        Returns:
            np.ndarray: The hypothetical frequency representations that we would have, if each customer were added.
        """
        num_customers = len(customers)

        rho_representations = repeat(
            self.vector, "num_cat -> num_customers num_cat", num_customers=num_customers
        ).astype(np.float64)
        mask = np.zeros((num_customers, self.num_categories), dtype=np.float64)
        for ii in range(self.num_categories):
            mask[:, ii] = self.categories[ii].check_inclusion(customers)

        rho_representations += mask.astype(np.float64)
        return rho_representations


class ProportionPortfolio(PortfolioBase):
    def __init__(self, categories: List[Category]):
        self.num_categories = len(categories)
        self.categories = categories
        self.frequencies = np.zeros(self.num_categories)
        self.total_number = 0

    def add_customers(self, customers: pd.DataFrame):
        self.total_number += len(customers)
        for ii in range(self.num_categories):
            self.frequencies[ii] += self.categories[ii].check_inclusion(customers).sum()

    def get_representation(self):
        if self.total_number == 0:
            return 0
        else:
            return self.frequencies / self.total_number


class LossBase(ABC):
    @abstractmethod
    def __call__(self, rho: np.ndarray, target_rho: np.ndarray) -> np.ndarray:
        pass


class L1Loss(LossBase):
    def __init__(self):
        self.loss_coefficient = 2000

    def __call__(self, rho: np.ndarray, target_rho: np.ndarray) -> np.ndarray:
        """
        Computes the loss. This is the average (over categories) of |f_i - f^*_i|/max(f_i,f^*_i)

        Args:
            rho (np.ndarray): The actual frequency representation.
            target_rho (np.ndarray): The target frequency representation.

        Returns:
            np.ndarray: The loss between actual and target frequency representations.
        """
        numerator = abs(rho - target_rho)
        denominator = (
            np.maximum(rho, target_rho) + 1e-7
        )  # Add on a small constant to prevent divide-by-zero errors

        fraction = numerator / denominator

        return self.loss_coefficient * np.mean(fraction, axis=-1)


class L2Loss(LossBase):
    def __init__(self):
        self.loss_coefficient = 2000

    def __call__(self, rho: np.ndarray, target_rho: np.ndarray) -> np.ndarray:
        """
        Computes the loss. This is the average (over categories) of |f_i - f^*_i|^2/max(f_i,f^*_i)

        Args:
            rho (np.ndarray): The actual frequency representation.
            target_rho (np.ndarray): The target frequency representation.

        Returns:
            np.ndarray: The loss between actual and target frequency representations.
        """
        numerator = (rho - target_rho) ** 2
        denominator = rho * target_rho + 1e-7

        fraction = numerator / denominator

        return self.loss_coefficient * np.mean(fraction, axis=-1)


##########################################################################


class ModulationFunction:
    def __init__(self, fn: Callable[[float, float], float]):
        self.fn = fn

    def __call__(
        self, inclusion: np.ndarray, average_rho: np.ndarray, target_rho: np.ndarray
    ) -> np.ndarray:
        """
        For each customer, we consider all those categories that the customer belongs to. For each of those categories, we compute self.fn applied to the (average_rho, target_rho) pair for that category. We then take the product across all categories the customer belongs to. This gives the corresponding entry in the output array.

        Args:
            inclusion (np.ndarray): array of shape num_customers x num_categories filled with 1s and 0s which indicate whether each customer belongs to each category
            average_rho (np.ndarray): array of shape num_categories
            target_rho (np.ndarray): array of shape num_categories

        Returns:
            np.ndarray: array of shape num_customers.
        """
        num_customers, num_categories = inclusion.shape

        # Apply self.fn to each pair of (average_rho, target_rho)
        category_values = np.array(
            [self.fn(avg, tgt) for avg, tgt in zip(average_rho, target_rho)]
        )

        # Reshape category_values to broadcast correctly
        category_values = category_values.reshape(1, num_categories)

        # Calculate the product for each customer
        customer_products = np.prod(
            np.where(inclusion == 1, category_values, 1), axis=1
        )

        return customer_products

from customer import Customer
from insurer import Insurer
import scipy.stats
import numpy as np
import pandas as pd
from collections import deque
from tqdm.notebook import tqdm
from typing import List, Tuple

from config import MarketConfig, InflationConfig, InsurerConfig


class Market:
    """
    The market has three key methods:
        1. step: This updates the price indices of the market and the time stamp
        2. observation: This generates a new collection of customers for the insurers to interact with
        3. response: This takes in a collection of offers for each customer and has the customers choose each offer.
    """

    def __init__(self):
        """
        The initialisation method for the market.

        Receives no arguments but pulls settings from MarketConfig.
        """
        # The following variables are used to track time and inflation
        self.time_stamp = (
            0.0  # Time since the initialisation of the market, measured in years
        )
        self.inflation_rate = (
            InflationConfig.mean_inflation  # Initialise the inflation rate at the mean value
        )
        self.insurer_price_index = 1  # Initialise a price index at 1 for the insurers
        self.customer_price_index = 1  # Initialise a price index at 1 for the customers
        self.base_index = 1  # Initialise a base price index at 1

        # Create a list object to store customers who interact with the market
        self.customer_dataframe_list = []
        # Store the total number of customers who have interacted with the market
        self.total_customers = 0
        # Store the number of customers who walk away
        self.walk_away = 0

    def step(self):
        """
        We update the time stamp of the market and the price indices.
        """
        # First update the inflation rate, insurer price index, and customer price index
        self.time_stamp += InflationConfig.dt

        self.inflation_rate = (
            (1 - InflationConfig.reversion * InflationConfig.dt) * self.inflation_rate
            + (
                InflationConfig.mean_inflation
                * InflationConfig.dt
                * InflationConfig.reversion
            )
            + (
                InflationConfig.inflation_sigma
                * np.sqrt(InflationConfig.dt)
                * np.random.randn()
            )
        )

        self.base_index = self.base_index * (
            1 + InflationConfig.dt * self.inflation_rate
        )

        self.insurer_price_index = self.insurer_price_index * (
            1
            + InflationConfig.dt * self.inflation_rate
            + InflationConfig.inflation_noise
            * np.sqrt(InflationConfig.dt)
            * np.random.randn()
        )
        self.customer_price_index = self.customer_price_index * (
            1
            + InflationConfig.dt * self.inflation_rate
            + InflationConfig.inflation_noise
            * np.sqrt(InflationConfig.dt)
            * np.random.randn()
        )

    def observation(self) -> Tuple[List[dict], np.ndarray]:
        """
        Generate the customer features for this time step.

        Returns:
            Tuple[List[dict], np.ndarray]: The function returns two outputs:
                1. customer_features (List[dict]): This is a list of length num_customers where each entry is a dictionary of customer features.
                2. expected_costs (np.ndarray): This is an array with shape (num_customers x num_insurers) which gives the expected cost of each customer, as estimated by each insurer.
        """
        customer_features = []
        self.current_customer_list = []

        pure_premiums = np.zeros(MarketConfig.customers_per_time_step)

        for ii in range(MarketConfig.customers_per_time_step):
            new_customer = Customer()
            # Generate MarketConfig.customers_per_time_step new customers at random and add to the customer list
            self.current_customer_list.append(new_customer)
            # Get the features of the customer and the pure premium
            customer_features.append(new_customer.get_profile())
            pure_premiums[ii] = new_customer.pure_premium

        expected_costs = (
            self.insurer_price_index
            * pure_premiums[:, np.newaxis]
            * scipy.stats.gamma.rvs(
                a=1 / MarketConfig.CoV2,
                scale=MarketConfig.CoV2,
                size=(MarketConfig.customers_per_time_step, MarketConfig.num_insurers),
            )
        )

        # Expected_costs has shape (customers_per_time_step, num_insurers)
        return customer_features, expected_costs

    def response(self, offers: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        This function generates the responses of the customers at the current time step

        Args:
            offers (np.ndarray): This is a numpy array with shape (num_customers x num_insurers).

        Returns:
            Tuple[np.ndarray, List[float]]: The function returns two outputs:
                1. responses (np.ndarray): This is a binary array with shape (num_customers x num_insurers).
                2. step_profits (List[float]): A list of length (num_insurers) containing profit generated for each company from this batch.
        """
        num_customers = offers.shape[0]
        num_insurers = offers.shape[1]

        responses = np.zeros_like(offers)
        step_profits = [0] * num_insurers

        for customer_index in range(num_customers):
            # Get the decision for the customer
            decision = self.current_customer_list[customer_index].decision(
                offers[customer_index, :],
                self.customer_price_index,
            )

            # Record if the customer walks away from the market
            if decision == num_insurers:
                self.walk_away += 1
            else:
                step_profits[decision] += (
                    offers[customer_index, decision] / self.base_index
                    - self.current_customer_list[customer_index].pure_premium
                )
                # Record the decision in the response lists
                responses[customer_index, decision] = 1

        # Finally, increase the customer counter
        self.total_customers += num_customers

        return responses, step_profits

    #########################################################################################

    def initialise_market(self):
        """
        This function runs the market to give the insurers training data, and then erases the market history, essentially resetting the statistics
        """
        # First run the market for as long as it takes to fill up the insurer customer history data
        self.run_market(InsurerConfig.retrain_length)
        # Next clear the market history
        self.clear_market_history()

    def clear_market_history(self):
        """
        Clears the market history by setting counters and customer stores to zero.
        This includes resetting counters for each firm in the market.
        """
        self.total_customers = 0
        self.walk_away = 0
        self.time_stamp = 0
        self.customer_dataframe_list = []
        for firm in self.insurer_list:
            firm.clear_counters()

    def proportion_walk_away(self):
        """
        This function gives the proportion of customers entering the market who decide not to buy any insurance.

        Returns:
            float: The proportion of customers who walk away.
        """
        if self.total_customers == 0:
            return 0
        else:
            return self.walk_away / self.total_customers

    def store_customers(
        self,
        customer_features: List[dict],
        pure_premiums: List[float],
        offers: np.ndarray,
        average_top_5_price_list: List[float],
        decision_list: List[int],
    ):
        """
        Takes in information about a collection of customers and adds that information to the data attribute.
        Intended to be called at the end of interact_one_step

        Args:
            customer_features (List[dict]): A list of the feature profiles for each customer
            pure_premiums (List[float]): A list of the pure premiums of each customer
            offers (np.ndarray): A numpy array which stores the offer made by each firm to each customer
            average_top_5_price_list (List[float]): A list which stores average price offered to the customer by the cheapest 5 firms
            decision_list (List[int]): A list which stores the decisions made by each customer
        """
        # First create a new dataframe from the customer features
        new_data = pd.DataFrame(customer_features)
        # Add the offers to the new dataset
        offer_columns = pd.DataFrame(
            offers,
            index=new_data.index,
            columns=[f"Offer {i}" for i in range(MarketConfig.num_insurers)],
        )
        new_data = pd.concat([new_data, offer_columns], axis=1)
        # Add the pure premium, average top 5 price, and decision list
        new_data["Pure premium"] = pure_premiums
        new_data["Average top 5 price"] = average_top_5_price_list
        new_data["Decision"] = decision_list
        # Store the context variables
        new_data["Time stamp"] = self.time_stamp
        new_data["Customer price index"] = self.customer_price_index
        new_data["Insurer price index"] = self.insurer_price_index
        new_data["Inflation rate"] = self.inflation_rate
        # Store the decision rank
        ranks = np.argsort(np.argsort(offers, axis=1), axis=1)
        decision_ranks = np.full(MarketConfig.customers_per_time_step, np.nan)
        valid_indices = [
            i for i, r in enumerate(decision_list) if r != MarketConfig.num_insurers
        ]
        decision_ranks[valid_indices] = ranks[
            np.arange(len(decision_list))[valid_indices],
            np.array(decision_list)[valid_indices],
        ]
        new_data["Decision rank"] = decision_ranks
        # Perform a one-hot encoding of the data
        new_data = pd.get_dummies(
            new_data, columns=["Occupation", "Location", "Decision"]
        )
        # Finally, add the data to the list of dataframes
        self.customer_dataframe_list.append(new_data)

    def return_dataset(self):
        """
        This returns a single pandas dataframe which contains all the data of all the customers that have interacted with the market
        """
        return pd.concat(self.customer_dataframe_list)

    def rank_statistics(self):
        """
        This returns a list of the proportion of times each different rank is chosen in the market
        """
        dataset = self.return_dataset()
        return dataset["Decision rank"].value_counts(normalize=True)

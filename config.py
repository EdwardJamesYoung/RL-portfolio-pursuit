from dataclasses import dataclass
import numpy as np


@dataclass
class InflationConfig:
    dt: float = 1 / (
        52 * 10000
    )  # Length of time corresponding to one time-step in the market, measured in years
    # mean_inflation: float = 0.02  # Average inflation rate (per year)
    # var_inflation: float = (
    #     0.004  # Variance of inflation rate (in units of years squared)
    # )
    # reversion: float = 2  # Mean reversion rate, measured in 1/years
    # inflation_sigma: float = np.sqrt(2 * reversion * var_inflation)
    # inflation_noise: float = (
    #     0.02  # Additional inflation noise added to the customer inflation and market inflation respectively
    # )
    mean_inflation: float = 0  # Average inflation rate (per year)
    var_inflation: float = 0  # Variance of inflation rate (in units of years squared)
    reversion: float = 1  # Mean reversion rate, measured in 1/years
    inflation_sigma: float = 0
    inflation_noise: float = (
        0  # Additional inflation noise added to the customer inflation and market inflation respectively
    )


@dataclass
class MarketConfig:
    success_rate: float = 0.1
    num_insurers: int = 6
    customers_per_time_step: int = (
        1  # The number of customers that interact with the market each time step.
    )
    CoV2: float = 1 / 1000


@dataclass
class InsurerConfig:
    retrain_epoch_num: int = 4


@dataclass
class ExperimentConfig:
    epoch_customers = 1000
    burn_in_epochs = 6
    test_epochs = 8


@dataclass
class CustomerConfig:
    sensitivity_multiplier: float = 1000
    num_loc: int = 4
    num_occ: int = num_loc + 1
    num_customer_features: int = 7 + num_loc + num_occ
    expected_offer_multiplier: float = 1.1

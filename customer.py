import numpy as np
import scipy.stats
import scipy.special
from config import MarketConfig, CustomerConfig
from typing import Dict


class Customer:
    def __init__(self):
        """
        Generates the customer features at random.
        """
        # Sample the age:
        self.age = int(16 + 84 * scipy.stats.beta.rvs(a=1.4, b=2.5))
        # Sample the location:
        self.loc = np.random.randint(0, CustomerConfig.num_loc)
        # Sample occupation:
        self.occ = scipy.stats.binom.rvs(
            CustomerConfig.num_loc,
            np.linspace(0.2, 0.8, CustomerConfig.num_loc)[self.loc],
        )
        # Sample income percentile for age based on occupation
        self.pctl = min(
            1.1
            * scipy.stats.beta.rvs(
                a=2 + 2 * self.occ, b=2 * CustomerConfig.num_occ - 2 * self.occ
            ),
            1,
        )
        # Map age to base income
        self.base_income = max(40000 - 25 * ((self.age - 45) ** 2), 20000)
        # Generate income by scaling base income
        self.salary = (
            3 * self.base_income * max(scipy.stats.beta.ppf(self.pctl, a=3, b=9), 0.3)
        )
        # round down to the nearest multiple of 1000
        self.salary = 1000 * int(self.salary / 1000)
        # Generate the probability that a person is married
        self.prob_married = 0.7 * (110 - self.age) * (self.age - 16) / (47**2)
        # Sample whether they are married
        self.married = scipy.stats.bernoulli.rvs(self.prob_married)
        # Find the probability they have a child living at home
        self.p_child = (
            (self.age - 16)
            * (self.age - 16)
            * (self.age - 66)
            * (self.age - 66)
            / (25**4)
        )
        if self.age > 66:
            self.p_child = 0
        if self.married:
            self.p_child = self.p_child * 0.8
        else:
            self.p_child = self.p_child * 0.2
        # transform so we can sample from a binomial
        self.p_child_binomial = 1 - ((1 - self.p_child) / 4) ** (1 / 4)
        # Sample number of children
        self.num_children = scipy.stats.binom.rvs(4, self.p_child_binomial)
        self.child_indicator = self.num_children > 0
        # Sample partner salary
        if self.married:
            self.partner_salary = 1000 * int(
                max(self.salary * scipy.stats.laplace.rvs(loc=1, scale=1 / 3), 0) / 1000
            )
        else:
            self.partner_salary = 0
        # Calculate salary per household member
        self.salary_per_household_member = (self.salary + self.partner_salary) / (
            1 + self.married + self.num_children
        )
        # Draw the number of years that a person has been driving
        self.years_driving = (
            int((self.age - 16) * scipy.stats.beta.rvs(a=3, b=100 / self.age - 1)) + 1
        )
        # Drawing level of risk
        self.risk = scipy.stats.beta.rvs(
            a=1 + self.loc, b=1 + CustomerConfig.num_loc - self.loc
        )
        # Drawing number of previous claims
        self.num_claims = scipy.stats.binom.rvs(self.years_driving, self.risk * 0.07)
        # Drawing risk perception
        self.risk_perception = (
            self.num_claims / (0.14 * self.years_driving) + self.risk / 2
        )
        # Drawing car valuation
        self.car_valuation = (
            scipy.stats.gamma.rvs(4, scale=1 / 8, loc=0.2)
            * self.salary
            / (1 + self.num_children / (1 + self.married))
        )
        # Set the pure premium for the customer.
        self.pure_premium = min((0.5 + self.risk) * (self.car_valuation / 10), 6000)

    def decision(self, offers: np.ndarray, customer_price_index: float) -> int:
        """
        Makes a decision for the customer.

        Args:
            offers (np.ndarray): A numpy array with each element corresponding to the offer made by a specific company
            customer_price_index (float): The price index for the customer

        Returns:
            int: The index of the offer that the customer decided to accept
        """

        # Set the price sensitivity
        self.beta = CustomerConfig.sensitivity_multiplier / (
            customer_price_index * self.salary_per_household_member
        )
        # Set the expected offer price variable
        self.expected_offer = (
            CustomerConfig.expected_offer_multiplier
            * customer_price_index
            * self.pure_premium
        )
        # Set the unmodulated customer base price
        self.unmodulated_base_price = self.expected_offer - (1 / self.beta) * np.log(
            1 / MarketConfig.success_rate - MarketConfig.num_insurers
        )
        # Rectify the unmodulate_base_price to be non-negative
        self.unmodulated_base_price = max(0, self.unmodulated_base_price)
        # Modulate the customer base price with information about the customer
        self.base_price = (
            (1.25 - 0.5 * self.pctl)
            * (1.25 - 0.5 * self.risk_perception)
            * self.unmodulated_base_price
        )
        # Rectify the modulate_base_price to be non-negative
        self.base_price = max(0, self.base_price)
        # Select a brand loyalty variable from an exponential
        self.loyalty_identity = np.random.randint(MarketConfig.num_insurers)
        self.loyalty_strength = scipy.stats.expon.rvs(
            scale=0.05 * (self.base_price + 20)
        )
        offers_with_loyalty = offers.copy()
        offers_with_loyalty[self.loyalty_identity] -= self.loyalty_strength

        # Add the base_price to the offer list
        offers_with_base = np.append(offers_with_loyalty, self.base_price)
        # Compute the logits
        logits = -self.beta * offers_with_base
        # Compute the probabilities using the softmax
        probabilities = scipy.special.softmax(logits)
        # Sample from the resulting categorical, returning N for `walking away'
        return np.random.choice(MarketConfig.num_insurers + 1, p=probabilities)

    def get_profile(self) -> Dict:
        """
        Returns the profile for the customer, as a dictionary.

        Returns:
            Dict: A dictionary containing the customers features
        """
        return {
            "Car valuation": self.car_valuation,
            "Number of previous claims": self.num_claims,
            "Years driving": self.years_driving,
            "Age": self.age,
            "Marital status": self.married,
            "Child indicator": self.child_indicator,
            "Occupation": self.occ,
            "Location": self.loc,
        }

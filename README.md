# RL for Portfolio Pursuit

The codebase accompanies our [paper](https://arxiv.org/abs/2408.00713) applying reinforcement learning ideas to portfolio pursuit in the context of insurance markets. It is designed to allow other research to build on our research, replicate our results, and use the market environment for their own work. It is licensed under the a [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/). 

## Paper walk-through (Accenture internal use)

For internal use at Accenture we have created a series of videos which walk through the different sections of our paper.
Accenture employees can find them via the following links:
1. [Introduction and problem statement](https://mediaexchange.accenture.com/media/t/1_lf2w7w0l)
2. [Problem formalism](https://mediaexchange.accenture.com/media/t/1_q6sit8p6)
3. [Standard industry pipeline](https://mediaexchange.accenture.com/media/t/1_6tuoev6y)
4. [Baseline method](https://mediaexchange.accenture.com/media/t/1_hbl2avdf)
6. [RL method](https://mediaexchange.accenture.com/media/t/1_sbzm3go2)
7. [Results](https://mediaexchange.accenture.com/media/t/1_x73thdor)
8. [Limitations and further research](https://mediaexchange.accenture.com/media/t/1_avxrmwzm)

## Motivation of the research 

**For more details of why we focus on this problem, see [this](https://mediaexchange.accenture.com/media/t/1_lf2w7w0l) walkthrough video**.

Our reseach focuses on applying ideas from reinforcement learning to the insurance market. We focus on what we call the *portfolio pursuit* problem. By a *portfolio*, we mean the collection of new customers that the insurer has insured over some specified time period. An insurer may have a *target portfolio*, as dictated by their longer-term goals. For example, they may want to avoid insuring too many customers, since this would lead to liquidity issues if many customers needed to be paid simultaneously. Alternatively, they may wish to establish a reputation among a specific demographic, and make generous offers to that specific demographic. Lastly, they might want to protect against correlated risk or systematic bias, and maintain a diverse portfolio. 

Suppose then that the insurer has in mind a target portfolio that they would like to achieve. The porftolio pursuit problem asks: given that target portfolio, how should we adjust the offers we make the individual customers to get closer to that ideal portfolio? Our research focuses on this issue. We apply ideas from reinforcement learning to devise an algorithm for portfolio pursuit. 

**To see how we formulate the portfolio pursuit problem, see [this](https://mediaexchange.accenture.com/media/t/1_q6sit8p6) walkthrough video**.

## Structure of the codebase

### The run experiment notebook 
The run experiment notebook allows a user to run specific experiments using the environment. It allows users to define a grid of model hyperparameters to perform a grid search over, and then run a number of trials (with pre-specified random seeds) to assess performance for those hyperparameters. The functions in this notebook also handle the interaction between insurers and the market 

### Config file
The config.py file contains dataclasses used to set hyperparameters for experiments, such as the length of an episode, the number of insurers in the market, and the number of training and testing episodes. Other hyperparameters for specific models used by insurers can be passed in when running experiments 

### Market file
The market.py file contains the Market class. This handles the interactions between insurers and customers. Customers are generated by the market. Their details are sent to insurers, who each make the customer an offer. On the basis of that, the customer makes a decision about which offer to accept. The market object also tracks various statistics related to market interactions. 

### Customer file
The customer.py file contains the Customer class. When initialised, the customer's features are drawn from a complex generative distribution. Features include occupation, location, age, marital status, income, and car valuation. These depend on each other in a realistic manner. Customers also have a method for making decisions, where they take in a collection of offers made by various insurance firms, and choose which offer to accept (or whether to reject all offers). Selection is random, with a preference towards lower offers.  

### Insurer file

The bulk of the code is found in the insurer file. We go into detail about how our insurance agents operate in our [paper](https://arxiv.org/abs/2408.00713). Here we briefly explain some key aspects of the insurance agents. 

The insurance agent can be one of three types: "Null", "Baseline", and "RL". These correspond to &sect;3.1, &sect;3.2, and &sect;3.3 respectively. The Null insurers do not implement portfolio pursuit - that is, they have no notion of a desired portfolio of customers towards which they are optimising. Instead, they optimise purely for profit.  

The operational flow of a Null insurer is as follows (see &sect;3.1 of the [paper](https://arxiv.org/abs/2408.00713)). These insurers have three core models:
1. *Market models* - these predict the behaviour of other firms within the market, for new customers.
2. *Conversion models* - these predict the likelihood of a customer accepting a given offers, given the other market behaviour.
3. *Bidding models* - these tell the insurer which offer to take for a new customer, given the other market behaviour.

The conversion and market models are trained using data from the insurer's previous market interactions. We use the conversion model to compute optimal actions on which to train the bidding model. We have classes corresponding to each model; these classes essentially operate as wrappers around scikit-learn models. The wrappers ensure that the inputs and outputs to each model are the same, and log various other aspects of performance to make debugging easier. When faced with a new customer, insurers use their market model to predict the behaviour of other firms within the market. On the basis of the predicted behaviour, the insurer uses the bidding model to make the customer an offer. **For further details of the operation of null insurers, see [this](https://mediaexchange.accenture.com/media/t/1_6tuoev6y) walkthrough video**.

The Baseline insurers (see &sect;3.2 of the [paper](https://arxiv.org/abs/2408.00713)) implement portfolio pursuit via a naive method. Offers are computed via the operational flow of Null insurers. A multiplier is then applied to the offers according to how appealing the customer is, from the perspective of the portfolio. The multipler is found as follows. We look at each category that the customer falls into that are relevant to the insurer's portfolio. For each of those categories, we ask whether the insurer wants to increase or decrease the number of customers they get in that category. If they want to increase the number, they lower the customer's offer. Otherwise, they lower the customer's offer. **For further details of the operation of baseline insurers, see [this](https://mediaexchange.accenture.com/media/t/1_hbl2avdf) walkthrough video**. 

The RL insurers (see &sect;3.3 of the [paper](https://arxiv.org/abs/2408.00713)) use a reinforcement learning approach to implement portfolio pursuit. The reinforcement learning approach uses a *portfolio value function*, which quantifies the value of having a particular portfolio at a particular time. These values are known at the final time step. Moreover, we can use a  [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) to find the values at any step, if we know the values at the next. Accordingly, to compute values everywhere, we can iterate backwards in time, starting with known values at the final step, and then using the Bellman equation to find values at the penultimate step, then the step before, and so on. This is the core of our reinforcement learning algorithm. Once we have value estimates everywhere, we train a neural network on those values. At inference, we use our value function to compute how a new customer will effect the value of the portfolio. Increased values leads to lower offers; deacreased values lead to higher offers. **For further details of the operation of the reinforcement learning insurers, see [this](https://mediaexchange.accenture.com/media/t/1_sbzm3go2) video**. 

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

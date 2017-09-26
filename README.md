# ApproxTS
Utilizing strategy from [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning), aims to construct decision making system for large decision problems. The general setting is that we want to estimate an optimal policy such that the mean accumulative utility (think it as reward) is maximized. We do not know how the enviroment reacts to actions applied on it, but we will assume a parametric model for it (model-based planning problem). Here, a variant of [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) is employed to estimate the optimal policy.

To illustrate the algorithm, I built two simulators:
1. Resource allocation for control of the spread of emerging infectious diseases. The code is [here](https://github.com/Tao-Hu/ApproxTS/blob/master/src/ABMcomponents.py).
2. Adaptive management of mallard populations in the U.S. The code is [here](https://github.com/Tao-Hu/ApproxTS/blob/master/src/AHMdef.py).

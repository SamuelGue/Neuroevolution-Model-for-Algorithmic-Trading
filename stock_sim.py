import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable

class worldEvent():
    def __init__(self, category, strength):
        self.category = category    #World event category affects some stocks more than others - e.g. War events effect defense stocks
        self.strength = strength    #Measures the severity of the event - e.g. all out war vs minor skirmish
    



class stock():
    def __init__(self,initial_price,drift,volatility,category,merton_jump_parameters={"lambda":None,"mu":None,"sigma":None}):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.category = category
        self.jump_lambda = merton_jump_parameters["lambda"]
        self.jump_mu = merton_jump_parameters["mu"]
        self.jump_sigma = merton_jump_parameters["sigma"]

    def simulate(self, simulation_time = 1000):
        prices = np.zeros(simulation_time)
        prices[0] = self.initial_price
        dt = 1/simulation_time

        for t in range(1,simulation_time):
            if prices[t-1] <= 0:
                prices[t] = 0
                continue
            
            dW = np.random.normal(0, np.sqrt(dt))
            gbm = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * dW

            jumps = np.random.poisson(self.jump_lambda * dt)
            if jumps > 0:
                jump_size = np.sum(np.random.normal(self.jump_mu,self.jump_sigma, jumps))
            else:
                jump_size = 0
            
            prices[t] = prices[t-1] * np.exp(gbm + jump_size)

            if prices[t] < 0:
                prices[t] = 0

        return prices
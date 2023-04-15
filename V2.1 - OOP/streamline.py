import numpy as np
import matplotlib.pyplot as plt


class Streamline():

    def __init__(self, seed=np.array([0.47,0.01])):
        self.seed = seed

    def _getSeed(self):
        return self.seed
    

    def plot(self, plot_type):
        
"""
Author: Yousef Nami

Description:
------------
This file contains some of the functions that are useful for generating datasets. They are created by me, except where otherwise specified
"""
# libraries
import numpy as np
def generate_circle(mu = 0, sd = 1, n_points = 100):
    theta = np.linspace(0, 2*np.pi, n_points)
    r = np.random.normal(mu, sd, n_points)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

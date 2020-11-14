"""
Author: Yousef Nami

Description:
------------
This file contains some of the functions that are useful for plotting purposes.
They are created by me, except where otherwise specified.
"""
import numpy as np
import matplotlib.pyplot as plt


def gridspacer(*gridparams, mu = [0,0]):
    """
    creates a meshgrid

    Attributes:
    -----------

    *gridparams: *[int, int, int]
        N lists of the following format [X_min, X_max, n_samples]
    """
    n = len(gridparams)
    lines = []
    grids = []
    axes = [] #this axes was added as a last minute save!
    total_size = 1
    for parameters, mean in zip(gridparams,mu):
        axes.append(np.linspace(*parameters))
        lines.append(np.linspace(*parameters)- mean)
        total_size *= parameters[2]

    for grid in np.meshgrid(*lines):
        grids.append(grid)


    X_grid = np.array(grids).reshape(n,total_size).T

    return lines, X_grid, axes


def plot_classes(X,y, ax = None, **kwargs):

    """ Given X and y, where X represents the X values (in 2D) and y the classes, plots outcomes
    with different colors.

    NOTE:
    -----
    - lacks the use of **kwargs *args properly
    - does not have functionality for x, y and title labels
    - does not have legend
    - only 2 dimensional

    """
    if ax:
        pass
    else:
        fig, ax = plt.subplots()
    for y_unique in np.unique(y):
        plt.plot(X[y == y_unique, 0], X[y == y_unique, 1], '.')

def reshape_by_component(f, *x):
    """
    reshapes a function f by the dimensions of it's consituent components *x
    """
    return f.reshape(*[len(i) for i in x])

# added comment for testing purposes
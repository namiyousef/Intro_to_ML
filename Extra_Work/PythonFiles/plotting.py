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


def plot_classes(X,y, vars = None, **kwargs):

    """
    Given X and y, where X represents the X values and y the classes, plots outcomes
    with different colors.

    I want to have the option of choosing which variables to plot against each other
    NOTE:
    -----
    - lacks the use of **kwargs *args properly
    - does not have functionality for x, y and title labels
    - does not have legend
    - only 2 dimensional

    """
    if vars:
        pass # here add code for custom variables
    else:
        ncols = int(X.shape[1] ** (1/2))
        nrows = X.shape[1] // ncols

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (nrows * 8, ncols * 8))
    feature_1 = 0
    feature_2 = feature_1 + 1
    for j, column in enumerate(axes):
        for i, row in enumerate(column):
            for y_unique in np.unique(y):
                axes[j,i].plot(
                    X[y == y_unique, feature_1],
                    X[y == y_unique, feature_2],
                    '.'
                )
                axes[j,i].set_xlabel('Feature {}'.format(feature_1))
                axes[j,i].set_ylabel('Feature {}'.format(feature_2))

            feature_2 += 1
            if feature_2 >= X.shape[1]:
                feature_1 += 1
                feature_2 = feature_1 + 1
    plt.legend([*np.unique(y)])
    return fig, axes

def reshape_by_component(f, *x):
    """
    reshapes a function f by the dimensions of it's consituent components *x
    """
    return f.reshape(*[len(i) for i in x])
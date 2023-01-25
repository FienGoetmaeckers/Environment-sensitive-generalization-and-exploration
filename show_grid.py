# -*- coding: utf-8 -*-
"""
code to visualize the grids
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


    
def show_bandit(bandit, W, L):
    """
    This function shows a heatmap of the grid
    the tiles show their reward value.
    
    Parameters
    ----------
    bandit :    numpy array
            the bandit list with the rewards.
    W:          int
            the width of the grid
    L:          int
            the length of the grid

    Returns
    -------
    plot the bandit

    """
    
    #the font and figure size grow along with the size of the grid
    plt.rcParams['font.size'] = str(W*4)
    plt.figure(figsize=(2*W, 2*L))
    #we round the rewards to integers
    rounded_bandit = np.array([int(value) for value in bandit])
    sns.heatmap(rounded_bandit.reshape((L, W)), linewidth=0.5, cmap = 'Reds', annot=True, fmt='g', square = True, cbar = True)
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pdb

def plot_2D_data_label(X,y):
    
    for labelValue in np.unique(y):
        plt.scatter(X[y==labelValue, 0], X[y==labelValue, 1], color=plt.cm.Set2(labelValue / np.size(np.unique(y))))
        
def plot_digits_embedding(X, y,title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / np.size(np.unique(y))),
                 fontdict={'weight': 'bold', 'size': 9})
    
        
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
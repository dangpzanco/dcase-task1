import os
import pathlib
import pickle
from datetime import datetime

import numpy as np
import numpy.random as rnd
import pandas as pd
import sklearn.metrics as skmetrics

import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_history(history):
    print(history.keys())

    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(history['loss'],'o-')
    plt.plot(history['val_loss'],'o-')
    plt.title('Loss')
    plt.xlabel('Epoch', labelpad=-10)
    plt.legend(['Training', 'Test'])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(history['acc'],'o-')
    plt.plot(history['val_acc'],'o-')
    plt.title('Accuracy')
    plt.xlabel('Epoch', labelpad=-10)
    plt.legend(['Training', 'Test'])
    plt.grid()

    plt.gcf().set_size_inches(15, 15)


def plot_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, cmap='YlGnBu'):
    """Plots a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Modified from `shaypal5's gist`.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    cmap: str
        Colormap for the heatmap (see `Colormaps in Matplotlib`). Defaults to YlGnBu.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure

    References
    ----------
    .. _shaypal5's gist:
       https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    
    .. _Colormaps in Matplotlib:
       https://matplotlib.org/tutorials/colors/colormaps.html
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap=cmap) 
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=fontsize, fontweight='bold')
    return fig



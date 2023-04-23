import matplotlib.pyplot as plt

def pltResults(logdir, data1, data2, title, labelx, labely):
    """
    Method to plot the results of the training and test
    Args:
        logdir: Directory where the results will be saved
        data1: Data to plot in the first graph
        data2: Data to plot in the second graph
        title: Title of the graph
        labelx: Label of the x axis
        labely: Label of the y axis
    Returns:
        None
    Usage:
        ```python
        pltResults(logdir, history.history['acc'], history.history['val_acc'], 'Model Accuracy', 'Epoch', 'Accuracy')
        ```
    """
    plt.plot(data1)
    plt.plot(data2)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(logdir + '/' + labely + '.png')
    plt.show()
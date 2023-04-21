import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    plt.close()

def confusionMatrix(y_test, y_pred):
    """
    Method to plot the confusion matrix
    Args:
        y_test: True labels
        y_pred: Predicted labels
    Returns:
        None
    Usage:
        confusionMatrix(y_test, y_pred)
    """
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['mantled_howler', 'patas_monkey', 'bald_uakari', 'japanese_macaque',\
                             'pygmy_marmoset', 'white_headed_capuchin', 'silvery_marmoset', \
                             'common_squirrel_monkey', 'black_headed_night_monkey', 'nilgiri_langur']); 
    ax.yaxis.set_ticklabels(['mantled_howler', 'patas_monkey', 'bald_uakari', 'japanese_macaque',\
                             'pygmy_marmoset', 'white_headed_capuchin', 'silvery_marmoset', \
                             'common_squirrel_monkey', 'black_headed_night_monkey', 'nilgiri_langur']);

    plt.show()
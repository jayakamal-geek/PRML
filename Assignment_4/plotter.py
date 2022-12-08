from sklearn.metrics import roc_curve
from sklearn.metrics import det_curve
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def plot_cmat (ytrue, ypred):
    mat = confusion_matrix (ytrue, ypred)
    plt.imshow (mat, cmap='hot', interpolation='nearest')

    for i in range (len (mat)):
        for j in range (len (mat[i])):
            plt.text (i, j, mat[i][j], c='green')

    plt.show ()

def plot_det (ytrue, yscore):
    a, b, _ = det_curve (ytrue, yscore)
    plt.plot (a, b)
    plt.xlim (0, 1)
    plt.ylim (0, 1)
    plt.show ()

def plot_roc (ytrue, yscore):
    a, b, _ = roc_curve (ytrue, yscore)
    plt.plot (a, b)
    plt.xlim (0, 1)
    plt.ylim (0, 1)
    plt.show ()
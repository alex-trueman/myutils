import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true, y_pred, labels, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    Plots an n x n confusion matrix for regression classification.

    Modified version from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Args:
    ----------
    y_true: Series or array-like (shape = n_samples) true values of the
        response variable usually from the test split of the data.
    y_pred: Series or array-like (shape = n_samples) predicted response
        values from the regression model.
    labels: List of labels to for the matrix (shape = n_categories).
    normalize: Normalize the results.
    title: String title for the plot.
    cmap: matplotlib pyplot color map.

    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    return ax

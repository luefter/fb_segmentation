import numpy as np
import matplotlib.pyplot as plt


def plot_results(data, x_lab: str, y_lab: str, title: str):
    """
    Function to plot the results of the training
    :param data: list of the training results
    :param x_lab: name of the x-axis
    :param y_lab: name of y-axis
    :param title: title of the plot
    :return: plot
    """
    plt.figure()
    x = np.arange(0, len(data))
    y = data
    plt.plot(x, y)
    plt.xlabel(xlabel=x_lab)
    plt.ylabel(ylabel=y_lab)
    plt.title(title)
    # if needed change filepath
    plt.savefig(f"./src/fbs/data/img/{title.strip(' ')}.png")
    plt.show(block=False)

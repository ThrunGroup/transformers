import matplotlib.pyplot as plt
from typing import Iterable


def scatter_plot(xs: Iterable, ys: Iterable, xlabel: str = None, ylabel: str = None):
    plt.figure(figsize=(6, 4.5))
    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.scatter(xs, ys)
    plt.show()



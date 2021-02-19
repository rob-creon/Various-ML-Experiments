from dataclasses import dataclass
from typing import List

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import dates as pldate

plot_dates_num_ticks = 20
plot_price_num_ticks = 20


def plot_stock_data(title, dates, prices):
    graphable_dates = [pldate.datestr2num(day) for day in dates]

    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(pldate.DateFormatter('%m/%d/%Y'))
    plt.plot(graphable_dates, prices)
    plt.title(title)

    plt.show()


def plot_fd_stock_data(title, floated_dates, y):
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(pldate.DateFormatter('%m/%d/%Y'))
    plt.plot(floated_dates, y)
    plt.title(title)

    plt.show()


def plot_model_comparison(title, name1, floated_dates1, y1, name2, floated_dates2, y2):
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(pldate.DateFormatter('%m/%d/%Y'))
    plt.plot(floated_dates1, y1, label=name1)
    plt.plot(floated_dates2, y2, label=name2)
    plt.title(title)
    plt.legend()

    plt.show()


@dataclass(order=True)
class GraphLine:
    label: str
    x: List[float]
    y: List[float]
    dots: bool = False
    color = None


def plot_n_comparison(title, graph_lines):
    fig = plt.figure(figsize=(24, 16))
    ax = fig.gca()

    ax.set_yticks(np.arange(0, 5000, 100))
    plt.xticks(rotation=90)
    ax.xaxis.set_major_formatter(pldate.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(graph_lines[0].x)/50))

    for line in graph_lines:
        plt.plot(line.x, line.y, label=line.label)
        if line.dots:
            plt.scatter(line.x, line.y)

    plt.title(title)
    plt.legend()
    plt.grid()

    plt.show()
    fig.savefig("test.svg", format='svg')


def plot_multiple_stocks(attr_name, stock_data, plot_title):
    for symbol in stock_data:

        data = stock_data[symbol]
        graphable_dates = [pldate.datestr2num(day) for day in data.dates]

        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(pldate.DateFormatter('%m/%d/%Y'))

        plt.plot(graphable_dates, getattr(stock_data[symbol], attr_name),
                 label=symbol)
    plt.title(plot_title)
    plt.legend()
    plt.show()

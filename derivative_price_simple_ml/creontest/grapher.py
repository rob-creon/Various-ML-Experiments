from dataclasses import dataclass
import typing

import creontest.utils as utils
import creontest.stockmarket as stockmarket
import matplotlib.dates as dates
import matplotlib.pyplot as plt


@dataclass
class GraphLine:
    label: str
    x: typing.Union[typing.List[float], None]
    y: typing.List[float]
    dots: bool = False


def plot(title, graph_lines, dimensions=(5, 5), save=False, show=True, img_format='svg', str_dates=True, file_name='test', auto_scale=False):

    fig = plt.figure(figsize=dimensions)
    ax = fig.gca()

    if not auto_scale:
        ax.set(ylim=(0.0, 1.0))

    plt.xticks(rotation=45)
    if str_dates:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%m/%d/%Y'))

    # ax.xaxis.set_major_locator(plt.MaxNLocator(len(graph_lines[0].x)/50))

    for line in graph_lines:
        if line.x is None:
            plt.plot(line.y, label=line.label)
        else:
            plt.plot(line.x, line.y, label=line.label)
        if line.dots:
            print(f'line has scatter: {line.label}')
            plt.scatter(line.x, line.y)

    plt.title(title)
    plt.legend()
    plt.grid()

    if show:
        plt.show()
    if save:
        fig.savefig(f"{file_name}.{img_format}", format=img_format)


def plot_stock_data(stock_data: stockmarket.StockData, begin_date=None, end_date=None):

    graph_dates = stock_data.dates[begin_date:end_date]
    adj_close_data = stock_data.adjusted_closes[begin_date:end_date]
    ema_50 = utils.ema(stock_data.adjusted_closes[begin_date:end_date], 50)
    ema_100 = utils.ema(stock_data.adjusted_closes[begin_date:end_date], 100)

    title = f'{stock_data.symbol} Data'
    if begin_date is None and end_date is None:
        title += ' all time'
    else:
        title += f' from {stock_data.dates_str[begin_date]} to {stock_data.dates_str[end_date]}'

    lines = [
        GraphLine('Adjusted Closing Price', graph_dates, adj_close_data),
        GraphLine('EMA 50', graph_dates, ema_50),
        GraphLine('EMA 100', graph_dates, ema_100),

    ]

    plot(title, lines, auto_scale=True)


def fft_example():
    plot('fft', [GraphLine('fft', None, utils.example_fft())], auto_scale=True)

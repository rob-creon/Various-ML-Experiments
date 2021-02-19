from matplotlib import pyplot as plt
from matplotlib import dates as pldate

plot_dates_num_ticks = 20
plot_price_num_ticks = 20


def plot_raw_stocks(stocks):
    for symbol in stocks:
        plot_single_raw_stock(stocks[symbol], symbol)


def plot_single_raw_stock(data, stock_name):
    plt.xticks(rotation=90)

    my_plot = plt.plot('date', 'price', data=data)

    my_plot[0].axes.xaxis.set_major_locator(plt.MaxNLocator(plot_dates_num_ticks))
    my_plot[0].axes.yaxis.set_major_locator(plt.MaxNLocator(plot_price_num_ticks))

    plt.title(f'{stock_name} adjusted price since November 1999')
    plt.show()

    plt.close()


def plot_combined_stocks_(stocks):
    for symbol in stocks:
        plt.xticks(rotation=90)

        my_plot = plt.plot('date', 'price', data=stocks[symbol], label=symbol)

        for line in my_plot:
            my_plot[0].axes.xaxis.set_major_locator(plt.MaxNLocator(plot_dates_num_ticks))
            my_plot[0].axes.yaxis.set_major_locator(plt.MaxNLocator(plot_price_num_ticks))

    # stocks_symbol_str = '%s' % ', '.join(map(str, stocks.keys()))

    plt.legend()
    plt.title('Adjusted Price Since November 1999')
    plt.show()
    plt.close()

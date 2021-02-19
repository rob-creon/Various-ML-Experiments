
import typing
from dataclasses import dataclass

from creontest import stockmarket
from creontest import utils

import numpy as np


@dataclass
class ModelParams:
    history_points: int
    input_size: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_all: np.ndarray
    y_all: np.ndarray
    x_linear_train: typing.List[float]
    x_linear_test: typing.List[float]
    final_day_x: np.ndarray


class Dataset:

    def __init__(self, history_points=100):
        self.history_points = history_points
        self._x_params = []
        self._y_param = []

    def _add_x_params(self, p: typing.List, normalized=True):
        for param in p:
            self._add_x_param(param, normalized=normalized)

    def _add_x_param(self, p, normalized=True):
        if normalized:
            param = utils.get_normalized_list(p)
        else:
            param = p
        self._x_params.append(param)

    def _add_y_param(self, p, normalized=True):
        if normalized:
            print('normalizing a y param')
            param = utils.get_normalized_list(p)
        else:
            param = p
        self._y_param.append(param)

    def to_model_params(self, test_split):
        x_param_array = np.array(self._x_params).transpose()
        y_param_array = np.array(self._y_param).transpose()

        last_x = x_param_array[-1]

        x_param_array = x_param_array[:-1]

        n = int(x_param_array.shape[0] * test_split)

        x_linear = x_param_array.copy()
        x_linear_train = x_linear[:n + self.history_points]
        x_linear_test = x_linear[n:]

        x_data = np.array(
            [x_param_array[i:i + self.history_points].copy()
             for i in range(len(x_param_array) + 1 - self.history_points)])

        y_data = np.array(
            [y_param_array[i + self.history_points]
             for i in range(len(y_param_array) - self.history_points)])

        return ModelParams(history_points=self.history_points,
                           input_size=x_data.shape[2],
                           x_all=x_data,
                           y_all=y_data,
                           x_train=x_data[:n],
                           y_train=y_data[:n],
                           x_test=x_data[n:],
                           y_test=y_data[n:],
                           final_day_x=last_x,
                           x_linear_train=x_linear_train,
                           x_linear_test=x_linear_test)


class BasicDataset(Dataset):
    def __init__(self, stock_data: stockmarket.StockData):
        super().__init__()
        print(f'creating standard dataset for {stock_data.symbol}')
        self._add_x_params([stock_data.dates,
                            stock_data.volumes,
                            stock_data.highs,
                            stock_data.lows,
                            stock_data.opens,
                            stock_data.closes,
                            stock_data.adjusted_closes,
                            ])
        self._add_y_param(stock_data.adjusted_closes)


class RelatedDataset(BasicDataset):
    def __init__(self, stock_data: stockmarket.StockData,
                 related_stocks: typing.List[stockmarket.StockData]):
        super().__init__(stock_data)

        # todo handle stocks of different size, need to crop stocks? or does numpy automatically do this? if so what behavior?

        print(f'creating related dataset for {stock_data.symbol}')
        for related_stock in related_stocks:
            print(f'parsing {related_stock.symbol} as related stock...')
            # if related_stock.dates[-1] != stock_data.dates[-1]:
            #     print('one stock is older than the other!!!')
            #     sys.exit()
            #
            # if len(related_stock.dates) > len(stock_data.dates):
            #     offset = 0
            #     while related_stock.dates[offset] < stock_data.dates[0]:
            #         if offset > len(related_stock.dates)-1:
            #             print('related stock doesnt ever exist at the same time as the main stock. wtf?')
            #         offset += 1
            #     print(f 'found the offset! related stock, {related_stock.symbol}, is exactly {offset} days older than {stock_data.symbol}')
            # elif len(related_stock.dates) < len(stock_data.dates):
            #     offset = 0
            #     while stock_data.dates[offset] < related_stock.dates[0]:
            #         if offset > len(stock_data.dates) - 1:
            #             print('related stock doesnt ever exist at the same time as the main stock. wtf?')
            #         offset += 1
            #     print(f 'found the offset! main stock, {stock_data.symbol}, is exactly {offset} days older than it\'s related stock, {related_stock.symbol}')
            #     print('last', related_stock.dates_str[len(related_stock.dates_str)-1])
            #     print('first', related_stock.dates_str[0])
            #     print(type(related_stock.dates_str))
            #     print(type(related_stock.dates))
            #     print(type(related_stock.adjusted_closes))

            # else:
            # They must be equal
            self._add_x_params([related_stock.volumes,
                                related_stock.highs,
                                related_stock.lows,
                                related_stock.opens,
                                related_stock.closes,
                                related_stock.adjusted_closes,
                                ])


class EmptyDataset(Dataset):
    def __init__(self, stock_data: stockmarket.StockData):
        super().__init__()
        self._add_x_param(stock_data.dates)
        self._add_y_param(stock_data.adjusted_closes)


class SimpleDataset(Dataset):
    def __init__(self, stock_data: stockmarket.StockData):
        super().__init__()
        self._add_x_params([stock_data.dates,
                            stock_data.adjusted_closes])
        self._add_y_param(stock_data.adjusted_closes)


class DerivativeDataset(Dataset):
    """
    Inputs are all per day: dates, adjusted closing price, high, low, volume, and the change in price for that day.
    The model is trained to output the change in price for the following day
    """
    def __init__(self, stock_data: stockmarket.StockData):
        super().__init__()

        adjusted_close_prime = utils.discrete_derivative(stock_data.adjusted_closes, 1)
        self._add_x_param(stock_data.dates)
        self._add_x_param(stock_data.adjusted_closes)
        self._add_x_param(stock_data.highs)
        self._add_x_param(stock_data.lows)
        self._add_x_param(stock_data.volumes)
        self._add_x_param(adjusted_close_prime)

        self._add_y_param(adjusted_close_prime, normalized=False)

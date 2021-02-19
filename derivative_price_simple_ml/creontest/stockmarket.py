# todo add company information from AlphaVantage, like stock overview

import json
import os.path
from dataclasses import dataclass
from typing import List

import alpha_vantage.timeseries
import pandas as pd
from creontest import utils


@dataclass(order=True)
class StockData:
    symbol: str
    dates_str: List[str]
    dates: List[float]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    adjusted_closes: List[float]
    volumes: List[float]


def get_stock_data(symbol):
    if not os.path.exists(f'./{get_data_csv_name(symbol)}'):
        download_data_to_csv(symbol)

    csv_data = load_data_from_csv(get_data_csv_name(symbol))
    return StockData(symbol=symbol,
                     dates_str=csv_data['date'].tolist(),
                     dates=utils.get_floated_dates(csv_data['date'].tolist()),
                     opens=csv_data['open'].tolist(),
                     highs=csv_data['high'].tolist(),
                     lows=csv_data['low'].tolist(),
                     closes=csv_data['close'].tolist(),
                     adjusted_closes=csv_data['adjusted_close'].tolist(),
                     volumes=csv_data['volume'].tolist())


def get_test_data(size):
    return StockData(symbol='TEST',
                     dates_str=list(range(0, size)),
                     dates=list(range(0, size)),
                     opens=list(range(0, size)),
                     highs=list(range(0, size)),
                     lows=list(range(0, size)),
                     closes=list(range(0, size)),
                     adjusted_closes=list(range(0, size)),
                     volumes=list(range(0, size)))


def get_dict_of_stock_data(symbols_list):
    stock_dict = {}
    for symbol in symbols_list:
        stock_data = get_stock_data(symbol)
        stock_dict[symbol] = stock_data
    return stock_dict


def get_list_of_stock_data(symbols_list):
    stock_list = []
    for symbol in symbols_list:
        stock_data = get_stock_data(symbol)
        stock_list.append(stock_data)
    return stock_list


def download_data_to_csv(symbol):
    print(f'downloading data for {symbol} from alpha_vantage...')
    credentials = json.load(open('creds.json', 'r'))
    api_key = credentials['alpha_vantage_api_key']
    ts = alpha_vantage.timeseries.TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
    data.to_csv(f'./{get_data_csv_name(symbol)}')


def load_data_from_csv(csv_path, drop_date_axis=False, drop_ipo=False):
    data = pd.read_csv(csv_path)
    data.rename(columns={'1. open': 'open',
                         '2. high': 'high',
                         '3. low': 'low',
                         '4. close': 'close',
                         '5. adjusted close': 'adjusted_close',
                         '6. volume': 'volume'},
                inplace=True)
    if drop_date_axis:
        data = data.drop('date', axis=1)
    if drop_ipo:
        data = data.drop(0, axis=0)
    data = data.iloc[::-1]
    return data


def get_data_csv_name(symbol):
    return f"data/{symbol}_daily_adjusted.csv"

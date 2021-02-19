from alpha_vantage.timeseries import TimeSeries
import pandas as pd

import json
import os.path

# todo get stock information from alphavantage, like stock overview


def load_stocks(symbols_list, use_cached=True):
    symbol_data_dict = {}
    for symbol in symbols_list:

        # Only download if there is no cached csv or if use_cached=False
        if not has_data(symbol) or not use_cached:
            download_data_to_csv(symbol)

        symbol_data_dict[symbol] = load_data_from_csv(get_data_csv_name(symbol))
        symbol_data_dict[symbol]['symbol'] = symbol

    return symbol_data_dict


def get_data_csv_name(symbol):
    return f"data/{symbol}_daily_adjusted.csv"


def has_data(symbol):
    return os.path.exists(f'./{get_data_csv_name(symbol)}')


def download_data_to_csv(symbol):
    print(f'downloading data for {symbol} from alpha_vantage...')
    credentials = json.load(open('creds.json', 'r'))
    api_key = credentials['alpha_vantage_api_key']

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')

    data.to_csv(f'./{get_data_csv_name(symbol)}')


def load_data_from_csv(csv_path, drop_date_axis=False, drop_ipo=False):
    data = pd.read_csv(csv_path)

    data.rename(columns={'1. open': 'open',
                         '2. high': 'high',
                         '3. low': 'low',
                         '4. close': 'close',
                         '5. adjusted close': 'adjusted_close',
                         '6. volume': 'volume',
                         '7. dividend amount': 'dividend_amount',
                         '8. split coefficient': 'split_coefficient'},
                inplace=True)

    if drop_date_axis:
        data = data.drop('date', axis=1)
    if drop_ipo:
        data = data.drop(0, axis=0)

    data = data.iloc[::-1]

    return data

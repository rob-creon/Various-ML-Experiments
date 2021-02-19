from matplotlib import dates as pldate
from dataclasses import dataclass
from typing import List
import numpy as np

history_points = 50


def sma(data, sma_range):
    sma_data = [0] * len(data)
    for n in range(len(data)):
        # If there are less than sma_range days of data, continue until there are enough days
        if n < sma_range:
            sma_data[n] = data[0]
            continue
        # Calculate the average of the past sma_range days
        total = 0
        for i in data[n - sma_range:n]:
            total += i
        avg = total / sma_range
        sma_data[n] = avg
    return sma_data


def ema(data, ema_range, smoothing=2):
    ema_data = [0] * len(data)

    initial = sma(data[:ema_range+1], ema_range)[-1]
    ema_data[ema_range-1] = initial

    weight = smoothing / (ema_range + 1)
    for n in range(len(data)):
        if n < ema_range:
            ema_data[n] = data[0]
        ema_data[n] = data[n] * weight + ema_data[n-1] * (1 - weight)

    return ema_data


def get_dataset_from_raw(raw):
    floated_dates = get_floated_dates(raw.dates)

    o_param = get_scaled_datas(raw.opens)
    h_param = get_scaled_datas(raw.highs)
    l_param = get_scaled_datas(raw.lows)
    v_param = get_scaled_datas(raw.volumes)
    d_param = get_scaled_datas(raw.dividend_amounts)
    ac_param = get_scaled_datas(raw.adjusted_closes)

    data_normalized = np.array([
        o_param.norm,
        h_param.norm,
        l_param.norm,
        ac_param.norm,
        v_param.norm,
        d_param.norm
    ])

    data_normalized = np.transpose(data_normalized)

    histories_normalized = np.array(
        [data_normalized[i:i + history_points].copy()
         for i in range(len(data_normalized) - history_points)]
    )

    next_day_open_values_normalized = np.array(
        [data_normalized[:, 0][i + history_points].copy()
         for i in range(len(data_normalized) - history_points)])

    next_day_open_values_normalized = np.expand_dims(
        next_day_open_values_normalized, -1)

    next_day_open_values = np.array(
        [raw.opens[i + history_points] for i in
         range(len(raw.opens) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    assert histories_normalized.shape[0] == \
           next_day_open_values_normalized.shape[0]

    return TrainingData(floated_dates=floated_dates,
                        opens=o_param,
                        highs=h_param,
                        lows=l_param,
                        volumes=v_param,
                        dividends=d_param,
                        closes=ac_param,
                        histories_normalized=histories_normalized,
                        next_day_opens=next_day_open_values_normalized)


def get_scaled_datas(data, scaled_low=0, scaled_high=1):
    return TrainingDataParameter(min_val=min(data), max_val=max(data),
                                 norm=scale_list(data, scaled_low, scaled_high),
                                 raw=data)


@dataclass(order=True)
class TrainingDataParameter:
    min_val: float
    max_val: float
    norm: List[float]
    raw: List[float]


@dataclass(order=True)
class TrainingData:
    floated_dates: List[float]
    opens: TrainingDataParameter
    highs: TrainingDataParameter
    lows: TrainingDataParameter
    volumes: TrainingDataParameter
    dividends: TrainingDataParameter
    closes: TrainingDataParameter
    histories_normalized: np.array
    next_day_opens: np.array


@dataclass(order=True)
class StockData:
    symbol: str
    dates: List[float]
    highs: List[float]
    lows: List[float]
    opens: List[float]
    closes: List[float]
    adjusted_closes: List[float]
    volumes: List[float]
    dividend_amounts: List[float]
    split_coefficients: List[float]


def get_floated_dates(dates):
    return [pldate.datestr2num(day) for day in dates]


def get_list_from_dict(my_dict):
    l = [my_dict[i] for i in range(len(my_dict))]
    l.reverse()
    return l


def generate_single_stock_data(stock_data):
    stock_dict = stock_data.to_dict()
    return StockData(
        symbol=stock_dict['symbol'][0],
        dates=get_list_from_dict(stock_dict['date']),
        highs=get_list_from_dict(stock_dict['high']),
        lows=get_list_from_dict(stock_dict['low']),
        opens=get_list_from_dict(stock_dict['open']),
        closes=get_list_from_dict(stock_dict['close']),
        adjusted_closes=get_list_from_dict(stock_dict['adjusted_close']),
        volumes=get_list_from_dict(stock_dict['volume']),
        dividend_amounts=get_list_from_dict(stock_dict['dividend_amount']),
        split_coefficients=get_list_from_dict(stock_dict['split_coefficient'])
    )


def generate_stock_data(stock_data_dict):
    output = {}
    for symbol in stock_data_dict:
        output[symbol] = generate_single_stock_data(stock_data_dict[symbol])
    return output


def scale_val(f, f_min, f_max, g_min, g_max):
    # https://www.desmos.com/calculator/h4g1esalm1
    if f_min == f_max:
        return g_min
    if g_min == g_max:
        return g_min

    return (g_max - g_min) * ((f - f_min) / (f_max - f_min)) + g_min


def scale_list(data, target_min=0, target_max=1):
    return [scale_val(item, min(data), max(data), target_min, target_max)
            for item in data]

# todo implement standardization

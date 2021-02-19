from math import sin

import numpy as np
import scipy.fftpack

from matplotlib import dates


def example_fft():
    y = [(sin(i / 10) * 100) + (sin(i / 5) * 50 + sin(i) * 25) for i in
         range(0, 1000)]
    return y


def fft(y):
    return (len(y) * np.abs(scipy.fftpack.fft(y)[:len(y)//2])).tolist()


def mse(predictions, real):
    total_error = 0
    for i, pred_x in enumerate(predictions):
        total_error += (real[i] - pred_x) ** 2
    return total_error / len(predictions)


def discrete_derivative(data, time_period):
    dd = [0] * len(data)
    for i in range(time_period, len(data)):
        dd[i] = data[i] - data[i-time_period]
    return dd


def integrate_from_normalized(starting_value, data):
    return integrate_from_starting_value(starting_value, data - 0.5)


def integrate_from_starting_value(starting_value, data):
    ig = [starting_value] * len(data)
    for i in range(1, len(data)):
        ig[i] = ig[i-1] + data[i-1]
    return ig


def windowed(x, points):
    """
    Written by Neil
    x is a 2D array whose first axis will be windowed.
    points is the number of points in the window.
    return value is a 3D array.
    """
    num_windows = x.shape[0] - points + 1
    return x[np.tile(np.arange(points), (num_windows, 1)) + np.tile(np.arange(num_windows)[:, np.newaxis], points), :]


def sma(data, sma_range):
    sma_data = [0] * len(data)
    for n in range(len(data)):
        # If there are less than sma_range days of data, continue until there are enough days
        if n < sma_range:
            sma_data[n] = data[1]
            continue
        # Calculate the average of the past sma_range days
        total = 0
        for i in data[n - sma_range:n]:
            total += i
        avg = total / sma_range
        sma_data[n] = avg
    return sma_data


def ema(data, ema_range, smoothing=2, remove_start=False):
    ema_data = [0] * len(data)

    initial = sma(data[:ema_range+1], ema_range)[-1]
    ema_data[ema_range-1] = initial

    weight = smoothing / (ema_range + 1)
    for n in range(len(data)):
        if n < ema_range:
            ema_data[n] = data[0]
        ema_data[n] = data[n] * weight + ema_data[n-1] * (1 - weight)

    if remove_start:
        return ema_data[ema_range:]

    return ema_data


def scale_val(f, f_min, f_max, g_min, g_max):
    if f_min == f_max:
        return g_min
    if g_min == g_max:
        return g_min

    # https://www.desmos.com/calculator/h4g1esalm1
    return (g_max - g_min) * ((f - f_min) / (f_max - f_min)) + g_min


def scale_list(data, target_min, target_max):
    return [scale_val(item, min(data), max(data), target_min, target_max)
            for item in data]


def scale_array(data, g_min, g_max):
    f_min = data.amin()
    f_max = data.amax()
    return (g_max - g_min) * ((data - f_min) / (f_max - f_min)) + g_min


def get_normalized_list(data):
    return scale_list(data, 0, 1)


def get_normalized_array(data):
    return scale_array(data, 0, 1)


def get_floated_date(str_date):
    return dates.datestr2num(str_date)


def get_floated_dates(str_dates):
    return [get_floated_date(day) for day in str_dates]


def get_unfloated_date(floated_date):
    return dates.num2date(floated_date).strftime('%Y-%m-%d')

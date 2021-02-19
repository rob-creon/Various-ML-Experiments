import datetime
import sys

from matplotlib import dates as pldate

from typing import Dict

import matplotlib
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers

import numpy as np

import grapher
import prep_old as prep
import stockmarket


tf.random.set_seed(4)
np.random.seed(4)

print('fetching stock data...')
raw_stocks = stockmarket.load_stocks(['AMD', 'MSFT', 'AAPL', 'SNAP', 'CRSP',
                                      'VXRT', 'TSLA', 'INTC', 'T', 'MCK', 'BLDP',
                                      'SNE', 'BA', 'ATO'])

stock_data: Dict[str, prep.StockData] = prep.generate_stock_data(raw_stocks)

print('creating dataset...')
raw = stock_data['ATO']

dataset = prep.get_dataset_from_raw(raw)

test_split = 0.9

n = int(dataset.histories_normalized.shape[0] * test_split)

x_train = dataset.histories_normalized[:n]
y_train = dataset.next_day_opens[:n]

x_test = dataset.histories_normalized[n:]
y_test = dataset.next_day_opens[n:]

unscaled_y_test = dataset.closes.raw[:n]

grapher.plot_fd_stock_data(f'model training data [{raw.symbol[0]}]',
                           dataset.floated_dates[:n], dataset.closes.norm[:n])
grapher.plot_fd_stock_data(f'model test data [{raw.symbol[0]}]',
                           dataset.floated_dates[n:], dataset.closes.norm[n:])

print('creating model...')

lstm_input = Input(shape=(prep.history_points, 6), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)
model = Model(inputs=lstm_input, outputs=output)

adam = optimizers.Adam(lr=0.0005)

model.compile(optimizer=adam, loss='mse')


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 != 0:
            return
        y_pred = self.model.predict(x_test)
        y_pred = np.transpose(y_pred)
        grapher.plot_model_comparison(
            f'model comparison to real at epoch{epoch}',

            'real',
            dataset.floated_dates[n:],
            dataset.closes.norm[n:],

            'modeled',
            dataset.floated_dates[
            n + prep.history_points:],
            y_pred[0]
            )


print('training...')

model.fit(x=x_train,
          y=y_train,
          batch_size=32,
          epochs=100,
          shuffle=True,
          validation_split=0.1,
          callbacks=[PredictionCallback()])

evaluation = model.evaluate(x_test, y_test)
print(evaluation)

y_predicted = model.predict(x_test)
y_predicted = np.transpose(y_predicted)

print('n', n)
print('prep.history_points',prep.history_points)
grapher.plot_model_comparison('model result',

                              'real',
                              dataset.floated_dates[n:-prep.history_points],
                              y_test,

                              'modeled',
                              dataset.floated_dates[n:-prep.history_points],
                              y_predicted[0]
                              )

print('the guess for tomorrow is:')
latest_predict = y_predicted[0][-1]
latest_real = y_test[-1]
latest_floated_date = dataset.floated_dates[-prep.history_points]
latest_date_str = matplotlib.dates.num2date(latest_floated_date)

scale_min = dataset.opens.min_val
scale_max = dataset.opens.max_val

print(f'normalized pred: {latest_predict}')
print(f'normalized real: {latest_real}')

latest_predict = prep.scale_val(latest_predict, 0, 1, scale_min, scale_max)
latest_real = prep.scale_val(latest_real, 0, 1, scale_min, scale_max)

print(f'latest prediction: {latest_predict}')
print(f'latest real: {latest_real}')
print(f'date: 10/22/2020')

num_shares = 10
cash = 0
print(f'calculating profits/loss from last {len(y_test)} days.')
print(f'starting with {num_shares} shares.')

for i in range(len(y_test)):

    real_open = prep.scale_val(y_test[i], 0, 1, scale_min, scale_max)
    pred_open = prep.scale_val(y_predicted[0][i], 0, 1, scale_min, scale_max)

    print(f'day {i}:')
    print(f'current money: ${cash}')
    print(f'current shares: {num_shares}')
    print(f'value of shares: ${num_shares *  real_open}')

    print('predicted opening cost:', pred_open)
    print('real opening cost:', real_open)

    if i > 0:
        yesterday_close = dataset.closes.raw[n + i - 1]
        print('today close:', yesterday_close)
        if num_shares > 0 and yesterday_close <= pred_open:
            # Hold or buy, we expect stock to go up
            if num_shares == 0:
                # Buy
                num_shares += cash / yesterday_close
                cost = yesterday_close * num_shares
                cash -= cost
                print(f'ACTION: bought {num_shares} for ${cost}.')

            else:
                # Hold
                print(f'ACTION: holding onto stock.')
        else:
            # Sell
            shares_to_sell = num_shares
            num_shares -= shares_to_sell
            cost = yesterday_close * shares_to_sell
            cash += cost
            print(f'ACTION: sold {shares_to_sell} for ${cost}.')

    print('')
    if cash < 0 or num_shares < 0:
        print(f'cash or num_shares are negative: cash={cash}, num_shares={num_shares}')
        sys.exit(-1)

print(f'INVERSE: calculating profits/loss from last {len(y_test)} days.')
print(f'starting with {num_shares} shares.')

for i in range(len(y_test)):

    real_open = prep.scale_val(y_test[i], 0, 1, scale_min, scale_max)
    pred_open = prep.scale_val(y_predicted[0][i], 0, 1, scale_min, scale_max)

    print(f'day {i}:')
    print(f'current money: ${cash}')
    print(f'current shares: {num_shares}')
    print(f'value of shares: ${num_shares *  real_open}')

    print('predicted opening cost:', pred_open)
    print('real opening cost:', real_open)

    if i > 0:
        yesterday_close = dataset.closes.raw[n + i - 1]
        print('today close:', yesterday_close)
        if num_shares > 0 and yesterday_close >= pred_open:
            # Hold or buy, we expect stock to go up
            if num_shares == 0:
                # Buy
                num_shares += cash / yesterday_close
                cost = yesterday_close * num_shares
                cash -= cost
                print(f'ACTION: bought {num_shares} for ${cost}.')

            else:
                # Hold
                print(f'ACTION: holding onto stock.')
        else:
            # Sell
            shares_to_sell = num_shares
            num_shares -= shares_to_sell
            cost = yesterday_close * shares_to_sell
            cash += cost
            print(f'ACTION: sold {shares_to_sell} for ${cost}.')

    print('')
    if cash < 0 or num_shares < 0:
        print(f'cash or num_shares are negative: cash={cash}, num_shares={num_shares}')
        sys.exit(-1)

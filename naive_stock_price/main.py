from typing import Dict

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers

import stockmarket
import prep
import grapher

history_points = 50

print('fetching stock data...')
stock_market = stockmarket.load_stocks(['AMD', 'MSFT', 'AAPL', 'AMZN', 'SNAP', 'CRSP', 'VXRT', 'TSLA', 'INTC', 'T', 'MCK', 'BLDP', 'SNE', 'BA', 'ATO', 'HOTH', 'AVGR',
                                        'NOG', 'AMC', 'NFLX'])
all_stocks_data: Dict[str, prep.StockData] = prep.generate_stock_data(stock_market)
my_stock_raw = all_stocks_data['AAPL']

print('graphing filtered data...')
window_to_graph = len(my_stock_raw.adjusted_closes)
prices = my_stock_raw.adjusted_closes[-window_to_graph:]
graph_floated_dates = prep.get_floated_dates(my_stock_raw.dates)[-window_to_graph:]
lines = [grapher.GraphLine('raw', graph_floated_dates, prices, dots=False),
         grapher.GraphLine('ema15', graph_floated_dates, prep.ema(prices, 15)),
         grapher.GraphLine('ema45', graph_floated_dates, prep.ema(prices, 45)),
         grapher.GraphLine('ema60', graph_floated_dates, prep.ema(prices, 60))]
grapher.plot_n_comparison(f'Filters of {my_stock_raw.symbol}', lines)

print('creating x data...')
normalized_volume = prep.get_scaled_datas(my_stock_raw.volumes).norm
normalized_dividends = prep.get_scaled_datas(my_stock_raw.dividend_amounts).norm
normalized_closes = prep.get_scaled_datas(my_stock_raw.closes).norm
x_linear = np.array([normalized_volume, normalized_dividends, normalized_closes])
x_linear = np.transpose(x_linear)
x_data = np.array([x_linear[i:i + history_points].copy() for i in range(len(x_linear) - history_points - 1)])  # why -1? This is because the y data is offset by a day

print('creating y data...')
normalized_filtered_prices = prep.get_scaled_datas(prep.ema(my_stock_raw.adjusted_closes, 15)).norm
y_data = np.array([normalized_filtered_prices[i + history_points + 1] for i in range(len(normalized_filtered_prices) - history_points - 1)])
y_data_raw = np.array([normalized_closes[i + history_points + 1] for i in range(len(normalized_filtered_prices) - history_points - 1)])
print(f'x_data.shape={x_data.shape}')
print(f'y_data.shape={y_data.shape}')

print('splitting data into training and testing...')
test_split = 0.9
num_of_training_elements = int(x_data.shape[0] * test_split)
x_train = x_data[:num_of_training_elements]
y_train = y_data[:num_of_training_elements]
x_test = x_data[num_of_training_elements:]
y_test = y_data[num_of_training_elements:]

print(f'x_train.shape={x_train.shape}')
print(f'y_train.shape={y_train.shape}')
print(f'x_test.shape={x_test.shape}')
print(f'y_test.shape={y_test.shape}')


print('creating model...')
lstm_input = Input(shape=(history_points, 3), name='lstm_input')
x = LSTM(history_points, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(65, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)
model = Model(inputs=lstm_input, outputs=output)
model.compile(optimizer=optimizers.Adam(lr=0.0005), loss='mse')
model.summary()

print('training model...')
model.fit(x=x_train,
          y=y_train,
          batch_size=32,
          epochs=5,
          shuffle=True,
          validation_split=0.1)
evaluation = model.evaluate(x_test, y_test)
print('evaluation of test data', evaluation)

evaluation = model.evaluate(x_data, y_data)
print('evaluation of all data=', evaluation)

print('generating results...')
prediction = model.predict(x_data)
prediction = np.transpose(prediction)

print('graphing results...')
floated_dates = prep.get_floated_dates(my_stock_raw.dates)
grapher.plot_model_comparison(f'model prediction for [{my_stock_raw.symbol}]',

                              'real [filtered]',
                              floated_dates[:-history_points-1],
                              y_data,

                              'modeled',
                              floated_dates[:-history_points-1],
                              prediction[0])

# 'real [raw]',
# floated_dates[:-history_points - 1],
# y_data,

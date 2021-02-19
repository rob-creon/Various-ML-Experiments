from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import datetime
# from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def discrete_derivative(data, time_period=1):
    dd = [0] * len(data)
    for i in range(time_period, len(data)):
        dd[i] = data[i] - data[i-time_period]
    return dd


df = pd.read_csv('input/CMP.csv')
size = (len(df['Date']))
print('size', size)

plt.figure()
lag_plot(df['Close'], lag=2)
plt.title('Transocean Stock - Autocorrelation plot with lag = 3')
plt.show()

x_interval = 30
y_interval = 50
max_y = max(df['Close'])

plt.figure(figsize=(10, 10))
plt.plot(df["Date"], df["Close"])
plt.xticks(np.arange(0, size, x_interval), df['Date'][0:size:x_interval], rotation=90)
plt.yticks(np.arange(0, int(max_y)+y_interval, y_interval), np.arange(0, int(max_y)+y_interval, y_interval))
plt.title("Transocean stock price over time")
plt.xlabel("time")
plt.ylabel("price")
plt.grid(True)
plt.show()

train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]

training_data = train_data['Close'].values
test_data = test_data['Close'].values

history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)

for time_point in range(N_test_observations):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
    print('{:.3f}% complete'.format(time_point/N_test_observations * 100))

MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))

test_size = int(len(df)*0.7)
test_set_range = df[test_size:].index
print(f'test_set_range={test_set_range}')
fig = plt.figure(figsize=(30, 30))
plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed', label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', marker='o', linestyle='dashed', label='Actual Price')
plt.title('Transocean Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.xticks(np.arange(test_size, size, x_interval), df.Date[test_size:size:x_interval])
plt.yticks(np.arange(0, int(max_y)+y_interval, y_interval), np.arange(0, int(max_y)+y_interval, y_interval))
plt.legend()
plt.show()
fig.savefig('preds.svg', format='svg')


dd_real = discrete_derivative(test_data)
dd_pred = discrete_derivative(model_predictions)
max_y = max(max(test_data), max(model_predictions))

fig = plt.figure(figsize=(30, 30))
plt.plot(test_set_range, dd_pred, color='blue', marker='o', linestyle='dashed', label='Predicted Price Change')
plt.plot(test_set_range, dd_real, color='red', marker='o', linestyle='dashed', label='Actual Price Change')
plt.title('Transocean Prices Prediction Delta Price')
plt.xlabel('Date')
plt.ylabel('Change in Price')
plt.xticks(np.arange(test_size, size, x_interval), df.Date[test_size:size:x_interval])
plt.yticks(np.arange(0, int(max_y)+y_interval, y_interval), np.arange(0, int(max_y)+y_interval, y_interval))
plt.legend()
plt.show()
fig.savefig('delta_preds.svg', format='svg')

Portfolio = namedtuple('Portfolio', ['money', 'shares', 'correct_signs'])

def print_portfolio(portfolio, portfolio_name):
    print(f'portfolio_name:')
    print(f'Ending account value: ${port.money + (port.shares * test_data[-1])}')
    print(f'Sign Value: {port.correct_signs / len(model_predictions) * 100}%')
    print()


arima_portfolio = Portfolio(10000, 0, 0)
hold_portfolio = Portfolio(10000, 0, 0)
coin_portfolio = Portfolio(10000, 0, 0)
perfect_portfolio = Portfolio(10000, 0, 0)

print('Evaluating returns of algorithm over test period.')

hold_portfolio.shares += hold_portfolio.money / test_data[0]

port = arima_portfolio
for i in range(len(model_predictions)-1):
    date = df.Date[test_size + i]

    today = test_data[i]
    pred = model_predictions[i+1]
    print(f'{date}, Price={today}, Prediction for Tomorrow:{pred}')
    if port.money > 0 and pred > today:
        shares_to_buy = port.money / today
        print(f'Bought {shares_to_buy} shares for ${port.money}.')
        money = 0
        port.shares += shares_to_buy
    elif port.shares > 0 and pred < today:
        cost_of_shares = port.shares * today
        print(f'Sold {port.shares} for ${cost_of_shares}.')
        shares = 0
        port.money += cost_of_shares
    else:
        print('No action today.')

    tomorrow = test_data[i+1]
    if (tomorrow > today and pred > today) or (tomorrow < today and pred < today):
        port.correct_signs += 1
    print()
print_portfolio('ARIMA portfolio', arima_portfolio)
print_portfolio('Hold portfolio', hold_portfolio)
print_portfolio('Coinflip portfolio', coin_portfolio)


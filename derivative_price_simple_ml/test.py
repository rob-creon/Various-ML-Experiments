import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


from creontest.dataset import DerivativeDataset
from creontest.grapher import GraphLine
from creontest.model import DeepLearningStockModel, CoinFlipModel, HoldModel
from creontest import grapher
from creontest import stockmarket
from creontest import utils
from creontest import stockmarket_sim


print('fetching stock data...')
stock_data = stockmarket.get_stock_data('GE')

print('plotting stock data...')
grapher.plot_stock_data(stock_data)
# stock_data = stockmarket.get_test_data(1000)

print('parsing stock data...')
params = DerivativeDataset(stock_data).to_model_params(0.9)

print('creating model...')
model = DeepLearningStockModel('ddx_model', params)
coin_flip_model = CoinFlipModel('coinflip_model', params)
hold_model = HoldModel('hold_model', params)

print('training model...')
train_loss, test_loss = model.train(epochs=5000, batch_size=32)
print('training_loss', train_loss)
print('test_loss', test_loss)

print('running model...')
result_train = model.predict_training_data()
result_test = model.predict_test_data()

print('graphing results...')
training_starting_price = stock_data.adjusted_closes[0]
testing_starting_price = stock_data.adjusted_closes[int(0.9 * len(stock_data.adjusted_closes))]

print('training_starting_price', training_starting_price)
print('testing_starting_price', testing_starting_price)

train_dates = params.x_linear_train[:, 0][params.history_points:-1]
test_dates = params.x_linear_test[:, 0][params.history_points:-1]

#################

train_real_ddx_line = GraphLine('real ddx', None, params.y_train)
train_real_int_line = GraphLine('real int', None, utils.integrate_from_starting_value(training_starting_price, params.y_train))

test_real_ddx_line = GraphLine('real ddx', None, params.y_test)
test_real_int_line = GraphLine('real int', None, utils.integrate_from_starting_value(testing_starting_price, params.y_test))

#################

train_model_ddx_line = GraphLine('model ddx', None, result_train)
train_model_int_line = GraphLine('model int', None, utils.integrate_from_starting_value(training_starting_price, result_train))

test_model_ddx_line = GraphLine('model ddx', None, result_test)
test_model_int_line = GraphLine('model int', None, utils.integrate_from_starting_value(testing_starting_price, result_test))

ema_adj = utils.ema(stock_data.adjusted_closes, 30, remove_start=True)
int_ema = utils.integrate_from_starting_value(ema_adj[0], params.y_all)
print('integrated_real_ending_price', int_ema[-1])
print('real_ending_price', ema_adj[-1])

grapher.plot('integral loss comparison', [GraphLine('raw', None, stock_data.adjusted_closes), GraphLine('integrated_ema', None, int_ema), GraphLine('ema', None, ema_adj)], auto_scale=True)


#################

grapher.plot('training ddx', [train_real_ddx_line, train_model_ddx_line], auto_scale=True)
grapher.plot('testing ddx', [test_real_ddx_line, test_model_ddx_line], auto_scale=True)

grapher.plot('training int', [train_real_int_line, train_model_int_line], auto_scale=True)
grapher.plot('testing int', [test_real_int_line, test_model_int_line], auto_scale=True)

#####################################################################################
# Show what the algorithm's predictions would look like with daily price adjustments#
#####################################################################################

dd_model_value = stockmarket_sim.do_stock_sim(result_test, params, stock_data) - 10000
coinflip_model_value = stockmarket_sim.do_stock_sim(coin_flip_model.predict_test_data(), params, stock_data, silent=True) - 10000
hold_model_value = stockmarket_sim.do_stock_sim(hold_model.predict_test_data(), params, stock_data, silent=True) - 10000
print()
print(f'With $10K, the ML model returned a {"profit" if dd_model_value > 0 else "loss"} of ${dd_model_value}.')
print(f'With $10K, the Coinflip model returned a {"profit" if coinflip_model_value > 0 else "loss"} of ${coinflip_model_value}.')
print(f'With $10K, the Buy-and-Hold model returned a {"profit" if hold_model_value > 0 else "loss"} of ${hold_model_value}.')

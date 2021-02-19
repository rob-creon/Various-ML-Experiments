import os
import sys

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from collections import namedtuple

from creontest.dataset import BasicDataset, Dataset, EmptyDataset, \
    SimpleDataset, RelatedDataset, DerivativeDataset
from creontest.grapher import GraphLine
from creontest.model import DeepLearningStockModel, ReturnToAvgModel
from creontest.stats_logger import StatsCSV
from creontest import grapher, stockmarket_sim
from creontest import stockmarket
from creontest import utils

import numpy as np

batch_size_to_test = 32
epochs_to_test = [50]
ModelDef = namedtuple('ModelDef', ['name', 'data_set'])
ModelResult = namedtuple('ModelResult', ['model_name', 'pred_train', 'pred_test', 'eval_train', 'eval_test'])


def run_model(model_def: ModelDef, epochs=50, deep=True):
    print(f'calculating model {model_def.name}...')
    model_params = model_def.data_set.to_model_params(0.9)
    if deep:
        model = DeepLearningStockModel(model_def.name, model_params)
        eval_train, eval_test = model.train(epochs=epochs, batch_size=batch_size_to_test)
    else:
        model = ReturnToAvgModel(model_def.name, model_params)
        eval_train, eval_test = model.train()

    print(f'{model_def.name} training loss:', eval_train)
    print(f'{model_def.name} testing loss:', eval_test)
    # print('saving model...')
    # model.save()
    return ModelResult(model_def.name, pred_train=model.predict_training_data(), pred_test=model.predict_test_data(), eval_train=eval_train, eval_test=eval_test)


print('fetching stock data...')
stock_data = stockmarket.get_stock_data('RIG')

print('fetching related stock datas...')
related_stocks = stockmarket.get_list_of_stock_data(['AAPL', 'MSFT', 'T'])

print('parsing stock data...')
sample_params = BasicDataset(stock_data).to_model_params(0.9)

print('defining models...')
model_defs = [#ModelDef('ddx', DerivativeDataset(stock_data)),
              # ModelDef('srs', RelatedDataset(stock_data, related_stocks)),
              # ModelDef('empty', EmptyDataset(stock_data)),
              # ModelDef('simple', SimpleDataset(stock_data)),
              ModelDef('standard', BasicDataset(stock_data)),
              ]

print('running models...')
results = {}
for e in epochs_to_test:
    print(f'running models for {e} epochs.')
    results[e] = [run_model(model_d, epochs=e) for model_d in model_defs]

    result = results[e]

    print(f'finished running models for {e} epochs. graphing results...')
    train_data_lines = [GraphLine('derivative', None, sample_params.y_train)]
    test_data_lines = [GraphLine('Real Delta Price', None, sample_params.y_test)]

    for r in results[e]:
        model_return_train = stockmarket_sim.do_stock_sim(r.pred_train, sample_params, stock_data) - 10000
        model_return_test = stockmarket_sim.do_stock_sim(r.pred_test, sample_params, stock_data) - 10000
        train_data_lines.append(GraphLine(f'delta {r.model_name}, eval={r.eval_train}, returns=${model_return}', None, r.pred_train))
        test_data_lines.append(GraphLine(f'delta {r.model_name}, eval={r.eval_test}, returns=${model_return}', None, r.pred_test))


    grapher.plot('Models vs Real [training data]', train_data_lines,
                 str_dates=False, dimensions=(16, 9), save=True,
                 file_name=f'predictions_training_batchsize{batch_size_to_test}_epoch{e}',
                 auto_scale=True)

    grapher.plot('Models vs Real [test data]', test_data_lines, str_dates=False,
                 dimensions=(16, 9), save=True, file_name=f'predictions_testing_batchsize{batch_size_to_test}_epoch{e}',
                 auto_scale=True)



print('finished running all models...')

print('logging results to csv...')
stats_csv_training = StatsCSV(f'loss_training_batchsize{batch_size_to_test}.csv', ['epochs'] + [model_d.name for model_d in model_defs])
stats_csv_testing = StatsCSV(f'loss_testing_batchsize{batch_size_to_test}.csv', ['epochs'] + [model_d.name for model_d in model_defs])

# result is a list of ModelResults
# 'ModelResult', ['model_name', 'eval_train', 'eval_test']
for epoch, model_set_result in results.items():
    models_training_losses = [str(single_model_result.eval_train) for single_model_result in model_set_result]
    models_testing_losses = [str(single_model_result.eval_test) for single_model_result in model_set_result]

    stats_csv_training.add_benchmark([str(epoch)] + models_training_losses)
    stats_csv_testing.add_benchmark([str(epoch)] + models_testing_losses)

stats_csv_training.write()
stats_csv_testing.write()

print('finished writing.')

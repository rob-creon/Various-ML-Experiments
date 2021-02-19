import datetime
import random
from abc import ABC, abstractmethod

from creontest import utils
from creontest.dataset import ModelParams
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

import tensorflow_addons as tfa

import numpy as np
np.random.seed(4)

import tensorflow as tf
tf.random.set_seed(4)


class StockModel(ABC):
    def __init__(self, name, model_params):
        self._name = name
        self._params = model_params

    def train(self):
        pass

    def predict_training_data(self):
        return self._predict_data(self._params.x_train)

    def predict_test_data(self):
        return self._predict_data(self._params.x_test)

    def predict_all_data(self):
        return self._predict_data(self._params.x_all)

    @abstractmethod
    def _predict_data(self, data):
        raise NotImplementedError()


class CoinFlipModel(StockModel):
    def __init__(self, name, model_params):
        super().__init__(name, model_params)

    def train(self):
        # todo return useful values, create the shallow learning model class
        return 0.5, 0.5

    def predict_training_data(self):
        return self._predict_data(self._params.x_linear_train)

    def predict_test_data(self):
        return self._predict_data(self._params.x_linear_test)

    def _predict_data(self, data):
        d = []
        for i in range(len(data) - self._params.history_points + 1):
            d.append(random.getrandbits(1)*2 - 1)
        return d


class HoldModel(StockModel):
    def __init__(self, name, model_params):
        super().__init__(name, model_params)

    def train(self):
        # todo return useful values, create the shallow learning model class
        return 0.5, 0.5

    def predict_training_data(self):
        return self._predict_data(self._params.x_linear_train)

    def predict_test_data(self):
        return self._predict_data(self._params.x_linear_test)

    def _predict_data(self, data):
        d = []
        for i in range(len(data) - self._params.history_points + 1):
            d.append(1)
        return d


class ReturnToAvgModel(StockModel):
    def __init__(self, name, model_params):
        super().__init__(name, model_params)
        self._avg_change = 0
        self._avg_price = 0

    def train(self):
        data = self._params.y_train
        dd = utils.discrete_derivative(self._params.y_train)

        self._avg_price = sum(data) / len(data)
        self._avg_change = abs(sum(dd) / len(dd))

        # todo move this to a shallow learning model base class
        train_predictions = self._predict_data(self._params.x_linear_train)
        test_predictions = self._predict_data(self._params.x_linear_test)

        return utils.mse(train_predictions, self._params.y_train), utils.mse(test_predictions, self._params.y_test)

    def predict_training_data(self):
        return self._predict_data(self._params.x_linear_train)

    def predict_test_data(self):
        return self._predict_data(self._params.x_linear_test)

    def _predict_data(self, data):
        dd = utils.discrete_derivative(data.transpose()[1], 1)

        while len(dd) < len(data):
            dd.append(0)
        pred = []
        for i in range(self._params.history_points, len(data)):
            day = data[i]
            price = day[1]
            if i - (self._params.history_points + 1) < 0:
                continue

            if price < self._avg_price:
                pred.append(price + self._avg_change)
            elif price > self._avg_price:
                pred.append(price - self._avg_change)
            else:
                pred.append(price)
        return pred


class DeepLearningStockModel(StockModel):

    def __init__(self, name, model_params: ModelParams, print_summary=False):
        super().__init__(name, model_params)

        lstm_input = Input(
            shape=(model_params.history_points, model_params.input_size),
            name='lstm_input')
        x = LSTM(model_params.history_points, name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        x = Dense(65, name='dense_0')(x)
        x = Activation('relu', name='relu_0')(x)
        #x = Activation('sigmoid', name='sigmoid_0')(x)
        x = Dense(1, name='dense_1')(x)
        output = Activation('linear', name='linear_output')(x)

        opt = tf.keras.optimizers.SGD(0.01)
        opt = tfa.optimizers.SWA(opt, start_averaging=0, average_period=10)

        self._model = Model(inputs=lstm_input, outputs=output)
        #self._model.compile(optimizer=optimizers.Adam(lr=0.0005), loss='mse')
        #self._model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss='mse')
        # self._model.compile(optimizer=optimizers.RMSprop(), loss='mse')
        self._model.compile(optimizer=opt, loss='mse')

        if print_summary:
            self._model.summary()

        self._name = name
        self._params = model_params

    def train(self, epochs=150, batch_size=256):
        self._model.fit(x=self._params.x_train,
                        y=self._params.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True)
        train_evaluation = self._model.evaluate(self._params.x_train, self._params.y_train)
        test_evaluation = self._model.evaluate(self._params.x_test, self._params.y_test)

        return train_evaluation, test_evaluation

    def _predict_data(self, data):
        return self._model.predict(data)

    def save(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._model.save(f'saved_models/{self._name}_{timestamp}')

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Subtract, Add
from keras.models import Model
from keras.optimizers import Adam


class N_Beats:
    def __init__(self):
        self.number_slacks = 2
        self.number_blocks = [6, 6]
        assert len(self.number_blocks) == self.number_slacks
        self.block_types = ['trend', 'trend', 'trend', 'seasonality', 'seasonality', 'seasonality']
        # self.block_types = ['generic', 'generic', 'generic', 'generic', 'generic', 'generic']
        assert len(self.block_types) == self.number_blocks[0]
        self.backcast_length = 10
        self.forecast_length = 1
        self.units = 256
        self.nb_poly = 3
        self.best_perf = 100.0
        self.epochs = 10001
        self.plot_results = 100

        X_ = Input(shape=(self.backcast_length,))
        x_ = Lambda(lambda x: x)(X_) 
        for i in range(self.number_slacks):
            for j in range(self.number_blocks[i]):
                d1 = Dense(self.units, activation='relu')(x_)
                d2 = Dense(self.units, activation='relu')(d1)
                d3 = Dense(self.units, activation='relu')(d2)
                d4 = Dense(self.units, activation='relu')(d3)
                if self.block_types[j] == 'generic':
                    theta_b = Dense(self.nb_poly, activation='linear')(d4)
                    theta_f = Dense(self.nb_poly, activation='linear')(d4)
                    backcast = Dense(self.backcast_length, activation='linear')(theta_b)
                    forecast = Dense(self.forecast_length, activation='linear')(theta_f)
                    x_ = Subtract()([x_, backcast])
                    if i == 0 and j == 0:
                        y_ = forecast
                    else :
                        y_ = Add()([y_, forecast])
                if self.block_types[j] == 'trend':
                    theta_f = theta_b = Dense(self.nb_poly, activation='linear')(d4)
                    backcast = Lambda(trend_model, arguments={"is_forecast": False, "length": self.backcast_length})(
                        theta_b)
                    forecast = Lambda(trend_model, arguments={"is_forecast": True, "length": self.forecast_length})(
                        theta_f)
                    x_ = Subtract()([x_, backcast])
                    if i == 0 and j == 0:
                        y_ = forecast
                    else :
                        y_ = Add()([y_, forecast])
                if self.block_types[j] == 'seasonality':
                    theta_b = theta_f = Dense(self.backcast_length, activation='linear')(d4)
                    # thetas_b = Dense(self.backcast_length, activation='linear')(d4) 
                    # theta_b and theta_f are shared even for seasonality bloc
                    backcast = Lambda(seasonality_model,
                                      arguments={"is_forecast": False, "length": self.backcast_length})(theta_b)
                    forecast = Lambda(seasonality_model,
                                      arguments={"is_forecast": True, "length": self.forecast_length})(theta_f)
                    x_ = Subtract()([x_, backcast])
                    if i == 0 and j == 0:
                        y_ = forecast
                    else :
                        y_ = Add()([y_, forecast])

        model = Model(X_, y_)
        model.summary()

        optimizer = Adam(lr=0.00001)
        self.nbeats = model
        self.nbeats.compile(loss='mae', optimizer=optimizer)

    def train_model(self):

        print('loading dev data')
        x_test, y_test = get_xtest_circular(self.backcast_length, self.forecast_length)
        print(x_test.shape)

        nb_epochs = self.epochs
        for step in range(nb_epochs):
            # print('loading train data')
            x, y = get_xtrain_circular(self.backcast_length, self.forecast_length)
            # print(x.shape)
            # print(y.shape)
            # history = self.nbeats.train_on_batch(x, y)
            self.nbeats.train_on_batch(x, y)

            if step % self.plot_results == 0:
                print(step)
                if not os.path.exists('results'):
                    os.makedirs('results')
                self.nbeats.save('results/n_beats_model_20_' + str(step) + '.h5')
                predictions = self.nbeats.predict(x_test)
                residual = y_test - predictions
                print(get_accuracy(y_test, predictions))
                if get_accuracy(y_test, predictions)[0] < self.best_perf:
                    self.best_perf = get_accuracy(y_test, predictions)[0]
                    self.nbeats.save('results/n_beats_model_20_ongoing.h5')
                print(predictions.shape)
                print(residual.shape)
                plt.figure()
                plt.grid(True)
                print(x.shape)
                print(y.shape)
                print(predictions.shape)
                print(residual.shape)
                x_y = np.concatenate([x[0, :], y[0, :]], axis=-1).flatten()
                plt.plot(list(range(self.backcast_length)), x[0, :].flatten(), color='b')
                plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), y[0, :].flatten(), color='g')
                plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), predictions[0, :].flatten(), color='r')
                plt.scatter(range(len(x_y)), x_y.flatten(),
                            color=['b'] * self.backcast_length + ['g'] * self.forecast_length)
                plt.scatter(list(range(len(x_y) - self.forecast_length, len(x_y))), predictions[0, :].flatten(),
                            color=['r'] * self.forecast_length)
                plt.legend(['backcast', 'forecast', 'predictions of forecast'])
                plt.savefig('results/' + str(step) + '.png')
                plt.close()
                validation_predictions = self.nbeats.predict(x_test)
                x_y = np.concatenate([x_test[0, :], y_test[0, :]], axis=-1).flatten()
                plt.figure()
                plt.plot(list(range(self.backcast_length)), x_test[0, :].flatten(), color='b')
                plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), y_test[0, :].flatten(), color='g')
                plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), validation_predictions[0, :].flatten(),
                         color='r')
                plt.scatter(range(len(x_y)), x_y.flatten(),
                            color=['b'] * self.backcast_length + ['g'] * self.forecast_length)
                plt.scatter(list(range(len(x_y) - self.forecast_length, len(x_y))),
                            validation_predictions[0, :].flatten(), color=['r'] * self.forecast_length)
                plt.legend(['backcast', 'forecast', 'predictions of forecast'])
                if not os.path.exists('results/test'):
                    os.makedirs('results/test')
                plt.savefig('results/test/' + str(step) + '.png')
                plt.close()

        self.nbeats.save('results/n_beats_model_20_small.h5')

        predictions = self.nbeats.predict(x)
        x_y = np.concatenate([x[100, :], y[100, :]], axis=-1).flatten()
        plt.plot(list(range(self.backcast_length)), x[100, :].flatten(), color='b')
        plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), y[100, :].flatten(), color='g')
        plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), predictions[100, :].flatten(), color='r')
        plt.scatter(range(len(x_y)), x_y.flatten(), color=['b'] * self.backcast_length + ['g'] * self.forecast_length)
        plt.scatter(list(range(len(x_y) - self.forecast_length, len(x_y))), predictions[100, :].flatten(),
                    color=['r'] * self.forecast_length)
        plt.legend(['backcast', 'forecast', 'predictions of forecast'])
        plt.savefig('results/' + str(nb_epochs) + '.png')

        validation_predictions = self.nbeats.predict(x_test)
        x_y = np.concatenate([x_test[10, :], y_test[10, :]], axis=-1).flatten()
        plt.figure()
        plt.plot(list(range(self.backcast_length)), x_test[10, :].flatten(), color='b')
        plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), y_test[10, :].flatten(), color='g')
        plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), validation_predictions[10, :].flatten(),
                 color='r')
        plt.scatter(range(len(x_y)), x_y.flatten(), color=['b'] * self.backcast_length + ['g'] * self.forecast_length)
        plt.scatter(list(range(len(x_y) - self.forecast_length, len(x_y))), validation_predictions[10, :].flatten(),
                    color=['r'] * self.forecast_length)
        plt.legend(['backcast', 'forecast', 'predictions of forecast'])
        plt.savefig('results/test/' + str(nb_epochs) + '.png')

        print(get_accuracy(y_test, validation_predictions))


def linear_space(length, fwd_looking=True):
    if fwd_looking:
        t = np.linspace(0, length - 1, length)
    else:
        t = np.linspace(-length, -1, length)
    t = t / length
    return t


def seasonality_model(thetas, length, is_forecast):
    p = thetas.shape[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(length, fwd_looking=is_forecast)
    s1 = np.stack([np.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)  # H/2-1
    s2 = np.stack([np.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    S = np.concatenate([s1, s2], axis=0)
    S = K.cast(S, np.float32)
    return K.dot(thetas, S)


def trend_model(thetas, length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(length, fwd_looking=is_forecast)
    T = K.transpose(tf.stack([t ** i for i in range(p)], axis=0))
    T = K.cast(T, np.float32)
    return K.dot(thetas, K.transpose(T))


def get_xtrain_circular(backcast_length, forecast_length):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    x_train = np.array([]).reshape(0, backcast_length)
    y_train = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    filename = 'data/train/'
    with open(filename + 'Daily-train.csv', "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
    x_tl_tl = np.array(x_tl)
    x_tl_tl = x_tl_tl[:, 1:]
    for i in range(x_tl_tl.shape[0]):
        if (len(x_tl_tl[i]) < backcast_length + forecast_length):
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
        time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
        j = np.random.randint(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length)
        time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
        time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        x_train = np.vstack((x_train, time_series_cleaned_forlearning_x))
        y_train = np.vstack((y_train, time_series_cleaned_forlearning_y))

    ts_min = np.min(x_train, axis = 1).reshape((x_train.shape[0], 1))
    ts_max = np.max(x_train, axis = 1).reshape((x_train.shape[0], 1))
    x_train = (x_train - ts_min) / (ts_max - ts_min + 1e-8)
    y_train = (y_train - ts_min) / (ts_max - ts_min + 1e-8)
    return x_train, y_train


def get_xtest_circular(backcast_length, forecast_length):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    # zero-size array to reduction operation minimum which has no identity
    x_test = np.array([]).reshape(0, backcast_length)
    y_test = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    filename = 'data/val/'
    with open(filename + 'Daily-test.csv', "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
    x_tl_tl = np.array(x_tl)
    x_tl_tl = x_tl_tl[:, 1:]
    for i in range(x_tl_tl.shape[0]):
        if (len(x_tl_tl[i]) < backcast_length + forecast_length):
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        time_series_cleaned_forlearning_x = np.zeros(
            (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
        time_series_cleaned_forlearning_y = np.zeros(
            (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
        for j in range(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length):
            time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series_cleaned[j - backcast_length:j]
            time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series_cleaned[j: j + forecast_length]
        x_test = np.vstack((x_test, time_series_cleaned_forlearning_x))
        y_test = np.vstack((y_test, time_series_cleaned_forlearning_y))
        
    ts_min = np.min(x_test, axis = 1).reshape((x_test.shape[0], 1))
    ts_max = np.max(x_test, axis = 1).reshape((x_test.shape[0], 1))
    x_test = (x_test - ts_min) / (ts_max - ts_min + 1e-8)
    y_test = (y_test - ts_min) / (ts_max - ts_min + 1e-8)

    return x_test, y_test


def get_accuracy(y_true, y_hat):
    preds = y_hat[:, 0]
    true_y = y_true[:, 0]
    print(preds)
    print(true_y)
    error = np.mean(np.square(true_y - preds))
    smape = np.mean(2 * np.abs(true_y - preds) / (np.abs(true_y) + np.abs(preds)))
    correl = np.corrcoef(true_y, preds)

    return smape, error, correl


def train():
    nbeats = N_Beats()
    print(nbeats)
    nbeats.train_model()


if __name__ == '__main__':
    train()
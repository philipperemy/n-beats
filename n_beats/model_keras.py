import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Subtract, Add
from keras.models import Model
from keras.optimizers import Adam


class NBeats:
    def __init__(self):
        self.number_slacks = 2
        self.number_blocks = [6, 6]
        assert len(self.number_blocks) == self.number_slacks
        # self.block_types = ['trend', 'trend', 'trend', 'seasonality', 'seasonality', 'seasonality']
        self.block_types = ['generic', 'generic', 'generic', 'generic', 'generic', 'generic']
        assert len(self.block_types) == self.number_blocks[0]
        self.backcast_length = 10
        self.forecast_length = 2
        self.units = 256
        self.nb_poly = 3
        self.best_perf = 100.0
        self.steps = 10001
        self.plot_results = 100
        self.learning_rate = 1e-5
        self.loss = 'mae'

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
                if self.block_types[j] == 'trend':
                    theta_f = theta_b = Dense(self.nb_poly, activation='linear')(d4)
                    backcast = Lambda(trend_model, arguments={"is_forecast": False, "backcast_length": self.backcast_length, "forecast_length": self.forecast_length})(
                        theta_b)
                    forecast = Lambda(trend_model, arguments={"is_forecast": True, "backcast_length": self.backcast_length, "forecast_length": self.forecast_length})(
                        theta_f)
                if self.block_types[j] == 'seasonality':
                    theta_b = theta_f = Dense(self.backcast_length, activation='linear')(d4)
                    # theta_b = Dense(self.backcast_length, activation='linear')(d4) 
                    # theta_b and theta_f are shared even for seasonality bloc
                    backcast = Lambda(seasonality_model,
                                      arguments={"is_forecast": False, "backcast_length": self.backcast_length, "forecast_length": self.forecast_length})(theta_b)
                    forecast = Lambda(seasonality_model,
                                      arguments={"is_forecast": True, "backcast_length": self.backcast_length, "forecast_length": self.forecast_length})(theta_f)
                x_ = Subtract()([x_, backcast])
                if i == 0 and j == 0:
                    y_ = forecast
                else :
                    y_ = Add()([y_, forecast])

        model = Model(X_, y_)
        model.summary()
        
        self.nbeats = model
        self.compile_model(loss=self.loss, learning_rate=self.learning_rate)

    def compile_model(self, loss, learning_rate):
        optimizer = Adam(lr=learning_rate)
        self.nbeats.compile(loss=loss, optimizer=optimizer)
        
    def train_model(self):
        
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('results/test'):
            os.makedirs('results/test')
            
        x_test, y_test = get_data(self.backcast_length, self.forecast_length, is_training=False)
        print('x_test.shape=', x_test.shape)

        for step in range(self.steps):
            x, y = get_data(self.backcast_length, self.forecast_length, is_training=True)
            self.nbeats.train_on_batch(x, y)
            if step % self.plot_results == 0:
                print('step=', step)
                self.nbeats.save('results/n_beats_model_' + str(step) + '.h5')
                predictions = self.nbeats.predict(x)
                validation_predictions = self.nbeats.predict(x_test)
                smape = get_metrics(y_test, validation_predictions)[0]
                print('smape=', smape)
                if smape < self.best_perf:
                    self.best_perf = smape
                    self.nbeats.save('results/n_beats_model_ongoing.h5')
                self.plot_model_predictions(False, step, x[0, :], y[0, :], predictions[0, :])
                self.plot_model_predictions(True, step, x_test[0, :], y_test[0, :], validation_predictions[0])

        self.nbeats.save('results/n_beats_model.h5')

        predictions = self.nbeats.predict(x)
        validation_predictions = self.nbeats.predict(x_test)
        self.plot_model_predictions(False, self.steps, x[100, :], y[100, :], predictions[100, :])
        self.plot_model_predictions(True, self.steps, x_test[10, :], y_test[10, :], validation_predictions[10, :])

        print('smape=', get_metrics(y_test, validation_predictions)[0])
        print('error=', get_metrics(y_test, validation_predictions)[1])

    def plot_model_predictions(self, is_test, step, backcast, forecast, prediction):
        legend = ['backcast', 'forecast', 'predictions of forecast']
        if is_test:
            title = 'results/test/' + str(step) + '.png'
        else:
            title = 'results/' + str(step) + '.png'
        plt.figure()
        plt.grid(True)
        x_y = np.concatenate([backcast, forecast], axis=-1).flatten()
        plt.plot(list(range(self.backcast_length)), backcast.flatten(), color='b')
        plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), forecast.flatten(), color='g')
        plt.plot(list(range(len(x_y) - self.forecast_length, len(x_y))), prediction.flatten(), color='r')
        plt.scatter(range(len(x_y)), x_y.flatten(), color=['b'] * self.backcast_length + ['g'] * self.forecast_length)
        plt.scatter(list(range(len(x_y) - self.forecast_length, len(x_y))), prediction.flatten(), color=['r'] * self.forecast_length)
        plt.legend(legend)
        plt.savefig(title)
        plt.close()  
    
    
def linear_space(backcast_length, forecast_length, fwd_looking=True):
    l = K.arange(-float(backcast_length), float(forecast_length), 1) / backcast_length
    if fwd_looking:
        l = l[backcast_length:]
    else:
        l = l[:backcast_length]
    return l
    

def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)  # H/2-1
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    S = K.concatenate([s1, s2], axis=0)
    S = K.cast(S, np.float32)
    return K.dot(thetas, S)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    T = K.transpose(K.stack([t ** i for i in range(p)], axis=0))
    T = K.cast(T, np.float32)
    return K.dot(thetas, K.transpose(T))


def get_data(backcast_length, forecast_length, is_training=True):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    if is_training:
        filename = 'data/train/Daily-train.csv'
    else:
        filename = 'data/val/Daily-test.csv'

    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:] 
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False 
    x_tl_tl = np.array(x_tl)
    for i in range(x_tl_tl.shape[0]):
        if (len(x_tl_tl[i]) < backcast_length + forecast_length):
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        if is_training:
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length)
            time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        else:
            time_series_cleaned_forlearning_x = np.zeros((time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
            for j in range(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length):
                time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series_cleaned[j - backcast_length:j]
                time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series_cleaned[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))
        
    return x, y


def get_metrics(y_true, y_hat):
    error = np.mean(np.square(y_true - y_hat))
    smape = np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))

    return smape, error


def train():
    nbeats = NBeats()
    nbeats.train_model()


if __name__ == '__main__':
    train()
    
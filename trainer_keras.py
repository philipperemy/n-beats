import os

import matplotlib.pyplot as plt
import numpy as np

from data import get_m4_data, dummy_data_generator
from nbeats_keras.model import NBeatsNet


def get_metrics(y_true, y_hat):
    error = np.mean(np.square(y_true - y_hat))
    smape = np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))
    return smape, error


def ensure_results_dir():
    if not os.path.exists('results/test'):
        os.makedirs('results/test')


def generate_data(backcast_length, forecast_length):
    x_train, y_train = next(dummy_data_generator(backcast_length, forecast_length,
                                                 signal_type='seasonality', random=True,
                                                 batch_size=6_000))
    x_test, y_test = next(dummy_data_generator(backcast_length, forecast_length,
                                               signal_type='seasonality', random=True,
                                               batch_size=1_000))
    return x_train, y_train, x_test, y_test


def train_model(model: NBeatsNet):
    ensure_results_dir()

    x, y, x_test, y_test = generate_data(model.backcast_length, model.forecast_length)

    # x_test, y_test = get_m4_data(model.backcast_length, model.forecast_length, is_training=False)
    print('x_test.shape=', x_test.shape)

    for step in range(model.steps):
        # x, y = get_m4_data(model.backcast_length, model.forecast_length, is_training=True)
        model.nbeats.train_on_batch(x, y)
        if step % model.plot_results == 0:
            print('step=', step)
            model.nbeats.save('results/n_beats_model_' + str(step) + '.h5')
            predictions = model.nbeats.predict(x)
            validation_predictions = model.nbeats.predict(x_test)
            smape = get_metrics(y_test, validation_predictions)[0]
            print('smape=', smape)
            if smape < model.best_perf:
                model.best_perf = smape
                model.nbeats.save('results/n_beats_model_ongoing.h5')
            plot_keras_model_predictions(model, False, step, x[0, :], y[0, :], predictions[0, :])
            plot_keras_model_predictions(model, True, step, x_test[0, :], y_test[0, :], validation_predictions[0])

    model.nbeats.save('results/n_beats_model.h5')

    predictions = model.nbeats.predict(x)
    validation_predictions = model.nbeats.predict(x_test)
    plot_keras_model_predictions(model, False, model.steps, x[100, :], y[100, :], predictions[100, :])
    plot_keras_model_predictions(model, True, model.steps, x_test[10, :], y_test[10, :], validation_predictions[10, :])

    print('smape=', get_metrics(y_test, validation_predictions)[0])
    print('error=', get_metrics(y_test, validation_predictions)[1])


def plot_keras_model_predictions(model, is_test, step, backcast, forecast, prediction):
    legend = ['backcast', 'forecast', 'predictions of forecast']
    if is_test:
        title = 'results/test/' + str(step) + '.png'
    else:
        title = 'results/' + str(step) + '.png'
    plt.figure()
    plt.grid(True)
    x_y = np.concatenate([backcast, forecast], axis=-1).flatten()
    plt.plot(list(range(model.backcast_length)), backcast.flatten(), color='b')
    plt.plot(list(range(len(x_y) - model.forecast_length, len(x_y))), forecast.flatten(), color='g')
    plt.plot(list(range(len(x_y) - model.forecast_length, len(x_y))), prediction.flatten(), color='r')
    plt.scatter(range(len(x_y)), x_y.flatten(), color=['b'] * model.backcast_length + ['g'] * model.forecast_length)
    plt.scatter(list(range(len(x_y) - model.forecast_length, len(x_y))), prediction.flatten(),
                color=['r'] * model.forecast_length)
    plt.legend(legend)
    plt.savefig(title)
    plt.close()


def train():
    model = NBeatsNet()
    model.compile_model(loss='mae', learning_rate=1e-5)
    train_model(model)


if __name__ == '__main__':
    train()

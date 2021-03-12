import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from data import dummy_data_generator_multivariate, get_m4_data_multivariate, get_nrj_data, get_kcg_data

from nbeats_keras.model import NBeatsNet


def get_script_arguments():
    parser = ArgumentParser()
    parser.add_argument('--task', choices=['m4', 'kcg', 'nrj', 'dummy'], required=True)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def get_metrics(y_true, y_hat):
    error = np.mean(np.square(y_true - y_hat))
    smape = np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))
    return smape, error


def ensure_results_dir():
    if not os.path.exists('results/test'):
        os.makedirs('results/test')


def reshape_array(x):
    assert len(x.shape) == 2, 'input np.array should be in the format: samples, timesteps'
    if len(x.shape) == 2:
        nb_samples, nb_timestamps = x.shape
        return x.reshape((nb_samples, nb_timestamps, 1))


def generate_data(backcast_length, forecast_length):
    def gen(num_samples):
        return next(dummy_data_generator_multivariate(backcast_length, forecast_length,
                                                      signal_type='seasonality', random=True, batch_size=num_samples))

    x_train, y_train = gen(6_000)
    x_test, y_test = gen(1_000)

    x_train, y_train, x_test, y_test = reshape_array(x_train), reshape_array(y_train), reshape_array(
        x_test), reshape_array(y_test)
    return x_train, None, y_train, x_test, None, y_test


def train_model(model: NBeatsNet, task: str, best_perf=np.inf, max_steps=10001, plot_results=100, is_test=False):
    ensure_results_dir()
    # if is_test then override max_steps argument
    if is_test:
        max_steps = 5

    if task == 'dummy':
        x_train, e_train, y_train, x_test, e_test, y_test = generate_data(model.backcast_length, model.forecast_length)
    elif task == 'm4':
        x_test, e_test, y_test = get_m4_data_multivariate(model.backcast_length, model.forecast_length,
                                                          is_training=False)
    elif task == 'kcg':
        x_test, e_test, y_test = get_kcg_data(model.backcast_length, model.forecast_length, is_training=False)
    elif task == 'nrj':
        x_test, e_test, y_test = get_nrj_data(model.backcast_length, model.forecast_length, is_training=False)
    else:
        raise ValueError('Invalid task.')

    print('x_test.shape=', x_test.shape)

    x_train, y_train, e_train = None, None, None
    for step in range(max_steps):
        if task == 'dummy':
            x_train, e_train, y_train, x_test, e_test, y_test = generate_data(model.backcast_length,
                                                                              model.forecast_length)
        elif task == 'm4':
            x_train, e_train, y_train = get_m4_data_multivariate(model.backcast_length, model.forecast_length,
                                                                 is_training=True)
        elif task == 'kcg':
            x_train, e_train, y_train = get_kcg_data(model.backcast_length, model.forecast_length, is_training=True)
        elif task == 'nrj':
            x_train, e_train, y_train = get_nrj_data(model.backcast_length, model.forecast_length, is_training=True)
        else:
            raise ValueError('Invalid task.')

        if model.has_exog():
            model.train_on_batch([x_train, e_train], y_train)
        else:
            model.train_on_batch(x_train, y_train)

        if step % plot_results == 0:
            print('step=', step)
            model.save('results/n_beats_model_' + str(step) + '.h5')
            if model.has_exog():
                predictions = model.predict([x_train, e_train])
                validation_predictions = model.predict([x_test, e_test])
            else:
                predictions = model.predict(x_train)
                validation_predictions = model.predict(x_test)
            smape = get_metrics(y_test, validation_predictions)[0]
            print('smape=', smape)
            if smape < best_perf:
                best_perf = smape
                model.save('results/n_beats_model_ongoing.h5')
            for k in range(model.input_dim):
                plot_keras_model_predictions(model, False, step, x_train[0, :, k], y_train[0, :, k],
                                             predictions[0, :, k], axis=k)
                plot_keras_model_predictions(model, True, step, x_test[0, :, k], y_test[0, :, k],
                                             validation_predictions[0, :, k], axis=k)

    model.save('results/n_beats_model.h5')

    if model.has_exog():
        predictions = model.predict([x_train, e_train])
        validation_predictions = model.predict([x_test, e_test])
    else:
        predictions = model.predict(x_train)
        validation_predictions = model.predict(x_test)

    for k in range(model.input_dim):
        plot_keras_model_predictions(model, False, max_steps, x_train[10, :, k], y_train[10, :, k],
                                     predictions[10, :, k], axis=k)
        plot_keras_model_predictions(model, True, max_steps, x_test[10, :, k], y_test[10, :, k],
                                     validation_predictions[10, :, k], axis=k)
    print('smape=', get_metrics(y_test, validation_predictions)[0])
    print('error=', get_metrics(y_test, validation_predictions)[1])


def plot_keras_model_predictions(model, is_test, step, backcast, forecast, prediction, axis):
    legend = ['backcast', 'forecast', 'predictions of forecast']
    if is_test:
        title = 'results/test/' + 'step_' + str(step) + '_axis_' + str(axis) + '.png'
    else:
        title = 'results/' + 'step_' + str(step) + '_axis_' + str(axis) + '.png'
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


def main():
    args = get_script_arguments()

    if args.task in ['m4', 'dummy']:
        model = NBeatsNet(backcast_length=10, forecast_length=1,
                          stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                          thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=128)
    elif args.task == 'kcg':
        model = NBeatsNet(input_dim=2, backcast_length=360, forecast_length=10,
                          stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK), nb_blocks_per_stack=3,
                          thetas_dim=(4, 8), share_weights_in_stack=False,
                          hidden_layer_units=256)
    elif args.task == 'nrj':
        model = NBeatsNet(input_dim=1, exo_dim=2, backcast_length=10, forecast_length=1,
                          stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK), nb_blocks_per_stack=2,
                          thetas_dim=(4, 8), share_weights_in_stack=False, hidden_layer_units=128,
                          nb_harmonics=10)
    else:
        raise ValueError('Unknown task.')

    model.compile(loss='mae', optimizer='adam')
    train_model(model, args.task, is_test=args.test)


if __name__ == '__main__':
    main()

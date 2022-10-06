import itertools
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from examples.data import dummy_data_generator
from nbeats_keras.model import NBeatsNet as NBeatsKeras
from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch

warnings.filterwarnings('ignore')

matplotlib.rcParams.update({'font.size': 7})


def main():
    backcast_length = 10
    forecast_length = 10

    data_gen = dummy_data_generator(
        backcast_length=backcast_length, forecast_length=forecast_length,
        signal_type='seasonality', random=False, batch_size=32
    )
    num_samples_train = 1000
    num_samples_test = 200
    batches = list(itertools.islice(data_gen, num_samples_train))
    x_train = np.vstack([b[0] for b in batches])
    y_train = np.vstack([b[1] for b in batches])

    batches = list(itertools.islice(data_gen, num_samples_test))
    x_test = np.vstack([b[0] for b in batches])
    y_test = np.vstack([b[1] for b in batches])

    sample_idx = 10
    sample_x = x_test[sample_idx:sample_idx + 1]
    sample_y = y_test[sample_idx]

    for backend in [NBeatsKeras, NBeatsPytorch]:
        backend_name = backend.name()
        print(f'Running the example for {backend_name}...')
        model = backend(
            backcast_length=backcast_length, forecast_length=forecast_length,
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.TREND_BLOCK, NBeatsKeras.SEASONALITY_BLOCK),
            nb_blocks_per_stack=2, thetas_dim=(4, 4, 4), hidden_layer_units=20
        )
        model.compile(loss='mae', optimizer='adam')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
        model.enable_intermediate_outputs()
        model.predict(sample_x)  # load intermediary outputs into our model object.
        # NOTE: g_pred + i_pred = pred.
        g_pred, i_pred, outputs = model.get_generic_and_interpretable_outputs()
        plot(target=sample_y, generic_predictions=g_pred, interpretable_predictions=i_pred, backend_name=backend_name)
        subplots(outputs, backend_name)
    plt.show()


def subplots(outputs: dict, backend_name: str):
    layers = [a[0] for a in outputs.items()]
    values = [a[1] for a in outputs.items()]
    assert len(layers) == len(values)
    n = len(layers)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15, 3))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.75, wspace=0.4, hspace=0.4)
    for i in range(n):
        axes[i].plot(values[i], color='deepskyblue')
        axes[i].set_title(layers[i])
        axes[i].set_xlabel('t')
        axes[i].grid(axis='x')
        axes[i].grid(axis='y')
    fig.suptitle(f'{backend_name} - Outputs of the generic and interpretable configurations', fontweight='bold')
    plt.draw()


def plot(target, generic_predictions, interpretable_predictions, backend_name):
    plt.figure()
    plt.plot(target, color='gold', linewidth=2)
    plt.plot(interpretable_predictions + generic_predictions, color='r', linewidth=2)
    plt.plot(interpretable_predictions, color='orange')
    plt.plot(generic_predictions, color='green')
    plt.grid()
    plt.legend(['ACTUAL', 'FORECAST-PRED', 'FORECAST-I', 'FORECAST-G'])
    plt.xlabel('t')
    plt.title(f'{backend_name} - Forecast - Actual vs Predicted')
    plt.draw()


if __name__ == '__main__':
    main()

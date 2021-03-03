# N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- *Implementation in Keras by @eljdos (Jean-Sébastien Dhr)*
- *Implementation in Pytorch by @philipperemy (Philippe Remy)*
- https://arxiv.org/abs/1905.10437

<p align="center">
  <img src="nbeats.png" width="600"><br/>
  <i>N-Beats at the beginning of the training</i><br><br>
</p>

Trust me, after a few more steps, the green curve (predictions) matches the ground truth exactly :-)

## Installation

### From PyPI

Install Keras: `pip install nbeats-keras`.

Install Pytorch: `pip install nbeats-pytorch`.

### From the sources

Installation is based on a MakeFile. Make sure you are in a virtualenv and have python3 installed.

Command to install N-Beats with Keras: `make install-keras`

Command to install N-Beats with Pytorch: `make install-pytorch`

### Run on the GPU

To force the utilization of the GPU (Tensorflow), run: `pip uninstall -y tensorflow && pip install tensorflow-gpu`.

## Example

Jupyter notebook: [NBeats.ipynb](examples/NBeats.ipynb): `make run-jupyter`.

Here is a toy example on how to use this model (train and predict):

```python
import warnings

import numpy as np

from nbeats_keras.model import NBeatsNet as NBeatsKeras
from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch

warnings.filterwarnings(action='ignore', message='Setting attributes')


def main():
    # https://keras.io/layers/recurrent/
    num_samples, time_steps, input_dim, output_dim = 50_000, 10, 1, 1

    # Definition of the model.
    model_keras = NBeatsKeras(backcast_length=time_steps, forecast_length=output_dim,
                              stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
                              nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
                              hidden_layer_units=64)

    model_pytorch = NBeatsPytorch(backcast_length=time_steps, forecast_length=output_dim,
                                  stack_types=(NBeatsPytorch.GENERIC_BLOCK, NBeatsPytorch.GENERIC_BLOCK),
                                  nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
                                  hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    model_keras.compile_model(loss='mae', learning_rate=1e-5)
    model_pytorch.compile_model(loss='mae', learning_rate=1e-5)

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    y = np.mean(x, axis=1, keepdims=True)

    # Split data into training and testing datasets.
    c = num_samples // 10
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]

    # Train the model.
    model_keras.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)
    model_pytorch.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)

    # Save the model for later.
    model_keras.save('n_beats_model.h5')
    model_pytorch.save('n_beats_pytorch.th')

    # Predict on the testing set.
    predictions_keras = model_keras.predict(x_test)
    predictions_pytorch = model_pytorch.predict(x_test)
    print(predictions_keras.shape)
    print(predictions_pytorch.shape)

    # Load the model.
    model_keras_2 = NBeatsKeras.load('n_beats_model.h5')
    model_pytorch_2 = NBeatsPytorch.load('n_beats_pytorch.th')

    np.testing.assert_almost_equal(predictions_keras, model_keras_2.predict(x_test))
    np.testing.assert_almost_equal(predictions_pytorch, model_pytorch_2.predict(x_test))


if __name__ == '__main__':
    main()
```

## Citation

```
@misc{NBeatsPRemy,
  author = {Philippe Remy},
  title = {N-BEATS: Neural basis expansion analysis for interpretable time series forecasting},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/philipperemy/n-beats}},
}
```

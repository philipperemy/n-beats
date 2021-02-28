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

To force the utilization of the GPU, run: `pip uninstall -y tensorflow && pip install tensorflow-gpu`.

## Example

Jupyter notebook: [NBeats.ipynb](examples/NBeats.ipynb): `make run-jupyter`.

Here is a toy example on how to use this model (train and predict):

```python
import numpy as np

from nbeats_keras.model import NBeatsNet


def main():
    # https://keras.io/layers/recurrent/
    num_samples, time_steps, input_dim, output_dim = 50_000, 10, 1, 1

    # Definition of the model.
    model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                      thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    model.compile_model(loss='mae', learning_rate=1e-5)

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    y = np.mean(x, axis=1, keepdims=True)

    # Split data into training and testing datasets.
    c = num_samples // 10
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]

    # Train the model.
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)

    # Save the model for later.
    model.save('n_beats_model.h5')

    # Predict on the testing set.
    predictions = model.predict(x_test)
    print(predictions.shape)

    # Load the model.
    model2 = NBeatsNet.load('n_beats_model.h5')

    predictions2 = model2.predict(x_test)
    np.testing.assert_almost_equal(predictions, predictions2)


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

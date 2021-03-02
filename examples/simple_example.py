import numpy as np
from tensorflow.keras.utils import plot_model

from nbeats_keras.model import NBeatsNet, GenericStackDef


def main():
    # https://keras.io/layers/recurrent/
    forecast_length = 1
    backcast_length = 10
    num_samples = 50_000

    # Definition of the model.
    model = NBeatsNet(
        forecast_length=forecast_length,  # H
        backcast_length=backcast_length,  # n in n*H
        stacks_def=tuple([GenericStackDef()] * 30)
    )

    plot_model(model.n_beats, to_file='model.png',
               show_shapes=True,
               show_layer_names=True,
               expand_nested=True)

    # Definition of the objective function and the optimizer.
    model.compile_model(loss='mae', learning_rate=1e-5)

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    x = np.random.uniform(size=(num_samples, backcast_length))
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

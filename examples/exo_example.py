import numpy as np

from nbeats_keras.model import NBeatsNet as NBeatsKeras


# As of today, exogenous variables have only been implemented in the Keras backend.
def main():
    # Let's consider a setup where we have [sunshine] and [rainfall] and we want to predict [rainfall].
    # [sunshine] will be our external variable (exogenous).
    # [rainfall] will be our internal variable (endogenous).
    # We assume that rainfall[t] depends on the previous values of rainfall[t-1], ... rainfall[t-N].
    # And we also think that rainfall[t] depends on sunshine.
    # Rainfall is 1-D so input_dim=1. We have just one exo variable so exo_dim=1.
    # Output_dim is also 1-D. It's rainfall[t]. Therefore, output_dim=1.
    # We have long sequences of rainfall[t], sunshine[t] (t>0) that we cut into length N+1.
    # N will be the history. and +1 is the one we want to predict.
    # N-Beats is not like an LSTM. It needs the history window to be finite (of size N<inf).
    # here N=time_steps. Let's say 20.
    # We end of having an arbitrary number of sequences (say 100) of length 20+1.
    num_samples, time_steps, input_dim, output_dim, exo_dim = 1000, 20, 1, 1, 1

    # Definition of the model.
    # NOTE: If you choose the Keras backend with input_dim>1, you have
    # to set the value here too (in the constructor).
    model_keras = NBeatsKeras(
        input_dim=input_dim,
        backcast_length=time_steps,
        forecast_length=output_dim,
        exo_dim=exo_dim
    )

    # from keras.utils.vis_utils import plot_model
    # plot_model(model_keras, 'exo.png')

    model_keras.compile(loss='mae', optimizer='adam')
    rainfall = np.random.uniform(size=(num_samples, time_steps + 1, input_dim))

    # predictors.
    x_rainfall = rainfall[:, 0:time_steps, :]
    x_sunshine = np.random.uniform(size=(num_samples, time_steps, 1))

    # target.
    y_rainfall = rainfall[:, time_steps:, :]

    model_keras.compile(loss='mae', optimizer='adam')
    model_keras.fit([x_rainfall, x_sunshine], y_rainfall, epochs=10)

    np.testing.assert_equal(model_keras.predict([x_rainfall, x_sunshine]).shape, (1000, 1, 1))


if __name__ == '__main__':
    main()

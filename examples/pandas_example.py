import numpy as np
import pandas as pd

from nbeats_keras.model import NBeatsNet as NBeatsKeras


# This is an example linked to this issue: https://github.com/philipperemy/n-beats/issues/60.
# Here the target variable is no longer part of the inputs.
# NOTE: it is also possible to solve this problem with exogenous variables.
# See example/exo_example.py.

def main():
    num_rows = 100
    num_columns = 4
    timesteps = 20
    d = pd.DataFrame(data=np.random.uniform(size=(num_rows, num_columns)), columns=['A', 'B', 'C', 'D'])
    print(d.head())

    # Use <A, B, C> to predict D.
    predictors = d[['A', 'B', 'C']]
    targets = d['D']

    # backcast length is timesteps.
    # forecast length is 1.
    predictors = np.array([predictors[i:i + timesteps] for i in range(num_rows - timesteps)])
    targets = np.array([targets[i:i + 1] for i in range(num_rows - timesteps)])[:, :, None]

    # noinspection PyArgumentEqualDefault
    model_keras = NBeatsKeras(
        input_dim=num_columns - 1,
        output_dim=1,
        forecast_length=1,
        nb_blocks_per_stack=1,
        backcast_length=timesteps
    )
    # plot_model(model_keras, 'pandas.png', show_shapes=True, show_dtype=True)
    model_keras.compile(loss='mae', optimizer='adam')

    model_keras.fit(predictors, targets, validation_split=0.2)

    num_predictions = len(predictors)
    predictions = model_keras.predict(predictors)
    np.testing.assert_equal(predictions.shape, (num_predictions, 1, 1))
    d['P'] = [np.nan] * (num_rows - num_predictions) + list(model_keras.predict(predictors).squeeze(axis=(1, 2)))
    print(d)


if __name__ == '__main__':
    main()

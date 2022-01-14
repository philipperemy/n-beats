import numpy as np
import pandas as pd
from keras.utils.vis_utils import plot_model

from nbeats_keras.model import NBeatsNet as NBeatsKeras


# This is an example linked to this issue: https://github.com/philipperemy/n-beats/issues/60.
# We deviate a bit from the original thoughts of NBeats.
# Here we:
# - drop the time dimension.
# - the target variable is no longer part of the inputs.

# NOTE: it is also possible to solve this problem with exogenous variables. See example/exo_example.py.

def main():
    num_rows = 1000
    num_columns = 4
    d = pd.DataFrame(data=np.random.uniform(size=(num_rows, num_columns)), columns=['A', 'B', 'C', 'D'])
    print(d.head())

    # Use <A, B, C> to predict D.
    predictors = d[['A', 'B', 'C']]
    targets = d['D']

    predictors = np.expand_dims(predictors, axis=1)  # "emulate" the time dimension
    targets = np.expand_dims(targets, axis=1)  # "emulate" the time dimension

    num_samples, time_steps, input_dim, output_dim = num_rows, 1, num_columns - 1, 1

    model_keras = NBeatsKeras(
        input_dim=input_dim,
        output_dim=output_dim,
        stack_types=(NBeatsKeras.GENERIC_BLOCK,),
        nb_blocks_per_stack=1,
        thetas_dim=(4,),
        backcast_length=time_steps
    )
    plot_model(model_keras, 'pandas.png', show_shapes=True, show_dtype=True)
    model_keras.compile(loss='mae', optimizer='adam')

    model_keras.fit(predictors, targets, validation_split=0.2)

    predictions = model_keras.predict(predictors)
    np.testing.assert_equal(predictions.shape, (num_samples, 1, 1))
    d['P'] = model_keras.predict(predictors).squeeze(axis=(1, 2))


if __name__ == '__main__':
    main()

# https://www.kaggle.com/stytch16/jena-climate-2009-2016
# https://www.kaggle.com/pankrzysiu/6-3-1-a-temperature-forecasting-problem
# https://www.tensorflow.org/tutorials/structured_data/time_series

# On the hourly temperature dataset and
# without much tuning, N-Beats can achieve a 23% reduction in loss compared to the last value benchmark.
#

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback

from nbeats_keras.model import NBeatsNet

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
print(csv_path)
print(df.head())
print(len(df))


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    data = np.array(data)
    labels = np.array(labels)
    return data.reshape(data.shape[0], data.shape[1], 1), labels.reshape(labels.shape[0], 1, 1)


TRAIN_SPLIT = 300000
tf.random.set_seed(13)

uni_data = df['T (degC)']
uni_data.index = df['Date Time']

uni_data = uni_data.values


class DataNormalizer:

    def __init__(self, train):
        self.uni_train_mean = train.mean()
        self.uni_train_std = train.std()

    def apply(self, x):
        return (x - self.uni_train_mean) / self.uni_train_std

    def apply_inv(self, x):
        return x * self.uni_train_std + self.uni_train_mean


dn = DataNormalizer(train=uni_data[:TRAIN_SPLIT])
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = dn.apply(uni_data)

univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

print('x_train_uni.shape=', x_train_uni.shape)
print('y_train_uni.shape=', y_train_uni.shape)
print('x_val_uni.shape=', x_val_uni.shape)
print('y_val_uni.shape=', y_val_uni.shape)

b_val_uni = np.mean(x_val_uni, axis=1)[..., 0]
print(np.mean(np.abs(b_val_uni - y_val_uni)))
print(np.mean(np.abs(dn.apply_inv(b_val_uni) - dn.apply_inv(y_val_uni))))

b2_val_uni = x_val_uni[:, -1, 0]
print(np.mean(np.abs(b2_val_uni - y_val_uni)))
print(np.mean(np.abs(dn.apply_inv(b2_val_uni) - dn.apply_inv(y_val_uni))))

# noinspection PyArgumentEqualDefault
m = NBeatsNet(
    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    nb_blocks_per_stack=3,
    forecast_length=1,
    backcast_length=univariate_past_history,
    thetas_dim=(15, 15, 15),
    share_weights_in_stack=False,
    hidden_layer_units=384)
m.compile(loss='mae', optimizer='adam')


class EvaluateModelCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        b3_val_uni = m.predict(x_val_uni)
        print(f'[{epoch}] b3_val_uni.shape=', b3_val_uni.shape)
        print(np.mean(np.abs(b3_val_uni - y_val_uni)))
        print(np.mean(np.abs(dn.apply_inv(b3_val_uni) - dn.apply_inv(y_val_uni))))
        print('*' * 80)


m.fit(x_train_uni, y_train_uni,
      epochs=20, validation_split=0.1, shuffle=True,
      callbacks=[EvaluateModelCallback()])

import csv

import numpy as np


def dummy_data_generator(num_samples, backcast_length, forecast_length, signal_type='seasonality', random=False):
    def get_x_y():
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        if random:
            offset = np.random.standard_normal() * 0.1
        else:
            offset = 1
        if signal_type == 'trend':
            a = lin_space + offset
        elif signal_type == 'seasonality':
            a = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
            a += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
            a += lin_space * offset + np.random.rand() * 0.1
        elif signal_type == 'cos':
            a = np.cos(2 * np.pi * lin_space)
        else:
            raise Exception('Unknown signal type.')

        x = a[:backcast_length]
        y = a[backcast_length:]

        min_x, max_x = np.minimum(np.min(x), 0), np.max(np.abs(x))

        x -= min_x
        y -= min_x

        x /= max_x
        y /= max_x

        return x, y

    def gen():
        while True:
            xx = []
            yy = []
            for i in range(num_samples):
                x, y = get_x_y()
                xx.append(x)
                yy.append(y)
            yield np.array(xx), np.array(yy)

    return gen()


def get_m4_data(backcast_length, forecast_length, is_training=True):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    if is_training:
        filename = 'data/m4/train/Daily-train.csv'
    else:
        filename = 'data/m4/val/Daily-test.csv'

    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
    x_tl_tl = np.array(x_tl)
    for i in range(x_tl_tl.shape[0]):
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        if is_training:
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length)
            time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        else:
            time_series_cleaned_forlearning_x = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
            time_series_cleaned_forlearning_y = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
            for j in range(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length):
                time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series_cleaned[j - backcast_length:j]
                time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series_cleaned[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y

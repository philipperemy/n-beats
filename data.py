import numpy as np


def get_data(num_samples, backcast_length, forecast_length, signal_type='seasonality', random=False):
    def get_x_y():
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        if random:
            offset = np.random.standard_normal() * 0.1
        else:
            offset = 1
        if signal_type == 'trend':
            x = lin_space + offset
        elif signal_type == 'seasonality':
            x = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
            x += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
            x += lin_space * offset + np.random.rand() * 0.1
        elif signal_type == 'cos':
            x = np.cos(2 * np.pi * lin_space)
        else:
            raise Exception('Unknown signal type.')
        x -= np.minimum(np.min(x), 0)
        x /= np.max(np.abs(x))
        x = np.expand_dims(x, axis=0)
        y = x[:, backcast_length:]
        x = x[:, :backcast_length]
        return x[0], y[0]

    while True:
        X = []
        Y = []
        for i in range(num_samples):
            x, y = get_x_y()
            X.append(x)
            Y.append(y)
        yield np.array(X), np.array(Y)

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Subtract, Add
from keras.models import Model
from keras.optimizers import Adam


class NBeatsNet:
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 stack_types=[TREND_BLOCK, SEASONALITY_BLOCK],
                 nb_blocks_per_stack=3,
                 forecast_length=2,
                 backcast_length=10,
                 thetas_dim=[4, 8],
                 share_weights_in_stack=False,
                 hidden_layer_units=256):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.units = hidden_layer_units
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        assert not share_weights_in_stack, 'Feature not implemented.'
        self.thetas_dim = thetas_dim
        self.best_perf = 100.0
        self.steps = 10001
        self.plot_results = 100

        X_ = Input(shape=(self.backcast_length,))
        x_ = Lambda(lambda x: x)(X_)
        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, stack_type, nb_poly)
                x_ = Subtract()([x_, backcast])
                if stack_id == 0 and block_id == 0:
                    y_ = forecast
                else:
                    y_ = Add()([y_, forecast])

        model = Model(X_, y_)
        model.summary()

        self.nbeats = model

    def create_block(self, x, stack_type, nb_poly):
        d1 = Dense(self.units, activation='relu')(x)
        d2 = Dense(self.units, activation='relu')(d1)
        d3 = Dense(self.units, activation='relu')(d2)
        d4 = Dense(self.units, activation='relu')(d3)
        if stack_type == 'generic':
            theta_b = Dense(nb_poly, activation='linear')(d4)
            theta_f = Dense(nb_poly, activation='linear')(d4)
            backcast = Dense(self.backcast_length, activation='linear')(theta_b)
            forecast = Dense(self.forecast_length, activation='linear')(theta_f)
        elif stack_type == 'trend':
            theta_f = theta_b = Dense(nb_poly, activation='linear')(d4)
            backcast = Lambda(trend_model,
                              arguments={"is_forecast": False, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})(
                theta_b)
            forecast = Lambda(trend_model,
                              arguments={"is_forecast": True, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})(
                theta_f)
        else:  # seasonality
            theta_b = theta_f = Dense(self.backcast_length, activation='linear')(d4)
            # theta_b = Dense(self.backcast_length, activation='linear')(d4)
            # theta_b and theta_f are shared even for seasonality bloc
            backcast = Lambda(seasonality_model,
                              arguments={"is_forecast": False, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})(theta_b)
            forecast = Lambda(seasonality_model,
                              arguments={"is_forecast": True, "backcast_length": self.backcast_length,
                                         "forecast_length": self.forecast_length})(theta_f)
        return backcast, forecast

    def compile_model(self, loss, learning_rate):
        optimizer = Adam(lr=learning_rate)
        self.nbeats.compile(loss=loss, optimizer=optimizer)


def linear_space(backcast_length, forecast_length, fwd_looking=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / backcast_length
    if fwd_looking:
        ls = ls[backcast_length:]
    else:
        ls = ls[:backcast_length]
    return ls


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)  # H/2-1
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    S = K.concatenate([s1, s2], axis=0)
    S = K.cast(S, np.float32)
    return K.dot(thetas, S)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    T = K.transpose(K.stack([t ** i for i in range(p)], axis=0))
    T = K.cast(T, np.float32)
    return K.dot(thetas, K.transpose(T))

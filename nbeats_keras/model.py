import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class StackDef:

    def __init__(self, width, blocks, blocks_layers, share_weights, degree, name):
        self.width = width
        self.blocks = blocks
        self.blocks_layers = blocks_layers
        self.share_weights = share_weights
        self.degree = degree
        self.name = name

    def __str__(self):
        s = f'[{self.name}] - '
        s += f'width={self.width}, '
        s += f'blocks={self.blocks}, '
        s += f'blocks_layers={self.blocks_layers}, '
        s += f'share_weights={self.share_weights}, '
        s += f'degree={self.degree}'
        return s


class GenericStackDef(StackDef):
    def __init__(self, width=512, blocks=1, blocks_layers=4, share_weights=False, degree=512):
        super().__init__(width, blocks, blocks_layers, share_weights, degree, 'generic')


class SeasonalityStackDef(StackDef):
    def __init__(self, width=2048, blocks=3, blocks_layers=4, share_weights=True, degree=4):
        super().__init__(width, blocks, blocks_layers, share_weights, degree, 'seasonality')


class TrendStackDef(StackDef):
    def __init__(self, width=256, blocks=3, blocks_layers=4, share_weights=True, degree=3):
        super().__init__(width, blocks, blocks_layers, share_weights, degree, 'trend')


class NBeatsNet:
    def __init__(
            self,
            forecast_length=2,  # H
            backcast_length=2 * 7,  # n in n*H
            stacks_def=tuple([GenericStackDef()] * 30)
    ):
        self.stacks_def = stacks_def
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = 1
        self.output_dim = 1
        self.input_shape = (self.backcast_length,)
        self.output_shape = (self.forecast_length,)
        self.weights = {}
        self.n_beats = None
        self.graph()

    def create_stack(self, x, y=None, stack_id=0, stack_def: StackDef = GenericStackDef()):
        for block_id in range(stack_def.blocks):
            backcast, forecast = self.create_block(x, stack_id, block_id, stack_def)
            x = Subtract()([x, backcast])
            y = forecast if y is None else Add()([y, forecast])
        return x, y

    def create_block(self, x, stack_id, block_id, stack_def: StackDef):
        # register weights (used when share_weights=True)
        def reg(layer):
            return self._r(stack_def.share_weights, layer, stack_id)

        # update name (used when share_weights=True)
        def n(layer_name):
            return '/'.join(['s' + str(stack_id), 'b' + str(block_id), stack_def.name, layer_name])

        fc_layers = []
        for i in range(stack_def.blocks_layers):
            fc_layers.append(reg(Dense(stack_def.width, activation='relu', name=n(f'd{i + 1}'))))

        if stack_def.name == 'generic':
            theta_b = reg(Dense(stack_def.degree, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(stack_def.degree, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_def.name == 'trend':
            theta_f = reg(Dense(stack_def.degree, activation='linear', use_bias=False, name=n('theta_b')))
            theta_b = reg(Dense(stack_def.degree, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(trend_model, arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
            forecast = Lambda(trend_model, arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
        elif stack_def.name == 'seasonality':
            theta_b = reg(Dense(2 * self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(2 * self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
        else:
            raise Exception(f'Undefined type for StackDef: {stack_def}.')

        for fc in fc_layers:
            x = fc(x)

        theta_f_output = theta_f(x)
        theta_b_output = theta_b(x)
        backcast_output = backcast(theta_b_output)
        forecast_output = forecast(theta_f_output)
        return backcast_output, forecast_output

    def graph(self):
        inputs = Input(shape=self.input_shape, name='input_variable')
        x = inputs
        y = None  # output
        for stack_id, stack_def in enumerate(self.stacks_def):
            x, y = self.create_stack(x, y, stack_id, stack_def)
        model = Model(inputs, y)
        model.summary()
        self.n_beats = model

    @staticmethod
    def load(file_path, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(file_path, custom_objects, compile)

    def _r(self, share_weights_in_stack, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def compile_model(self, loss, learning_rate):
        optimizer = Adam(lr=learning_rate)
        self.compile(loss=loss, optimizer=optimizer)

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.n_beats, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            return attr(*args, **kwargs)

        return wrapper


def linear_space(backcast_length, forecast_length, fwd_looking=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    if fwd_looking:
        ls = ls[backcast_length:]
    else:
        ls = ls[:backcast_length]
    return ls


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)], axis=0))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))

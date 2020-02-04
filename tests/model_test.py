import unittest

from nbeats_keras.model import NBeatsNet


class ModelTest(unittest.TestCase):
    def test_share_weights_count_params(self):
        m = NBeatsNet(stack_types=(
            NBeatsNet.TREND_BLOCK,
            NBeatsNet.SEASONALITY_BLOCK
        ),
            nb_blocks_per_stack=3,
            forecast_length=2,
            backcast_length=10,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            hidden_layer_units=64)

        self.assertEqual(m.count_params(), 80512)

        m2 = NBeatsNet(stack_types=(
            NBeatsNet.TREND_BLOCK,
            NBeatsNet.SEASONALITY_BLOCK
        ),
            nb_blocks_per_stack=3,
            forecast_length=2,
            backcast_length=10,
            thetas_dim=(4, 8),
            share_weights_in_stack=True,  # just change it to True.
            hidden_layer_units=64)

        self.assertEqual(m2.count_params(), (80512 + 128) // 3)  # nb_blocks_per_stack=3

        m3 = NBeatsNet(stack_types=(
            NBeatsNet.TREND_BLOCK,
            NBeatsNet.SEASONALITY_BLOCK,
            NBeatsNet.GENERIC_BLOCK,
        ),
            nb_blocks_per_stack=3,
            forecast_length=2,
            backcast_length=10,
            thetas_dim=(4, 8, 4),
            share_weights_in_stack=True,  # just change it to True.
            hidden_layer_units=64)

        self.assertEqual(len(m3.weights), len(m3.stack_types))

    def test_thetas_stack_types_same_length(self):

        try:
            NBeatsNet(stack_types=(
                NBeatsNet.TREND_BLOCK,
                NBeatsNet.SEASONALITY_BLOCK,
                NBeatsNet.GENERIC_BLOCK,
            ),
                nb_blocks_per_stack=3,
                forecast_length=2,
                backcast_length=10,
                thetas_dim=(4, 8),
                share_weights_in_stack=True,  # just change it to True.
                hidden_layer_units=64)
            raise Exception('Test fail.')
        except AssertionError:
            pass

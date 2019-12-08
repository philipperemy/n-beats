# N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- *Implementation in Pytorch*
- *Implementation in Keras by @eljdos*
- https://arxiv.org/abs/1905.10437

<p align="center">
  <img src="nbeats.png" width="600"><br/>
  <i>N-Beats at the beginning of the training</i><br><br>
</p>

Trust me, after a few more steps, the green curve (predictions) matches the ground truth exactly :-)

## Installation

### From PyPI

Install Keras: `pip install nbeats-keras`.

Install Pytorch: `pip install nbeats-pytorch`.

### From the sources

Installation is based on a MakeFile. Make sure you are in a virtualenv and have python3 installed.

Command to install N-Beats with Keras: `make install-keras`

Command to install N-Beats with Pytorch: `make install-pytorch`

## Example

Jupyter notebook: [NBeats.ipynb](examples/NBeats.ipynb): `make run-jupyter`.



## Model

Pytorch and Keras have the same model arguments:

```python
class NBeatsNet:
    def __init__(self,
                 stack_types=[TREND_BLOCK, SEASONALITY_BLOCK],
                 nb_blocks_per_stack=3,
                 forecast_length=2,
                 backcast_length=10,
                 thetas_dim=[2, 8],
                 share_weights_in_stack=False,
                 hidden_layer_units=128):
        pass                
```

Which would translate in this model:

```
--- Model ---
| N-Beats
| --  Stack Trend (#0) (share_weights_in_stack=False)
     | -- TrendBlock(units=128, thetas_dim=2, backcast_length=50, forecast_length=10, share_thetas=True) at @4500902576
     | -- TrendBlock(units=128, thetas_dim=2, backcast_length=50, forecast_length=10, share_thetas=True) at @4779951944
     | -- TrendBlock(units=128, thetas_dim=2, backcast_length=50, forecast_length=10, share_thetas=True) at @4779952280
| --  Stack Seasonality (#1) (share_weights_in_stack=False)
     | -- SeasonalityBlock(units=128, thetas_dim=8, backcast_length=50, forecast_length=10, share_thetas=True) at @4779952616
     | -- SeasonalityBlock(units=128, thetas_dim=8, backcast_length=50, forecast_length=10, share_thetas=True) at @4779952952
     | -- SeasonalityBlock(units=128, thetas_dim=8, backcast_length=50, forecast_length=10, share_thetas=True) at @4779953288
```

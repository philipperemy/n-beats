import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from data import get_data
from model import NBeatsNet


def test(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    b, f = net(torch.tensor(x, dtype=torch.float))
    b, f, x, y = b.numpy()[0], f.numpy()[0], x[0], target[0]
    x_y = np.concatenate([x, y], axis=-1).flatten()
    plt.plot(range(0, backcast_length), x, color='b')
    plt.plot(range(backcast_length, backcast_length + forecast_length), y, color='g')
    plt.plot(range(backcast_length, backcast_length + forecast_length), f, color='r')
    # plt.scatter(range(0, backcast_length + forecast_length), x_y.flatten(),
    #             color=['b'] * backcast_length + ['g'] * forecast_length)
    # plt.scatter(range(backcast_length, backcast_length + forecast_length), f.flatten(),
    #             color=['r'] * forecast_length)
    plt.grid(True)
    plt.title(grad_step)
    plt.show()


def train():
    forecast_length = 10
    backcast_length = 5 * forecast_length
    batch_size = 100

    test_starts_at = backcast_length

    data_gen = get_data(batch_size, backcast_length, forecast_length,
                        test_starts_at, signal_type='seasonality', random=True)

    net = NBeatsNet(nb_stacks=2, units=64,
                    nb_thetas=8, nb_blocks=3,
                    backcast_length=backcast_length,
                    forecast_length=forecast_length)

    optimiser = optim.Adam(net.parameters())

    for grad_step, (x, target) in enumerate(data_gen):
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float))
        loss.backward()
        optimiser.step()
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                print(f'{str(grad_step).zfill(6)} {loss.item():.6f}')
                test(net, x, target, backcast_length, forecast_length, grad_step)
        if grad_step > 10000:
            print('Finished.')
            break


if __name__ == '__main__':
    train()

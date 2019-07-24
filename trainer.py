import os

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F

from data import get_data
from model import NBeatsNet


def train():
    forecast_length = 10
    backcast_length = 5 * forecast_length
    batch_size = 10

    data_gen = get_data(batch_size, backcast_length, forecast_length,
                        signal_type='trend', random=True)

    print('--- Model ---')
    net = NBeatsNet(stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[2, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=128,
                    share_weights_in_stack=False)

    # net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
    #                 forecast_length=forecast_length,
    #                 thetas_dims=[7, 8],
    #                 nb_blocks_per_stack=3,
    #                 backcast_length=backcast_length,
    #                 hidden_layer_units=128,
    #                 share_weights_in_stack=False)

    optimiser = optim.Adam(net.parameters())

    print('--- Training ---')
    initial_grad_step = load(net, optimiser)
    for grad_step, (x, target) in enumerate(data_gen):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float))
        loss.backward()
        optimiser.step()
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                print(f'{str(grad_step).zfill(6)} {loss.item():.6f}')
                save(net, optimiser, grad_step)
                test(net, x, target, backcast_length, forecast_length, grad_step)
        if grad_step > 10000:
            print('Finished.')
            break


def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, 'nbeats-checkpoint.th')
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimiser.state_dict(), 'optimiser.pth')


def load(model, optimiser):
    if os.path.exists('nbeats-checkpoint.th'):
        checkpoint = torch.load('nbeats-checkpoint.th')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print('Restored checkpoint from nbeats-checkpoint.th.')
        return grad_step
    return 0


def test(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    b, f = net(torch.tensor(x, dtype=torch.float))
    b, f, x, y = b.numpy()[0], f.numpy()[0], x[0], target[0]
    plt.plot(range(0, backcast_length), x, color='b')
    plt.plot(range(backcast_length, backcast_length + forecast_length), y, color='g')
    plt.plot(range(backcast_length, backcast_length + forecast_length), f, color='r')
    plt.grid(True)
    plt.title(grad_step)
    plt.show()


if __name__ == '__main__':
    train()

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F

from data import get_m4_data, dummy_data_generator
from nbeats_pytorch.model import NBeatsNet

CHECKPOINT_NAME = 'nbeats-training-checkpoint.th'


def get_script_arguments():
    parser = ArgumentParser(description='N-Beats')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
    parser.add_argument('--task', choices=['m4', 'dummy'], required=True)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


def main():
    args = get_script_arguments()
    device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
    forecast_length = 10
    backcast_length = 5 * forecast_length
    batch_size = 4  # greater than 4 for viz

    if args.task == 'm4':
        data_gen = batcher(get_m4_data(backcast_length, forecast_length), batch_size=batch_size, infinite=True)
    elif args.task == 'dummy':
        data_gen = dummy_data_generator(backcast_length, forecast_length,
                                        signal_type='seasonality', random=True,
                                        batch_size=batch_size)
    else:
        raise Exception('Unknown task.')

    print('--- Model ---')
    net = NBeatsNet(device=device,
                    stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dim=[2, 8, 3],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=1024,
                    share_weights_in_stack=False,
                    nb_harmonics=None)

    optimiser = optim.Adam(net.parameters())

    def plot_model(x, target, grad_step):
        if not args.disable_plot:
            print('plot()')
            plot(net, x, target, backcast_length, forecast_length, grad_step)

    max_grad_steps = 10000
    if args.test:
        max_grad_steps = 5

    simple_fit(net, optimiser, data_gen, plot_model, device, max_grad_steps)


def simple_fit(net, optimiser, data_generator, on_save_callback=None, device=torch.device('cpu'), max_grad_steps=10000):
    print('--- Training ---')
    initial_grad_step = load(net, optimiser)
    for grad_step, (x, target) in enumerate(data_generator):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                save(net, optimiser, grad_step)
                if on_save_callback is not None:
                    on_save_callback(x, target, grad_step)
        if grad_step > max_grad_steps:
            print('Finished.')
            break


def save(model, optimiser, grad_step=0):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0


def plot(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        # plt.title(f'step #{grad_step} ({i})')

    output = 'n_beats_{}.png'.format(grad_step)
    plt.savefig(output)
    plt.clf()
    print('Saved image to {}.'.format(output))


if __name__ == '__main__':
    main()

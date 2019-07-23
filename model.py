import torch
from torch import nn
from torch.nn import functional as F


class GenericBlock(nn.Module):

    def __init__(self, units, nb_thetas, backcast_length=10, forecast_length=5, input_size=69):
        super(GenericBlock, self).__init__()

        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)

        self.theta_b_fc = nn.Linear(units, nb_thetas)
        self.theta_f_fc = nn.Linear(units, nb_thetas)

        self.backcast_fc = nn.Linear(nb_thetas, backcast_length)
        self.forecast_fc = nn.Linear(nb_thetas, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class GenericNet(nn.Module):

    def __init__(self, nb_stacks=2, units=256,
                 nb_thetas=10, nb_blocks=3, backcast_length=10,
                 forecast_length=5):
        super(GenericNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.stacks = []
        self.parameters = []
        for stack_id in range(nb_stacks):
            blocks = []
            for block_id in range(nb_blocks):
                block = GenericBlock(units, nb_thetas, backcast_length, forecast_length)
                blocks.append(block)
                self.parameters.extend(block.parameters())
            self.stacks.append(blocks)
        self.parameters = nn.ParameterList(self.parameters)

    def forward(self, backcast):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        return backcast, forecast

import torch.nn as nn
import torch.functional as F

from ape_x_dqn.utilities import (
    model_state_bsize,
    model_state_to_ndarray,
    ndarray_to_model_state,
    model_states_equal,
)


class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestStateUtilities(object):
    def test_state_to_ndarray(self):
        net = SampleNet()
        state_ndarray = model_state_to_ndarray(net.state_dict())
        state_restored = ndarray_to_model_state(state_ndarray, net.state_dict())

        assert model_states_equal(state_restored, net.state_dict())
        assert model_state_bsize(state_restored) == 248024

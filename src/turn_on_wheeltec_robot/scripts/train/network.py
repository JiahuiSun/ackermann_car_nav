import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml

with open('scripts/train/conf/ppo.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# input shape is (1, input_size), output shape is (1, output_size), 2 conv layers, 1 fc layer
class CNN2D(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(CNN2D, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_size // 4) * (input_size // 4), hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SharedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loc_net = MLP(5, 128)
        self.goal_net = MLP(2, 128)
        self.map_net = CNN2D(self.input_dim, 256)
    
    def forward(self, obs):
        # loc = obs[:, :5]
        # goal = obs[:, 5:7]
        # map = obs[:, 7:]
        # batch_size, _ = map.shape
        # map = map.view(batch_size, self.input_dim, self.input_dim)
        
        # x = self.loc_net(loc)
        # y = self.goal_net(goal)
        z = self.map_net(obs)
        return z


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=512):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.CNN2D = CNN2D(input_dim, middle_dim)
        self.share = SharedNetwork(input_dim, middle_dim)
        self.fc1 = nn.Linear(middle_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.net = nn.Sequential(
        #     nn.Linear(middle_dim, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 256),
        #     nn.Tanh()
        # )
        self.mu = nn.Linear(512, output_dim)
        self.sigma = nn.Parameter(torch.zeros(output_dim, 1))
        nn.init.constant_(self.sigma, -0.1)
        
        if config['orth_init']:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mu, gain=0.01)
    
    def forward(self, obs):
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits = self.CNN2D(obs)
        # x = torch.tanh(self.fc1(x))
        # logits = torch.tanh(self.fc2(x))
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return mu, sigma


class RecurrentActor(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=512, hidden_layer_size=256,
                 layer_num=1):
        super(RecurrentActor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.share = SharedNetwork(input_dim, middle_dim)
        self.fc1 = nn.Linear(middle_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.rnn_hidden = None
        if config['use_gru']:
            self.rnn = nn.GRU(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num,
                               batch_first=True)
        self.mu = nn.Linear(256, output_dim)
        self.sigma = nn.Parameter(torch.zeros(output_dim, 1))
        nn.init.constant_(self.sigma, -0.1)
        
        if config['orth_init']:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            # orthogonal_init(self.rnn)
            orthogonal_init(self.mu, gain=0.01)
    
    def forward(self, obs):
        """Mapping: obs -> logits -> (mu, sigma)."""
        x = self.share(obs)
        x = torch.tanh(self.fc1(x))
        logits = torch.tanh(self.fc2(x))
        # if len(logits.shape) == 2:
        #     logits = logits.unsqueeze(-2)
        self.rnn.flatten_parameters()
        output, self.rnn_hidden = self.rnn(logits, self.rnn_hidden)
        mu = self.mu(output)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return mu, sigma
    
    def hidden_reset(self):
        self.rnn_hidden = None


class ActorBeta(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=512):
        super(ActorBeta, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.share = SharedNetwork(input_dim, middle_dim)
        self.net = nn.Sequential(
            nn.Linear(middle_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.alpha_layer = nn.Linear(256, output_dim)
        self.beta_layer = nn.Linear(256, output_dim)
    
    def forward(self, obs):
        """Mapping: obs -> logits -> (alpha, beta)."""
        x = self.share(obs)
        logits = self.net(x)
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(logits)) + 1.0
        beta = F.softplus(self.beta_layer(logits)) + 1.0
        return alpha, beta


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=512):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.CNN = CNN2D(input_dim, middle_dim)
        # self.share = SharedNetwork(input_dim, middle_dim)
        self.fc1 = nn.Linear(middle_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        if config['orth_init']:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
    
    def forward(self, obs):
        x = self.CNN(obs)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits


class RecurrentCritic(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=512, hidden_layer_size=256,
                 layer_num=1):
        super(RecurrentCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.share = SharedNetwork(input_dim, middle_dim)
        self.fc1 = nn.Linear(middle_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.rnn_hidden = None
        if config['use_gru']:
            self.rnn = nn.GRU(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=hidden_layer_size, hidden_size=hidden_layer_size, num_layers=layer_num,
                               batch_first=True)
        self.fc3 = nn.Linear(256, output_dim)
        if config['orth_init']:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            # orthogonal_init(self.rnn)
            orthogonal_init(self.fc3)
    
    def forward(self, obs):
        x = self.share(obs)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        output, self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        logits = self.fc3(output)
        return logits
    
    def hidden_reset(self):
        self.rnn_hidden = None

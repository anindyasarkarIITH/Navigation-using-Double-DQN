import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

'''
####  Duelling DQN architecture for calculating Q value Q(s,a).
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=256):
        super(QNetwork, self).__init__()

        self.num_actions = action_size
        self.seed = torch.manual_seed(seed)
        
        #self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        #self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        #self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3_adv = nn.Linear(fc2_units, out_features=512)
        self.fc3_val = nn.Linear(fc2_units, out_features=512)

        self.fc4_adv = nn.Linear(in_features=512, out_features=action_size)
        self.fc4_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1(x))  ## (b_s,64) common layer
        x = F.relu(self.fc2(x))  ## (b_s,256) common layer
        
        adv = F.relu(self.fc3_adv(x)) # (b_s,512) adv. computation layer1
        val = F.relu(self.fc3_val(x)) # (b_s,512) val. computation layer1

        #print ("Duelling Q network :")
        #print (self.fc4_val(val).shape)
        adv = self.fc4_adv(adv)        #(b_s,4) adv. computation layer2

        val = self.fc4_val(val).expand(x.size(0), self.num_actions)  ##(b_s,4) val. computation layer2(expanded from (b_s,1))..repeats V(s) for all actions in the state 
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

'''


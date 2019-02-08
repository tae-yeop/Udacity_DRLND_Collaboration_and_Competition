import torch
import torch.nn.functional as F
import torch.nn.init as I
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Build model and Intialize it
        
        Params
        ======
            state_size (int) : State space size
            action_size (int) : Action space size
            seed (int) : Random seed
            fc1_unit (int)  
            fc2_unit (int)
        """
        # super(Actor, self).__init__()
        # Use this constructor instead because of autoreload issue. 
        super().__init__()
        torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()
        
    def reset_parameters_xavier(self):
        """
        Initialize parameters of the layers
        xavier_normal is used.
        See "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010) for details.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)

    def reset_parameters_uniform(self):
        """
        Use the intialization techniqe from Lillicrap et al.
        See http://arxiv.org/abs/1509.02971 for details.
        """
        I.uniform_(self.fc1.weight, *self.get_fan_in(self.fc1))
        I.uniform_(self.fc2.weight, *self.get_fan_in(self.fc2))
        I.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        lim = 1/np.sqrt(layer.in_features)
        return (-lim, lim)

    def forward(self, state):
        """
        Forward pass state -> action
        
        Parmas
        =======
            state (Torch Tensor) [batch_size, state_size]
        Returns
        ====== 
            F.tanh(x) (Torch Tensor) [batch_size, action_size] : range of (-1,1)
        """
        #x = self.bn1(state)
        x = F.leaky_relu(self.fc1(state))
        #x = self.bn2(x)
        x = F.leaky_relu(self.fc2(x))
        #x = self.bn3(x)
        x = self.fc3(x)
        return F.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, num_agents=2):
        """
        Build model and Intialize it
        
        Params
        ======
            state_size (int) : State space size
            action_size (int) : Action space size
            seed (int) : Random seed
            fc1_unit (int) : 
            fc2_unit (int) 
        """
        # super(Critic, self).__init__()
        super().__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear((state_size+action_size)*num_agents, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.reset_parameters()

    def get_fan_in(self, layer):
        """
        Get the fan-in in each layer.
        """
        lim = 1/np.sqrt(layer.in_features)
        return (-lim, lim)
        
    def reset_parameters(self):
        """
        Initialize parameters of the layers
        xavier_normal is used.
        See "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010) for details.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                I.xavier_normal_(m.weight)
    
    def reset_parameters_uniform(self):
        """
        Use the intialization techniqe from Lillicrap et al.
        See http://arxiv.org/abs/1509.02971 for details.
        """
        I.uniform_(self.fc1.weight, *self.get_fan_in(self.fc1))
        I.uniform_(self.fc2.weight, *self.get_fan_in(self.fc2))
        I.uniform_(self.fc3.weight, -3e-3, 3e-3)


    def forward(self, all_states, action_collections):
        """
        Forward pass state, action -> q_value 
        
        Params
        =======
            all_states (torch tensor) [batch_size, num_agents, state_size]
            action_collections (torch tensor) [batch_size, num_agents, action_size]
        Retunrs
        ======
            x (torch tensor) [batch_size, 1] : Q-value
        """
        #[batch_size, num_agents, state_size] -> [batch_size, num_agents*state_size]
        states_flatten = all_states.view(all_states.shape[0], -1)
        # action_collections : [batch_size, num_agents, action_size] -> [batch_size, num_agents*action_size]
        actions_flatten = action_collections.view(action_collections.shape[0],-1)
        # concat : [bact_size, num_agents*(state_size, action_size)]
        x = torch.cat((actions_flatten,states_flatten), dim=1).to(device)
        
        x = F.leaky_relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        #x = self.bn2(x)
        x = self.fc3(x)
        return x
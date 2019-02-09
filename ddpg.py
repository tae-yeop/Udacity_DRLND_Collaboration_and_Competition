import torch
import numpy as np
from network import Actor
from network import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG():
    """
    Individual Agent.
    """
    def __init__(self, state_size, action_size, params):
        """
        Build model, random process and intilize it.
        """
        torch.manual_seed(params['SEED'])
        self.random_process = OUNoise(action_size, params)

        self.local_actor = Actor(state_size, action_size, params['SEED'], params['FC1'], params['FC2']).to(device)
        self.target_actor = Actor(state_size, action_size, params['SEED'], params['FC1'], params['FC2']).to(device)
        # Initialize target networks weights with local networks
        self.hard_copy(self.local_actor, self.target_actor)
        # Optimizer for local actor networks
        self.actor_optimizer = torch.optim.Adam(self.local_actor.parameters(), params['LR_ACTOR'])

        self.local_critic = Critic(state_size, action_size, params['SEED'], params['FC1'], params['FC2']).to(device)
        self.target_critic = Critic(state_size, action_size, params['SEED'], params['FC1'], params['FC2']).to(device)
        # Initialize target networks weights with local networks
        self.hard_copy(self.local_critic, self.target_critic)
        # Optimizer for local critic networks
        self.critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), params['LR_CRITIC'])
    
    def reset_noise(self):
        """
        Reset the noise state every episode
        """
        self.random_process.reset()
    
    
    def hard_copy(self, local, target):
        """
        hard copy the weights of the local network to the target network
        """
        for local_param, target_param in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_copy(self, tau):
        """
        soft update target network
        𝜃_target = 𝜏*𝜃_local + (1 - 𝜏)*𝜃_target 
        """
        for local_param, target_param in zip(self.local_actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
        for local_param, target_param in zip(self.local_critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

class OUNoise():
    def __init__(self, action_size, params, mu=.0):
        """
        Set initial random process state.
        
        Params
        =====
        action_size (int)
        mu (float) : center that noise will move around.
        params (dict) : paramters
            seed (int) : For determinsitc random process. 
            sacle (flaot) : scale factor
            theta(flaot) : parameter for the process
            sigma(float) : parameter for the process
        """
        #self.noise_state = np.ones(action_size)*mu
        self.mu = np.ones(action_size)*mu
        self.theta = params['THETA']
        self.sigma = params['SIGMA']
        self.scale = params['SCALE']
        np.random.seed(params['SEED'])
        self.reset()
    
    def reset(self):
        """
        Reset the noise state to the mu.
        """
        self.noise_state = self.mu
    def sample(self):
        """
        Sample the noise state

        Returns
        =====
        noise_state (numpy array) [action_size,]
        """
        x = self.noise_state
        dx = self.theta*(self.mu-x) + self.sigma*(np.random.randn(len(x)))
        self.noise_state = x+ dx
        return (self.noise_state)*self.scale
import torch
import numpy as np
from network import Actor
from network import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hard_copy(local, target):
        """
        hard copy the weights of the local network to the target network
        """
        for local_param, target_param in zip(local.parameters(), target.parameters()):
            target_param.data.copy_(local_param.data)

class DDPG():
    """
    Individual Agent.
    """
    def __init__(self, state_size, action_size, seed, lr):
        """
        Build model, random process and intilize it.
        """
        
        self.lr = lr
        self.random_process = OUNoise(action_size, seed)

        self.local_critic = Critic(state_size, action_size, seed).to(device)
        self.target_critic = Critic(state_size, action_size, seed).to(device)
        hard_copy(self.local_critic, self.target_critic)


        # Initialize target networks weights with local networks
        # self.local_actor = Actor(state_size, action_size, seed).to(device)
        # self.target_actor = Actor(state_size, action_size, seed).to(device)
        # self.hard_copy(self.local_actor, self.target_actor)
        
        
        # Optimizer for local networks
        
        self.critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), self.lr)
    def init_actor(self):

        self.local_actor = Actor(state_size, action_size, seed).to(device)
        self.target_actor = Actor(state_size, action_size, seed).to(device)
        hard_copy(self.local_actor, self.target_actor)
        self.actor_optimizer = torch.optim.Adam(self.local_actor.parameters(), self.lr)

    def set_actor(self, local, target, optimizer):

        self.local_actor = local
        self.target_actor = target
        self.actor_optimizer = optimizer# torch.optim.Adam(self.local_actor.parameters(), self.lr)

    def act_local(self, state):
        """
        select action with the local actor network.
        
        Parmas
        ====
            state (torch Tensor) [batch_size, state_size]
        Retuns
        ====
            action (torch Tensor) [batch_size, action_size]
        """
        # BatchNorm1d need 2D shape
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # self.local_actor.eval()
        # with torch.no_grad():
        action = (self.local_actor(state) + 
                torch.from_numpy(self.random_process.sample()).float().to(device)
                )
        #self.local_actor.train()
        
        return action # torch.clamp(action, min=-1.0, max=1.0)
    
    def act_target(self, state):
        """
        select action with the target actor network
        
        Parmas
        ====
            state (torch Tensor) [batch_size, state_size]
        Retuns
        ====
            action (torch Tensor) [action_size]
        """
        # BatchNorm1d need 2D shape
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #print(state.shape)
        # self.target_actor.eval()
        # with torch.no_grad():
        action = (self.target_actor(state) + 
                torch.tensor(self.random_process.sample()).float().to(device)
        )       
        # self.target_actor.train()
        
        return action# torch.clamp(action, min=-1.0, max=1.0)
    
    def reset_noise(self):
        """
        Reset the noise state every episode
        """
        self.random_process.reset()
    
    
            
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
    def __init__(self, action_size, seed, scale=0.9,mu=.0, theta=0.15, sigma=0.15):
        """
        Set initial random process state.
        
        Params
        =====
            action_size (int)
            seed (int) : For determinsitc random process. 
            mu (float) : center that noise will move around.
            sacle (flaot) : scale factor
            theta(flaot) : parameter for the process
            sigma(float) : parameter for the process
        """
        #self.noise_state = np.ones(action_size)*mu
        self.mu = np.ones(action_size)*mu
        self.theta = theta
        self.sigma = sigma
        self.scale = scale
        np.random.seed(seed)
        self.reset()
    
    def reset(self):
        """
        Reset the noise state to the mu.
        """
        self.noise_state = self.mu
    def sample(self):
        """
        Returns
        =====
            noise_state (numpy array) [action_size,]
        """
        x = self.noise_state
        dx = self.theta*(self.mu-x) + self.sigma*(np.random.randn(len(x)))
        self.noise_state = x+ dx
        return (self.noise_state)*self.scale
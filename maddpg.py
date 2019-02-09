import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from ddpg import DDPG
import random
from network import Actor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    def __init__(self,num_agents, state_size, action_size, params):
        """
        Intialize Multi Agents Class
        
        Params
        ======
        num_agents (int) : # of agents
        state_size (int)
        action_size (int)
        params (dict) : hyperparmeters
        """
        torch.manual_seed(params['SEED'])

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.buf_size = params['BUFFER_SIZE']
        self.batch_size = params['BATCH_SIZE']
        self.train_freq = params['TRAIN_FREQ']
        self.train_iter = params['TRAIN_ITER']
        self.clipping = params['CLIPPING']

        self.agents_list = [] # list of multi-agents
        for i in range(num_agents):
                self.agents_list.append(DDPG(self.state_size, self.action_size, params))
            
        # shared replay buffer    
        self.memory = ReplayBuffer(self.buf_size, self.batch_size, params['SEED'])   
        self.gamma = params['GAMMA']
        self.tau = params['TAU']
        self.avg_critic_history = [] # tracking ctiric loss
        self.avg_actor_history = [] # tracking actor loss
        # condition to decide whether to update networks.
        self.step_count = 0 
        
    def act(self, states):
        """
        Multi-agent select actions with the current policies and explorations.
        each agent's action : a_i = µ_θ_i(o_i) + N_t
        See Lowe et al. http://arxiv.org/abs/1706.02275  for more details.

        Params
        =====
        states (numpy array) [num_agents, state_size]

        Retruns
        ====
        actions_array (numpy array) [num_agents, action_size]
        """
        # Convert numpy states into torch tensor.
        states_tensor = torch.from_numpy(states).float().to(device)

        actions = np.zeros((self.num_agents, self.action_size))
        for i in range(self.num_agents):
            # Set eval mode
            self.agents_list[i].local_actor.eval()
            # Deactivate autograd engine
            with torch.no_grad():
                # Get action and transform into numpy
                each_action = self.agents_list[i].local_actor(states_tensor[i]).cpu().data.numpy()
            # Back to train mode
            self.agents_list[i].local_actor.train()
            actions[i] = each_action

        return actions

    def reset(self):
        """
        Reset noise process.
        """
        for agent in self.agents_list:
            agent.reset_noise()

    def step(self, states, actions, rewards, next_states, dones):
        """
        Store trajactories to the shared replay buffer and do learning procedure.
        
        Params
        =====
        states (numpy array) [num_agents, state_size]
        actions (numpy array) [num_agents, action_size]
        rewards (list of floats) [num_agents]
        next_states (numpy array) [num_agents, state_size]
        dones (booean list) [num_agent]
        """
        # Add a trajactory into shared buffer
        self.memory.add(states, actions, rewards, next_states, dones)
        
        self.step_count += 1
        # If condition satisfied, then do learning with the number of iterations.
        if self.train_condition_check():
            for i in range(self.train_iter):
                self.learn()
    
    def train_condition_check(self):
        """
        Check the condition whether to train the network or not this time
        Two conditions must be satisfied.
        1. Replay buffer has at least data larger than the batch_size
        2. Is this time to update the network based on train frequency
        """
        memory_check = len(self.memory)>=self.batch_size
        train_check = self.step_count % self.train_freq == 0

        # if both condition are True, then return True
        return np.all([memory_check, train_check])

    def get_target_actions(self, states):
        """
        To calculate critic loss, target actions(a_prime_1, . . . , a_prime_N) are needed

        Params
        =====
        states (torch Tensor) [batch_size, num_agents, state_size]
        
        Retunrs
        =====
        target_actions (torch Tensor) [bact_size, num_agents, action_size]
        """
        
        target_actions = torch.zeros(self.batch_size, self.num_agents, self.action_size)
        for i in range(self.num_agents):
            # next_action : [batch_size, action_size]
            next_action = self.agents_list[i].target_actor(states[:,i,:])
            target_actions[:,i,:] = next_action

        return target_actions

    def learn(self):
        """
        Update local actor and critic networks and target actor network.
        """
        critic_loss_list = []
        actor_loss_list = []
        for i in range(self.num_agents):
            # states shape: [ batch_size, num_agents,state_size]
            # actions shape: [batch_size, num_agents, actions_size]
            # rewards shape: [ batch_size, num_agents]
            # next_states shape: [batch_size, num_agents, state_size]
            # dones shape: [batch_size,num_agents]
            states, actions, rewards, next_states, dones = self.memory.sample()

            # target_actions shape: [batch_size, num_agents, actions_size]
            target_actions = self.get_target_actions(states) 

            # ---------Update local critic---------#
            self.agents_list[i].critic_optimizer.zero_grad()
            # Calculate q_target with target critic network
            with torch.no_grad():
                # q_next shape: [batch_size, 1]
                q_next = self.agents_list[i].target_critic(next_states, target_actions.to(device))  
            # q_target shape: [batch_size,]
            q_target = rewards.t()[i] + self.gamma* q_next.view(-1) * (1-dones.t()[i])

            self.agents_list[i].local_critic.train()
            # q_value shape: [batch_size,]
            q_value = self.agents_list[i].local_critic(states, actions).view(-1)

            critic_loss = F.mse_loss(q_value,target=q_target.detach())
            critic_loss.backward()
            # Gradient clipping
            if self.clipping:
                torch.nn.utils.clip_grad_norm_(self.agents_list[i].local_critic.parameters(), 1)
      
            self.agents_list[i].critic_optimizer.step()
            critic_loss_list.append(critic_loss.cpu().data.numpy())
            
            # -----------------Update local actor------------------#
            self.agents_list[i].actor_optimizer.zero_grad()
            # pred_action shape : [batch_size, action_size]
            pred_action = self.agents_list[i].local_actor(states[:,i,:])
    
            # insert pred_action to the actions samples
            actions_clone = actions.clone()
            actions_clone[:,i,:] = pred_action

            actor_loss = -self.agents_list[i].local_critic(states, actions_clone).mean()
            actor_loss.backward()
            self.agents_list[i].actor_optimizer.step()
            actor_loss_list.append(actor_loss.cpu().data.numpy())

        avg_critic_loss = np.mean(critic_loss_list)
        avg_actor_loss = np.mean(actor_loss_list)
        
        self.avg_critic_history.append(avg_critic_loss)
        self.avg_actor_history.append(avg_actor_loss)

        # Update target nework
        for i in range(self.num_agents):
            self.agents_list[i].soft_copy(self.tau)


    def load_model(self, path='model'):
        """
        Load the model parameters
        """
        if torch.cuda.is_available():
            for i in range(self.num_agents):
                self.agents_list[i].local_actor.load_state_dict(torch.load(path+'/agent{}_actor_local.pth'.format(i)))
                self.agents_list[i].target_actor.load_state_dict(torch.load(path+'/agent{}_actor_target.pth'.format(i)))
                self.agents_list[i].local_critic.load_state_dict(torch.load(path+'/agent{}_critic_local.pth'.format(i)))
                self.agents_list[i].target_critic.load_state_dict(torch.load(path+'/agent{}_critic_target.pth'.format(i)))
        else:
            for i in range(self.num_agents):
                self.agents_list[i].local_actor.load_state_dict(torch.load(path+'/agent{}_actor_local.pth'.format(i), map_location='cpu'))
                self.agents_list[i].target_actor.load_state_dict(torch.load(path+'/agent{}_actor_target.pth'.format(i), map_location='cpu'))
                self.agents_list[i].local_critic.load_state_dict(torch.load(path+'/agent{}_critic_local.pth'.format(i), map_location='cpu'))
                self.agents_list[i].target_critic.load_state_dict(torch.load(path+'/agent{}_critic_target.pth'.format(i), map_location='cpu'))

    def save_model(self, path='model'):
        """
        Save model parameters.
        """
        for i in range(self.num_agents):
            torch.save(self.agents_list[i].local_actor.state_dict(), path+'/agent{}_actor_local.pth'.format(i))
            torch.save(self.agents_list[i].local_critic.state_dict(), path+'/agent{}_critic_local.pth'.format(i))
            torch.save(self.agents_list[i].target_actor.state_dict(), path+'/agent{}_actor_target.pth'.format(i))
            torch.save(self.agents_list[i].target_critic.state_dict(), path+'/agent{}_critic_target.pth'.format(i))      
            

class ReplayBuffer():
    
    def __init__(self, buf_size, batch_size, seed):
        """
        Set memory and bacth_size
        
        Params
        =====
        buf_size (int) 
        batch_size (int)
        seed (int)
        """
        self.memory = deque(maxlen=buf_size)
        self.batch_size = batch_size
        random.seed(seed)
        
    def __len__(self):
        return len(self.memory)
        
    def add(self, states, actions, rewards, next_states, dones):
        """
        Add trajecotries into replay buffer
        
        Params
        =====
        states (numpy array) [num_agents, state_size]
        actions (numpy array) [num_agents, action_size]
        rewards (list) [num_agents]
        next_states (numpy array) [num_agents, state_size]
        dones (list) [num_agents]
        """
        
        e = (states, actions, rewards, next_states, dones)
        self.memory.append(e)
        
    def sample(self):
        """
        Sample the trajecoties and convert it to torch float tensor.
        
        Returns
        ======
        Tuple of Torch Tensors :
            each tensor's outermost dimension is batch_size, 
            and the second dimension is num_agents
        """
        # list of tuples
        batch_list = random.sample(self.memory, self.batch_size)
        # Convert the element into torch float tensors.
        states_tensor = torch.from_numpy(np.stack([i[0] for i in batch_list], axis=0)).float().to(device)
        actions_tensor = torch.from_numpy(np.stack([i[1] for i in batch_list],axis=0)).float().to(device)
        rewards_tensor = torch.from_numpy(np.stack([i[2] for i in batch_list],axis=0)).float().to(device)
        next_states_tensor = torch.from_numpy(np.stack([i[3] for i in batch_list],axis=0)).float().to(device)
        dones_tensor = torch.from_numpy(np.stack([i[4] for i in batch_list],axis=0).astype(np.uint8)).float().to(device)
        
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)       
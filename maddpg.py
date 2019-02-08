import torch
import numpy as np
from collections import deque
from ddpg import DDPG
from ddpg import hard_copy
import random
from network import Actor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TAU = 0.01
LR = 0.001
GAMMA = 0.95
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
UP_FREQ = 10



class MADDPG():
    def __init__(self,num_agents, state_size, action_size, seed, shared_actor=True, lr=LR, tau=TAU, gamma=GAMMA):
        """
        Intialize Multi Agents Class
        
        Params
        ======
            num_agents (int) : # of agents
            state_size (int)
            action_size (int)
            seed (int) : random seed
            lr (float) : learning rate for optimizer
            tau (float) : interpolation rate for soft update
            
         """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        self.agents_list = [] # list of multi-agents
        for i in range(num_agents):
                self.agents_list.append(DDPG(self.state_size, self.action_size, seed, LR))


        if shared_actor:
            actor = Actor(self.state_size, self.action_size, seed).to(device)
            actor_target = Actor(self.state_size, self.action_size, seed).to(device)
            hard_copy(actor, actor_target)
            actor_optimizer = torch.optim.Adam(actor.parameters(), LR)
            for agent in self.agents_list:
                agent.set_actor(actor, actor_target, actor_optimizer)
        else:
            for agent in self.agents_list:
                agent.init_actor()
            
          
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed) # shared replay buffer  
        self.gamma = gamma
        self.tau = tau
        self.avg_critic_list = [] # tracking ctiric loss
        self.avg_actor_list = [] # tracking actor loss
        self.step_count = 0 # condition to decide whether to update networks.
        
    def act(self, states):
        """
        Multi-agent select actions with the current policies and explorations.
        a_i = µ_θ_i(o_i) +N_t
        Params
        =====
            states (numpy array) [num_agents, state_size]
        Retruns
        ====
            actions_array (numpy array) [num_agents, action_size]
        """
        # Convert numpy states into torch tensor.
        states_tensor = torch.from_numpy(states).float().to(device)
        
        # for i in range(self.num_agents):  
        #     action_list.append(self.agents_list[i].act_local(states_tensor[i]))

        # state : [state_size,]
        action_list = [agent.act_local(state.view(1,-1)).detach() \
                        for agent,state in zip(self.agents_list, states_tensor)]
        # actions_tensor : [num_agents, action_size]    
        #actions_tensor = torch.cat(action_list, dim=1)
        #actions_array = actions_tensor.data.cpu().numpy()

        # action_list : list of tensors whose shape is [1,action_size] 
        actions_array = np.concatenate(action_list,axis=0).reshape(self.num_agents, -1)
        
        return actions_array

    def reset(self):
        """
        Reset noise process.
        """
        for agent in self.agents_list:
            agent.reset_noise()
    def step(self, states, actions, rewards, next_states, dones):
        """
        Store trajactories to shared replay buffer and do learning procedure.
        
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
        
        
        # after every BATCH_SIZE smaples added to the replay buffer, network parameters are updated
        self.step_count = (self.step_count + 1) % UP_FREQ
        # To perform learning, at least we need that memory size is larger than BATCH_SIZE
        if len(self.memory)>BATCH_SIZE:
            if self.step_count==0:
                for i in range(10):
                    self.learn()
            
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
        
    
        states = states.permute(1,0,2).contiguous()
        # state : [batch_size, action_size]
        target_actions_list = [agent.act_target(state) for agent, state in zip(self.agents_list, states)]
        # Cocat target_actions_list dim=0 to [num_agents * batch_size ,action_size]
        # Cocat target_actions_list dim=1 to [batch_size ,num_agents *action_size]
        target_actions = torch.cat(target_actions_list, dim=1).to(device)
        # reshape target_actions into [batch_size, num_agents, action_size]
        target_actions = target_actions.view(-1, self.num_agents, self.action_size)

        return target_actions

    def learn(self):
        """
        Update local actor and critic networks and target actor network.
        """
        critic_loss_list = []
        actor_loss_list = []
        for i in range(self.num_agents):
            # states : [ batch_size, num_agents,state_size]
            # actions : [batch_size, num_agents, actions_size]
            # rewards : [ batch_size, num_agents]
            # next_states : [batch_size, num_agents, state_size]
            # dones : [batch_size,num_agents]
            states, actions, rewards, next_states, dones = self.memory.sample()
            
            target_actions = self.get_target_actions(states) #states

            # ---------Update local critic---------#
            self.agents_list[i].critic_optimizer.zero_grad()
            # Calculate q_target with target critic network
            with torch.no_grad():
                # q_next size : [batch_size, 1]
                # print(next_states.shape)
                # print(target_actions.shape)
                q_next = self.agents_list[i].target_critic(next_states, target_actions) 

            q_target = rewards.t()[i] + self.gamma* q_next.view(-1) * (1-dones.t()[i])
            q_value = self.agents_list[i].local_critic(states, actions).view(-1)
            # input and target shapes do not match: input [128 x 1], target [128 x 128]
            huber_loss = torch.nn.SmoothL1Loss()
            critic_loss = huber_loss(q_value,target=q_target.detach())
            #critic_loss = F.mse_loss(q_value,target=q_target.detach())
            critic_loss.backward()
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.agents_list[i].local_critic.parameters(), 0.5)
            #nn.utils.clip_grad_value_(self.agents_list[i].local_critic.parameters(),1.0)
            self.agents_list[i].critic_optimizer.step()
            critic_loss_list.append(critic_loss.cpu().data.numpy())
            
            # -----------------Update local actor------------------#
            self.agents_list[i].actor_optimizer.zero_grad()
            pred_action = self.agents_list[i].local_actor(states.permute(1,0,2)[i]) # pred_action shape : [batch_size, action_size]
            # .contiguous()
            # [num_agetns, batch_size, action_size]
            actions = actions.permute(1,0,2)
            #print(actions.shape)
            # insert pred_action to the actions samples
            #print(pred_action.shape)
            actions[i] = pred_action

            actor_loss = -self.agents_list[i].local_critic(states, actions.permute(1,0,2)).mean()
            # .contiguous()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.agents_list[i].local_actor.parameters(), 0.5)
            self.agents_list[i].actor_optimizer.step()
            actor_loss_list.append(actor_loss.cpu().data.numpy())

        avg_critic_loss = np.mean(critic_loss_list)
        avg_actor_loss = np.mean(actor_loss_list)
        
        self.avg_critic_list.append(avg_critic_loss)
        self.avg_actor_list.append(avg_actor_loss)
        # Update target nework
        for i in range(self.num_agents):
            self.agents_list[i].soft_copy(self.tau)
            
            

class ReplayBuffer():
    
    def __init__(self, buf_size, batch_size,seed):
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
            
                states_tensor, rewards_tensor, dones_tensor:
                tensor's outermost dimension is num_agents, 
                and the second dimension is batch_size
                
                actions_tensor, next_states_tenosr:
                outermost dimension is batch_size
                and the second dimension is num_agents
        """
        # list of tuples
        batch_list = random.sample(self.memory, self.batch_size)
        
        # Convert the element into torch float tensors.
        
        # batch of states [batch_size, num_agents, state_size]
        
        #print(torch.from_numpy(np.vstack([i[0] for i in batch_list])).float().to(device).shape) # torch.Size([128, 24])
        states_tensor = torch.from_numpy(np.stack([i[0] for i in batch_list], axis=0)).float().to(device)#.permute(1,0,2)
        actions_tensor = torch.from_numpy(np.stack([i[1] for i in batch_list],axis=0)).float().to(device)
        rewards_tensor = torch.from_numpy(np.stack([i[2] for i in batch_list],axis=0)).float().to(device)#.permute(1,0)
        # preserve the shape [bactch_size, num_agents, state_size]
        next_states_tensor = torch.from_numpy(np.stack([i[3] for i in batch_list],axis=0)).float().to(device)
        # [bath_size, num_agents]
        dones_tensor = torch.from_numpy(np.stack([i[4] for i in batch_list],axis=0).astype(np.uint8)).float().to(device)#.permute(1,0)
        
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)       
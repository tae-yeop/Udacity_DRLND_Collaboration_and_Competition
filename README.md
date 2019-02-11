[//]: # (Image References)

[image1]:
https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif 
"Trained Agent"

# Project 3: Collaboration and Competition

### Project Details

In this project, You can train two agents to control rackets and bounce a ball over a net. 


![Trained Agent][image1]

A reward of +0.1 is provided to the agent If it hits the ball over the net. And a reward of -0.01 is provided to the agent If it lets a ball hit the ground or hits the ball out of bounds.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Getting Started

To run this project, You need several python packages, Unity ML-Agents Toolkit and the environment.

- numpy(>=1.11)
- pytorch(>=0.4)
- matplotlib(>=1.11)

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
2. Clone the udacity nanodegree repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies. This is for installing [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) and all the needed python packages.
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
3. Download the unity environment from one of the links below. In this case you will download the Tennis environment.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the data folder, and uzip the file.

### Instructions

There are 7 main elements in this project. 

- Report.ipynd
- network.py
- ddpg.py
- maddpg.py
- model_checkpoint.pth (in ./model)
- params.json (in ./model)
- scores.json (in ./model)

*Report.ipynd* includes simple summary of the algorithm and codes for training the agent, visualizing the rewards graphs and running the agent. You can try experiments by setting different hyperparameters in this Report.ipynd file.

You can modify the actor and critic network model via network.py. 

*ddpg.py* includes base ddpg agent model and noise model. 

*maddpg.py* includes maddpg model and replay buffer model. maddpg model have references to each individual ddpg agent and shared replay buffer. 

*model_checkpoint.pth* is parameters of the agent's networks. You can check the all the checkpoint.pth in ./model folder. Instead of training, You can use this checkpoint directly to see how the agents interact with the each other. To see how the agent behave, You can run the cell in Report.ipynd

*params.json* is the log file for the hyperparmeters of particular maddpg model.

*scores.json* is the log file for the particular maddpg model.

After several experiments, I've found that model_18 is the best model. Please check the ./model folder.

### References

- Lowe et al., [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](http://arxiv.org/abs/1706.02275)

### License

This project is covered under the [MIT License.](./LICENSE)






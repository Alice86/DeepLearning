##########################################
# Stat232A&CS266A Project 3:
# Solving CartPole with Deep Q-Network
##########################################

#import gym
#env = gym.make('CartPole-v0').unwrapped
#env.reset()
#env.render(mode='rgb_array')

import argparse
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
    
parser = argparse.ArgumentParser(description='DQN_AGENT')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
					help='number of epochs to train (default: 300)')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
					help='batch size for training (default: 32)')
parser.add_argument('--memory-size', type=int, default=10000, metavar='M',
					help='memory length (default: 10000)')
parser.add_argument('--max-step', type=int, default=250,
					help='max steps allowed in gym (default: 250)')
parser.add_argument('--image', type=bool, default=False,
					help='whether input is image pixels or state (default: False)')
parser.add_argument('--gamma', type=float, default=0.8,
					help='discount rate for Q-learning (default: 0.8)')


args = parser.parse_args([])
args.cuda = torch.cuda.is_available()

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class DQN(nn.Module):
###################################################################
# Image input network architecture and forward propagation. Dimension
# of output layer should match the number of actions.
###################################################################
    # Define your network structure here    
    def __init__(self):
        super(DQN, self).__init__()   
        # (no need to have conv block for state input)
        if args.image:
            self.conv_block = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=5, stride=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=5, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=5, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                    )
            self.fc_block = nn.Sequential(
                    nn.Linear(448, 2)
                    )
###################################################################
# State vector input network architecture and forward propagation.
# Dimension of output layer should match the number of actions.
##################################################################
        else:
            self.state_block = nn.Sequential(
                nn.Linear(4, 60),
                nn.BatchNorm1d(60),
                nn.ReLU(),
                nn.Linear(60, 30),
                nn.BatchNorm1d(30),
                nn.ReLU(),
                nn.Linear(30, 2)
                )
            self.state_block.train(False)
 		
    # Define your forward propagation function here
    def forward(self, x):
        if args.image:
            out_conv = self.conv_block(x)
            out = self.fc_block(out_conv.view(out_conv.size(0), -1))
        else:
            out = self.state_block(x) #if size(x)[0]!=1 else 
        return out
        

class DQNagent():
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=args.memory_size)
#        memory = deque(maxlen=3)
#        memory.append((2,3))
        self.gamma = args.gamma
        self.epsilon_start = 0.99
        self.epsilon_min = 0.05
        self.epsilon_decay = 200
    ###################################################################
    # remember() function
    # remember function is for the agent to get "experience". Such experience
    # should be storaged in agent's memory. The memory will be used to train
    # the network. The training example is the transition: (state, action,
    # next_state, reward). There is no return in this function, instead,
    # you need to keep pushing transition into agent's memory. For your
    # convenience, agent's memory buffer is defined as deque.
    ###################################################################
    def remember(self, state, action, next_state, reward):
        self.memory.append([state, action, next_state, reward]) # deque of lists

    ###################################################################
    # act() fucntion
    # This function is for the agent to act on environment while training.
    # You need to integrate epsilon-greedy in it. Please note that as training
    # goes on, epsilon should decay but not equal to zero. We recommend to
    # use the following decay function:
    # epsilon = epsilon_min+(epsilon_start-epsilon_min)*exp(-1*global_step/epsilon_decay)
    # act() function should return an action according to epsilon greedy. 
    # Action is index of largest Q-value with probability (1-epsilon) and 
    # random number in [0,1] with probability epsilon.
    ###################################################################
    def act(self, state):
        global steps_done
        eps_threshold = self.epsilon_min \
                        +(self.epsilon_start-self.epsilon_min) \
                        *np.exp(-1*steps_done/self.epsilon_decay)
        steps_done  += 1
        # Index of largest Q-value with probability (1-epsilon)
        # random number in [0,1] with probability epsilon
        sample = random.random()
        if sample > eps_threshold:
            return self.model( \
                              Variable(state, volatile=True).type(torch.FloatTensor) \
                             ).data.view(1,2).max(1)[1].view(1,1) # convert size 1 to 1*2
        else:
            return torch.LongTensor([random.randrange(2)]).view(1,1)

    ###################################################################
    # replay() function
    # This function performs an one step replay optimization. It first
    # samples a batch from agent's memory. Then it feeds the batch into 
    # the network. After that, you will need to implement Q-Learning. 
    # The target Q-value of Q-Learning is Q(s,a) = r + gamma*max_{a'}Q(s',a'). 
    # The loss function is distance between target Q-value and current
    # Q-value. We recommend to use F.smooth_l1_loss to define the distance.
    # There is no return of act() function.
    # Please be noted that parameters in Q(s', a') should not be updated.
    # You may use Variable().detach() to detach Q-values of next state 
    # from the current graph.
    ###################################################################
    def replay(self, batch_size):
        batchs = random.sample(list(self.memory), batch_size)
        transitions = list(zip(*batchs))
        return transitions
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).


#################################################################
# Functions 'getCartLocation' and 'getGymScreen' are designed for 
# capturing current renderred image in gym. You can directly take 
# the return of 'getGymScreen' function, which is a resized image
# with size of 3*40*80.
#################################################################
def getCartLocation():
    world_width = env.x_threshold*2
    scale = 600/world_width
    return int(env.state[0]*scale+600/2.0)

def getGymScreen():
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = getCartLocation()
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    elif cart_location > (600-view_width//2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width//2, cart_location+view_width//2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.FloatTensor(screen)
    return resize(screen).unsqueeze(0)

def plot_durations(durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 30 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
        
def main(args):
    global env
    env = gym.make('CartPole-v0').unwrapped
    env._max_episode_steps = args.max_step
    print('env max steps:{}'.format(env._max_episode_steps))
    global steps_done
    steps_done = 0
    agent = DQNagent()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=1e-3)
    durations = []
    rewards_epoch = []

    ################################################################
    # training loop
    # You need to implement the training loop here. In each epoch, 
    # play the game until trial ends. At each step in one epoch, agent
    # need to remember the transitions in self.memory and perform
    # one step replay optimization. Use the following function to 
    # interact with the environment:
    #   env.step(action)
    # It gives you infomation about next step after taking the action.
    # The return of env.step() is (next_state, reward, done, info). You
    # do not need to use 'info'. 'done=1' means current trial ends.
    # if done equals to 1, please use -1 to substitute the value of reward.
    ################################################################
    for epoch in range(args.epochs+1):
        # steps = 0
        reward_epoch = 0
    ################################################################
    # Image input. We recommend to use the difference between two
    # images of current_screen and last_screen as input image.
    ################################################################
        env.reset()
        if args.image:
            last_screen = getGymScreen()
            current_screen = getGymScreen()
            state = current_screen - last_screen
        else:
            action = torch.LongTensor([random.randrange(2)]).view(1,1)
            s, _, _, _ = env.step(action[0, 0])
            state = torch.FloatTensor(s).view(1,4)
            
        for t in range(args.max_step+1):
            # Select and perform an action
            action = agent.act(state)
            next_s, reward, done, _ = env.step(action[0, 0])
            next_state = torch.FloatTensor(next_s).view(1,4)
            if done:
                reward = torch.Tensor([-100])
            else:
                reward = torch.Tensor([reward])    
            reward_epoch += reward
                        
            # Observe new state
            if done:
                    next_state = None
            else:             
                if args.image:
                    last_screen = current_screen
                    current_screen = getGymScreen()
                    next_state = current_screen - last_screen
                        
    
            # Store the transition in memory
            agent.remember(state, action, next_state, reward)
    
            # Move to the next state
            if not done:
                state = next_state     
    
            # Perform one step of the optimization (on the target network)
            # last_sync = 0
            if len(agent.memory) >= args.batch_size:
                transitions = agent.replay(args.batch_size)
                names = ['state', 'action', 'next_state', 'reward']
                batch = dict(zip(names, transitions)) 
                
                # Compute a mask of non-final states and concatenate the batch elements
                # map: *apply* a function to all the items in an input_list
                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, list(batch['next_state']))))
                # We don't want to backprop through the expected action values
                # volatile will save us on temporarily changing the model parameters'
                # requires_grad to False ?
                # .cat: Concatenates the given sequence of seq tensors
                non_final_next_states = Variable(torch.cat([s for s in batch['next_state']
                                            if s is not None]), volatile=True)
                state_batch = Variable(torch.cat(batch['state']))
                action_batch = Variable(torch.cat(batch['action']))
                reward_batch = Variable(torch.cat(batch['reward']))
            
                # Compute Q(s_t, a) - the model computes Q(s_t)
                # then we select columns (dim=1) of actions ([[1],[0],...]) taken
                # .gather: Gathers values along an axis specified by dim
                state_action_values = agent.model(state_batch).gather(1, action_batch)
            
                # Compute V(s_{t+1}) for all next states.
                next_state_values = Variable(torch.zeros(args.batch_size).type(torch.Tensor))
                next_state_values[non_final_mask] = agent.model(non_final_next_states).max(1)[0]
                # Now, we don't want to mess up the loss with a volatile flag, so let's
                # clear it. After this, we'll just end up with a Variable that has
                # requires_grad=False
                next_state_values.volatile = False
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * agent.gamma) + reward_batch
            
                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            
                # Optimize the model
                optimizer.zero_grad()
                # .zero_grad Clears the gradients of all optimized Variables
                loss.backward()
                for param in agent.model.parameters():
                    # clamp_: in-place version of clamp(), min if data<min, max if data>max, data otherwise 
                    param.grad.data.clamp_(-1, 1)
                # Performs a single optimization step
                optimizer.step()
            
            if done: # else to next t
                durations.append(t + 1)
                if is_ipython:
                    display.clear_output(wait=True)
                plot_durations(durations)
                # print(reward_epoch)
                rewards_epoch.append(reward_epoch) 
                break # to next epoch
    
    # display.clear_output()
    # plot_durations(durations)
    ################################################################

if __name__ == "__main__":
    main(args)

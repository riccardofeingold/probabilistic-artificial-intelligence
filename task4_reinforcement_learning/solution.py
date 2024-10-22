import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
from copy import deepcopy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DONE: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.layers = nn.ModuleList()

        # add input layer
        self.layers.append(nn.Linear(input_dim, hidden_size, device=self.device))

        # add hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, device=self.device))
        
        # add output layer for mu and sigma
        self.output_layer = nn.Linear(hidden_size, output_dim, device=self.device)
        
        # activation function
        self.activation_fn = getattr(nn.functional, activation, nn.functional.relu)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # DONE: Implement the forward pass for the neural network you have defined.
        x = s
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.output_layer(x)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # DONE: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py.
        self.actor_network = NeuralNetwork(
            self.state_dim,
            2*self.action_dim, # since we want to have a mu and stdv
            self.hidden_size,
            self.hidden_layers,
            activation="relu"
        ).to(self.device)

        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        pass

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # DONE: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        state = state.to(self.device)
        # forward pass through actor
        mean, log_stdv = self.actor_network(state).chunk(2, dim=-1)
        log_stdv = self.clamp_log_std(log_stdv)

        # create a normal distribution and sample actions
        std = log_stdv.exp()
        normal_dist = Normal(mean, std)
        if deterministic:
            # bound the mean using tanh
            action = torch.tanh(mean)
            log_prob = torch.zeros((state.shape[0], self.action_dim))
        else:
            # we choose rsample because it allows for backpropagation
            actions = normal_dist.rsample()
            action = torch.tanh(actions)
            # Eq 21 from Paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
            # the 1e-6 value should prevent a log(0) error
            log_prob = (normal_dist.log_prob(actions) - torch.log(1 - action.pow(2) + 1e-6)).sum(1, keepdim=True)

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # DONE: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.critic_network = NeuralNetwork(
            self.state_dim + self.action_dim,
            1,
            self.hidden_size,
            self.hidden_layers,
            activation="relu"
        )

        self.optimizer = optim.Adam(self.critic_network.parameters(), self.critic_lr)
        pass
    
    def forward(self, state, action):
        assert state.size(0) == action.size(0)
        
        state_action_pair = torch.cat([state, action], dim=-1)
        return self.critic_network(state_action_pair)
class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in [-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # DONE: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.
        # learning rate
        self.lr = 3e-4
        # reward scale 
        self.reward_scale = 5
        # entropy scale
        self.alpha = TrainableParameter(init_param=1/self.reward_scale, lr_param=self.lr, train_param=True, device=self.device)
        # discount factor
        self.gamma = 0.99
        # soft update parameter
        self.tau = 0.005
        # entropy target (used to calculate the loss value of self.alpha)
        self.entropy_target = -self.action_dim

        # initialize actor neural network
        self.actor = Actor(
            hidden_size=256,
            hidden_layers=2,
            actor_lr=self.lr,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )

        # initialize first critic networks
        self.critics1 = Critic(hidden_size=256, hidden_layers=2, critic_lr=self.lr,
                          state_dim=self.state_dim, action_dim=self.action_dim,
                          device=self.device)
        
        # initialize second critic networks
        self.critics2 = Critic(hidden_size=256, hidden_layers=2, critic_lr=self.lr,
                          state_dim=self.state_dim, action_dim=self.action_dim,
                          device=self.device)
        
        # initializing both target critic network
        self.target_critics1 = Critic(hidden_size=256, hidden_layers=2, critic_lr=self.lr,
                          state_dim=self.state_dim, action_dim=self.action_dim,
                          device=self.device)
        self.target_critics2 = Critic(hidden_size=256, hidden_layers=2, critic_lr=self.lr,
                          state_dim=self.state_dim, action_dim=self.action_dim,
                          device=self.device)
        pass

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # DONE: Implement a function that returns an action from the policy for the state s.
        state = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        if train:
            action, _ = self.actor.get_action_and_log_prob(state, deterministic=False)
        else:
            action, _ = self.actor.get_action_and_log_prob(state, deterministic=True)
        action = action.detach().cpu().numpy()[0]

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # DONE: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # ensure that these tensor are sent to the right device
        s_batch = s_batch.to(self.device)
        a_batch = a_batch.to(self.device)
        r_batch = r_batch.to(self.device)
        s_prime_batch = s_prime_batch.to(self.device)

        # DONE: Implement Critic(s) update here.
        # implementing Eq 7 from paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
        ## the soft Q-function parameters can be trained to minimize the soft bellman residual
        with torch.no_grad():
            next_actions, log_probs = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)
            Q1_target = self.target_critics1.forward(s_prime_batch, next_actions)
            Q2_target = self.target_critics2.forward(s_prime_batch, next_actions)
            min_Q_target_next = torch.min(Q1_target, Q2_target) - self.alpha.get_param() * log_probs
            Q_target = r_batch + self.gamma * min_Q_target_next
        
        Q1_current = self.critics1.forward(s_batch, a_batch)
        Q2_current = self.critics2.forward(s_batch, a_batch)
        
        critic1_loss = 0.5 * torch.nn.functional.mse_loss(Q1_current, Q_target)
        critic2_loss = 0.5 * torch.nn.functional.mse_loss(Q2_current, Q_target)
        
        # after calculating the critic loss based on Eq 7, we do a gradient step
        self.run_gradient_update_step(self.critics1, critic1_loss)
        self.run_gradient_update_step(self.critics2, critic2_loss)
        
        # do the soft update for the target networks with polyak averaging 
        self.critic_target_update(self.critics1.critic_network, self.target_critics1.critic_network, tau=self.tau, soft_update=True)
        self.critic_target_update(self.critics2.critic_network, self.target_critics2.critic_network, tau=self.tau, soft_update=True)
        
        # DONE: Implement Policy update here
        # implementation of Eq. 12 from paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
        new_actions, log_probs = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        Q1_new = self.critics1.forward(s_batch, new_actions)
        Q2_new = self.critics2.forward(s_batch, new_actions)
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = -(Q_new - self.alpha.get_param() * log_probs).mean()
        
        # gradient step for actor network
        self.run_gradient_update_step(self.actor, actor_loss)

        # improve the temperatur scale alpha based on Eq. 18 paper "Soft Actor-Critic Algorithms and Applications"
        self.alpha.optimizer.zero_grad()
        temperature_loss = (self.alpha.get_param() * (-log_probs - self.entropy_target).detach()).mean()
        temperature_loss.backward()
        self.alpha.optimizer.step()

        

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()

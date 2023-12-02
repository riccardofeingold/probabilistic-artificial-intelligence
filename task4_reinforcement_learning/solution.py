import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

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

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.layers = nn.ModuleList()

        # add input layer
        self.layers.append(nn.Linear(input_dim, hidden_size))

        # add hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # add output layer
        self.output_layer = nn.Linear(hidden_size, output_dim)

        # activation function
        self.activation_fn = getattr(nn.functional, activation, nn.functional.relu)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
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
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py.
        self.actor_network = NeuralNetwork(
            self.state_dim,
            self.action_dim,
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
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        state = state.to(self.device)
        # forward pass through actor 
        output = self.actor_network(state)

        # split output into mean and log_stdv
        mean, log_stdv = torch.chunk(output, 2, dim=-1)
        log_stdv = self.clamp_log_std(log_stdv)

        # create a normal distribution and sample actions
        std = log_stdv.exp()
        normal_dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            # we choose rsample because it allows for backpropagation
            action = normal_dist.rsample()
        
        log_prob = normal_dist.log_prob(action).sum(axis=-1, keepdim=True)

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
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.critic_network_one = NeuralNetwork(
            self.state_dim + self.action_dim,
            1,
            self.hidden_size,
            self.hidden_layers,
            activation="relu"
        )

        self.critic_network_two = NeuralNetwork(
            self.state_dim + self.action_dim,
            1,
            self.hidden_size,
            self.hidden_layers,
            activation="relu"
        )

        self.optimizer_one = optim.Adam(self.critic_network_one.parameters(), self.critic_lr)
        self.optimizer_two = optim.Adam(self.critic_network_two.parameters(), self.critic_lr)
        pass

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
        self.action_dim = 1  # [torque] in[-1,1]
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
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.
        self.actor = Actor(
            hidden_size=256,
            hidden_layers=2,
            actor_lr=0.0003,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )

        self.critic1 = Critic(hidden_size=256, hidden_layers=2, critic_lr=0.0003,
                          state_dim=self.state_dim, action_dim=self.action_dim,
                          device=self.device)
        self.critic2 = Critic(hidden_size=256, hidden_layers=2, critic_lr=0.0003,
                            state_dim=self.state_dim, action_dim=self.action_dim,
                            device=self.device)

        # Target Critic networks for stable Q-value estimates
        self.critic1_target = torch.deepcopy(self.critic1)
        self.critic2_target = torch.deepcopy(self.critic2)
        pass

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
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
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # TODO: Implement Critic(s) update here.

        # TODO: Implement Policy update here
        # Convert to tensors
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        a_batch = torch.FloatTensor(a_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        s_prime_batch = torch.FloatTensor(s_prime_batch).to(self.device)

        # Compute target Q-values using target critic networks and Bellman equation
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=False)
            Q1_target_next = self.critic1_target(s_prime_batch, next_action)
            Q2_target_next = self.critic2_target(s_prime_batch, next_action)
            min_Q_target_next = torch.min(Q1_target_next, Q2_target_next) - self.alpha * next_log_prob
            Q_target = r_batch + self.gamma * min_Q_target_next  # Assuming a discount factor gamma

        # Compute actual Q-values using critic networks
        Q1_current = self.critic1(s_batch, a_batch)
        Q2_current = self.critic2(s_batch, a_batch)

        # Compute critic loss
        critic1_loss = nn.functional.mse_loss(Q1_current, Q_target)
        critic2_loss = nn.functional.mse_loss(Q2_current, Q_target)

        # Update critic networks
        self.run_gradient_update_step(self.critic1, critic1_loss)
        self.run_gradient_update_step(self.critic2, critic2_loss)

        # Compute actor loss and update actor network
        # In SAC, this is typically the negative of the Q-value estimated by the critic for the
        # action chosen by the actor, minus a term for entropy regularization
        new_action, log_prob = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        Q1_new = self.critic1(s_batch, new_action)
        Q2_new = self.critic2(s_batch, new_action)
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = (self.alpha * log_prob - Q_new).mean()

        self.run_gradient_update_step(self.actor, actor_loss)

        # Soft update target networks
        self.critic_target_update(self.critic1, self.critic1_target, tau=self.tau, soft_update=True)
        self.critic_target_update(self.critic2, self.critic2_target, tau=self.tau, soft_update=True)

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

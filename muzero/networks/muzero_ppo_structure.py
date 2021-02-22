"""
This code is MuZero but with the structure of PPO.
What I mean by structure is to not only use the PPO framework but to also use the structure found in Dafna.
For example in Dafna. The trajectories for PPO and A3C are very similar so I need to use the same structure for MuZero.
"""

import random
import math
import typing
import os
import json
import tensorflow as tf
import numpy as np
from copy import deepcopy

import gym

from typing import Dict, List, Callable, Optional
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from abc import ABC, abstractmethod
from config import MuZeroConfig
from tensorboardX import SummaryWriter
# from game.game import Action, AbstractGame, ActionHistory, Player

# tf.compat.v1.enable_eager_execution()

print(tf.executing_eagerly())


ENV = 'LunarLander-v2'
CONTINUOUS = False
# ENV = 'LunarLanderContinuous-v2'
# CONTINUOUS = True


# EPISODES = 100000
# EPISODES_START_WATCH = 90000
# EPISODES_WATCH_STEP = 100
#
# # losses
# CFG_CLIP_EPSILON = 0.2  # clip loss
# CFG_BETA = 1e-3         # exploration - loss entropy beta for discrete action space
# CFG_SIGMA = 1.0         # loss exploration noise
#
# NOISE = 1.0         # exploration noise for continues action space
#
# MEMORY_SIZE = 256
# GAMMA = 0.99        # discount rewards
#
# NEURONS = 128
# NUM_LAYERS = 2
#
# EPOCHS = 10
# BATCH_SIZE = 64
#
# LR = 1e-4 # Lower lr stabilises training greatly

MAXIMUM_FLOAT_VALUE = float('inf')


########################################################
# environment for Open AI gym
########################################################

class GymEnvironment:
    ''' openAI gym environment for single player '''
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.env = gym.make(self.problem)

        self.__state = None
        self.__reward = 0
        self.__action = None
        self.__done = False
        pass

    def get_state_size(self):
        ''' return size of states '''
        return self.env.observation_space.shape[0]

    def get_action_size(self):
        ''' return number of possible actions '''
        return self.env.action_space.n

    def get_action_space(self):
        return self.env.action_space.shape, self.env.action_space.low, self.env.action_space.high

    def render(self):
        ''' render the game - can be empty '''
        self.env.render()
        pass

    def reset(self):
        ''' reset the game '''
        # gym reset returns initial state
        state = deepcopy(self.env.reset())
        self.__state = state

        self.__done = False
        pass

    def state(self, agentId):
        ''' return state '''
        return self.__state

    def agent_reward(self, agentId):
        ''' return last action reward '''
        #pop reward
        reward = self.__reward
        self.__reward = 0
        return reward

    def order(self, agentId, action):
        ''' set order to player '''
        # store the order
        self.__action = action
        pass

    def pulse(self):
        ''' implement all given orders '''
        # apply action
        action = self.__action
        state, reward, done, info = self.env.step(action)
        state = deepcopy(state)
        self.__state, reward, self.__done, info = (state, reward, done, info)
        self.__reward += reward
        # ignore info
        pass

    def Done(self):
        ''' check if game is finished '''
        return self.__done


########################################################
# Model
########################################################

''' This model should be used in regard to the Muzero models we have and not the Actor critic models.'''


# class Action(object):
#     """ Class that represent an action of a game."""
#
#     def __init__(self, index: int):
#         self.index = index
#
#     def __hash__(self):
#         return self.index
#
#     def __eq__(self, other):
#         return self.index == other.index
#
#     def __gt__(self, other):
#         return self.index > other.index


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: typing.Optional[List[float]]

    @staticmethod
    def build_policy_logits(policy_logits):
        return {Action(i): logit for i, logit in enumerate(policy_logits[0])}


class AbstractNetwork(ABC):

    def __init__(self):
        self.training_steps = 0

    @abstractmethod
    def initial_inference(self, image) -> NetworkOutput:
        pass

    @abstractmethod
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        pass


class UniformNetwork(AbstractNetwork):
    """policy -> uniform, value -> 0, reward -> 0"""

    def __init__(self, action_size: int):
        super().__init__()
        self.action_size = action_size

    def initial_inference(self, image) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)


class InitialModel(Model):
    """Model that combine the representation and prediction (value+policy) network."""

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, image):
        hidden_representation = self.representation_network(image)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class RecurrentModel(Model):
    """Model that combine the dynamic, reward and prediction (value+policy) network."""

    def __init__(self, dynamic_network: Model, reward_network: Model, value_network: Model, policy_network: Model):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden):
        hidden_representation = self.dynamic_network(conditioned_hidden)
        reward = self.reward_network(conditioned_hidden)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, reward, value, policy_logits


class BaseNetwork(AbstractNetwork):
    """Base class that contains all the networks and models of MuZero."""

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model,
                 dynamic_network: Model, reward_network: Model):
        super().__init__()
        # Networks blocks
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network

        # Models for inference and training
        self.initial_model = InitialModel(self.representation_network, self.value_network, self.policy_network)
        self.recurrent_model = RecurrentModel(self.dynamic_network, self.reward_network, self.value_network,
                                              self.policy_network)

    def initial_inference(self, image: np.array) -> NetworkOutput:
        """representation + prediction function"""

        hidden_representation, value, policy_logits = self.initial_model.predict(np.expand_dims(image, 0))
        output = NetworkOutput(value=self._value_transform(value),
                               reward=0.,
                               policy_logits=NetworkOutput.build_policy_logits(policy_logits),
                               hidden_state=hidden_representation[0])
        return output

    def recurrent_inference(self, hidden_state: np.array, action: Action) -> NetworkOutput:
        """dynamics + prediction function"""

        conditioned_hidden = self._conditioned_hidden_state(hidden_state, action)
        hidden_representation, reward, value, policy_logits = self.recurrent_model.predict(conditioned_hidden)
        output = NetworkOutput(value=self._value_transform(value),
                               reward=self._reward_transform(reward),
                               policy_logits=NetworkOutput.build_policy_logits(policy_logits),
                               hidden_state=hidden_representation[0])
        return output

    @abstractmethod
    def _value_transform(self, value: np.array) -> float:
        pass

    @abstractmethod
    def _reward_transform(self, reward: np.array) -> float:
        pass

    @abstractmethod
    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        pass

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables


class LunarLanderNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        regularizer = regularizers.l2(weight_decay)
        representation_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                             Dense(representation_size, activation=representation_activation,
                                                   kernel_regularizer=regularizer)])
        value_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])
        policy_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                     Dense(action_size, kernel_regularizer=regularizer)])
        dynamic_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(representation_size, activation=representation_activation,
                                            kernel_regularizer=regularizer)])
        reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                     Dense(1, kernel_regularizer=regularizer)])

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        if value is None:
            raise ValueError

        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        # If the value is unknow, by default we set it to the minimum possible value
        if value is None:
            return 0.0

        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):
    """A class that represent nodes inside the MCTS tree"""

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> Optional[float]:
        if self.visit_count == 0:
            return None
        return self.value_sum / self.visit_count


def softmax_sample(visit_counts, actions, t):
    counts_exp = np.exp(visit_counts) * (1 / t)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]
########################################################
# Exploration, Noise policy
########################################################


class GreedyQPolicy:
    def __init__(self):
        super().__init__()
        pass

    def SelectAction(self, q_values: np.array):
        assert isinstance(q_values, (np.ndarray))
        assert q_values.ndim == 1, "q_values.ndim = " + str(q_values.ndim)
        return np.argmax(q_values)

    def Tick(self):
        pass


class PPOExporationPolicy:
    def __init__(self, **kwargs):

        super().__init__()
        self.interval = kwargs.get('interval', 100)
        self.__cnt = kwargs.get('cnt', 0)
        self.__r = kwargs.get('initial', True)

    def CheckIfRandom(self):
        return self.__r

    def Tick(self):
        ''' optional tick '''
        self.__cnt += 1
        if self.__cnt % self.interval == 0:
            self.__r = False
        else:
            self.__r = True
        pass


class PPONoisePolicyDiscrete:
    def __init__(self, action_size, **kwargs):
        super().__init__()
        self.action_size = action_size

    def SelectAction(self, policy):
        action = np.random.choice(self.action_size, p=np.nan_to_num(policy))
        return action

    def Tick(self):
        pass


class PPOPolicy:
    def __init__(self, **kwargs):
        self.is_frozen = False
        self.action = None
        self.__Initialize(
            kwargs.get('explorationPolicy'),
            kwargs.get('qPolicy'),
            kwargs.get('noisePolicy')
        )
        pass

    def __Initialize(self, explorationPolicy, qPolicy, noisePolicy,
                     ):
        self.explorationPolicy = explorationPolicy
        self.qPolicy = qPolicy
        self.noisePolicy = noisePolicy
        pass

    def CheckIfRandom(self):
        if self.explorationPolicy is not None:
            return self.explorationPolicy.CheckIfRandom()
        return False

    def SelectAction(self, q_values: np.array, node: Node, num_moves: int, network: BaseNetwork, mode: str = 'softmax'):
        visit_counts = [child.visit_count for child in node.children.values()]
        actions = [action for action in node.children.keys()]
        if self.CheckIfRandom():
            if mode == 'softmax':
                t = config.visit_softmax_temperature_fn( # The config parameter will change for the final code.
                    num_moves=num_moves, training_steps=network.training_steps)
                q_values = softmax_sample(visit_counts, actions, t)
        else:
            if mode == 'max':
               q_values, _ = max(node.children.items(), key=lambda item: item[1].visit_count)
        return q_values

    def Tick(self):
        if self.explorationPolicy is not None:
            self.explorationPolicy.Tick()
        if self.noisePolicy is not None:
            self.noisePolicy.Tick()
        if self.qPolicy is not None:
            self.qPolicy.Tick()
        pass

########################################################
# Trajectory
########################################################


class PPOGameMemory:
    def __init__(self, **kwargs):

        self.gamma = kwargs.get('gamma', 0.99)

        self.mem = [[], [], [], []]

        self.states = []
        self.actions = []
        self.predicted_actions = []
        self.rewards = []
        pass

    def append(self, state, action, reward, predicted, done):
        self.states.append(state)
        self.actions.append(action)
        self.predicted_actions.append(predicted)
        self.rewards.append(reward)

        if done is True:
            self.apply_final_reward(reward)
            for i in range(len(self.states)):
                self.mem[0].append(self.states[i])
                self.mem[1].append(self.actions[i])
                self.mem[2].append(self.predicted_actions[i])
                self.mem[3].append(self.rewards[i])

            self.states = []
            self.actions = []
            self.predicted_actions = []
            self.rewards = []
        pass

    def HasBatch(self, batch_size):
        return (len(self.mem[0]) >= batch_size)

    def apply_final_reward(self, reward):
        for j in range(len(self.rewards) - 2, -1, -1):
            self.rewards[j] += self.rewards[j + 1] * self.gamma
        pass

    def GetBatch(self, batch_size):
        obs = np.array(self.mem[0])
        actions = np.array(self.mem[1])
        preds = np.array(self.mem[2])
        # preds = np.reshape(preds, (preds.shape[0], preds.shape[2]))
        rewards = np.reshape(np.array(self.mem[3]), (len(self.mem[3]), 1))
        obs, actions, preds, rewards = obs[:batch_size], actions[:batch_size], preds[:batch_size], rewards[:batch_size]
        return obs, actions, preds, rewards

    def PopBatch(self, batch_size):
        res = self.GetBatch(batch_size)
        self.mem = [[], [], [], []]
        return res


########################################################
# Agent
########################################################


class Agent:
    def __init__(self, actor, critic, policy, trajectory, memory_size, writer):
        self.critic = critic
        self.actor = actor
        self.policy = policy
        self.trajectory = trajectory
        self.memory_size = memory_size
        self.gradient_steps = 0
        self.writer = writer
        pass

    @property
    def continuous(self):
        return self.actor.continuous

    @property
    def action_size(self):
        return self.actor.action_size

    @property
    def state_size(self):
        return self.actor.state_size

    def get_action(self, p):
        return self.policy.SelectAction(p)

    def get_action_matrix(self, action):
        if self.continuous is False:
            action_matrix = np.zeros(self.action_size)
            action_matrix[action] = 1
            return action_matrix
        else:
            return action

    def predict(self, state, return_q_values=False):
        """ get input state and return action in array if return_q_values is True.
            use to check accuracy
            always use trained model to do it (no random)
            return action as number if return_q_values is False
        """
        # p = self.shared.predict_actor(state)
        # p = self.shared.actor_model.predict(state)
        p = self.actor.predict(state)
        if return_q_values is True:
            return p
        return self.get_action(p)

    def observe(self, state, action, reward, predicted_action, done):
        """ store in memory  """
        self.trajectory.append(state, action, reward, predicted_action, done)

        # self.__r = self.policy.CheckIfRandom()

        if done:
            self.policy_tick()
        pass

    def policy_tick(self):
        """
        Manual tick policy
        """
        self.policy.Tick()

    def __replay(self):
        # replay
        obs, actions, old_preds, rewards = self.trajectory.PopBatch(self.memory_size)

        pred_values = self.critic.model.predict(obs)
        advantages = rewards - pred_values
        actor_loss = self.actor.model.fit(
            [obs, advantages, old_preds],
            [actions],
            batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        critic_loss = self.critic.model.fit(
            [obs],
            [rewards],
            batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
        self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
        self.gradient_steps += 1

    def replay(self):
        """ replay stored memory"""
        if self.trajectory.HasBatch(self.memory_size) is True:
            self.__replay()
            return
        pass

    pass    # Agent


########################################################
# Trainer
########################################################


class Trainer:
    def __init__(self, game, agent, writer):
        self.agent = agent
        self.game = game
        self.writer = writer

        self.episode = 0
        self.game.reset()
        pass

    def run_once(self, do_train, do_render):
        self.game.reset()
        rewards = []
        while not self.game.Done():
            state = self.game.state(0)

            predicted_action = self.agent.predict(state, return_q_values=True)
            action = self.agent.get_action(predicted_action)
            action_matrix = self.agent.get_action_matrix(action)

            self.game.order(0, action)
            self.game.pulse()

            reward = self.game.agent_reward(0)
            done = self.game.Done()
            rewards.append(reward)

            if do_render:
                self.game.render()

            if do_train:
                self.agent.observe(state, action_matrix, reward, predicted_action, done)
            pass
        else:
            if do_train:
                self.writer.add_scalar('Episode reward', np.array(rewards).sum(), self.episode)
                self.agent.replay()
            else:
                self.writer.add_scalar('Val episode reward', np.array(rewards).sum(), self.episode)
                print('Episode #', self.episode, 'finished with reward', round(np.array(rewards).sum()))
                self.agent.policy_tick()
            self.episode += 1
        pass

    def run(self, episodes, episodes_start_watch):
        while self.episode < episodes:

            if self.agent.policy.CheckIfRandom():
                do_train = True
                if self.episode > episodes_start_watch:
                    do_render = True
                else:
                    do_render = False
            else:
                do_train = False
                do_render = True

            self.run_once(do_train, do_render)
        pass

    pass    # Trainer

########################################################
# __main__
########################################################


def get_name(env, is_continuous):
    name = 'AllRuns/'
    if is_continuous is True:
        name += 'continous/'
    else:
        name += 'discrete/'
    name += env
    return name


# use config to save run id
class Config:
    def __init__(self, fname):
        self.fname = fname
        if os.path.exists(fname):
            with open(fname, 'r') as json_file:
                json_dict = json.load(json_file)
            self.initialize(**json_dict)
        else:
            self.initialize()
        pass

    def initialize(self, **kwargs):
        self.run = kwargs.get('run', 0)
        pass

    def save(self):
        json_str = self.toJSON()
        with open(self.fname, 'w') as outfile:
            #    json.dump(json_dict, outfile)
            outfile.write(json_str)

    @property
    def __dict__(self):
        return {
            'run': self.run
        }

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, separators=(',', ':'), indent=4)
        # return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, separators=(',', ':'))

    def initializeFromJSON(self, s: str):
        json_dict = json.loads(s)
        self.initialize(**json_dict)
        pass


def train():

    game = GymEnvironment(ENV)

    config = Config("config.json")
    config.run += 1
    config.save()
    writer = SummaryWriter(get_name(ENV + "__" + str(config.run), CONTINUOUS))

    if CONTINUOUS is True:
        print(game.get_action_space(), 'action_space', game.get_state_size(), 'state size')
        state_size = game.get_state_size()
        action_space, space_min, space_max = game.get_action_space()
        # action_size = action_space[0]
        action_size = action_space[0] * 2
        print('action_size=', action_size, 'state_size=', state_size)

        critic_model = CriticModel(state_size, action_size, batch_size=BATCH_SIZE,
                                   neurons=NEURONS,
                                   hidden_layers=NUM_LAYERS,
                                   learnig_rate=LR)
        actor_model = ActorModelContinuous(state_size, action_size, batch_size=BATCH_SIZE,
                                           neurons=NEURONS,
                                           hidden_layers=NUM_LAYERS,
                                           learnig_rate=LR, activation_last='tanh')
        policy = PPOPolicy(
            explorationPolicy=PPOExplorationPolicy(interval=EPISODES_WATCH_STEP),
            qPolicy=None,
            noisePolicy=PPONoisePolicyContinuous(noise=NOISE))
        pass
    else:
        print(game.get_action_size(), 'action_size', game.get_state_size(), 'state size')
        state_size = game.get_state_size()
        action_size = game.get_action_size()

        critic_model = CriticModel(state_size, action_size, batch_size=BATCH_SIZE,
                                   neurons=NEURONS,
                                   hidden_layers=NUM_LAYERS,
                                   learnig_rate=LR)
        actor_model = ActorModelDiscrete(state_size, action_size, batch_size=BATCH_SIZE,
                                         neurons=NEURONS,
                                         hidden_layers=NUM_LAYERS,
                                         learnig_rate=LR, activation_last='softmax')
        policy = PPOPolicy(
            explorationPolicy=PPOExplorationPolicy(interval=EPISODES_WATCH_STEP),
            qPolicy=GreedyQPolicy(),
            noisePolicy=PPONoisePolicyDiscrete(action_size))
        pass

    agent = Agent( actor = actor_model,
                   critic = critic_model,
                   policy = policy,
                   trajectory=PPOGameMemory(gamma=GAMMA),
                   memory_size=MEMORY_SIZE,
                   writer=writer
                   )

    tr = Trainer(game,agent,writer)
    tr.run(EPISODES,EPISODES_START_WATCH)
    pass


if __name__ == '__main__':
    train()
    pass
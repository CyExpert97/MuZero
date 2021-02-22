#!/usr/bin/env python

import os
import json
import tensorflow as tf
import numpy as np
from copy import deepcopy
import typing
import random
from itertools import zip_longest
from self_play.self_play import run_selfplay, run_eval

from abc import ABC, abstractmethod
from typing import Dict, List, Callable

from tensorflow.python.keras.models import Model

from game.game import Action, AbstractGame, ActionHistory, Player
from game.gym_wrappers import ScalingObservationWrapper

from config import MuZeroConfig, make_lunarlander_config
from self_play.utils import MinMaxStats, Node, softmax_sample
from networks.shared_storage import SharedStorage

import gym
import math

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.losses import MSE
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


from tensorboardX import SummaryWriter

# from ppo_loss import get_ppo_actor_loss_clipped_obj, get_ppo_actor_loss_clipped_obj_continuous, get_ppo_critic_loss

# tf.compat.v1.enable_eager_execution()

print(tf.executing_eagerly())

#ENV = 'LunarLander-v2'
# CONTINUOUS = False
ENV = 'LunarLanderContinuous-v2'
CONTINUOUS = True


EPISODES = 100000
EPISODES_START_WATCH = 90000
EPISODES_WATCH_STEP = 100

# losses
CFG_CLIP_EPSILON = 0.2  # clip loss
CFG_BETA = 1e-3         # exploration - loss entropy beta for discrete action space
CFG_SIGMA = 1.0         # loss exploration noise

NOISE = 1.0         # exploration noise for continues action space

MEMORY_SIZE = 256
GAMMA = 0.99        # discount rewards

NEURONS = 128
NUM_LAYERS = 2

EPOCHS = 10
BATCH_SIZE = 64

LR = 1e-4  # Lower lr stabilises training greatly

########################################################
# environment for Open AI gym
########################################################


class LunarLanderDiscrete(AbstractGame):
    """The Gym LunarLander environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('LunarLander-v2')
        self.env = ScalingObservationWrapper(self.env, low=[-1, -1, -1, -1, -1, -1, -1, -1], high=[1, 1, 1, 1, 1, 1, 1, 1])
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""

        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        return reward

    def terminal(self) -> bool:
        """Is the game finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]


########################################################
# Model
########################################################


''' This model should be used in regard to the Muzero models we have and not the Actor critic models.'''


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

########################################################
# MuZero Configuration
########################################################


# KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
#
#
# class MuZeroConfig(object):
#
#     def __init__(self,
#                  game,
#                  nb_training_loop: int,
#                  nb_episodes: int,
#                  nb_epochs: int,
#                  network_args: Dict,
#                  network,
#                  action_space_size: int,
#                  max_moves: int,
#                  discount: float,
#                  dirichlet_alpha: float,
#                  num_simulations: int,
#                  batch_size: int,
#                  td_steps: int,
#                  visit_softmax_temperature_fn,
#                  lr: float,
#                  known_bounds: Optional[KnownBounds] = None):
#         ### Environment
#         self.game = game
#
#         ### Self-Play
#         self.action_space_size = action_space_size
#         # self.num_actors = num_actors
#
#         self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
#         self.max_moves = max_moves
#         self.num_simulations = num_simulations
#         self.discount = discount
#
#         # Root prior exploration noise.
#         self.root_dirichlet_alpha = dirichlet_alpha
#         self.root_exploration_fraction = 0.25
#
#         # UCB formula
#         self.pb_c_base = 19652
#         self.pb_c_init = 1.25
#
#         # If we already have some information about which values occur in the
#         # environment, we can use them to initialize the rescaling.
#         # This is not strictly necessary, but establishes identical behaviour to
#         # AlphaZero in board games.
#         self.known_bounds = known_bounds
#
#         ### Training
#         self.nb_training_loop = nb_training_loop
#         self.nb_episodes = nb_episodes  # Nb of episodes per training loop
#         self.nb_epochs = nb_epochs  # Nb of epochs per training loop
#
#         # self.training_steps = int(1000e3)
#         # self.checkpoint_interval = int(1e3)
#         self.window_size = int(1e6)
#         self.batch_size = batch_size
#         self.num_unroll_steps = 5
#         self.td_steps = td_steps
#
#         self.weight_decay = 1e-4
#         self.momentum = 0.9
#
#         self.network_args = network_args
#         self.network = network
#         self.lr = lr
#         # Exponential learning rate schedule
#         # self.lr_init = lr_init
#         # self.lr_decay_rate = 0.1
#         # self.lr_decay_steps = lr_decay_steps
#
#     def new_game(self) -> AbstractGame:
#         return self.game(self.discount)
#
#     def new_network(self) -> BaseNetwork:
#         return self.network(**self.network_args)
#
#     def uniform_network(self) -> UniformNetwork:
#         return UniformNetwork(self.action_space_size)
#
#     def new_optimizer(self) -> tf.keras.optimizers:
#         return tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum)
#
#
# def make_lunarlander_config() -> MuZeroConfig:
#     def visit_softmax_temperature(num_moves, training_steps):
#         return 1.0
#
#     return MuZeroConfig(
#         game=LunarLanderDiscrete,
#         nb_training_loop=50,
#         nb_episodes=20,
#         nb_epochs=20,
#         network_args={'action_size': 4,
#                       'state_size': 4,
#                       'representation_size': 4,
#                       'max_value': 500},
#         network=LunarLanderNetwork,
#         action_space_size=4,
#         max_moves=1000,
#         discount=0.99,
#         dirichlet_alpha=0.25,
#         num_simulations=11,  # Odd number perform better in eval mode
#         batch_size=512,
#         td_steps=10,
#         visit_softmax_temperature_fn=visit_softmax_temperature,
#         lr=0.05)

########################################################
# Exploration, Noise policy
########################################################


''' These policies should be used for the policies in Muzero'''


def add_exploration_noise(config: MuZeroConfig, node: Node):
    """
    At the start of each search, we add dirichlet noise to the prior of the root
    to encourage the search to explore new actions.
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory, network: BaseNetwork):
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state, history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats)


def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    """
    Select the child with the highest UCB score.
    """
    # When the parent visit count is zero, all ucb scores are zeros, therefore we return a random child
    if node.visit_count == 0:
        return random.sample(node.children.items(), 1)[0]

    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child


def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
    """
    The score for a node is based on its value, plus an exploration bonus based on
    the prior.
    """
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
    """
    We expand a node using the value, reward and policy prediction obtained from
    the neural networks.
    """
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    """
    At the end of a simulation, we propagate the evaluation all the way up the
    tree to the root.
    """
    for node in search_path[::-1]:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: BaseNetwork, mode: str = 'softmax'):
    """
    After running simulations inside in MCTS, we select an action based on the root's children visit counts.
    During training we use a softmax sample for exploration.
    During evaluation we select the most visited child.
    """
    visit_counts = [child.visit_count for child in node.children.values()]
    actions = [action for action in node.children.keys()]
    action = None
    if mode == 'softmax':
        t = config.visit_softmax_temperature_fn(
            num_moves=num_moves, training_steps=network.training_steps)
        action = softmax_sample(visit_counts, actions, t)
    elif mode == 'max':
        action, _ = max(node.children.items(), key=lambda item: item[1].visit_count)
    return action



########################################################
# Trajectory
########################################################

''' With Muzero trajectories should be used for Monte Carlo tree searches'''


class Action(object):
    """ Class that represent an action of a game."""

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class Player(object):
    """
    A one player class.
    This class is useless, it's here for legacy purpose and for potential adaptations for a two players MuZero.
    """

    def __eq__(self, other):
        return True


class ActionHistory(object):
    """
    Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()


class AbstractGame(ABC):
    """
    Abstract class that allows to implement a game.
    One instance represent a single episode of interaction with the environment.
    """

    def __init__(self, discount: float):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def apply(self, action: Action):
        """Apply an action onto the environment."""

        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        """After each MCTS run, store the statistics generated by the search."""

        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        """Generate targets to learn from during the network training."""

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        """Return the current player."""
        return Player()

    def action_history(self) -> ActionHistory:
        """Return the actions executed inside the search."""
        return ActionHistory(self.history, self.action_space_size)

    # Methods to be implemented by the children class
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        pass

    @abstractmethod
    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """Is the game is finished?"""
        pass

    @abstractmethod
    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        pass

    @abstractmethod
    def make_image(self, state_index: int):
        """Compute the state of the game."""
        pass


########################################################
# Agent
########################################################
''' This part here should be used in the replay functions'''


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        # Generate some sample of data to train on
        games = self.sample_games()
        game_pos = [(g, self.sample_position(g)) for g in games]
        game_data = [(g.make_image(i), g.history[i:i + num_unroll_steps],
                      g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                     for (g, i) in game_pos]

        # Pre-process the batch
        image_batch, actions_time_batch, targets_batch = zip(*game_data)
        targets_init_batch, *targets_time_batch = zip(*targets_batch)
        actions_time_batch = list(zip_longest(*actions_time_batch, fillvalue=None))

        # Building batch of valid actions and a dynamic mask for hidden representations during BPTT
        mask_time_batch = []
        dynamic_mask_time_batch = []
        last_mask = [True] * len(image_batch)
        for i, actions_batch in enumerate(actions_time_batch):
            mask = list(map(lambda a: bool(a), actions_batch))
            dynamic_mask = [now for last, now in zip(last_mask, mask) if last]
            mask_time_batch.append(mask)
            dynamic_mask_time_batch.append(dynamic_mask)
            last_mask = mask
            actions_time_batch[i] = [action.index for action in actions_batch if action]

        batch = image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch
        return batch

    def sample_games(self) -> List[AbstractGame]:
        # Sample game from buffer either uniformly or according to some priority.
        return random.choices(self.buffer, k=self.batch_size)

    def sample_position(self, game: AbstractGame) -> int:
        # Sample position from game either uniformly or according to some priority.
        return random.randint(0, len(game.history))

    # def __replay(self):
    #     # replay
    #     obs, actions, old_preds, rewards = self.trajectory.PopBatch(self.memory_size)
    #
    #     pred_values = self.critic.model.predict(obs)
    #     advantages = rewards - pred_values
    #     actor_loss = self.actor.model.fit(
    #         [obs, advantages, old_preds],
    #         [actions],
    #         batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
    #     critic_loss = self.critic.model.fit(
    #         [obs],
    #         [rewards],
    #         batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
    #     self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
    #     self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
    #     self.gradient_steps += 1
    #
    # def replay(self):
    #     """ replay stored memory"""
    #     if self.trajectory.HasBatch(self.memory_size) is True:
    #         self.__replay()
    #         return
    #     pass
    #
    # pass    # Agent


########################################################
# Trainer
########################################################
''' Should be used to train the whole model'''


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, epochs: int):
    network = storage.current_network
    optimizer = storage.optimizer

    for _ in range(epochs):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch)
        storage.save_network(network.training_steps, network)


def update_weights(optimizer: tf.keras.optimizers, network: BaseNetwork, batch):
    def scale_gradient(tensor, scale: float):
        """Trick function to scale the gradient in tensorflow"""
        return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor

    def loss():
        loss = 0
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch = batch

        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch = network.initial_model(np.array(image_batch))

        # Only update the element with a policy target
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)
        mask_policy = list(map(lambda l: bool(l), target_policy_batch))
        target_policy_batch = list(filter(lambda l: bool(l), target_policy_batch))
        policy_batch = tf.boolean_mask(policy_batch, mask_policy)

        # Compute the loss of the first pass
        loss += tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size))
        loss += tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch))

        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch,
                                                                    mask_time_batch, dynamic_mask_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # Only execute BPTT for elements with an action
            representation_batch = tf.boolean_mask(representation_batch, dynamic_mask)
            target_value_batch = tf.boolean_mask(target_value_batch, mask)
            target_reward_batch = tf.boolean_mask(target_reward_batch, mask)
            # Creating conditioned_representation: concatenate representations with actions batch
            actions_batch = tf.one_hot(actions_batch, network.action_size)

            # Recurrent step from conditioned representation: recurrent + prediction networks
            conditioned_representation_batch = tf.concat((representation_batch, actions_batch), axis=1)
            representation_batch, reward_batch, value_batch, policy_batch = network.recurrent_model(
                conditioned_representation_batch)

            # Only execute BPTT for elements with a policy target
            target_policy_batch = [policy for policy, b in zip(target_policy_batch, mask) if b]
            mask_policy = list(map(lambda l: bool(l), target_policy_batch))
            target_policy_batch = tf.convert_to_tensor([policy for policy in target_policy_batch if policy])
            policy_batch = tf.boolean_mask(policy_batch, mask_policy)

            # Compute the partial loss
            l = (tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size)) +
                 MSE(target_reward_batch, tf.squeeze(reward_batch)) +
                 tf.math.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)))

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)

        return loss

    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.training_steps += 1


def loss_value(target_value_batch, value_batch, value_support_size: int):
    batch_size = len(target_value_batch)
    targets = np.zeros((batch_size, value_support_size))
    sqrt_value = np.sqrt(np.abs(target_value_batch))
    floor_value = np.floor(sqrt_value).astype(int)
    rest = sqrt_value - floor_value
    targets[range(batch_size), floor_value.astype(int)] = 1 - rest
    targets[range(batch_size), floor_value.astype(int) + 1] = rest

    return tf.nn.softmax_cross_entropy_with_logits(logits=value_batch, labels=targets)

########################################################
# __main__
########################################################


''' The process of running the whole model together'''


def muzero(config: MuZeroConfig):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """
    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    replay_buffer = ReplayBuffer(config)

    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        train_network(config, storage, replay_buffer, config.nb_epochs)

        print("Train score:", score_train)
        print("Eval score:", run_eval(config, storage, 50))
        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    return storage.latest_network()


if __name__ == '__main__':
    # config = make_cartpole_config()
    config = make_lunarlander_config()
    muzero(config)

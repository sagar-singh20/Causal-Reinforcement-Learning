import contextlib
import io
import sys

import gym
import pyro
import torch
from gym import spaces
from gym.utils import seeding
from pyro.distributions import Categorical, Delta
from torch.nn.functional import one_hot

from rl_parsers.mdp import parse

from .errors import InternalStateError


class PyroMDP(gym.Env):  # pylint: disable=abstract-method
    metadata = {'render.modes': ['human']}

    def __init__(self, text, *, episodic, seed=None):
        self.model = parse(text)
        self.episodic = episodic
        self.seed(seed)

        if self.model.values == 'cost':
            raise ValueError('Unsupported `cost` values')

        self.discount = self.model.discount
        self.state_space = spaces.Discrete(len(self.model.states))
        self.action_space = spaces.Discrete(len(self.model.actions))
        self.reward_range = self.model.R.min(), self.model.R.max()

        if self.model.start is None:
            self.start = torch.ones(self.state_space.n) / self.state_space.n
        else:
            self.start = torch.from_numpy(self.model.start.copy())
        self.T = torch.from_numpy(self.model.T.transpose(1, 0, 2).copy())
        self.R = torch.from_numpy(self.model.R.transpose(1, 0, 2).copy())

        self.D = None
        if episodic:
            # only if episodic
            self.D = torch.from_numpy(self.model.reset.T.copy())

        self.__time_step = None
        self.state = None
        self.done = None
        self.action_prev = None
        self.reward_prev = None

    @property
    def states(self):
        return self.model.states

    @property
    def actions(self):
        return self.model.actions

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self, keep_state=False):  # pylint: disable=arguments-differ
        self.__time_step = 0

        state_probs = (
            one_hot(self.state, self.state_space.n).float()
            if keep_state
            else self.start
        )
        state_dist = Categorical(state_probs)
        self.state = pyro.sample(f'S_{self.__time_step}', state_dist)
        self.done = torch.tensor(0)
        self.action_prev = None
        self.reward_prev = None

        return self.state

    def step(self, action):
        assert self.__time_step >= 0
        assert 0 <= self.state < self.state_space.n

        if not 0 <= action < self.action_space.n:
            raise ValueError(
                f'Action should be an integer in {{0, ..., {self.action_space.n}}}'
            )

        if self.done is None or self.__time_step is None:
            raise InternalStateError(
                'The environment must be reset before being used'
            )

        if self.done:
            raise InternalStateError(
                'The previous episode has ended and the environment must reset'
            )

        self.__time_step += 1

        state_next_dist = Categorical(self.T[self.state, action])
        state_next = pyro.sample(f'S_{self.__time_step}', state_next_dist)

        reward_dist = Delta(self.R[self.state, action, state_next])
        reward = pyro.sample(f'R_{self.__time_step}', reward_dist)

        if self.episodic:
            done = self.D[self.state, action]
        else:
            done = torch.tensor(False)

        done_probs = one_hot(done.long(), 2).float()
        done_dist = Categorical(done_probs)
        done = pyro.sample(f'D_{self.__time_step}', done_dist)

        info = {
            'T': self.T[self.state, action],  # transition distribution
            'R': self.R[self.state, action],  # stochastic rewards
        }

        self.state = state_next
        self.action_prev = action
        self.reward_prev = reward

        return state_next, reward, done, info

    def render(  # pylint: disable=inconsistent-return-statements
        self, mode='human'
    ):
        if mode not in ('human', 'ansi'):
            raise ValueError('Only `human` and `ansi` modes are supported')

        # stream where to send the string representation of the env
        outfile = sys.stdout if mode == 'human' else io.StringIO()

        if self.action_prev is not None:
            ai = self.action_prev.item()
            print(f'action: {self.model.actions[ai]} (#{ai})', file=outfile)

        if self.reward_prev is not None:
            print(f'reward: {self.reward_prev.item()}', file=outfile)

        si = self.state.item()
        print(f'state: {self.model.states[si]} (#{si})', file=outfile)

        if mode == 'ansi':
            with contextlib.closing(outfile):
                return outfile.getvalue()

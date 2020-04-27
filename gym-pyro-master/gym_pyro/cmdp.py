import contextlib
import io
import sys
import types

import gym
import pyro
import torch
from gym import spaces
from gym.utils import seeding
from pyro.distributions import Categorical, Delta
from torch.nn.functional import one_hot

from .cmdp_parser import parse
from .errors import InternalStateError


class PyroCMDP(gym.Env):  # pylint: disable=abstract-method
    metadata = {'render.modes': ['human']}

    def __init__(self, text, *, episodic, seed=None):
        self.model = parse(text)
        self.episodic = episodic
        self.seed(seed)

        if self.model.values == 'cost':
            raise ValueError('Unsupported `cost` values')

        self.discount = self.model.discount
        self.confounder_space = spaces.Discrete(len(self.model.confounders))
        self.state_space = spaces.Discrete(len(self.model.states))
        self.action_space = spaces.Discrete(len(self.model.actions))
        self.reward_range = self.model.R.min(), self.model.R.max()

        if self.model.U is None:
            self.U = (
                torch.ones(self.confounder_space.n) / self.confounder_space.n
            )
        else:
            self.U = torch.from_numpy(self.model.U.copy())

        if self.model.start is None:
            self.start = torch.ones(self.state_space.n) / self.state_space.n
        else:
            self.start = torch.from_numpy(self.model.start.copy())
        self.T = torch.from_numpy(self.model.T.transpose(1, 0, 2).copy())
        self.R = torch.from_numpy(self.model.R.transpose(0, 2, 1, 3).copy())

        self.D = None
        if episodic:
            # only if episodic
            self.D = torch.from_numpy(self.model.reset.T.copy())

        self.__time = None
        self.confounder = None
        self.state = None
        self.done = None
        self.action_prev = None
        self.reward_prev = None

    @property
    def confounders(self):
        return self.model.confounders

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
        self.__time = 0

        if keep_state:
            state_probs = one_hot(self.state, self.state_space.n).float()
        else:
            state_probs = self.start

        self.confounder = pyro.sample('U', Categorical(self.U))
        self.state = pyro.sample(f'S_{self.__time}', Categorical(state_probs))
        self.done = torch.tensor(0)
        self.action_prev = None
        self.reward_prev = None

        return self.state, self.confounder

    def step(self, action):
        assert self.__time >= 0
        assert 0 <= self.state < self.state_space.n

        if not 0 <= action < self.action_space.n:
            raise ValueError(
                f'Action should be an integer in {{0, ..., {self.action_space.n}}}'
            )

        if self.done is None or self.__time is None:
            raise InternalStateError(
                'The environment must be reset before being used'
            )

        if self.done:
            raise InternalStateError(
                'The previous episode has ended and the environment must reset'
            )

        self.__time += 1

        state_next_dist = Categorical(self.T[self.state, action])
        state_next = pyro.sample(f'S_{self.__time}', state_next_dist)

        reward_dist = Delta(
            self.R[self.confounder, self.state, action, state_next]
        )
        reward = pyro.sample(f'R_{self.__time}', reward_dist)

        if self.episodic:
            done = self.D[self.state, action]
        else:
            done = torch.tensor(False)

        done_probs = one_hot(done.long(), 2).float()
        done_dist = Categorical(done_probs)
        done = pyro.sample(f'D_{self.__time}', done_dist)

        info = {
            'T': self.T[self.state, action],
            'R': self.R[self.confounder, self.state, action],
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

        ui = self.confounder.item()
        print(f'confounder: {self.model.confounders[ui]} (#{ui})', file=outfile)

        si = self.state.item()
        print(f'state: {self.model.states[si]} (#{si})', file=outfile)

        if mode == 'ansi':
            with contextlib.closing(outfile):
                return outfile.getvalue()


def make_cmdp(path, *args, **kwargs):
    """make_cmdp

    Creates a PyroCMDP instance based on the given CMDP file, injecting the
    respective custom renderer if found in `renders.py`.

    :param path: path to CMDP file
    :param *args: arguments to PyroCMDP
    :param **kwargs: keyword arguments to PyroCMDP
    """
    with open(path) as f:
        env = PyroCMDP(
            f.read(), *args, **kwargs
        )  # pylint: disable=missing-kwoa

    basename = path.split('/')[-1]
    try:
        render = renders.get_render(basename)
    except KeyError:
        pass
    else:
        env.render = types.MethodType(render, env)

    return env

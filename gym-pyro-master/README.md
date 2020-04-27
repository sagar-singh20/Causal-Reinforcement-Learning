# gym-pyro

OpenAI Gym environments for MDPs, POMDPs, and confounded-MDPs implemented as
pyro-ppl probabilistic programs.

## Installation

This package is dependent on the
[rl_parsers](https://github.com/abaisero/rl_parsers) package.  Install
`rl_parsers` first, then install the packaged in `requirements.txt`.

## Contents

This repository provides the `PyroMDP`, `PyroPOMDP`, and `PyroCMDP`
environments, whose dynamics are respectively loaded from the `.mdp`, `.pomdp`
and `.cmdp` file formats.

See [Cassandra's POMDP page](https://pomdp.org/code/pomdp-file-spec.html) for
the specifications of the POMDP file format.  The MDP and CMDP formats follow
suit.  Also see the `examples/` folder for sample files in the `.mdp`, `.pomdp`
and `.cmdp` file format.

## Useful Attributes and Methods

### PyroMDP

##### Attributes:

`env.states`
: tuple of state names

`env.actions`
: tuple of action names

##### Methods:

`env.reset(keep_state=False)`
: Resets the time-index to 0 and the system state by sampling site `S_0` from
the starting distribution.  If `keep_state` is True, then the time-index is
reset, but the current system state is kept and resampled as site `S_0`;  This
is useful to run simulations from the current environment context.

`env.step(action)` 
: Performs the action, advances the time-index and the system state.  State,
reward and done variables are respectively sampled as sites `S_{t}`, `R_{t}`,
and `D_{t}`, where `t` is the current time-index.

`env.render(mode='human')`
: Renders the previous action and reward, and the current system state.
Accepts modes `human` and `ansi`.

### PyroPOMDP

##### Attributes:

`env.states`
: tuple of state names

`env.actions`
: tuple of action names

`env.observations`
: tuple of observation names

##### Methods:

`env.reset(keep_state=False)`
: Resets the time-index to 0 and the system state by sampling site `S_0` from
the starting distribution.  If `keep_state` is True, then the time-index is
reset, but the current system state is kept and resampled as site `S_0`;  This
is useful to run simulations from the current environment context.

`env.step(action)` 
: Performs the action, advances the time-index and the system state.  State,
observation, reward and done variables are respectively sampled as sites
`S_{t}`, `O_{t}`, `R_{t}`, and `D_{t}`, where `t` is the current time-index.

`env.render(mode='human')`
: Renders the previous action and reward, and the current system state.
Accepts modes `human` and `ansi`.

### PyroCMDP

##### Attributes:

`env.confounders`
: tuple of confounder names

`env.states`
: tuple of state names

`env.actions`
: tuple of action names

##### Methods:

`env.reset(keep_state=False)`
: Resets the time-index to 0, the confounder by sampling site `U`, and the
system state by sampling site `S_0` from the starting distribution.  If
`keep_state` is True, then the time-index is reset, but the confounder and the
current system state are kept and resampled as sites `U` and  `S_0`;  This is
useful to run simulations from the current environment context.

`env.step(action)` 
: Performs the action, advances the time-index and the system state.  State,
reward and done variables are respectively sampled as sites `S_{t}`, `R_{t}`,
and `D_{t}`, where `t` is the current time-index.

`env.render(mode='human')`
: Renders the previous action and reward, and the current system state.
Accepts modes `human` and `ansi`.

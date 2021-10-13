import copy
from typing import List, Optional, Text, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents import networks, policies, trajectories
from tf_agents.typing import types

from connectfour.game.tictactoe import TicTacToeEnvironment
from connectfour.utils.mcts import MCTSTree

class MctsPolicy(policies.actor_policy.ActorPolicy):
  def __init__(
    self,
    env: TicTacToeEnvironment,
    time_step_spec: trajectories.TimeStep,
    action_spec: types.NestedTensorSpec,
    actor_network: networks.network.Network,
    training: bool = False,
    name: Optional[Text] = None
  ):
    super(MctsPolicy, self).__init__(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network,
      training=training,
      name=name
    )

    self._env = env
  
  def _step(self, action: np.ndarray, time_step: trajectories.TimeStep) -> trajectories.TimeStep:
    self._env.set_state(copy.deepcopy(time_step))
    return self._env.step(action)
  
  def _legal_actions(self, time_step: trajectories.TimeStep) -> List[np.ndarray]:
    return TicTacToeEnvironment.legal_actions(TicTacToeEnvironment.to_board(time_step.observation))
  
  def _evaluation(self, time_step: trajectories.TimeStep) -> Tuple:
    indices = [3 * action[0] + action[1] for action in self._legal_action(time_step)]
    probabilities, values = self._apply_actor_network(
      time_step.observation,
      step_type=time_step.step_type,
      policy_state=(),
      mask=None
    )

  def _mcts(self, time_step: trajectories.TimeStep):



  def _distribution(self, time_step: trajectories.TimeStep, policy_state: None) -> trajectories.PolicyStep:
    # Actor network outputs nested structure of distributions or actions.

    

    output, policy_state = self._apply_actor_network(
      time_step.observation,
      step_type=time_step.step_type,
      policy_state=policy_state,
      mask=None
    )

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution, output)
    return trajectories.PolicyStep(distributions, policy_state)

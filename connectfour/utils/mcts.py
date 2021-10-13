from __future__ import annotations
from collections import namedtuple
import copy
from typing import Callable, List, Optional, Tuple

import numpy as np
from tf_agents import trajectories
from tf_agents.trajectories.time_step import TimeStep
import tensorflow_probability as tfp


Stats = namedtuple('Stats', ['probability', 'visits', 'rewards'])


STEP_FN = Callable[[np.ndarray, trajectories.TimeStep], trajectories.TimeStep]
LEGAL_ACTIONS_FN = Callable[[np.ndarray], List[Tuple[int, np.ndarray]]]
EVALUATION_FN = Callable[[np.ndarray], Tuple[np.ndarray, float]]


class MCTSNode(object):
  def __init__(
    self,
    parent: Optional[MCTSNode],
    player: int,
    state: trajectories.TimeStep,
    stats: Stats,
    actions: List[Tuple[int, np.ndarray]],
    probabilities: List[Tuple[int, float]]
  ) -> None:
    self.parent = parent
    self.children = {}
    self.player = player

    self.state = state
    self.stats = stats
    self.untried_actions = actions
    self.valid_actions = len(actions)
    self.probabilities = probabilities

  @property
  def __is_expanded(self) -> bool:
    """
    Returns whether the node is fully expanded.

    :return: Boolean
    """

    return len(self.children) == self.valid_actions

  @property
  def __is_terminal(self) -> bool:
    """
    Returns whether the node is a leaf node.

    :return: Boolean
    """

    return self.valid_actions == 0

  def score(self, player: int, visits: int, exploration_constant: float) -> float:
    """
    Returns the node's score.

    :param visits: Number of the parent node's visits
    :param exploration_constant: Exploration constant
    :return: The score
    """

    action_value = self.stats.rewards / self.stats.visits * player
    upper_confidence_value = exploration_constant * self.stats.probability * np.sqrt(visits / (1 + self.stats.visits))

    return action_value + upper_confidence_value

  def select(self, exploration_constant: float) -> MCTSNode:
    """
    Returns the best child node recursively.

    :param exploration_constant: Exploration constant
    :return: The best child node
    """

    if self.__is_expanded and not self.__is_terminal:
      scores = {
        action: child.score(self.player, self.stats.visits, exploration_constant)
        for action, child in self.children.items()
      }

      return self.children[max(scores, key=scores.get)].select(exploration_constant)

    else:
      return self

  def expand(self, step_fn: STEP_FN, legal_actions_fn: LEGAL_ACTIONS_FN, evaluation_fn: EVALUATION_FN) -> MCTSNode:
    """
    Returns the node itself if it is fully expanded or the created child node

    :param step: The game's step method
    :param valid_actions: The game's valid_actions method
    :param evaluate: An evaluation method
    :return: The node itself or the created child node
    """

    if not self.__is_expanded:
      action_index, action = self.untried_actions.pop()

      time_step = step_fn(action, self.state)
      actions = legal_actions_fn(time_step.observation)
      probability_index, probability = self.probabilities.pop()

      assert action_index == probability_index, 'Action did not match the probability'

      if actions:
        probabilities, value = evaluation_fn(time_step.observation)
        reward = float(time_step.reward) * -1 if time_step.step_type == trajectories.StepType.LAST else value

        dirichlet_noise = np.random.dirichlet(np.repeat(.03, probabilities.shape), 1)[0]
        noisey_probabilities = (.75 * probabilities + .25 * dirichlet_noise)

        self.children[action_index] = MCTSNode(
          parent=self,
          player=self.player * -1,
          state=copy.deepcopy(time_step),
          stats=Stats(probability, 1, reward),
          actions=actions,
          probabilities=[(i, noisey_probabilities[i]) for i, _ in actions]
        )

        return self.children[action_index]
      
      else:
        self.children[action_index] = MCTSNode(
          parent=self,
          player=self.player * -1,
          state=copy.deepcopy(time_step),
          stats=Stats(probability, 1, float(time_step.reward) * -1),
          actions=actions,
          probabilities=[]
        )

        return self.children[action_index]
    
    else:
      return self

  def simulate(self) -> float:
    """
    Returns the simulated winner.

    :return: Simulated winner
    """

    return self.stats.rewards / self.stats.visits * self.player

  def backup(self, winner: int) -> None:
    """
    Propagates the winner up the tree.

    :param winner: Simulated winner
    """

    self.stats = Stats(self.stats.probability, self.stats.visits + 1, self.stats.rewards + winner * self.player)

    if self.parent:
      self.parent.backup(winner)


class MCTSTree:
  def __init__(self, step_fn: STEP_FN, legal_actions_fn: LEGAL_ACTIONS_FN, evaluation_fn: EVALUATION_FN):

    self.step_fn = step_fn
    self.legal_actions_fn = legal_actions_fn
    self.evaluation_fn = evaluation_fn

    self.root = None

  def reset(self, state: TimeStep, player: int) -> None:
    """
    Resets the tree's root state given a new state.

    :param state: New state
    """

    actions = self.legal_actions_fn(state.observation)
    probabilities, _ = self.evaluation_fn(state.observation)

    dirichlet_noise = np.random.dirichlet(np.repeat(.03, probabilities.shape), 1)[0]
    noisey_probabilities = (.75 * probabilities + .25 * dirichlet_noise)

    self.root = MCTSNode(
      parent=None,
      player=player,
      state=state,
      stats=Stats(probability=1, visits=0, rewards=0.),
      actions=actions,
      probabilities=[(i, noisey_probabilities[i]) for i, _ in actions]
    )

  def simulate(self, num_simulations: int, num_actions: int, exploration_constant: float, temperatur: float) -> np.ndarray:
    """
    Returns probabilities over all valid actions based on simulation.

    :param num_simulations: Number of simulations
    :param num_simulations: Number of actions in action space
    :param exploration_constant: Exploration constant
    :param temperature: Softmax temperature
    :return: Action probabilities
    """

    for _ in range(num_simulations):
      selected = self.root.select(exploration_constant)
      expanded = selected.expand(self.step_fn, self.legal_actions_fn, self.evaluation_fn)
      reward = expanded.simulate()
      selected.backup(reward)

    if temperatur > 0:


      visits = {key: child.stats.visits for key, child in self.root.children.items()}

      logits = np.zeros(num_actions)
      logits[list(visits.keys())] = list(visits.values())
      logits = np.power(logits, 1 / temperatur)

      return tfp.distributions.Multinomial(logits, total_counts=1)
    
    else:
      index = np.argmax([child.stats.visits for child in self.root.children])



      probabilities = np.zeros(len(self.root.children), dtype=np.float32)
      probabilities[index] = 1.0

      return probabilities[::-1]

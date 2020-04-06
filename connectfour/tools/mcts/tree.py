from typing import Any, Callable, Dict, Tuple, TypeVar

import numpy as np
from scipy.special import softmax

from connectfour.tools.mcts import stats as sts
from connectfour.tools.mcts.node import MCTSNode


State = TypeVar('State')
Action = TypeVar('Action')

Step = Callable[[State, Action], Tuple[State, Any]]
ValidActions = Callable[[State], np.ndarray]
Evaluate = Callable[[State], Tuple[np.ndarray, float]]


class MCTSTree:
    def __init__(self, step: Step, valid_actions: ValidActions, evaluate: Evaluate):
        self.step = step
        self.valid_actions = valid_actions
        self.evaluate = evaluate

        self.root = None

    def reset(self, state: State) -> None:
        """
        Resets the tree's root state given a new state.

        :param state: New state
        """

        probabilities, _ = self.evaluate(state)

        self.root = MCTSNode(
            parent=None,
            children={},
            state=state,
            actions=self.valid_actions(state),
            probabilities=probabilities,
            stats=sts.empty(1.0)
        )

    def simulate(self,
                 num_simulations: int,
                 exploration_constant: float,
                 temperature: float,
                 player: int) -> Dict[Action, float]:
        """
        Returns probabilities over all valid actions based on simulation.

        :param num_simulations: Number of simulations
        :param exploration_constant: Exploration constant
        :param temperature: Softmax temperature
        :param player: Current player
        :return: Action probabilities
        """

        for _ in range(num_simulations):
            selected = self.root.select(exploration_constant, player)
            expanded = selected.expand(self.step, self.valid_actions, self.evaluate)
            reward = expanded.simulate(player)
            selected.backup(reward)

        visits = np.array([child.stats.visits for child in self.root.children.values()])
        probabilities = softmax(visits / (np.sum(visits) * temperature))

        return dict(zip(
            self.root.children.keys(),
            probabilities
        ))

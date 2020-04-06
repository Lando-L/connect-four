from __future__ import annotations
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

import numpy as np

from connectfour.tools.mcts import stats as sts


State = TypeVar('State')
Action = TypeVar('Action')

Step = Callable[[State, Action], Tuple[State, Any]]
ValidActions = Callable[[State], np.ndarray]
Evaluate = Callable[[State], Tuple[np.ndarray, float]]


class MCTSNode((Generic[State, Action])):
    def __init__(self,
                 parent: Optional[MCTSNode],
                 children: Dict[int, MCTSNode],
                 state: State,
                 actions: np.ndarray,
                 probabilities: np.ndarray,
                 stats: sts.MCTSNodeStats) -> None:

        self.parent = parent
        self.children = children

        self.state = state
        self.untried_actions = actions.tolist()
        self.valid_actions = len(actions)
        self.probabilities = probabilities.tolist()

        self.stats = stats

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

    def score(self, visits: int, exploration_constant: float, player: int) -> float:
        """
        Returns the node's score.

        :param visits: Number of the parent node's visits
        :param exploration_constant: Exploration constant
        :param player: Current player
        :return: The score
        """

        action_value = self.stats.rewards / self.stats.visits
        upper_confidence = self.stats.probability * np.sqrt(visits / (1 + self.stats.visits))

        return action_value + player * exploration_constant * upper_confidence

    def select(self, exploration_constant: float, player: int) -> MCTSNode:
        """
        Returns the best child node recursively.

        :param exploration_constant: Exploration constant
        :param player: Current player
        :return: The best child node
        """

        if self.__is_expanded and not self.__is_terminal:
            scores = {
                action: player * child.score(self.stats.visits, exploration_constant, player)
                for action, child in self.children.items()
            }

            return self.children[max(scores, key=scores.get)].select(exploration_constant, player)

        else:
            return self

    def expand(self, step: Step, valid_actions: ValidActions, evaluate: Evaluate) -> MCTSNode:
        """
        Returns the node itself if it is fully expanded or the created child node

        :param step: The game's step method
        :param valid_actions: The game's valid_actions method
        :param evaluate: An evaluation method
        :return: The node itself or the created child node
        """

        if self.__is_expanded:
            return self

        else:
            action = self.untried_actions.pop()
            state, outcome = step(self.state, action)
            actions = valid_actions(state)
            probability = self.probabilities[action]
            probabilities, value = evaluate(state)

            probability_noise = np.random.normal(0, .01, probabilities.shape)
            reward = value if outcome is None else outcome

            self.children[action] = MCTSNode(
                parent=self,
                children={},
                state=state,
                actions=actions,
                probabilities=probability_noise + probabilities,
                stats=sts.MCTSNodeStats(probability, 1, reward),
            )

            return self.children[action]

    def simulate(self, player) -> float:
        """
        Returns the simulated return.

        :param player: Current player
        :return: Simulated reward
        """

        return self.score(0, 0., player)

    def backup(self, reward: float) -> None:
        """
        Propagates the received reward up the tree.

        :param reward: Received reward
        """

        self.stats = sts.update(self.stats, reward)

        if self.parent:
            self.parent.backup(reward)

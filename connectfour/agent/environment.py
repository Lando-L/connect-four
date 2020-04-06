import os
from typing import Tuple

import numpy as np

from connectfour.agent.memory import AgentMemory
from connectfour.agent.model import AgentModel
from connectfour.game import gamestate
from connectfour.tools.mcts.tree import MCTSTree


class AgentEnvironment:
    def __init__(self, observation_space: Tuple, action_space: int, value_space: int) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.value_space = value_space

        self.memory = AgentMemory()
        self.model = AgentModel(observation_space, action_space, value_space)
        self.tree = MCTSTree(gamestate.step, gamestate.valid_actions, self.model.evaluate)

        self.histories = []

    def evaluate(self, num_simulations: int, exploration_constant: float, temperature: float) -> None:
        state = gamestate.empty()

        episode = []
        outcome = None

        while outcome is None:
            self.tree.reset(state)
            probabilities = self.tree.simulate(num_simulations, exploration_constant, temperature, state.player)
            actions = np.array([probabilities.get(i, 0) for i in range(self.action_space)])
            action = np.random.choice(np.arange(self.action_space), p=actions)

            episode.append((state, actions))

            state, outcome = gamestate.step(state, action)

        for state, action in episode:
            self.memory.store((state, action, outcome))
            self.memory.store((
                gamestate.GameState(
                    board=np.fliplr(state.board),
                    depths=state.depths,
                    player=state.player
                ), action, outcome
            ))

    def improve(self, num_samples: int, batch_size: int, epochs: int) -> None:
        states, actions, outcomes = self.memory.sample(num_samples)
        history = self.model.train(states, actions, outcomes, batch_size, epochs)

        self.histories.append(history)

    def iterate(self,
                num_iterations: int,
                checkpoint: int,
                path: str,
                num_simulations: int,
                exploration_constant: float,
                temperature: float,
                samples: int,
                batch_size: int,
                epochs: int) -> None:

        for i in range(1, num_iterations + 1):
            self.evaluate(num_simulations, exploration_constant, temperature)
            self.improve(samples, batch_size, epochs)

            if i % checkpoint == 0:
                print(f'Saving model at checkpoint {i}')
                self.model.save(os.path.join(path, f'checkpoint_{i}_model.h5'))

        self.model.save(os.path.join(path, 'model.h5'))
        self.memory.save(os.path.join(path, 'memory.pickle'))

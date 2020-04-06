from collections import deque
import pickle
import random
from typing import List, Tuple

import numpy as np

from connectfour.game.gamestate import GameState


class AgentMemory:
    def __init__(self):
        self.memory = None

    def build(self, memory_size: int) -> None:
        """
        Builds a new memory.

        :param memory_size: Memory size
        """

        self.memory = deque(maxlen=memory_size)

    def load(self, path) -> None:
        """
        Loads memory.

        :param path: File path
        """

        self.memory = pickle.load(open(path, 'rb'))

    def save(self, path) -> None:
        """
        Save the current memory.

        :param path: File path
        """

        pickle.dump(self.memory, open(path, 'wb'))

    def store(self, sample: Tuple[GameState, np.ndarray, float]) -> None:
        """
        Stores a new sample.

        :param sample: Sample
        """

        self.memory.append(sample)

    def sample(self, num_samples: int) -> Tuple[List[GameState], List[np.ndarray], List[int]]:
        """
        Returns samples from the memory.

        :param num_samples: Number of samples
        :return: Samples
        """

        return zip(*random.sample(self.memory, min(num_samples, len(self.memory))))

from collections import namedtuple
from itertools import groupby, chain
from typing import Iterator, Optional, Tuple

import numpy as np


GameState = namedtuple('GameState', 'board depths player')
Player = int

PLAYER_ONE = 1
PLAYER_TWO = -1

PLAYER_TO_IDX = {1: -2, -1: -1}


def empty() -> GameState:
    """
    Returns the initial game state.

    :return: Empty GameState
    """
    return GameState(
        board=np.zeros((6, 7, 10), dtype=np.int),
        depths=np.zeros(7, dtype=np.int),
        player=PLAYER_ONE,
    )


def result(state: GameState) -> Optional[int]:
    """
    Returns the winner of the given GameState:
    - Draw: 0
    - Player One won: 1
    - Player Two won: -1
    - Or when the game is not over yet: None

    :param state: Current GameState
    :return: Winner of the game
    """

    def diagonal_top_bottom(board: np.ndarray) -> Iterator[int]:
        for di in ([(j, i - j) for j in range(6)] for i in range(12)):
            yield [board[i, j] for i, j in di if 0 <= i < 6 and 0 <= j < 7]

    def diag_bottom_top(board: np.ndarray) -> Iterator[int]:
        for di in ([(j, i - 6 + j + 1) for j in range(6)] for i in range(12)):
            yield [board[i, j] for i, j in di if 0 <= i < 6 and 0 <= j < 7]

    for player, idx in PLAYER_TO_IDX.items():
        for line in chain(
                state.board[:, :, idx].tolist(),
                state.board[:, :, idx].T.tolist(),
                diagonal_top_bottom(state.board[:, :, idx]),
                diag_bottom_top(state.board[:, :, idx])
        ):
            for key, group in groupby(line):
                if key == 1 and len(list(group)) >= 4:
                    return player

    if np.all(
        np.logical_or(state.board[:, :, PLAYER_TO_IDX[PLAYER_ONE]], state.board[:, :, PLAYER_TO_IDX[PLAYER_TWO]])
    ):
        return 0

    else:
        return None


def valid_actions(state: GameState) -> np.ndarray:
    """
    Returns all valid actions for a given GameState.

    :param state: Current GameState
    :return: Valid actions
    """
    if result(state) is not None:
        return np.array([])
    else:
        return np.array([idx for idx, depth in enumerate(state.depths) if depth < 6])


def step(state: GameState, action: int) -> Tuple[GameState, Optional[Player]]:
    """
    Returns the next state as well as the potential winner of the next state, if there is one.

    :param state: Current GameState
    :param action: Action to take
    :return: Next GameState and  winner
    """
    if state.depths[action] >= state.board.shape[0]:
        raise Exception(f'Column {action} is full')

    else:
        board = np.copy(state.board[:, :, -2:])
        board[state.depths[action], action, PLAYER_TO_IDX[state.player]] = 1

        depths = np.copy(state.depths)
        depths[action] += 1

        player = PLAYER_TWO if state.player == PLAYER_ONE else PLAYER_ONE
        next_state = GameState(
            board=np.concatenate([state.board[:, :, 2:], board], axis=-1),
            depths=depths,
            player=player
        )

        return next_state, result(next_state)


def to_string(state: GameState) -> str:
    """
    Returns a printable string representation of the GameState

    :param state: Current GameState
    :return: String representation
    """

    def player_to_ascii(player: Player) -> str:
        if player == PLAYER_ONE:
            return 'X'
        elif player == PLAYER_TWO:
            return 'O'
        else:
            return ' '

    board_1 = state.board[:, :, PLAYER_TO_IDX[PLAYER_ONE]] * PLAYER_ONE
    board_2 = state.board[:, :, PLAYER_TO_IDX[PLAYER_TWO]] * PLAYER_TWO
    board = np.flipud(board_1 + board_2)

    return '\n'.join(['| ' + ' '.join(map(player_to_ascii, board[i])) + ' |' for i in range(len(board))])

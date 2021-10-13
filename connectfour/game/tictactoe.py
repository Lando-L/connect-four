import copy
from typing import List, Tuple

import numpy as np
from tf_agents import environments, specs, trajectories


class TicTacToeEnvironment(environments.py_environment.PyEnvironment):
  # Players
  PLAYER_ONE = -1
  PLAYER_TWO = 1
  
  # Rewards
  REWARD_WIN = np.asarray(1., dtype=np.float32)
  REWARD_WIN.setflags(write=False)
  
  REWARD_LOSS = np.asarray(-1., dtype=np.float32)
  REWARD_LOSS.setflags(write=False)
  
  REWARD_DRAW_OR_NOT_FINAL = np.asarray(0., dtype=np.float32)
  REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)
  
  REWARD_ILLEGAL_MOVE = np.asarray(-.001, dtype=np.float32)
  REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)
  
  @staticmethod
  def legal_actions(board: np.ndarray) -> List[Tuple[int, int]]:
    return list(map(np.asarray, zip(*np.nonzero(board == 0))))
  
  @staticmethod
  def result(board: np.ndarray, player: int) -> Tuple[bool, int]:
    seqs = np.array([
      # rows
      board[0, :], board[1, :], board[2, :],
      # columns
      board[:, 0], board[:, 1], board[:, 2],
      # diagonals
      board[(0, 1, 2), (0, 1, 2)],
      board[(2, 1, 0), (0, 1, 2)],
    ]).tolist()
    
    if [player] * 3 in seqs:
      return True, TicTacToeEnvironment.REWARD_WIN
    
    if [-player] * 3 in seqs:
      return True, TicTacToeEnvironment.REWARD_LOSS
    
    if 0 in board:
      return False, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL
    
    return True, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL 
    
  @staticmethod
  def to_string(board: np.ndarray):
    def player_to_ascii(player: int) -> str:
      if player == TicTacToeEnvironment.PLAYER_ONE:
        return 'X'
      elif player == TicTacToeEnvironment.PLAYER_TWO:
        return 'O'
      else:
        return ' '

    return '\n'.join(['| ' + ' '.join(map(player_to_ascii, board[i])) + ' |' for i in range(len(board))])
  
  @staticmethod
  def to_board(states: np.ndarray):
    player_one_board = states[:, :, 0] * TicTacToeEnvironment.PLAYER_ONE
    player_two_board = states[:, :, 1] * TicTacToeEnvironment.PLAYER_TWO
    
    return player_one_board + player_two_board
  
  def __init__(self, discount: float = 1.) -> None:
    super(TicTacToeEnvironment, self).__init__(handle_auto_reset=True)
    self._states = None
    self._discount = np.asarray(discount, dtype=np.float32)
  
  def action_spec(self):
    # Actions the x and y coordinates of the board 
    return specs.array_spec.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=0, maximum=2, name='action')

  def observation_spec(self):
    # 3 planes constitute of 2 Features (Stones of player 1 and player 2) plus one plane indicating the current players colour
    return specs.array_spec.BoundedArraySpec(shape=(3, 3, 3), dtype=np.int32, minimum=-1, maximum=1, name='observation')

  def _reset(self) -> trajectories.TimeStep:
    self._states = np.zeros((3, 3, 3), np.int32)
    self._states[:, :, -1].fill(TicTacToeEnvironment.PLAYER_ONE)
    return trajectories.restart(self._states)
  
  def _step(self, action: np.ndarray) -> trajectories.TimeStep:
    player_tuple = tuple(action)
    player_index = (self._states[0, 0, -1] + 1) // 2
    player_action = player_tuple + (player_index,)
    
    if self._states[player_tuple][0] != 0 or self._states[player_tuple][1] != 0:
      self._states[:, :, -1].fill(-self._states[0, 0, -1])
      return trajectories.termination(self._states, TicTacToeEnvironment.REWARD_ILLEGAL_MOVE)
    
    self._states[player_action] = 1
    
    is_final, reward = TicTacToeEnvironment.result(
      TicTacToeEnvironment.to_board(self._states),
      self._states[0, 0, -1]
    )
    
    self._states[:, :, -1].fill(-self._states[0, 0, -1])
    
    if is_final:
      return trajectories.termination(self._states, reward)
    
    else:
      return trajectories.transition(self._states, reward, self._discount)
  
  def get_state(self) -> trajectories.TimeStep:
    return copy.deepcopy(self._current_time_step)

  def set_state(self, time_step: trajectories.TimeStep) -> None:
    self._current_time_step = time_step
    self._states = time_step.observation

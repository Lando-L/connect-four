import numpy as np
from tf_agents import environments, trajectories, utils

from connectfour.game.tictactoe import TicTacToeEnvironment

class TicTacToeEnvironmentTest(utils.test_utils.TestCase):
  def setUp(self):
    super(TicTacToeEnvironmentTest, self).setUp()
    self.discount = np.asarray(1., dtype=np.float32)
    self.env = TicTacToeEnvironment()
    step = self.env.reset()

    expected = np.zeros((3, 3, 3), dtype=np.float32)
    expected[:, :, -1].fill(-1)
    
    np.testing.assert_array_equal(expected, step.observation)
  
  def test_validate_specs(self):
    environments.utils.validate_py_environment(self.env, episodes=10)

  def test_result(self):
    # Not Final
    self.assertEqual(
      (False, 0.),
      TicTacToeEnvironment.result(
        np.array([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, -1]
        ]),
        -1
      )
    )

    # Loss
    self.assertEqual(
      (True, -1.),
      TicTacToeEnvironment.result(
        np.array([
          [1, 1, 1],
          [0, 0, 0],
          [0, 0, 0]
        ]),
        -1
      )
    )

    # Win
    self.assertEqual(
      (True, 1.),
      TicTacToeEnvironment.result(
        np.array([
          [0, 0, -1],
          [0, 0, -1],
          [0, 0, -1]
        ]),
        -1
      )
    )

    # Draw
    self.assertEqual(
      (True, 0.),
      TicTacToeEnvironment.result(
        np.array([
          [1, -1, 1],
          [-1, -1, 1],
          [1, 1, -1]
        ]),
        -1
      )
    )

  def test_legal_actions(self):
    self.assertEqual(
      [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
      TicTacToeEnvironment.legal_actions(np.array([[0, 0, 0], [-1, 0, 0], [1, -1, 0]]))
    )
  
  def test_step_win(self):
    self.env.set_state(
      trajectories.transition(
        np.array([
          [
            [0, 1, -1],
            [0, 1, -1],
            [0, 0, -1]
          ],
          [
            [0, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
          ],
          [
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1]
          ]
        ]),
        TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL,
        np.asarray(1., dtype=np.float32)
      )
    )

    current_time_step = self.env.current_time_step()
    self.assertEqual(trajectories.StepType.MID, current_time_step.step_type)

    # Winning

    step = self.env.step(np.array([1, 0]))

    np.testing.assert_array_equal(
      np.array([
        [
          [0, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ],
        [
          [1, 0, 1],
          [1, 0, 1],
          [1, 0, 1]
        ],
        [
          [0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]
        ]
      ]),
      step.observation
    )

    self.assertEqual(trajectories.StepType.LAST, step.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_WIN, step.reward)

    # Resetting

    step = self.env.step(np.array([2, 0]))
    
    self.assertEqual(trajectories.StepType.FIRST, step.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL, step.reward)
  
  def test_step_illegal_move(self):
    self.env.set_state(
      trajectories.transition(
        np.array([
          [
            [0, 1, -1],
            [0, 1, -1],
            [0, 0, -1]
          ],
          [
            [0, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
          ],
          [
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1]
          ]
        ]),
        TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL,
        np.asarray(1., dtype=np.float32)
      )
    )

    current_time_step = self.env.current_time_step()
    self.assertEqual(trajectories.StepType.MID, current_time_step.step_type)

    # Taking an illegal move.
    step = self.env.step(np.array([1, 1]))

    np.testing.assert_array_equal(
      np.array([
        [
          [0, 1, 1],
          [0, 1, 1],
          [0, 0, 1]
        ],
        [
          [0, 0, 1],
          [1, 0, 1],
          [1, 0, 1]
        ],
        [
          [0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]
        ]
      ]),
      step.observation
    )

    self.assertEqual(trajectories.StepType.LAST, step.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_ILLEGAL_MOVE, step.reward)

    # Resetting
    step = self.env.step(np.array([2, 0]))
    
    self.assertEqual(trajectories.StepType.FIRST, step.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL, step.reward)
  
  def test_step_draw(self):
    self.env.set_state(
      trajectories.transition(
        np.array([
          [
            [1, 0, -1],
            [0, 1, -1],
            [1, 0, -1]
          ],
          [
            [0, 1, -1],
            [0, 1, -1],
            [1, 0, -1]
          ],
          [
            [1, 0, -1],
            [0, 0, -1],
            [0, 1, -1]
          ]
        ]),
        TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL,
        np.asarray(1., dtype=np.float32)
      )
    )

    current_time_step = self.env.current_time_step()
    self.assertEqual(trajectories.StepType.MID, current_time_step.step_type)

    # Drawing
    step = self.env.step(np.array([2, 1]))

    np.testing.assert_array_equal(
      np.array([
        [
          [1, 0, 1],
          [0, 1, 1],
          [1, 0, 1]
        ],
        [
          [0, 1, 1],
          [0, 1, 1],
          [1, 0, 1]
        ],
        [
          [1, 0, 1],
          [1, 0, 1],
          [0, 1, 1]
        ]
      ]),
      step.observation
    )

    self.assertEqual(trajectories.StepType.LAST, step.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL, step.reward)

    # Resetting
    step = self.env.step(np.array([2, 0]))
    
    self.assertEqual(trajectories.StepType.FIRST, step.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL, step.reward)
  

if __name__ == '__main__':
  utils.test_utils.main()

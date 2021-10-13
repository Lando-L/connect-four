class ConnectFourEnvironment(environments.py_environment.PyEnvironment):
    # Board
    ROWS = 6
    COLUMNS = 7
    
    # Player
    PLAYER_ONE = 1
    PLAYER_TWO = 2
    
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
    def legal_actions(depths: np.ndarray) -> np.ndarray:
        return np.where(depths < ConnectFourEnvironment.HEIGHT)[0]
    
    @staticmethod
    def result(board: np.ndarray, player: int) -> Tuple[bool, np.ndarray]:
        def horizontal(board: np.ndarray) -> List[int]:
            return board.tolist()
            
        def vertical(board: np.ndarray) -> List[int]:
            return board.T.tolist()
        
        def diagonal_top_bottom(board: np.ndarray) -> Iterator[int]:
            for di in ([(j, i - j) for j in range(ConnectFourEnvironment.ROWS)] for i in range(12)):
                yield [states[i, j] for i, j in di if 0 <= i < ConnectFourEnvironment.ROWS and 0 <= j < ConnectFourEnvironment.COLUMNS]

        def diagonal_bottom_top(board: np.ndarray) -> Iterator[int]:
            for di in ([(j, i - ConnectFourEnvironment.ROWS + j + 1) for j in range(ConnectFourEnvironment.ROWS)] for i in range(12)):
                yield [states[i, j] for i, j in di if 0 <= i < ConnectFourEnvironment.ROWS and 0 <= j < ConnectFourEnvironment.COLUMNS]
                
        def is_winner(board: np.ndarray) -> Optional[int]:
            for line in itertools.chain(horizontal(board), vertical(board), diagonal_top_bottom(board), diagonal_bottom_top(board)):
                for colour, group in itertools.groupby(line):
                    if len(list(group)) >= 4:
                        return colour
        
        def is_full(board: np.ndarray) -> bool:
            return bool(len(board.nonzero()[0]))
        
        winner = is_winner(board)
        
        if winner:
            if winner == player:
                return True, ConnectFourEnvironment.REWARD_WIN
            else:
                return True, ConnectFourEnvironment.REWARD_LOSS
        
        elif is_full(board):
            return True, ConnectFourEnvironment.REWARD_DRAW_OR_NOT_FINAL
        
        else:
            return False, ConnectFourEnvironment.REWARD_DRAW_OR_NOT_FINAL
        
    
    def __init__(self, discount: float = 1.) -> None:
        super(ConnectFourEnvironment, self).__init__()
        self._states = None
        self._depths = None
        self._discount = discount
    
    def action_spec(self) -> specs.array_spec.BoundedArraySpec:
        # Actions resemble the seven slots of the board 
        return specs.array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=6, name='action')

    def observation_spec(self) -> specs.array_spec.BoundedArraySpec:
        # 17 planes constitute of 2 Features (Stones of player 1 and player 2) for 8 previous turns, plus one plane indicating the current players colour
        return specs.array_spec.BoundedArraySpec(shape=(ConnectFourEnvironment.ROWS, ConnectFourEnvironment.COLUMNS, 17), dtype=np.int32, minimum=0, maximum=1, name='observation')
    
    def _reset(self) -> trajectories.TimeStep:
        self._states = np.zeros((ConnectFourEnvironment.ROWS, ConnectFourEnvironment.COLUMNS, 17), dtype=np.int32)
        self._states[:, :, 0].fill(ConnectFourEnvironment.PLAYER_ONE)
        self._depths = np.zeros(ConnectFourEnvironment.COLUMNS, dtype=np.int32)
        return trajectories.restart(self._states)
    
    def _step(self, action: int) -> trajectories.TimeStep:
        if self._depths[action] >= ConnectFourEnvironment.ROWS:
            trajectories.termination(self._states, ConnectFourEnvironment.REWARD_ILLEGAL_MOVE)
            
        board = np.copy(self._states[:, :, 1:3])
        player = self._states[0, 0, 0]
        
        is_final, reward = ConnectFourEnvironment.result(board[:, :, 0] * ConnectFourEnvironment.PLAYER_ONE + board[:, :, 1] * ConnectFourEnvironment.PLAYER_TWO, player)
        
        board[self._depths[action], action, player - 1] = 1
        player = ConnectFourEnvironment.PLAYER_TWO if player == ConnectFourEnvironment.PLAYER_ONE else ConnectFourEnvironment.PLAYER_ONE
        
        self._depths[action] += 1
        self._states[:, :, 0].fill(player)
        self._states[:, :, 1:] = np.concatenate([board, self._states[:, :, 1:-2]], axis=-1)
        
        if is_final:
            return trajectories.termination(self._states, reward)
        
        else:
            return trajectories.transition(self._states, reward, self._discount)
    
    def get_state(self) -> trajectories.TimeStep:
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: trajectories.TimeStep) -> None:
        self._current_time_step = time_step
        self._states = time_step.observation
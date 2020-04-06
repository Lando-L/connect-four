from collections import namedtuple

Player = int
PlayerOne = 1
PlayerTwo = -1

GameState = namedtuple('GameState', 'board depths player move')
GameStep = namedtuple('GameStep', 'observation reward done info')

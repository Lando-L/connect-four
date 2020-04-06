from collections import namedtuple


MCTSNodeStats = namedtuple('MCTSNodeStats', 'probability visits rewards')


def empty(probability: float) -> MCTSNodeStats:
    """
    Returns an initial MCTSNodeStats tuple

    :param probability: Node probability
    :return: MCTSNodeStats
    """

    return MCTSNodeStats(
        probability=probability,
        visits=0,
        rewards=0.
    )


def update(stats: MCTSNodeStats, reward: float) -> MCTSNodeStats:
    """
    Returns the updated MCTSNodeStats given the reward

    :param stats: Current MCTSNodeStats
    :param reward: Received reward
    :return: Updated MCTSNodeStats
    """

    return MCTSNodeStats(
        probability=stats.probability,
        visits=stats.visits + 1,
        rewards=stats.rewards + reward
    )

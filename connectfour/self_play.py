import argparse

from connectfour.agent import environment


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('output')

    parser.add_argument('-m', '--memory-size', type=int, default=10000)
    parser.add_argument('-l', '--learning-rate', type=float, default=.01)
    parser.add_argument('-d', '--dropout', type=float, default=.25)

    parser.add_argument('-i', '--iterations', type=int, default=100)
    parser.add_argument('-p', '--checkpoint', type=int, default=25)
    parser.add_argument('-s', '--simulations', type=int, default=100)
    parser.add_argument('-c', '--exploration-constant', type=float, default=.8)
    parser.add_argument('-t', '--temperature', type=float, default=0.2)

    parser.add_argument('-r', '--replay', type=int, default=500)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    return parser


def main() -> None:
    args = arg_parser().parse_args()

    env = environment.AgentEnvironment((6, 7, 11), 7, 3)
    env.memory.build(args.memory_size)
    env.model.build(args.learning_rate, args.dropout)

    env.iterate(args.iterations,
                args.checkpoint,
                args.output,
                args.simulations,
                args.exporlation_constant,
                args.temperature,
                args.replay,
                args.batch_size,
                args.epochs)


if __name__ == '__main__':
    main()

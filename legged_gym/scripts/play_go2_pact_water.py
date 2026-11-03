"""Compatibility entrypoint for the shared Go1/Go2 liquid player."""
from legged_gym.scripts.play_pact_water import configure, load_policy, parser, main


if __name__ == '__main__':
    main(parser().parse_args())

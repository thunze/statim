import logging
import sys


def main() -> None:
    """Main function."""
    # prepare logging
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        style='{',
        format='{asctime:<19}  {levelname:<8}  {name:<25}  {message}',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


if __name__ == '__main__':
    main()

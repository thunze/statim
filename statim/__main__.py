import logging
import sys


def main() -> None:
    """Main function."""
    # prepare logging
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        style='{',
        format='{asctime:<19}  {threadName:<24}  {levelname:<8}  {name:<20}  {message}',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger('requests').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)


if __name__ == '__main__':
    main()

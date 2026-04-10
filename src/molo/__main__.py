import sys

from molo.drivers.yolo import App


def main() -> None:
    argv = sys.argv
    app = App()

    try:
        app(argv)
    except Exception as e:
        print(f"{str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

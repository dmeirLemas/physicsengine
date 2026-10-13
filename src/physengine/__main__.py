from physengine.core.app import Window
from physengine.modules.registry import REGISTRY


def main():
    window = Window(640, 480, REGISTRY)
    window.run()


if __name__ == "__main__":
    main()

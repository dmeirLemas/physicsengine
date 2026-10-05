from physengine.core.app import Window
from physengine.modules.triangle import Triangle


def main():
    triangle = Triangle()
    window = Window(640, 480, triangle)
    window.run()


if __name__ == "__main__":
    main()

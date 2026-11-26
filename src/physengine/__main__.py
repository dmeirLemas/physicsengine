from physengine.core.app import Window
from physengine.core.tour import Tour
from physengine.modules.registry import REGISTRY
from physengine.modules.tour_stops import COLLAPSE, STOPS


def main():
    tour = Tour(STOPS, collapse=COLLAPSE)
    window = Window(1280, 800, REGISTRY, title="Physics engine", director=tour)
    window.run()


if __name__ == "__main__":
    main()

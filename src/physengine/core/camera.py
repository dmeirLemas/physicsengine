import numpy as np

from physengine.core.transforms import look_at, pers_proj


class OrbitCamera:
    def __init__(self, W, H) -> None:
        self.target = np.array([0.0, 0.0, 0.0])

        self.fov = 1.0  # in radians

        self.near = 1.0
        self.far = 1000.0

        self.aspect_ratio = W / H

        self.distance = 5.0
        self.yaw = 0.0
        self.pitch = 0.0

        self.drag_sensitivity = 0.005
        self.zoom_sensitivity = 1.1

    def on_resize(self, W, H):
        self.aspect_ratio = W / H if H != 0 else 0.0

    def view(self):
        eye = (
            self.distance
            * np.array(
                [
                    np.cos(self.pitch) * np.sin(self.yaw),
                    np.sin(self.pitch),
                    np.cos(self.pitch) * np.cos(self.yaw),
                ]
            )
            + self.target
        )

        return look_at(eye, self.target, (0, 1, 0))

    def projection(self):
        return pers_proj(self.fov, self.aspect_ratio, self.near, self.far)

    def view_proj(self):
        return self.projection() @ self.view()

    def orbit(self, dx, dy):
        self.yaw -= dx * self.drag_sensitivity
        self.pitch -= dy * self.drag_sensitivity

    def zoom(self, amount):
        self.distance *= self.zoom_sensitivity**-amount
        self.distance = max(self.distance, 1.0)

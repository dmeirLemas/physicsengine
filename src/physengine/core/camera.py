import numpy as np

from physengine.core.transforms import look_at, pers_proj


class OrbitCamera:
    def __init__(self, W, H) -> None:
        self.target = np.zeros(3)

        self.fov = 1.0  # in radians

        # Clip planes follow the zoom (see the near/far properties), so the
        # camera can get close to small things without them being cut off.
        self.near_factor = 0.01
        self.far_factor = 1000.0
        self.min_distance = 1e-4

        self.aspect_ratio = W / H
        self.height = H

        # Stop just short of looking straight down or up. At exactly +-90
        # degrees the view direction is parallel to the up vector and look_at
        # is undefined; past it the image flips.
        self.max_pitch = np.pi / 2 - 0.01

        self.drag_sensitivity = 0.005
        self.zoom_sensitivity = 1.1

        self.reset()

    def reset(self):
        """Back to the starting view. Modifies target in place."""
        self.target[:] = 0.0
        self.distance = 5.0
        self.yaw = 0.0
        self.pitch = 0.0

    @property
    def near(self):
        return self.near_factor * self.distance

    @property
    def far(self):
        return self.far_factor * self.distance

    def on_resize(self, W, H):
        self.aspect_ratio = W / H if H != 0 else 0.0
        self.height = H

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

    def screen_ray(self, x, y, W, H):
        """Eye position and unit direction of the view ray through the screen
        point (x, y), given in pixels from the top-left of a W x H window."""
        f = 1 / np.tan(self.fov / 2)
        ndc_x = 2.0 * x / W - 1.0
        ndc_y = 1.0 - 2.0 * y / H

        view = self.view()
        rot = view[:3, :3]
        eye = -rot.T @ view[:3, 3]

        direction = rot.T @ np.array([ndc_x * (W / H) / f, ndc_y / f, -1.0])
        return eye, direction / np.linalg.norm(direction)

    def screen_to_plane(self, x, y, W, H, tilt_limit=0.17):
        """World point under the screen point (x, y), on a plane through the target.

        The plane is the horizontal-in-world z = target.z plane, unless the
        camera looks along it at a grazing angle (|forward . z| < tilt_limit,
        about 10 degrees), where the intersection becomes wild; then the plane
        facing the camera is used. Returns None if the ray misses the plane.
        """
        eye, direction = self.screen_ray(x, y, W, H)

        forward = self.target - eye
        forward /= np.linalg.norm(forward)

        normal = np.array([0.0, 0.0, 1.0])
        if abs(forward @ normal) < tilt_limit:
            normal = forward

        denom = direction @ normal
        if abs(denom) < 1e-9:
            return None
        s = ((self.target - eye) @ normal) / denom
        if s <= 0.0:
            return None
        return eye + s * direction

    def world_to_screen(self, point, W, H):
        """Pixel position of a world point, or None if it is behind the camera."""
        clip = self.view_proj() @ np.array([point[0], point[1], point[2], 1.0])
        if clip[3] <= 0.0:
            return None
        return (
            (clip[0] / clip[3] + 1.0) * 0.5 * W,
            (1.0 - clip[1] / clip[3]) * 0.5 * H,
        )

    def orbit(self, dx, dy):
        self.yaw -= dx * self.drag_sensitivity
        self.pitch -= dy * self.drag_sensitivity
        self.pitch = float(np.clip(self.pitch, -self.max_pitch, self.max_pitch))

    def pan(self, dx, dy):
        """Slide the target in the screen plane by a mouse drag of (dx, dy) pixels.

        Scaled so that a point at the target's depth follows the cursor: the
        visible height at distance d is 2 d tan(fov / 2), spread over `height`
        pixels.
        """
        if self.height == 0:
            return

        world_per_pixel = 2.0 * self.distance * np.tan(self.fov / 2) / self.height

        view = self.view()
        right, up = view[0, :3], view[1, :3]

        # Screen y grows downward. Dragging right moves the scene right, so
        # the target moves left.
        self.target += world_per_pixel * (-dx * right + dy * up)

    def zoom(self, amount):
        self.distance *= self.zoom_sensitivity**-amount
        self.distance = max(self.distance, self.min_distance)

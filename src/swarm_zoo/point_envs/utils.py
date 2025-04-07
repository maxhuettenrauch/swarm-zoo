import numpy as np


def get_distances(x0, x1, torus=False, world_size=None):
    if torus:
        assert world_size is not None
    delta = np.abs(x0 - x1)
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    return dist


def get_angle(x0, x1, torus=False, world_size=None, positive=False):
    delta = x0 - x1
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
        delta = np.where(delta < -world_size / 2, delta + world_size, delta)
    angle = np.arctan2(delta[:, 1], delta[:, 0])

    if positive:
        angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    return angle


def get_distance_matrix(points: np.ndarray,
                        world_size: np.ndarray | None = None,
                        torus: bool = False,
                        add_to_diagonal=0):
    distance_matrix = np.vstack([get_distances(points, p, torus=torus, world_size=world_size) for p in points])
    distance_matrix = distance_matrix + np.diag(add_to_diagonal * np.ones(points.shape[0]))
    return distance_matrix

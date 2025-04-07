import numpy as np

from swarm_zoo.point_envs.rendezvous import get_angle


def test_get_angle():
    agent = np.array([1, 0])
    other = np.array([0, 0])
    assert get_angle(other[None, :], agent) == np.pi

    agent = np.array([2, 0])
    other = np.array([1, 0])
    assert get_angle(other[None, :], agent) == np.pi

    agent = np.array([1, 0])
    other = np.array([1, 1])
    assert get_angle(other[None, :], agent) == np.pi / 2

    agent = np.array([1, 0])
    other = np.array([1, -1])
    assert get_angle(other[None, :], agent) == -np.pi / 2

    agent = np.array([1, 0])
    other = np.array([2, 0])
    assert get_angle(other[None, :], agent) == 0



if __name__ == '__main__':
    test_get_angle()
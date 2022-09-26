from numpy import arctan2, arcsin, power, cos, sin, pi, tensordot
import numpy as np


def qua2euler(w, x, y, z):
    roll = arctan2(2 * (w * x + y * z), 1 - 2 * (power(x, 2) + power(y, 2)))
    pitch = arcsin(2 * (w * y - x * z))
    yaw = arctan2(2 * (w * z + x * y), 1 - 2 * (power(y, 2) + power(z, 2)))
    return roll, pitch, yaw


def euler2qua(yaw, pitch, roll):
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return w, x, y, z


def qua2matrix(w, x, y, z):
    rot = np.zeros([w.shape[0], 3, 3])
    rot[:, 0, 0] = 1 - 2 * (power(y, 2) + power(z, 2))
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (w * y + x * z)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (power(x, 2) + power(z, 2))
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (w * x + y * z)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def main():
    print(qua2euler(1, 0, 0, 0))
    yaw = np.linspace(0, pi / 3, 100)
    pitch = np.zeros(100)
    roll = np.zeros(100)
    w, x, y, z = euler2qua(yaw, pitch, roll)
    rot = qua2matrix(w, x, y, z)
    v = np.zeros([100, 1, 3])
    v[:, 0, 0] = 1
    for i in range(100):
        v[i, 0, :] = np.dot(rot[i, :, :].T, v[i, 0, :])
        pass

    import matplotlib.pyplot as plt
    # plt.quiver(np.zeros(100), np.zeros(100), v[:, 0, 0], v[:, 0, 1])
    plt.plot(v[:, 0, 0], v[:, 0, 1])
    plt.axis('equal')
    plt.show()
    pass


if __name__ == "__main__":
    main()

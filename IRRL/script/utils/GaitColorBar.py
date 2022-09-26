"""
this script is used to plot robot state color bar
using color and pose to visualize the period information in detail
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib


def cubicBezier(p0, pf, phase):
    pdiff = pf - p0
    bezier = phase * phase * phase + 3.0 * (phase * phase * (1.0 - phase))
    return p0 + bezier * pdiff


def Bezier2(p0, pf, phase, left):
    pdiff = pf - p0
    bezier = phase * phase * phase + 3.0 * (phase * phase * (1.0 - phase))
    p = p0 + bezier * pdiff
    p[1] = p0[1] + left * np.exp(-(phase - 0.5) ** 2 / (2 * (1 / 6) ** 2))
    return p


def GaitBar(ax, phase=np.array([0.5, 0.5, 0, 0]), N=8, colormapname='coolwarm', body_len=0.4, up_len=0.2, low_len=0.2,
            lift=0.1, lam=0.5):
    '''
    ax is the figure gca
    phase is the gait phase of four leg, FR,FL,HR,HL
    N is used to determine how many anime will be plotted
    colormap is used to determine the color style of the anime
    body_len,up_len and low_len is the geometry size of robot, by now, this should be default
    '''
    left_body_point = np.zeros([N, 6, 2])
    right_body_point = np.zeros([N, 6, 2])

    for i in range(N):
        current_time = i / (N - 1)
        real_phase = (phase + current_time) % 1
        if real_phase[0] < lam:
            p0 = np.array([0.1, -0.25])
            pf = np.array([-0.1, -0.25])
            toe = cubicBezier(p0, pf, real_phase[0] / lam)
            pass
        else:
            pf = np.array([0.1, -0.25])
            p0 = np.array([-0.1, -0.25])
            toe = Bezier2(p0, pf, (real_phase[0] - lam) / (1-lam), lift)
            pass
        right_body_point[i, 0, :] = toe + np.array([body_len / 2, 0])
        right_body_point[i, 1, :] = np.array([toe[1], -toe[0]]) * np.sqrt(
            low_len ** 2 / (toe[0] ** 2 + toe[1] ** 2) - 0.25) + toe / 2 + np.array([body_len / 2, 0])
        right_body_point[i, 2, :] = + np.array([body_len / 2, 0])

        if real_phase[2] < lam:
            p0 = np.array([0.1, -0.25])
            pf = np.array([-0.1, -0.25])
            toe = cubicBezier(p0, pf, real_phase[2] / lam)
            pass
        else:
            pf = np.array([0.1, -0.25])
            p0 = np.array([-0.1, -0.25])
            toe = Bezier2(p0, pf, (real_phase[2] - lam) / (1-lam), lift)
            pass
        right_body_point[i, 5, :] = toe + np.array([-body_len / 2, 0])
        right_body_point[i, 4, :] = np.array([toe[1], -toe[0]]) * np.sqrt(
            low_len ** 2 / (toe[0] ** 2 + toe[1] ** 2) - 0.25) + toe / 2 + np.array([-body_len / 2, 0])
        right_body_point[i, 3, :] = + np.array([-body_len / 2, 0])

        if real_phase[1] < lam:
            p0 = np.array([0.1, -0.25])
            pf = np.array([-0.1, -0.25])
            toe = cubicBezier(p0, pf, real_phase[1] / lam)
            pass
        else:
            pf = np.array([0.1, -0.25])
            p0 = np.array([-0.1, -0.25])
            toe = Bezier2(p0, pf, (real_phase[1] - lam) / (1-lam), lift)
            pass
        left_body_point[i, 0, :] = toe + np.array([body_len / 2, 0])
        left_body_point[i, 1, :] = np.array([toe[1], -toe[0]]) * np.sqrt(
            low_len ** 2 / (toe[0] ** 2 + toe[1] ** 2) - 0.25) + toe / 2 + np.array([body_len / 2, 0])
        left_body_point[i, 2, :] = + np.array([body_len / 2, 0])

        if real_phase[3] < lam:
            p0 = np.array([0.1, -0.25])
            pf = np.array([-0.1, -0.25])
            toe = cubicBezier(p0, pf, real_phase[3] / lam)
            pass
        else:
            pf = np.array([0.1, -0.25])
            p0 = np.array([-0.1, -0.25])
            toe = Bezier2(p0, pf, (real_phase[3] - lam) / (1-lam), lift)
            pass
        left_body_point[i, 5, :] = toe + np.array([-body_len / 2, 0])
        left_body_point[i, 4, :] = np.array([toe[1], -toe[0]]) * np.sqrt(
            low_len ** 2 / (toe[0] ** 2 + toe[1] ** 2) - 0.25) + toe / 2 + np.array([-body_len / 2, 0])
        left_body_point[i, 3, :] = + np.array([-body_len / 2, 0])
        y_offset = -i * 0.435 * np.ones([6])
        left_body_point[i, :, 1] = left_body_point[i, :, 1] + y_offset
        right_body_point[i, :, 1] = right_body_point[i, :, 1] + y_offset
        pass

    for i in range(N):
        cmap = matplotlib.cm.get_cmap(colormapname)
        ax.plot(right_body_point[i, :, 0], right_body_point[i, :, 1], color=cmap(i / (N - 1)),
                lw=3)
        ax.plot(left_body_point[i, :, 0], left_body_point[i, :, 1], linestyle='--', color=cmap(i / (N - 1)),
                lw=3)
        ax.scatter(right_body_point[i, :, 0], right_body_point[i, :, 1], color=cmap(i / (N - 1)),
                   s=20)
        ax.scatter(left_body_point[i, :, 0], left_body_point[i, :, 1], color=cmap(i / (N - 1)),
                   s=20)
        ax.axis('equal')
        ax.set_axis_off()

        pass

    pass


def main():
    fig = plt.figure()
    ax = fig.subplots()
    GaitBar(ax, N=10, phase=np.array([0, 0.5, 0.5, 0.0]))
    plt.show()
    pass


if __name__ == "__main__":
    main()

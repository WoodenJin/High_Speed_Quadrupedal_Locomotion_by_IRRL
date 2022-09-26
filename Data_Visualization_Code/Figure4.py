"""
This script is used to plot the figure 4
"""
import numpy as np
from numpy import arctan2, arcsin, power, cos, sin, pi, tensordot
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import os

class RobotBodyInfo(object):
    def __init__(self, bin_file, param_file, flag_normalized=False):
        # load parameter
        with open(param_file) as f:
            self.cfg = yaml.load(f, Loader=SafeLoader)
        seg_len = self.cfg["seg_len"]
        self.NoE = self.cfg["NoE"]
        self.FoE = self.cfg["FoE"]
        self.NoEnv = self.cfg["Num_Of_Env"]
        self.noise = np.array([self.cfg["z_noise"], self.cfg["roll_noise"], self.cfg["pitch_noise"],
                               self.cfg["z_dot_noise"], self.cfg["roll_dot_noise"], self.cfg["pitch_dot_noise"]])
        self.skip = self.cfg["skip_frame"]
        # load and reshape data
        original_data = np.fromfile(bin_file, dtype=np.float32)
        total_len = self.NoE * int(self.FoE/self.skip) * self.NoEnv
        temp_head_idx = np.arange(0, total_len, seg_len)
        temp_tail_idx = temp_head_idx + seg_len
        temp_tail_idx[-1] = total_len

        data = np.empty([13, total_len])

        for i in range(len(temp_tail_idx)):
            data[:, temp_head_idx[i]:temp_tail_idx[i]] = \
                original_data[temp_head_idx[i] *
                              13:temp_tail_idx[i]*13].reshape([13, -1])
            pass
        del original_data
        data = data.transpose()

        self.flag_normalized = flag_normalized

        # Get Information in body frame
        temp_rot = RobotBodyInfo.qua2matrix(
            data[:, 3], data[:, 4], data[:, 4], data[:, 6])
        self.vel_body = np.matmul(np.transpose(
            temp_rot, (0, 2, 1)), data[:, 7:10].reshape([-1, 3, 1])).reshape([-1, 3])
        self.omega_body = np.matmul(np.transpose(
            temp_rot, (0, 2, 1)), data[:, 10:13].reshape([-1, 3, 1])).reshape([-1, 3])
        # self.omega_body = data[:,10:13]
        self.posture = RobotBodyInfo.qua2euler(
            data[:, 3], data[:, 4], data[:, 5], data[:, 6])
        self.z_axis = temp_rot[:, 2, :]
        # print(self.z_axis.shape)
        del temp_rot

        self.position = data[:, 0:3]
        self.quat = data[:, 3:7]

        # self.omega_o = data[10:13,:]
        del data
        # convention
        # the forth index is feature, [x,y,z,q0,q1,q2,q3,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot]
        # the first index is the index of exploration
        # the second index is the frame index of each exploration
        # the third index is the index of environment
        # self.data = self.data.reshape([13, self.NoE, self.FoE, self.NoEnv])

        pass

    @property
    def vel_formatted(self):
        return self.vel_body.reshape([self.NoEnv, int(self.FoE/self.skip), self.NoE, 3])

    @property
    def omega_formatted(self):
        return self.omega_body.reshape([self.NoEnv, int(self.FoE/self.skip), self.NoE, 3])

    @property
    def posture_formatted(self):
        return self.posture.reshape([self.NoEnv, int(self.FoE/self.skip), self.NoE, 3])

    @property
    def position_formatted(self):
        return self.position.reshape([self.NoEnv, int(self.FoE/self.skip), self.NoE, 3])

    @property
    def quat_formatted(self):
        return self.quat.reshape([self.NoEnv, int(self.FoE/self.skip), self.NoE, 4])

    @property
    def z_axis_formatted(self):
        return self.z_axis.reshape([self.NoEnv, int(self.FoE/self.skip), self.NoE, 3])

    @property
    def x(self):
        temp = np.concatenate((self.position_formatted[:, :, :, 2:3], self.posture_formatted[:, :, :, 0:2],
                               self.vel_formatted[:, :, :, 2:3], self.omega_formatted[:, :, :, 0:2]), axis=3)
        # temp = np.concatenate((self.position_formatted[:, :, :, 2:3], self.z_axis_formatted[:, :, :, 0:2],
        #                        self.vel_formatted[:, :, :, 2:3], self.omega_formatted[:, :, :, 0:2]), axis=3)
        # print(np.mean(temp, axis=(0, 1, 2)))
        # temp = temp - np.mean(temp, axis=(0, 1, 2))
        # temp[1] = -temp[1]
        # temp[2] = -temp[2]
        # temp = temp / np.array([0.03,0.1,0.1,1.5,3.0,3.0])
        if(self.flag_normalized):
            temp_mean = np.mean(temp, axis=(0, 1, 2))
            temp_std = np.std(temp, axis=(0, 1, 2))
            temp = (temp - temp_mean) / temp_std / 3
            pass

        return temp

    @staticmethod
    def qua2matrix(w, x, y, z):
        rot = np.zeros([w.shape[0], 3, 3])
        rot[:, 0, 0] = 1 - 2 * (power(y, 2) + power(z, 2))
        # rot[:, 0, 0] = power(w, 2) + power(x, 2) - power(y, 2) - power(z, 2)
        rot[:, 0, 1] = 2 * (x * y - w * z)
        rot[:, 0, 2] = 2 * (w * y + x * z)
        rot[:, 1, 0] = 2 * (x * y + w * z)
        rot[:, 1, 1] = 1 - 2 * (power(x, 2) + power(z, 2))
        rot[:, 1, 2] = 2 * (y * z - w * x)
        rot[:, 2, 0] = 2 * (x * z - w * y)
        rot[:, 2, 1] = 2 * (w * x + y * z)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot

    @staticmethod
    def qua2euler(w, x, y, z):
        roll = arctan2(2 * (w * x + y * z), 1 - 2 *
                       (power(x, 2) + power(y, 2)))
        pitch = arcsin(2 * (w * y - x * z))
        yaw = arctan2(2 * (w * z + x * y), 1 - 2 * (power(y, 2) + power(z, 2)))

        return np.array([roll, pitch, yaw]).transpose()

    pass

def gradient_image(ax, x, y, x_key, c_key):

    # cmap = mpl.cm.coolwarm_r
    cmap = mpl.cm.RdYlBu
    dx = x[1]-x[0]
    color = (x<=x_key[0]).astype(np.int32) * x/x_key[0]*c_key[0]
    color += (x>x_key[0]).astype(np.int32)* (x<=x_key[1]).astype(np.int32) * ((c_key[1]-c_key[0])/(x_key[1]-x_key[0])*(x-x_key[0])+c_key[0])
    color += (x>x_key[1]).astype(np.int32) * ((1-c_key[1])/(np.max(x)-x_key[1])*(x-x_key[1])+c_key[1])
    # print(cmap(1))
    for i in range(len(x)-1):
        ax.fill_between([x[i],x[i+1]-dx/100], y[i:i+2],[-20,-20], 
                        color=cmap(color[i]), edgecolor=cmap(color[i]), alpha=1, zorder=-1)
        pass
    ax.fill_between([x[0],x[-1]+dx/100],[10,10],[-10,-10],facecolor='w',zorder=0,alpha=0.7)
    
    pass

def entropy(data, lb, ub, precision):
    # data is narray like, (obs, dim)
    temp = np.clip(data, lb, ub)
    _, frequency = np.unique(
        (temp/precision).astype(np.int32), axis=0, return_counts=True)
    frequency = frequency / (data.shape[0])
    return -np.sum(frequency * np.log(frequency))


def piecewise_func3(x, a, b, c, d):
    temp = (x<=a).astype(np.int32) * b
    temp += (a < x).astype(np.int32)*(x <= c).astype(np.int32) * (d * (x - a) + b)
    temp += (x>c).astype(np.int32) * (d * (c - a) + b)
    return temp

def plot_poincare(fig_handler, g):
    # load test result without disturbance
    date = []

    # date.append("2021-07-22-15-28-36")     #  0 latency
    # date.append("2021-07-22-15-29-52")     #  8 latency                       
    # date.append("2021-07-22-15-30-10")     # 10 latency
    date.append("2021-07-22-16-07-01")     #  6 latency
    date.append("2021-07-22-16-07-19")     #  8 latency
    date.append("2021-07-22-16-07-38")     # 10 latency

    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-" + d + ".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i])
             for i in range(len(date))]
    
    offset = 50
    selected_index = np.arange(robot[0].FoE,step=int(100/robot[0].skip)) + offset
    
    # vx_b = [ro.vel_formatted[0, selected_index, 0, 0] for ro in robot]
    # roll_dot_b = [ro.omega_formatted[0, selected_index, 0, 0] for ro in robot]
    # lb = [[4, 4, 3.0],[-0.1, -5, -10]]
    # ub = [[5, 5, 4.5],[0.3, 5, 10]]
    vx_b = [ro.position_formatted[0, selected_index, 0, 2] for ro in robot]
    roll_dot_b = [ro.posture_formatted[0, selected_index, 0, 0] for ro in robot]
    lb = [[0.27, 0.27, 0.25], [-0.02, -0.05, -0.15]]
    ub = [[0.29, 0.29, 0.32], [0.02, 0.05, 0.15]]


    ax = [[fig_handler.add_subplot(g[j,i]) for i in range(3)] for j in range(2)]

    traj_id_x = [5, 6, 6, 7, 7, 8, 8]
    traj_id_y = [6, 6, 7, 7, 8, 8, 9]

    for i in range(3):
        ax[0][i].set_yticklabels([])
        ax[1][i].set_yticklabels([])

        # ax[0][i].title.set_text('{} ms'.format(2*(i+3)))

        ax[0][i].set_xticks([lb[0][i],ub[0][i]])
        ax[0][i].set_yticks([lb[0][i],ub[0][i]])
        ax[0][i].set_xticklabels([lb[0][i],ub[0][i]],fontsize=5)
        ax[1][i].set_xticks([lb[1][i],ub[1][i]])
        ax[1][i].set_yticks([lb[1][i],ub[1][i]])
        ax[1][i].set_xticklabels([lb[1][i],ub[1][i]],fontsize=5)

        # ax[0][i].set_xlabel(r"$v_{n}$",fontsize=6)
        # ax[0][i].set_ylabel(r"$v_{n+1}$",fontsize=6)
        ax[0][i].set_xlabel(r"$z_{n}\ (\mathrm{m})$",fontsize=5)
        ax[0][i].set_ylabel(r"$z_{n+1}$",fontsize=5)
        ax[0][i].xaxis.set_label_coords(0.5,-0.25)
        ax[0][i].yaxis.set_label_coords(-0.1,0.5)

        # ax[1][i].set_xlabel(r"$\omega_{n}$",fontsize=6)
        # ax[1][i].set_ylabel(r"$\omega_{n+1}$",fontsize=6)
        ax[1][i].set_xlabel(r"$\theta_{\mathrm{r}n}\ (\mathrm{rad})$",fontsize=5)
        ax[1][i].set_ylabel(r"$\theta_{\mathrm{r}n+1}$",fontsize=5)
        ax[1][i].xaxis.set_label_coords(0.5,-0.25)
        ax[1][i].yaxis.set_label_coords(-0.1,0.5)

        ax[0][i].plot([-10,10], [-10,10], color='C0', lw=0.5)
        ax[1][i].plot([-10,10], [-10,10], color='C0', lw=0.5)

        ax[0][i].scatter(vx_b[i][5:99], vx_b[i][6:100], marker='x', s=3, color='C1', alpha=0.5, linewidth=0.5)
        ax[1][i].scatter(roll_dot_b[i][5:99], roll_dot_b[i][6:100], marker='x', s=3, color='C1', alpha=0.5, linewidth=0.5)
        
        ax[0][i].plot(vx_b[i][traj_id_x], vx_b[i][traj_id_y], ls='--', lw=0.5, zorder=0)
        ax[1][i].plot(roll_dot_b[i][traj_id_x], roll_dot_b[i][traj_id_y], ls='--', lw=0.5, zorder=0)

        dx = vx_b[i][traj_id_x[1]]-vx_b[i][traj_id_x[0]]
        dy = vx_b[i][traj_id_y[1]]-vx_b[i][traj_id_y[0]]
        dl = np.sqrt(dx**2 + dy**2)
        ax[0][i].arrow((vx_b[i][traj_id_x[0]]+vx_b[i][traj_id_x[1]])/2,
                       (vx_b[i][traj_id_y[0]]+vx_b[i][traj_id_y[1]])/2,
                       (dx/dl)*0.05*(ub[0][i]-lb[0][i]),
                       (dy/dl)*0.05*(ub[0][i]-lb[0][i]),
                       head_width=0.05*(ub[0][i]-lb[0][i]), shape='full',
                       lw=0, length_includes_head=True, zorder=10)
        
        dx = roll_dot_b[i][traj_id_x[1]]-roll_dot_b[i][traj_id_x[0]]
        dy = roll_dot_b[i][traj_id_y[1]]-roll_dot_b[i][traj_id_y[0]]
        dl = np.sqrt(dx**2 + dy**2)
        ax[1][i].arrow((roll_dot_b[i][traj_id_x[0]]+roll_dot_b[i][traj_id_x[1]])/2,
                       (roll_dot_b[i][traj_id_y[0]]+roll_dot_b[i][traj_id_y[1]])/2,
                       (dx/dl)*0.05*(ub[1][i]-lb[1][i]),
                       (dy/dl)*0.05*(ub[1][i]-lb[1][i]),
                       head_width=0.05*(ub[1][i]-lb[1][i]), shape='full',
                       lw=0, length_includes_head=True, zorder=10)

        ax[0][i].set_xlim([lb[0][i], ub[0][i]])
        ax[0][i].set_ylim([lb[0][i], ub[0][i]])
        ax[1][i].set_xlim([lb[1][i], ub[1][i]])
        ax[1][i].set_ylim([lb[1][i], ub[1][i]])
        # ax[0][i].plot(vx_b[i][0:30],marker='x',lw=0.5)
        # ax[1][i].plot(roll_dot_b[i][0:30],marker='x',lw=0.5)
        pass

    pass

def plot_latency(fig_handler, g):
    ax = fig_handler.add_subplot(g)
    # ----------------------------------------
    # load data
    date = []
    date.append("2021-06-22-15-07-36")
    date.append("2021-06-22-15-04-15")
    date.append("2021-06-22-15-00-52")
    date.append("2021-06-22-22-46-32")
    date.append("2021-06-22-14-53-46")
    date.append("2021-06-22-14-47-55")
    bin_file = ["Exp_Raw_Data//body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data//Param-" + d + ".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i])
             for i in range(len(date))]
    # [print(ro.cfg["delay"]) for ro in robot]

    sf = [0, 1, 2, 3, 4, 5]  # selected feature
    ub = np.array([0.5, 3.14, 1.57, 10, 10, 10])
    lb = np.array([0.0, -3.14, -1.57, -10, -10, -10])
    # related with the number of sample
    precision = np.array([0.005, 0.02, 0.02, 0.005, 0.025, 0.025])

    x = [ro.x[:, :, :, sf] for ro in robot]
    FoE = int(robot[0].FoE/robot[0].skip)
    t = np.arange(0, robot[0].FoE, robot[0].skip) * 0.002

    label = ["{:>2d}ms".format(ro.cfg["delay"]*2) for ro in robot]
    del robot

    # ---------------------------------------------
    # calculate the entropy
    ent = []
    for xx in x:
        ent.append([])
        for i in range(FoE):
            ent[-1].append(entropy(xx[0, i, :, :],
                           lb[sf], ub[sf], precision[sf]))
            pass
        pass

    popt = []
    pcov = []
    # para_ub = np.array([1, 10, 2, 20])
    # para_lb = np.array([0, 5, 1, 0])
    para_ub = np.array([1, 10, 2, 2])
    para_lb = np.array([0, 5, 1, -20])

    label2 = []
    dE = []
    dE_up = []
    dE_low = []

    for e in ent:
        # temp_p, temp_cov = curve_fit(piecewise_func, t, np.asarray(e), bounds=(para_lb, para_ub))
        temp_p, temp_cov = curve_fit(piecewise_func3, t, np.asarray(e), bounds=(para_lb, para_ub))
        temp_cov = np.sqrt(np.diag(temp_cov))
        popt.append(temp_p)
        pcov.append(temp_cov)
        dE.append(temp_p[3])
        dE_up.append(temp_p[3]+temp_cov[3])
        dE_low.append(temp_p[3]-temp_cov[3])
        pass

    print(dE)
    print(dE_up)

    # load test result without disturbance
    date = []
    date.append("2021-06-22-16-48-33")
    date.append("2021-06-22-16-48-55")
    date.append("2021-06-22-16-49-18")
    date.append("2021-06-22-16-49-38")
    date.append("2021-06-22-16-50-00")
    date.append("2021-06-22-16-50-21")

    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-" + d + ".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i])
             for i in range(len(date))]
    vx_b = [ro.vel_formatted[0, :, 0, 0] for ro in robot]
    roll_dot_b = [ro.omega_formatted[0, :, 0, 0] for ro in robot]
    vx_mean = [np.mean(v) for v in vx_b]
    vx_error = [np.std(v) for v in vx_b]

    print(vx_mean)
    print(dE)

    tt = np.arange(0, robot[0].FoE, 1) * 0.002

    delay = np.linspace(0, 10, 6)
    ax.errorbar(delay, dE, yerr=(np.asarray(dE_up)-np.asarray(dE))*3,
                marker='o', markersize=4, capsize=4, capthick=2, lw=2, color='C0')
    ax.set_ylabel(r"$\kappa\ (\log_\mathrm{e}/\mathrm{s})$", color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.yaxis.set_label_coords(-0.22,0.5)
    ax.xaxis.set_label_coords(0.5,-0.11)
    ax.set_xticks([0,2,4,6,8,10])
    ax.set_xticklabels([0,2,4,6,8,10])

    ax_perf = ax.twinx()
    ax_perf.errorbar(delay, vx_mean, yerr=np.asarray(vx_error)*3,
                     marker='s', markersize=4, capsize=4, capthick=2, lw=2,
                     color='C3', alpha=0.8)
    ax_perf.set_ylabel(r"$v^\mathrm{B}_\mathrm{x}\ (\mathrm{m/s})$", color='C3')
    ax_perf.yaxis.set_label_coords(1.15,0.5)

    ax_perf.tick_params(axis='y', labelcolor='C3')
    ax.set_xlabel("Latency (ms)")

    ax.set_ylim([-10, 2])
    ax.set_xlim([-0.5, 10.5])
    ax_perf.set_ylim([3.5, 5.5])

    gradient_image(ax, np.linspace(-2, 12, 100),
                   np.ones(100)*20,
                   [6, 8], [0.25, 0.55])

    pass

def plot_mu(fig_handler, g, gc):

    from scipy import interpolate
    from matplotlib import colors, cm

    date = []
    date.append("2021-06-21-15-36-04")
    date.append("2021-06-21-23-14-03")
    
    data = [np.load("Exp_Raw_Data\\"+date[i]+".npy") for i in range(2)]

    Num_Mu = int(data[0][-6])
    Num_V = int(data[0][-5])
    Mu_Min = int(data[0][-4])
    Mu_Max = int(data[0][-3])
    V_Min = int(data[0][-2])
    V_Max = int(data[0][-1])

    dE = [data[i][0:Num_Mu*Num_V] for i in range(2)]

    f = []
    for i in range(2):
        temp_f = interpolate.interp2d(np.linspace(V_Min, V_Max, Num_V),
                                      np.linspace(Mu_Min, Mu_Max, Num_Mu),
                                      np.asarray(dE[i]).reshape(
                                          Num_Mu, Num_V).transpose(),
                                      kind='linear')
        f.append(temp_f)
        pass

    temp_mu = np.linspace(Mu_Min, Mu_Max,100)
    f_v5 = [f[i](5,temp_mu) for i in range(2)]
    
    print(temp_mu[np.argmax(f_v5[0])],temp_mu[np.argmin(f_v5[0])])
    print(temp_mu[np.argmax(f_v5[1])],temp_mu[np.argmin(f_v5[1])])

    norm = colors.Normalize(vmin=-12, vmax=0)

    ax = [fig_handler.add_subplot(g[i]) for i in range(2)]
    con = ax[0].contourf(np.linspace(V_Min, V_Max, 100),
                         np.linspace(Mu_Min, Mu_Max, 100),
                         f[0](np.linspace(V_Min, V_Max, 100),
                              np.linspace(Mu_Min, Mu_Max, 100)),
                         100, cmap='magma_r', norm=norm)
    
    for c in con.collections:
        c.set_edgecolor("face")

    con = ax[1].contourf(np.linspace(V_Min, V_Max, 100),
                         np.linspace(Mu_Min, Mu_Max, 100),
                         f[1](np.linspace(V_Min, V_Max, 100),
                              np.linspace(Mu_Min, Mu_Max, 100)),
                         100, cmap='magma_r', norm=norm)

    for c in con.collections:
        c.set_edgecolor("face")
    
    ax[0].text(4.65, temp_mu[np.argmin(f_v5[0])]+0.05, "i",ha="center", va="center", color="w")
    ax[0].text(4.65, temp_mu[np.argmax(f_v5[0])]+0.05, "ii",ha="center", va="center", color="w")
    ax[0].scatter([4.9, 4.9],[0.4, 0.05], s=4, color='w')
    ax[1].text(4.65, temp_mu[np.argmin(f_v5[1])]+0.05, "iii",ha="center", va="center", color="w")
    ax[1].text(4.65, temp_mu[np.argmax(f_v5[1])]+0.05, "iv",ha="center", va="center", color="w")
    ax[1].scatter([4.9, 4.9],[0.6, 0.05], s=4, color='w')

    # ax[0].set_xlabel('Velocity (m/s)')
    ax[0].set_ylabel(r"$\mu$")
    ax[1].set_xlabel('Velocity (m/s)')
    ax[1].set_ylabel(r"$\mu$")

    ax[0].set_xticks([0, 1, 2, 3, 4, 5])
    ax[0].set_xticklabels([0, 1, 2, 3, 4, 5])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[1].set_xticklabels([0, 1, 2, 3, 4, 5])
    ax[1].xaxis.set_label_coords(0.5,-0.15)

    cbaxes = fig_handler.add_subplot(gc)
    sm = cm.ScalarMappable(cmap='magma_r', norm=norm)
    sm.set_array([])
    cb = fig_handler.colorbar(sm, cax=cbaxes, shrink=0.6, orientation="horizontal",
                    ticks=[-12, -8, -4, 0], label=r'$\kappa\ \log_\mathrm{e}/\mathrm{s}$')
    cbaxes.xaxis.set_label_coords(0.5,-0.85)
    pass

def plot_recurrence(fig_handler, g, gc):

    from scipy.spatial.distance import pdist, squareform

    date = []
    date.append("2021-07-23-09-19-38")
    date.append("2021-07-23-09-19-15")
    date.append("2021-07-23-09-23-35")
    date.append("2021-07-23-09-23-05")
    

    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-" + d + ".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i])
             for i in range(len(date))]

    lb = np.array([0.2, -1, -1, -5, -5, -5])
    ub = np.array([0.4, 1, 1, 5, 5, 5])

    st_id = 2000
    ed_id = 2800

    x = [ro.x.reshape([-1, 6])[st_id:ed_id,:] for ro in robot]
    x = [(xx-(lb+ub)/2)/(ub-lb) for xx in x]
    
    def rec_plot(s, eps=0.0010, steps=40):
        d = pdist(s)
        d = np.floor(d/eps)
        d[d>steps] = steps
        Z = squareform(d)
        return Z*eps

    z = [rec_plot(xx) for xx in x]

    ax = [[fig_handler.add_subplot(g[i][j]) for j in range(2)]for i in range(2)]
    ax_vector = [[fig_handler.add_subplot(g[i][j]) for j in range(2)]for i in range(2)]
    
    x,y=np.meshgrid(np.arange(ed_id-st_id)*0.002,np.arange(ed_id-st_id)*0.002)
    colormap = 'GnBu_r'
    pm=ax[0][0].pcolormesh(x,y,z[0],cmap=colormap)
    pm=ax[0][1].pcolormesh(x,y,z[1],cmap=colormap)
    pm=ax[1][0].pcolormesh(x,y,z[2],cmap=colormap)
    pm=ax[1][1].pcolormesh(x,y,z[3],cmap=colormap)

    # [[ax[i][j].set_xticks([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    # [[ax[i][j].set_xticklabels([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    # [[ax[i][j].set_yticks([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    # [[ax[i][j].set_yticklabels([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    [[ax[i][j].set_xticks([]) for j in range(2)] for i in range(2)]
    [[ax[i][j].set_yticks([]) for j in range(2)] for i in range(2)]
    [[ax_vector[i][j].set_xticks([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    [[ax_vector[i][j].set_xticklabels([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    [[ax_vector[i][j].set_yticks([0,0.8,1.6]) for j in range(2)] for i in range(2)]
    [[ax_vector[i][j].set_yticklabels([0,0.8,1.6]) for j in range(2)] for i in range(2)]

    plt.setp(ax[0][0].get_xticklabels(), visible=False)
    plt.setp(ax[0][1].get_xticklabels(), visible=False)
    plt.setp(ax[0][1].get_yticklabels(), visible=False)
    plt.setp(ax[1][1].get_yticklabels(), visible=False)
    plt.setp(ax_vector[0][0].get_xticklabels(), visible=False)
    plt.setp(ax_vector[0][1].get_xticklabels(), visible=False)
    plt.setp(ax_vector[0][1].get_yticklabels(), visible=False)
    plt.setp(ax_vector[1][1].get_yticklabels(), visible=False)

    # ax[1][0].set_xlabel('Time (s)')
    # ax[1][1].set_xlabel('Time (s)')
    # ax[0][0].set_ylabel('Time (s)')
    # ax[1][0].set_ylabel('Time (s)')
    # ax[1][0].xaxis.set_label_coords(0.5,-0.15)
    # ax[1][1].xaxis.set_label_coords(0.5,-0.15)
    # ax[0][0].yaxis.set_label_coords(-0.25,0.5)
    # ax[1][0].yaxis.set_label_coords(-0.25,0.5)
    ax_vector[1][0].set_xlabel('Time (s)')
    ax_vector[1][1].set_xlabel('Time (s)')
    ax_vector[0][0].set_ylabel('Time (s)')
    ax_vector[1][0].set_ylabel('Time (s)')
    ax_vector[1][0].xaxis.set_label_coords(0.5,-0.15)
    ax_vector[1][1].xaxis.set_label_coords(0.5,-0.15)
    ax_vector[0][0].yaxis.set_label_coords(-0.25,0.5)
    ax_vector[1][0].yaxis.set_label_coords(-0.25,0.5)

    cbaxes = fig_handler.add_subplot(gc)
    cb = plt.colorbar(pm, cax = cbaxes, shrink=0.6,label=r'$\|\mathbf{x}_i-\mathbf{x}_j\|$', orientation="horizontal") 
    cbaxes.xaxis.set_label_coords(0.5,-0.85)
    
    ax[0][0].set_rasterized(True)
    ax[0][1].set_rasterized(True)
    ax[1][0].set_rasterized(True)
    ax[1][1].set_rasterized(True)

    pass

def plot_push_recovery_test(fig_handler, g):

    # load data
    date = "2022-04-26-14-07-16"
    data = np.load("Exp_Raw_Data\\"+date+".npy")
    vis_data = []
    for j in [0]:
        for i in [0, 1, 2, 3, 4, 5]:
            # vis_data.append(data[i][j][np.where(data[i][j] > 0.01)[0]]*0.9)
            vis_data.append(data[i][j][:]*0.9)
            pass
        pass
    vis_data1 = []
    for j in [4]:
        for i in [0, 1, 2, 3, 4, 5]:
            # vis_data1.append(data[i][j][np.where(data[i][j] > 0.01)[0]]*0.9)
            vis_data1.append(data[i][j][:]*0.9)
            pass
        pass

    label = [r"$0$", r"$2$", r"$4$", r"$6$", r"$8$", r"$10$"]
    from matplotlib import colors as c
    ax_pr = [fig_handler.add_subplot(g[i]) for i in range(2)]
    bplot = ax_pr[0].boxplot(vis_data, notch=False, vert=True,
                          patch_artist=True, labels=label,
                          showfliers=True,flierprops=dict(markersize=1))
    bplot1 = ax_pr[1].boxplot(vis_data1, notch=False, vert=True,
                              patch_artist=True, labels=label,
                              showfliers=True,flierprops=dict(markersize=1))

    for bp in [bplot, bplot1]:
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        for patch, l,point, color in zip(bp['boxes'], bp['medians'],bp['fliers'], colors):
            temp = c.to_rgba(color)
            temp = (temp[0], temp[1], temp[2], 0.2)
            patch.set_facecolor(temp)
            # patch.set_alpha(0.2)
            patch.set_edgecolor((0, 0, 0, 1))
            point.set_markeredgecolor(color)
            l.set_color(color)
            l.set_linewidth(2)
            pass
        pass

    [a.set_xlabel("Latency (ms)") for a in ax_pr]
    [a.xaxis.set_label_coords(0.5,-0.11) for a in ax_pr]
    [a.set_ylim(0,1.2) for a in ax_pr]
    ax_pr[0].set_ylabel("Impluse"+r"$\ (mv^\mathrm{B}_\mathrm{xT})$")
    plt.setp(ax_pr[1].get_yticklabels(), visible=False)
    pass

def main5():

    plt.style.use("science")
    params = {
    'text.usetex': False,
    'font.size': 7,
    }
    mpl.rcParams.update(params)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams["mathtext.fontset"]='dejavusans'

    fig = plt.figure(figsize=(12.1/2.54, 16/2.54), dpi=600)

    gs = fig.add_gridspec(4,1,height_ratios=[1,1.2,1.8,0.15], hspace=0.28)

    g_push = gs[0].subgridspec(1,2,width_ratios=[1.2, 2])

    g_latency = gs[1].subgridspec(1, 2, width_ratios=[1.5, 2.2], wspace=0.3)

    gmu = gs[2].subgridspec(2, 2, wspace=0.3,hspace=0.15, width_ratios=[1.5, 2.2], height_ratios=[1,1])

    gmumu = gmu[:,1].subgridspec(2,2,wspace=0.15, width_ratios=[1, 1],hspace=0.15)

    gc = gs[3].subgridspec(1, 2, wspace=0.3, width_ratios=[1.5, 2.2])

    plot_push_recovery_test(fig, g_push[1].subgridspec(1,2,wspace=0.15))
    plot_latency(fig, g_latency[0])
    plot_poincare(fig, g_latency[1].subgridspec(2, 3, wspace=0.5, hspace=0.5))
    plot_mu(fig, [gmu[0,0],gmu[1,0]], gc[0])
    plot_recurrence(fig, [[gmumu[0,0],gmumu[0,1]],[gmumu[1,0],gmumu[1,1]]], gc[1])

    # fig.savefig("src\\img2\\Fig4_temp_.jpg",dpi=600)
    # fig.savefig("src\\img2\\Fig4_temp_.pdf",dpi=600)

    fig.savefig(os.getcwd()+'/Fig4.pdf',dpi=600,transparent=True)
    # fig.savefig(os.getcwd()+'/src/Fig4.png',dpi=300,transparent=True)

    pass

if __name__ == "__main__":
    main5()          # version 2
    pass
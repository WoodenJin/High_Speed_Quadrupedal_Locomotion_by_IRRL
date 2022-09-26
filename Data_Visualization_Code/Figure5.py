"""
This script is used to visualize the data for figure 5
the test data on hardware
"""

from math import tau
from pathlib import WindowsPath
from symbol import factor
import numpy as np
from numpy import arctan2, arcsin, power, cos, sin, pi, tensordot
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy
from scipy.signal import savgol_filter
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import colors, cm
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
import types
import pandas as pd
from matplotlib.lines import Line2D
import os


class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'

def custom_new_gc(self):
    return GC()


RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)

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

    def load_torque(self, power_file):
        
        seg_len = self.cfg["seg_len"]
        total_len = self.NoE * int(self.FoE/self.skip) * self.NoEnv
        temp_head_idx = np.arange(0, total_len, seg_len)
        temp_tail_idx = temp_head_idx + seg_len
        temp_tail_idx[-1] = total_len

        original_data = np.fromfile(power_file, dtype=np.float32)

        Num_sub_loop = int(original_data.shape[0]/total_len/2/12)
        self.num_of_period = int(Num_sub_loop * 100 / self.skip)
        Num_col = Num_sub_loop * 24
        data = np.empty([Num_col, total_len])

        for i in range(len(temp_tail_idx)):
            data[:, temp_head_idx[i]:temp_tail_idx[i]] = \
                original_data[temp_head_idx[i] *
                              Num_col:temp_tail_idx[i]*Num_col].reshape([Num_col, -1])
            pass

        del original_data

        data = data.transpose()
        self.torque_power = data[:,0:12*Num_sub_loop].reshape([-1, 12])
        self.vel_power = data[:, 12*Num_sub_loop:].reshape([-1, 12])
        self.torque_power[:, [2, 5, 8, 11]] = self.torque_power[:, [2, 5, 8, 11]] / 1.55
        self.vel_power[:, [2, 5, 8, 11]] = self.vel_power[:, [2, 5, 8, 11]] * 1.55
        pass

    @property
    def torque_power_formatted(self):
        return self.torque_power.reshape([self.NoEnv, -1, self.NoE, 12])

    @property
    def vel_power_formatted(self):
        return self.vel_power.reshape([self.NoEnv, -1, self.NoE, 12])

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

def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    ax.set_xticklabels([]) # labels 
    ax.set_yticklabels([])
    # ax.xaxis.set_ticks_position('none') # tick markers
    # ax.yaxis.set_ticks_position('none')
    ax.set_xticks([-41.7, -14.2, 0, 14.2, 41.7])
    ax.set_xticklabels([-41.7, -14.2, 0, 14.2, 41.7])
    ax.set_yticks([-18,0,18])
    ax.set_yticklabels([-18,0,18])

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='gray', ec='gray', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='gray', ec='gray', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)
    pass

def motor_capicity_plot(ax, cmap='coolwarm', nlevel=20, alpha=1.0):
    motor_property = [np.array([18.1, 18.1, 18.1, 0, -18.1, -18.1, -18.1, 0, 18.1]),
                      np.array([-41.67, -14.2, 14.2, 41.67, 41.67, 14.67, -14.67, -41.67, -41.67])]
    power_speed = np.linspace(-41.67, 41.67, 500)
    power_torque = np.linspace(-18, 18, 500)
    X,Y = np.meshgrid(power_speed, power_torque)
    power = X * Y
    power[np.where(power>284)] = 284

    mask1_vert = [(-41.67, 18), (14.2, 18),
                  (41.67, 0), (41.67, -18),
                  (50, -18), (50, 20),
                  (-41.67, 20), (-41.67, 18)]
    mask2_vert = [(-41.67, 20), (-41.67, 0),
                  (-14.2, -18), (50, -18),
                  (50, -20), (-50, -20),
                  (-50, 20), (-41.67, 20)]
    codes = [Path.MOVETO, Path.LINETO,
             Path.LINETO, Path.LINETO,
             Path.LINETO, Path.LINETO,
             Path.LINETO, Path.CLOSEPOLY]
    mask1 = Path(mask1_vert, codes)
    mask2 = Path(mask2_vert, codes)
    norm = colors.Normalize(vmin=-40*18, vmax=40*18)
    # cs = ax.contourf(X, Y, power, nlevel, alpha=alpha, cmap=cmap, norm=norm, zorder=1)
    # cs = ax.pcolormesh(X,Y,power,alpha=alpha,cmap=cmap,norm=norm,zorder=1)

    con = ax.contourf(X,Y,power,100,alpha=1,cmap=cmap,norm=norm,zorder=-1)
    for c in con.collections:
        c.set_edgecolor("face")
    ax.fill_between([-50,50],[-20,-20],[20,20],facecolor='w',zorder=0,alpha=1-alpha)

    # cs = 0
    ax.add_patch(patches.PathPatch(mask1, facecolor='white', lw=0))
    ax.add_patch(patches.PathPatch(mask2, facecolor='white', lw=0))
    ax.plot(motor_property[1], motor_property[0], color='k', linestyle='--', lw=1)
    ax.set_xlim([-45,45])
    ax.set_ylim([-20,20])
    
    return con

def color_line_segment(x, y, color, alpha=0.5, cmap='magma', lw=1, flag_close=False):
    if flag_close:
        x = np.append(x,x[0])
        y = np.append(y,y[0])
        color = np.append(color,color[0])
        pass
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
    lc.set_array(color)
    lc.set_alpha(alpha)
    lc.set_linewidth(lw)
    return lc

def add_period_span(ax, start, end, step, c=['C0','C2'], alpha=0.2):
    s = start
    idx = 0
    while(s<end):
        ax.axvspan(s,s+step,facecolor=c[idx % len(c)], alpha=alpha, zorder=1)
        s = s + step
        idx += 1
        pass
    pass

def plot_torque_vel2(fig_handler, g, g_state):
    from scipy import signal

    # load simulation data
    date = "2021-07-07-08-25-45"
    power_file = "Exp_Raw_Data\\power-" + date + ".bin"
    param_file = "Exp_Raw_Data\\Param-" + date + ".txt"
    bin_file = "Exp_Raw_Data\\body-center-" + date + ".bin"

    robot = RobotBodyInfo(bin_file,param_file)
    robot.load_torque(power_file)
    torque_sim = robot.torque_power_formatted[0,:,0,:]
    vel_sim = robot.vel_power_formatted[0,:,0,:]

    # load real data
    ratio = [1,1,1.55]
    real_spi_data = pd.read_excel("Exp_Raw_Data\\spi_real.xlsx")
    tau_real = [real_spi_data['tau'+str(i)].to_numpy()/ratio[i%3] for i in range(12)]
    qd_real = [real_spi_data['qd'+str(i)].to_numpy()*ratio[i%3] for i in range(12)]
    q_real = [real_spi_data['q'+str(i)].to_numpy() for i in range(12)]
    del real_spi_data

    # correct the coding errors
    for i in range(12):
        for j in range(len(tau_real[0])):
            tau_candicate = np.array([tau_real[i][j],tau_real[i][j]+36,tau_real[i][j]-36])
            idx = np.argmin(np.abs(tau_candicate-torque_sim[(j*4+400)%800, i]))
            tau_real[i][j] = tau_candicate[idx]
            # if(torque_sim[(j*4+400+1)%800, i]>10 and tau_real[i][j]<-10):
            #     tau_real[i][j] = tau_real[i][j] + 36
            #     pass
            # if(torque_sim[(j*4+400+1)%800, i]<-10 and tau_real[i][j]>10):
            #     tau_real[i][j] = -36 + tau_real[i][j]
            #     pass
            pass
        pass
    
    b, a = signal.butter(4, 0.1)
    # tau_filter = [savgol_filter(tau_real[i],7,2) for i in range(12)]
    tau_filter = [signal.filtfilt(b, a, tau_real[i]) for i in range(12)]
    vel_filter = [signal.filtfilt(b, a, qd_real[i]) for i in range(12)]
    # visualization
    gs = g.subgridspec(2, 2, hspace=0.5, wspace=0.1, height_ratios=[10,1])

    ax = [fig_handler.add_subplot(gs[0, i]) for i in range(2)]
    cax = [fig_handler.add_subplot(gs[1, i]) for i in range(2)]
    
    norm = colors.Normalize(vmin=0, vmax=1)
    sm1 = cm.ScalarMappable(cmap='winter', norm=norm)
    sm1.set_array([])
    sm2 = cm.ScalarMappable(cmap='autumn', norm=norm)
    sm2.set_array([])
    cbx = fig_handler.colorbar(sm1, cax=cax[0], orientation="horizontal",
                         ticks=[0,0.2,0.4,0.6,0.8,1.0])
    cbx.ax.tick_params(labelsize=6)
    cbx.set_label("Sim Traj", fontsize=8)

    cbx=fig_handler.colorbar(sm2, cax=cax[1], orientation="horizontal",
                         ticks=[0,0.2,0.4,0.6,0.8,1.0])
    cbx.ax.tick_params(labelsize=6)
    cbx.set_label("Real Traj", fontsize=8)
    care_id = [7,8]
    n = 1

    for i in range(2):
        cs = motor_capicity_plot(ax[i],cmap='coolwarm',nlevel=200,alpha=0.3)

        vel_temp = vel_sim[0:robot.num_of_period,care_id[i]].reshape([100,-1])[:,0]
        tau_temp = torque_sim[0:robot.num_of_period,care_id[i]].reshape([100,-1])[:,0]
        color_temp = (np.linspace(0, 1, len(vel_temp)) + 0.5) % 1.0
        l_sim = color_line_segment(vel_temp,tau_temp,color_temp,alpha=1,cmap='winter',lw=0.5,flag_close=True)

        l_real = color_line_segment(vel_filter[care_id[i]][200:200+n*200],
                                    tau_filter[care_id[i]][200:200+200*n],
                                    (np.arange(200*n)%200)/200,
                                    alpha=0.6, cmap='autumn',lw=1.5,flag_close=True)
        ax[i].add_collection(l_sim)
        ax[i].add_collection(l_real)
        # arrowed_spines(fig_handler,ax[i])
        ax[i].grid('on')
        pass

    ax[0].set_xticks([-42,-28, -14, 0, 14, 28,42])
    ax[0].set_xticklabels([-42,-28, -14, 0, 14, 28,42])
    ax[0].set_yticks([-18,-12,-6,0,6,12,18])
    ax[0].set_yticklabels([-18,-12,-6,0,6,12,18])
    ax[1].set_xticks([-42,-28, -14, 0, 14, 28,42])
    ax[1].set_xticklabels([-42,-28, -14, 0, 14, 28,42])
    ax[1].set_yticks([-18,-12,-6,0,6,12,18])
    ax[1].set_yticklabels([-18,-12,-6,0,6,12,18])

    ax[0].set_xlabel(r"$\omega\ (\mathrm{rad/s})$")
    ax[0].set_ylabel(r"$\tau\ (\mathrm{Nm})$")
    ax[1].set_xlabel(r"$\omega\ (\mathrm{rad/s})$")

    ax[0].xaxis.set_label_coords(0.5, -0.12)
    ax[0].yaxis.set_label_coords(-0.12, 0.5)
    ax[1].xaxis.set_label_coords(0.5, -0.12)

    plt.setp(ax[1].get_yticklabels(), visible=False)

    # --------------------------------
    # plot state

    real_nn_data = pd.read_excel("Exp_Raw_Data\\nn_real.xlsx")

    zaxis = [real_nn_data['input'+str(i)].to_numpy() for i in range(29, 32)]
    del real_nn_data

    roll = np.arctan2(zaxis[1],zaxis[2])
    pitch = np.arctan2(-zaxis[0],np.sqrt(np.power(zaxis[1],2)+np.power(zaxis[2],2)))

    gs = g_state.subgridspec(4, 2, hspace=0.2, wspace=0.2, height_ratios=[1,4,4,4])

    ax = [fig_handler.add_subplot(gs[i+1,0]) for i in range(3)]
    ax_ = [fig_handler.add_subplot(gs[i+1,1]) for i in range(3)]
    [plt.setp(ax[i].get_xticklabels(), visible=False) for i in range(2)]
    [plt.setp(ax_[i].get_xticklabels(), visible=False) for i in range(2)]
    [plt.setp(ax_[i].get_yticklabels(), visible=False) for i in range(3)]

    n = 3

    offset = int(robot.num_of_period/2)
    temp_tau_sim = [torque_sim[0+offset:offset+n*robot.num_of_period,care_id[i]] for i in range(2)]
    temp_vel_sim = [vel_sim[0+offset:offset+n*robot.num_of_period,care_id[i]] for i in range(2)]
    time_sim = np.linspace(0,n * 0.2,n * robot.num_of_period)

    temp_tau_real = [tau_filter[care_id[i]][0: 200 * n] for i in range(2)]
    # temp_tau_real = tau_real[care_id[0]][0:200*n]
    temp_vel_real = [qd_real[care_id[i]][0: 200 * n] for i in range(2)]
    time_real = np.linspace(0, n * 0.2, 200 * n)

    power = [temp_tau_real[i] * temp_vel_real[i] for i in range(2)]
    power = [np.convolve(power[i],np.ones(5),'same')/5 for i in range(2)]

    ax[0].plot(np.linspace(0,0.2*n,n*100), robot.posture_formatted[0,50:50+n*100,0,0], lw=0.5)
    ax[0].plot(np.linspace(0,0.2*n,n*100), roll[5000:5000+n*100], lw=1, alpha=0.6)
    ax[0].set_ylim([-0.06,0.06])
    ax[0].set_xlim([0,0.6])
    ax[0].set_ylabel(r"$\theta_\mathrm{r}\ (\mathrm{rad})$")
    ax[0].yaxis.set_label_coords(-0.27, 0.5)
    
    ax_[0].plot(np.linspace(0,0.2*n,n*100), robot.posture_formatted[0,50:50+n*100,0,1], lw=0.5)
    ax_[0].plot(np.linspace(0,0.2*n,n*100), pitch[5000:5000+n*100], lw=1, alpha=0.6)
    ax_[0].set_ylim([-0.06,0.06])
    ax_[0].set_xlim([0,0.6])
    ax_[0].set_ylabel(r"$\theta_\mathrm{p}\ (\mathrm{rad})$")
    ax_[0].yaxis.set_label_coords(-0.04, 0.5)

    ax[1].plot(time_sim, temp_vel_sim[0], lw=0.5)
    ax[1].plot(time_real, temp_vel_real[0], lw=1, alpha=0.6)

    ax_[1].plot(time_sim, temp_vel_sim[1], lw=0.5)
    ax_[1].plot(time_real, temp_vel_real[1], lw=1, alpha=0.6)

    ax[1].set_xlim([0,0.6])
    ax_[1].set_xlim([0,0.6])
    ax[1].set_ylim([-40,40])
    ax_[1].set_ylim([-40,40])
    ax[1].set_ylabel(r"$\dot{\theta}_\mathrm{h}\ (\mathrm{rad/s})$")
    ax_[1].set_ylabel(r"$\dot{\theta}_\mathrm{k}\ (\mathrm{rad/s})$")
    ax[1].yaxis.set_label_coords(-0.27, 0.5)
    ax_[1].yaxis.set_label_coords(-0.04, 0.5)

    ax[2].plot(time_sim, temp_tau_sim[0], lw=0.5)
    ax[2].plot(time_real, temp_tau_real[0], lw=1, alpha=0.6)

    ax_[2].plot(time_sim, temp_tau_sim[1], lw=0.5)
    ax_[2].plot(time_real, temp_tau_real[1], lw=1, alpha=0.6)

    ax[2].set_xlim([0,0.6])
    ax_[2].set_xlim([0,0.6])
    ax[2].set_ylim([-20,20])
    ax_[2].set_ylim([-20,20])
    ax[2].set_ylabel(r"$\tau_\mathrm{h}\ (\mathrm{Nm})$")
    ax_[2].set_ylabel(r"$\tau_\mathrm{k}\ (\mathrm{Nm})$")
    ax[2].yaxis.set_label_coords(-0.27, 0.5)
    ax_[2].yaxis.set_label_coords(-0.04, 0.5)
    ax[2].set_xlabel("Time (s)")
    ax_[2].set_xlabel("Time (s)")
    ax[2].xaxis.set_label_coords(0.5, -0.3)
    ax_[2].xaxis.set_label_coords(0.5, -0.3)

    cmap = cm.get_cmap('coolwarm')
    [add_period_span(ax[i], 0, 0.6, 0.2, c=[cmap(0.2),cmap(0.7)],alpha=0.2) for i in range(3)]
    [add_period_span(ax_[i], 0, 0.6, 0.2, c=[cmap(0.2),cmap(0.7)],alpha=0.2) for i in range(3)]
    [ax[i].grid(axis='y') for i in range(3)]
    [ax_[i].grid(axis='y') for i in range(3)]

    label = ["Sim","Real"]
    legend_elements = [Line2D([0],[0],color='C0',lw=0.5,label=label[0]),
                       Line2D([0],[0],color='C1',lw=1,alpha=0.6,label=label[1])]
    fig_handler.legend(bbox_to_anchor=(0.15,0.8,0.25,0.1),loc="lower left",mode="expand",
                       borderaxespad=0, ncol=2,handles=legend_elements)
    pass


def main():

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

    fig = plt.figure(figsize=(7.2, 2.6), dpi=600)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0,1.2], wspace=0.15)
    # gs = fig.add_gridspec(1, 2, width_ratios=[1,1.8], wspace=0.2)

    plot_torque_vel2(fig, gs[1], gs[0])
    fig.savefig(os.getcwd()+'/Fig5.pdf',dpi=600,transparent=True)
    # plt.show()

    pass

if __name__ == "__main__":
    main()
    pass
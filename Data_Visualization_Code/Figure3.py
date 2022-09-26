"""
This script is used to plot Figure3
"""

import numpy as np
from numpy import arctan2, arcsin, power, cos, sin, pi, tensordot
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
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

def entropy(data, lb, ub, precision):
    # data is narray like, (obs, dim)
    temp = np.clip(data, lb, ub)
    _, frequency = np.unique(
        (temp/precision).astype(np.int), axis=0, return_counts=True)
    frequency = frequency / (data.shape[0])
    return -np.sum(frequency * np.log(frequency))

def piecewise_func(x, a, b, c, d):
    temp = (x <= a).astype(np.int) * b + (a < x).astype(np.int)*(x <
                                                                 c).astype(np.int)*((x-a)*((d-b)/(c-a))+b) + (x >= c).astype(np.int)*d
    return temp

def piecewise_func2(x, a, b, c, d, e):
    temp = (x <= a).astype(np.int) * b
    temp = temp + (a < x).astype(np.int)*(x <c).astype(np.int)*((x-a)*((d-b)/(c-a))+b)
    temp = temp + (x >= c).astype(np.int)*(d+e*(x-c))
    return temp

def gradient_image(ax,x,y,x_key,c_key):

    # cmap = mpl.cm.coolwarm_r
    cmap = mpl.cm.RdYlBu
    dx = x[1]-x[0]
    color = (x<=x_key[0]).astype(np.int) * x/x_key[0]*c_key[0]
    color += (x>x_key[0]).astype(np.int)* (x<=x_key[1]).astype(np.int) * ((c_key[1]-c_key[0])/(x_key[1]-x_key[0])*(x-x_key[0])+c_key[0])
    color += (x>x_key[1]).astype(np.int) * ((1-c_key[1])/(np.max(x)-x_key[1])*(x-x_key[1])+c_key[1])
    # print(cmap(1))
    for i in range(len(x)-1):
        ax.fill_between([x[i],x[i+1]+dx/100],y[i:i+2],facecolor=cmap(color[i]),edgecolor=cmap(color[i]),zorder=-1)
        pass

    ax.fill_between([x[0],x[-1]+dx/100],[0,0],[10,10],facecolor='w',zorder=0,alpha=0.7)
    
    pass

def plot_time_series_his(fig_handler, g, gc):
    # load data and preprocess
    date = []
    date.append("2021-06-21-12-07-06")
    date.append("2021-06-23-19-59-50")

    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-"+d+".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i], flag_normalized=False) for i in range(len(date))]

    x = robot[0].x
    x2 = robot[1].x
    t = np.arange(0, robot[0].FoE, robot[0].skip) * 0.002

    tt = np.tile(t,x.shape[2])
    yy = np.transpose(x, (0,2,1,3))[0,:,:,:]

    lb = [0.2, -1, -1, -5, -5, -5]
    ub = [0.4, 1, 1, 5, 5, 5]

    # title1 = [r"$z\ (m)$",r"$\theta_r\ (rad)$",r"$\theta_p\ (rad)$"]
    # title2 = [r"$\dot z\ (m/s)$",r"$\omega_r\ (rad/s)$",r"$\omega_p\ (rad/s)$"]
    title1 = [r"$z$"+"\n"+"(m)",r"$\theta_\mathrm{r}$"+"\n"+"(rad)",r"$\theta_\mathrm{p}$"+"\n"+"(rad)"]
    title2 = [r"$\dot z$"+"\n"+"(m/s)",r"$\omega_\mathrm{r}$"+"\n"+"(rad/s)",r"$\omega_\mathrm{p}$"+"\n"+"(rad/s)"]

    del robot

    gs = g.subgridspec(3,2,wspace=0.25, hspace=0.2)

    index1 = ["i","ii","iii"]
    index2 = ["iv","v","vi"]

    for i in range(3):
        ax1 = fig_handler.add_subplot(gs[i,0])
        cmap = copy(plt.cm.magma)
        # cmap = copy(plt.cm.Blues)
        cmap.set_bad(cmap(0))
        h,xedges,yedges = np.histogram2d(tt, yy[:,:,i].reshape(-1),bins=[100, 101],range=[[0,2],[lb[i],ub[i]]])
        # pcm = ax1.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=1e4), rasterized=True)  # version1
        pcm = ax1.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=Normalize(vmax=2e2), rasterized=True)  # version2
        # ax1.title.set_text(title1[i])
        ax1.set_ylabel(title1[i],fontsize=5)
        ax1.set_ylim([lb[i], ub[i]])
        ax1.set_yticks([lb[i], ub[i]])
        ax1.yaxis.set_label_coords(-0.05,0.5)

        ax2 = fig_handler.add_subplot(gs[i,1])
        h,xedges,yedges = np.histogram2d(tt, yy[:,:,i+3].reshape(-1),bins=[100, 101],range=[[0,2],[lb[i+3],ub[i+3]]])
        # pcm = ax2.pcolormesh(xedges, yedges, h.T, cmap=cmap,norm=LogNorm(vmax=1e4), rasterized=True)  # version 1
        pcm = ax2.pcolormesh(xedges, yedges, h.T, cmap=cmap,norm=Normalize(vmax=2e2), rasterized=True)  # version 2
        # fig_handler.colorbar(pcm, ax=ax2, pad=0)
        # ax2.title.set_text(title2[i])
        ax2.set_ylabel(title2[i],fontsize=5)
        ax2.set_ylim([lb[3+i], ub[3+i]])
        ax2.set_yticks([lb[3+i], ub[3+i]])
        ax2.yaxis.set_label_coords(-0.05,0.5)

        if i!=2:
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            pass
        else:
            ax1.set_xlabel("Time (s)")
            ax2.set_xlabel("Time (s)")
            # ax1.xaxis.set_label_coords(0.5,-0.2)
            # ax2.xaxis.set_label_coords(0.5,-0.2)
            pass

        ax1.text(0.9,0.85,index1[i],transform=ax1.transAxes,color='w')
        ax2.text(0.9,0.85,index2[i],transform=ax2.transAxes,color='w')

        pass

    # cbaxes = fig_handler.add_axes([0.08, 0.35, 0.87, 0.03])
    cbaxes = fig_handler.add_subplot(gc)
    # cb = plt.colorbar(pcm, cax = cbaxes, orientation="horizontal", ticks=[0,50,100,150,200],label='Probability of State (\%)')   
    # cb.ax.set_xticklabels([r"$0.0$",r"$0.5$",r"$1.0$",r"$1.5$",r"$\ge 2$"])
    # cb = plt.colorbar(pcm, cax = cbaxes, orientation="vertical", ticks=[0,50,100,150,200],label='Probability of State (\%)')   
    cb = plt.colorbar(pcm, cax = cbaxes, orientation="vertical", ticks=[0,50,100,150,200])   
    cb.ax.set_yticklabels(["0.0%","0.5%","1.0%","1.5%",u'\N{GREATER-THAN OR EQUAL TO}'+"2%"],Fontsize=5)

    pass

def plot_time_series_his_with_zoom(fig_handler, g, gc):
    # load data and preprocess
    date = []
    date.append("2021-06-21-12-07-06")
    date.append("2021-06-23-19-59-50")

    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-"+d+".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i], flag_normalized=False) for i in range(len(date))]

    x = robot[0].x
    x2 = robot[1].x
    t = np.arange(0, robot[0].FoE, robot[0].skip) * 0.002

    tt = np.tile(t,x.shape[2])
    yy = np.transpose(x, (0,2,1,3))[0,:,:,:]

    lb = [0.2, -1, -1, -5, -5, -5]
    ub = [0.4, 1, 1, 5, 5, 5]

    title1 = [r"$z$",r"$\theta_r$",r"$\theta_p$"]
    title2 = [r"$\dot z$",r"$\omega_r$",r"$\omega_p$"]

    del robot

    gs = g.subgridspec(3,2,wspace=0.2, hspace=0.2)
    vmax = 2e2

    for i in range(3):
        ax1 = fig_handler.add_subplot(gs[i,0])
        cmap = copy(plt.cm.magma)
        # cmap = copy(plt.cm.Blues)
        cmap.set_bad(cmap(0))
        h,xedges,yedges = np.histogram2d(tt, yy[:,:,i].reshape(-1),bins=[100, 101],range=[[0,2],[lb[i],ub[i]]])
        # pcm = ax1.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=1e4), rasterized=True)  # version1
        pcm = ax1.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=Normalize(vmax=vmax), rasterized=True)  # version2
        # ax1.title.set_text(title1[i])
        ax1.set_ylabel(title1[i])
        ax1.set_ylim([lb[i], ub[i]])
        ax1.set_yticks([lb[i], ub[i]])
        ax1.yaxis.set_label_coords(-0.05,0.5)

        # add zoom windows
        axins1 = ax1.inset_axes([0.5,0.7,0.3,0.3])
        axins1.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=Normalize(vmax=vmax), rasterized=True)
        axins1.set_xlim(0.8, 1.2)
        axins1.set_ylim((lb[i]+ub[i])/2-0.1*(ub[i]-lb[i]),(lb[i]+ub[i])/2+0.1*(ub[i]-lb[i]))
        ax1.indicate_inset_zoom(axins1, edgecolor="white")


        ax2 = fig_handler.add_subplot(gs[i,1])
        h,xedges,yedges = np.histogram2d(tt, yy[:,:,i+3].reshape(-1),bins=[100, 101],range=[[0,2],[lb[i+3],ub[i+3]]])
        # pcm = ax2.pcolormesh(xedges, yedges, h.T, cmap=cmap,norm=LogNorm(vmax=1e4), rasterized=True)  # version 1
        pcm = ax2.pcolormesh(xedges, yedges, h.T, cmap=cmap,norm=Normalize(vmax=vmax), rasterized=True)  # version 2
        # fig_handler.colorbar(pcm, ax=ax2, pad=0)
        # ax2.title.set_text(title2[i])
        ax2.set_ylabel(title2[i])
        ax2.set_ylim([lb[3+i], ub[3+i]])
        ax2.set_yticks([lb[3+i], ub[3+i]])
        ax2.yaxis.set_label_coords(-0.05,0.5)

        if i!=2:
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            pass
        else:
            ax1.set_xlabel("Time (s)")
            ax2.set_xlabel("Time (s)")
            # ax1.xaxis.set_label_coords(0.5,-0.2)
            # ax2.xaxis.set_label_coords(0.5,-0.2)
            pass

        pass

    # cbaxes = fig_handler.add_axes([0.08, 0.35, 0.87, 0.03])
    cbaxes = fig_handler.add_subplot(gc)
    cb = plt.colorbar(pcm, cax = cbaxes, orientation="horizontal")   

    pass

def plot_3d_state(fig_handler, g):
    pass

def plot_entropy(fig_handler, g):
    from scipy.optimize import curve_fit

    date = []
    date.append("2021-06-21-12-07-06")
    date.append("2021-06-23-19-59-50")
    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-"+d+".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i], flag_normalized=False) for i in range(len(date))]
    x = robot[0].x
    x2 = robot[1].x
    t = np.arange(0, robot[0].FoE, robot[0].skip) * 0.002
    del robot

    sf = [0, 1, 2, 3, 4, 5]  # selected feature
    ub = np.array([0.5, 3.14, 1.57, 10, 10, 10])
    lb = np.array([0.0, -3.14, -1.57, -10, -10, -10])
    # related with the number of sample
    # precision = np.array([0.002, 0.02, 0.02, 0.01, 0.05, 0.05])
    precision = np.array([0.005, 0.02, 0.02, 0.005, 0.025, 0.025])

    ent = []
    for i in range(x.shape[1]):
        ent.append(entropy(x[0, i, :, :], lb[sf], ub[sf], precision[sf]))
        pass

    ent2 = []
    for i in range(x2.shape[1]):
        ent2.append(entropy(x2[0, i, :, :], lb[sf], ub[sf], precision[sf]))
        pass

    para_ub = np.array([1, 10, 2, 5, 0])
    para_lb = np.array([0, 5, 1, 0, -5])
    # temp_p, _ = curve_fit(piecewise_func, t, np.asarray(ent), bounds=(para_lb, para_ub))
    temp_p, _ = curve_fit(piecewise_func2, t, np.asarray(ent), bounds=(para_lb, para_ub),p0=[0.2,9.2,1.5,2.5,-2.5])
    label2 = r"$\kappa_1=$"+"{:.2f}, ".format((temp_p[3]-temp_p[1])/(temp_p[2]-temp_p[0]))
    label2 = label2 + r"$\kappa_2=$" + "{:.2f}".format(temp_p[4])
    # label2 = label2 + r"$\kappa_2={:f}$".format(num2tex(temp_p[4],precision=2))

    temp_p2, _ = curve_fit(piecewise_func2, t, np.asarray(ent2), bounds=(para_lb, para_ub),p0=[0.2,9.2,1.5,2.5,-2.5])
    label22 = r"$\kappa_1=$"+"{:.2f}, ".format((temp_p2[3]-temp_p2[1])/(temp_p2[2]-temp_p2[0]))
    label22 = label22 + r"$\kappa_2=$"+"{:.2f}".format(temp_p2[4])
    # label22 = label22 + r"$\kappa_2={:f}$".format(num2tex(temp_p2[4],precision=2))

    ax = fig_handler.add_subplot(g)
    ax.scatter(t, ent, s=10, alpha=0.5, marker='o', linewidth=0, label=r"$v^\mathrm{B}_\mathrm{xT}=5m/s,\ \mu=0.8$")
    ax.plot(t, piecewise_func2(t, temp_p[0], temp_p[1], temp_p[2],
             temp_p[3],temp_p[4]), lw=2)#, label=label2)
    ax.scatter(t, ent2, s=10, alpha=0.5, marker='o', linewidth=0, label=r"$v^\mathrm{B}_\mathrm{xT}=5m/s,\ \mu=0.1$")
    ax.plot(t, piecewise_func2(t, temp_p2[0], temp_p2[1], temp_p2[2],temp_p2[3],temp_p2[4]), lw=2)#, label=label22)

    ax.set_xlim([0, 2.0])
    ax.set_ylim([0, 10.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Entropy "+r"$(\log_\mathrm{e})$")
    ax.text(0.05, 8.0, "9.21", color='r')
    ax.text(0.3, 4.5, r"$\kappa_1=$"+"{:.1f}".format((temp_p[3]-temp_p[1])/(temp_p[2]-temp_p[0])),rotation=-45.)
    ax.text(1.1, 0.5, r"$\kappa_2=$" + "{:.1f}".format(temp_p[4]),rotation=-4)

    ax.text(0.1, 0.5, "I", color='r')
    ax.text(0.6, 0.5, "II", color='r')
    ax.text(1.5, 0.5, "III", color='r')
    ax.legend(loc=1,frameon=True)

    gradient_image(ax, np.linspace(0,2,1000),piecewise_func2(np.linspace(0,2,1000), temp_p[0], temp_p[1], temp_p[2],
             temp_p[3],temp_p[4]),[temp_p[0],temp_p[2]],[0.25,0.55] )

    pass

def plot_entropy2(fig_handler, g):
    from scipy.optimize import curve_fit

    date = []
    date.append("2021-06-21-12-07-06")
    bin_file = ["Exp_Raw_Data\\body-center-" + d + ".bin" for d in date]
    param_file = ["Exp_Raw_Data\\Param-"+d+".txt" for d in date]
    robot = [RobotBodyInfo(bin_file[i], param_file[i], flag_normalized=False) for i in range(len(date))]
    x = robot[0].x
    t = np.arange(0, robot[0].FoE, robot[0].skip) * 0.002
    del robot

    sf = [0, 1, 2, 3, 4, 5]  # selected feature
    ub = np.array([0.5, 3.14, 1.57, 10, 10, 10])
    lb = np.array([0.0, -3.14, -1.57, -10, -10, -10])
    # related with the number of sample
    # precision = np.array([0.002, 0.02, 0.02, 0.01, 0.05, 0.05])
    precision = np.array([0.005, 0.02, 0.02, 0.005, 0.025, 0.025])

    ent = []
    for i in range(x.shape[1]):
        ent.append(entropy(x[0, i, :, :], lb[sf], ub[sf], precision[sf]))
        pass

    para_ub = np.array([1, 10, 2, 5, 0])
    para_lb = np.array([0, 5, 1, 0, -5])
    # temp_p, _ = curve_fit(piecewise_func, t, np.asarray(ent), bounds=(para_lb, para_ub))
    temp_p, _ = curve_fit(piecewise_func2, t, np.asarray(ent), bounds=(para_lb, para_ub),p0=[0.2,9.2,1.5,2.5,-2.5])
    label2 = r"$\kappa_1=$"+"{:.2f}, ".format((temp_p[3]-temp_p[1])/(temp_p[2]-temp_p[0]))
    label2 = label2 + r"$\kappa_2=$" + "{:.2f}".format(temp_p[4])
    # label2 = label2 + r"$\kappa_2={:f}$".format(num2tex(temp_p[4],precision=2))

    ax = fig_handler.add_subplot(g)
    ax.scatter(t, ent, s=10, alpha=0.5, marker='o', linewidth=0, label=r"$^Bv_{Tx}=5m/s,\ \mu=0.8$")
    ax.plot(t, piecewise_func2(t, temp_p[0], temp_p[1], temp_p[2],
             temp_p[3],temp_p[4]), lw=2)#, label=label2)

    ax.set_xlim([0, 2.0])
    ax.set_ylim([0, 10.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Entropy "+r"$(\log_\mathrm{e})$")
    ax.text(0.05, 8.0, "9.21", color='r')
    ax.text(0.3, 4.5, r"$\kappa_1=$"+"{:.1f}".format((temp_p[3]-temp_p[1])/(temp_p[2]-temp_p[0])),rotation=-45.)
    ax.text(1.1, 0.5, r"$\kappa_2=$" + "{:.1f}".format(temp_p[4]),rotation=-4)
    ax.text(0.1, 0.5, "I", color='r')
    ax.text(0.6, 0.5, "II", color='r')
    ax.text(1.5, 0.5, "III", color='r')
    leg = ax.legend(loc=1,frameon=True,fancybox=True,shadow=True)
    leg.get_frame().set_alpha(0.8)

    gradient_image(ax, np.linspace(0,2,100),piecewise_func2(np.linspace(0,2,100), temp_p[0], temp_p[1], temp_p[2],
             temp_p[3],temp_p[4]),[temp_p[0],temp_p[2]],[0.25,0.55] )

    pass

def main4():
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

    fig = plt.figure(figsize=(3.4645669, 5), dpi=600, constrained_layout=False)
    # gs = fig.add_gridspec(3, 1, height_ratios=[3,0.12,2.2], hspace=0.30)
    gs = fig.add_gridspec(2, 1, height_ratios=[3,2.2], hspace=0.25)
    g_sub = gs[0].subgridspec(1,2,wspace=0.06, width_ratios=[15,1])

    plot_time_series_his(fig, g_sub[0], g_sub[1])
    # plot_time_series_his_with_zoom(fig, gs[0], gs[1])
    # plot_3d_state(fig, g_sub[0])
    plot_entropy(fig, gs[1])

    # plt.tight_layout()
    # plt.draw()
    fig.savefig(os.getcwd()+'/Fig3.pdf',dpi=600,transparent=True)
    # plt.show()

    pass

if __name__ == "__main__":
    # main()
    # main2()       # version 1
    # main3()
    main4()    # version 2
    pass
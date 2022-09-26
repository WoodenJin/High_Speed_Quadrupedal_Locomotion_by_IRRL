import numpy as np
import pandas as pd
from numpy import arctan2, arcsin, power, cos, sin, pi, tensordot, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpltern.ternary.datasets import get_triangular_grid
from matplotlib.lines import Line2D
import os


class RobotInfo(object):
    def __init__(self, filename, is_wildcat=False):

        self.flag_is_wildcat = is_wildcat

        df = pd.read_csv(filename, sep=" ")

        self.z = df['z'].to_numpy()
        self.quat = np.asarray([df['quat'+str(i)].to_numpy()
                               for i in range(4)]).transpose()
        self.vel = np.asarray([df['vel'+str(i)].to_numpy()
                              for i in range(3)]).transpose()
        self.omega = np.asarray([df['omega'+str(i)].to_numpy()
                                for i in range(3)]).transpose()
        self.q = np.asarray([df['q'+str(i)].to_numpy()
                            for i in range(12)]).transpose()
        self.dq = np.asarray([df['dq'+str(i)].to_numpy()
                             for i in range(12)]).transpose()
        self.tau = np.asarray([df['t'+str(i)].to_numpy()
                              for i in range(12)]).transpose()

        self.tau[:, [0, 1, 3, 4, 6, 7, 9, 10]
                 ] = self.tau[:, [0, 1, 3, 4, 6, 7, 9, 10]] * 18
        self.tau[:, [2, 5, 8, 11]] = self.tau[:, [2, 5, 8, 11]] * 18*1.55

        self.contact = np.asarray([df['c'+str(i)].to_numpy()
                                  for i in range(4)]).transpose()
        del df
        pass

    def phase_jagged(self, period=0.2, sample_freq=500):
        temp = np.arange(len(self.z))
        temp = np.mod(temp, int(period/(1/sample_freq))) / \
            int(period/(1/sample_freq))
        return temp

    @property
    def motor_torque(self):
        temp_tau = self.tau.copy()
        temp_tau[:, [2, 5, 8, 11]] = temp_tau[:, [2, 5, 8, 11]] / 1.55
        return temp_tau
        # return self.tau[:,:]

    @property
    def motor_velocity(self):
        temp_vel = self.dq.copy()
        temp_vel[:, [2, 5, 8, 11]] = temp_vel[:, [2, 5, 8, 11]] * 1.55
        return temp_vel
        # return self.dq[:,:]

    @property
    def power(self):
        return np.sum(self.dq*self.tau, axis=1)

    @property
    def Vb(self):
        rot = RobotInfo.qua2matrix(
            self.quat[:, 0], self.quat[:, 1], self.quat[:, 2], self.quat[:, 3])
        body_vel = np.matmul(np.transpose(rot, (0, 2, 1)),
                             self.vel.reshape([-1, 3, 1])).reshape([-1, 3])
        if self.flag_is_wildcat:
            body_vel[:, 0] = -body_vel[:, 0]
            pass
        return body_vel

    @property
    def Omegab(self):
        rot = RobotInfo.qua2matrix(
            self.quat[:, 0], self.quat[:, 1], self.quat[:, 2], self.quat[:, 3])
        omega_body = np.matmul(np.transpose(
            rot, (0, 2, 1)), self.omega.reshape([-1, 3, 1])).reshape([-1, 3])
        return omega_body

    @property
    def roll(self):
        temp, _, _ = RobotInfo.qua2euler(
            self.quat[:, 0], self.quat[:, 1], self.quat[:, 2], self.quat[:, 3])
        return temp

    @property
    def pitch(self):
        _, temp, _ = RobotInfo.qua2euler(
            self.quat[:, 0], self.quat[:, 1], self.quat[:, 2], self.quat[:, 3])
        return temp

    @property
    def Contact_filtered(self):
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        contact_flag = []
        for i in range(4):
            contact_flag.append(np.convolve(
                self.contact[:, i], kernel, mode='same'))
            pass
        contact_flag = np.asarray(contact_flag).transpose()
        contact_flag = contact_flag.astype(np.bool)

        for i in range(4):
            for j in range(len(contact_flag[:, i])-11):
                if(contact_flag[j, i] == 1 and contact_flag[j+1, i] == 0):
                    contact_flag[j+1, i] = contact_flag[j+1:j+11, i].any()
                    pass
                pass
            pass

        return contact_flag

    @staticmethod
    def qua2euler(w, x, y, z):
        roll = arctan2(2 * (w * x + y * z), 1 - 2 *
                       (power(x, 2) + power(y, 2)))
        pitch = arcsin(2 * (w * y - x * z))
        yaw = arctan2(2 * (w * z + x * y), 1 - 2 * (power(y, 2) + power(z, 2)))
        return roll, pitch, yaw

    @staticmethod
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

    @staticmethod
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

    pass

def add_period_span(ax, start, end, step, c=['C5','C4'], alpha=0.6):
    s = start
    idx = 0
    while(s<end):
        ax.axvspan(s,s+step,facecolor=c[idx%len(c)],alpha=alpha,zorder=1)
        s += step
        idx += 1
        pass
    pass

def plot_performance(fig_handler, g):
    # plot vel, body information, TCOT

    g_body = g[1].subgridspec(3, 1)

    ax_v = fig_handler.add_subplot(g[0])  # ax for vel
    # ax_b = [fig_handler.add_subplot(g_body[i]) for i in range(3)]  # ax for body information
    ax_b = []
    ax_b.append(fig_handler.add_subplot(g_body[0]))
    ax_b.append(fig_handler.add_subplot(g_body[1],sharex=ax_b[0]))
    ax_b.append(fig_handler.add_subplot(g_body[2],sharex=ax_b[0]))
    plt.setp(ax_b[0].get_xticklabels(), visible=False)
    plt.setp(ax_b[1].get_xticklabels(), visible=False)

    grid_power = g[2].subgridspec(2, 1, height_ratios=[1,2], hspace=0.05)
    # ax_p = fig_handler.add_subplot(g[2])  # ax for power
    ax_p = [fig_handler.add_subplot(grid_power[i]) for i in range(2)]

    # ----------------------------------------
    # power data process and visualization
    file_name = []
    file_name.append("Exp_Raw_Data\\info_trot_155_stair.csv")
    file_name.append("Exp_Raw_Data\\info_trot_186_stair.csv")
    file_name.append("Exp_Raw_Data\\info_trot_177_stair.csv")
    file_name.append("Exp_Raw_Data\\info_trot_combine_stair.csv")

    robot = [RobotInfo(file) for file in file_name]

    # data process
    vel_x = [ro.Vb[:, 0] for ro in robot]
    time = np.arange(len(vel_x[0]))*0.002

    power = [np.mean(np.reshape(ro.power, (-1, 100)), axis=1) for ro in robot]
    vel = [np.mean(np.reshape(v, (-1, 100)), axis=1) for v in vel_x]

    care_id = [0, 1, 2, 3]
    label = ["Controller "+r'$\mathbf{\Theta}^f$',
             "Controller "+r'$\mathbf{\Theta}^m$',
             "Controller "+r'$\mathbf{\Theta}^v$',
             "Controller "+r'$\mathbf{\Theta}^l$',
             "Reference"]
    
    TCOT_mean = []
    TCOT_std = []
    m = 10
    g = 9.8

    TCOT_origin_data = []

    for i in range(len(care_id)):
        # temp_vel = vel[i].reshape(-1, 100)[:, 10:]
        # temp_pow = power[i].reshape(-1, 100)[:, 10:]
        temp_vel = vel[i].reshape(-1, 100)[:, -10:]
        temp_pow = power[i].reshape(-1, 100)[:, -10:]
        TCOT_mean.append(np.mean(temp_pow/temp_vel/m/g, axis=1))
        TCOT_std.append(np.std(temp_pow/temp_vel/m/g, axis=1))
        TCOT_origin_data.append(temp_pow/temp_vel/m/g)
        pass
    del robot

    # ----------------------------
    # visualization
    color = ['C0','C3','C1','gold','dimgray']
    x = np.arange(5)
    width = 0.22
    # plot barchart
    for ap in ax_p:
        for i in range(len(care_id)):
            ap.bar(x+width*i-1.5*width, TCOT_mean[i], width*1.0,
                        yerr=TCOT_std[i], label=label[i], facecolor=color[i], zorder=5,edgecolor='None')
            ap.errorbar(x+width*i-1.5*width, TCOT_mean[i],yerr=TCOT_std[i]*2,capsize=1.5,ls='none',zorder=20,color='k')
            for j in range(5):
                # ap.scatter((x[j]+width*i-1.5*width)*np.ones(len(TCOT_origin_data[i][j])),TCOT_origin_data[i][j][:],s=1,zorder=10,color='gray')
                ap.scatter((x[j]+width*i-1.5*width)+np.linspace(-width/2*0.8,width/2*0.8,len(TCOT_origin_data[i][j])),TCOT_origin_data[i][j][:],s=0.4,zorder=10,color='gray',alpha=0.5)
                pass
            pass
        pass

    ax_p[0].set_ylim([1.75, 2.0])
    ax_p[1].set_ylim([0, 0.5])

    # hide the spines between ax and ax2
    ax_p[0].spines.bottom.set_visible(False)
    ax_p[1].spines.top.set_visible(False)
    ax_p[0].xaxis.tick_top()
    ax_p[0].tick_params(labeltop=False)  # don't put tick labels at the top
    ax_p[1].xaxis.tick_bottom()

    ax_p[1].set_xticks(np.arange(5))
    ax_p[1].set_xticklabels(['1.0', '2.0', '3.0', '4.0', '5.0'])
    ax_p[1].set_xlabel('Command Velocity (m/s)')
    # ax_p[1].set_ylabel('TCoT (W/mgV)')
    ax_p[1].set_ylabel('TCoT')
    ax_p[1].yaxis.set_label_coords(-0.12, 0.8)
    # ax_p.grid('on')
    ax_p[0].grid(axis='y')
    ax_p[1].grid(axis='y')

    # ----------------------------------------------------------
    # visualize body information
    # filename
    file_name = []
    file_name.append("Exp_Raw_Data\\info_trot_f1.csv")
    file_name.append("Exp_Raw_Data\\info_trot_m1.csv")
    file_name.append("Exp_Raw_Data\\info_trot_v1.csv")
    file_name.append("Exp_Raw_Data\\info_trot_m65_v25_f10.csv")

    robot = [RobotInfo(file) for file in file_name]
    vel_x = [ro.Vb[:, 0] for ro in robot]
    vel_ref = np.zeros_like(vel_x[0])
    for i in range(len(vel_ref)-1):
        vel_ref[i+1] = 0.999 * vel_ref[i] + 0.001 * 5
        pass
    vel_x.append(vel_ref)
    time = np.arange(len(vel_ref))*0.002

    for i in range(5):
        ax_v.plot(time, vel_x[i], color=color[i], label=label[i], lw=1)
        pass
    ax_v.vlines(x=9.2, ymin=-10, ymax=10, color='k', linestyle='--')

    ax_v.set_xlim(0, 20)
    ax_v.set_ylim(-0.5, 5.5)
    ax_v.set_ylabel(r"$v^\mathrm{B}_x\ (m/s)$")
    ax_v.set_xlabel("Time (s)")
    ax_v.grid(axis='y')

    # ------------------------------
    # visualize body information
    [ax_b[0].plot(time[4600:5100],robot[i].z[4600:5100],color=color[i]) for i in range(4)]
    [ax_b[1].plot(time[4600:5100],robot[i].roll[4600:5100],color=color[i]) for i in range(4)]
    [ax_b[2].plot(time[4600:5100],robot[i].pitch[4600:5100],color=color[i]) for i in range(4)]
    # [add_period_span(ax_b[i],9.2,10.2,0.2) for i in range(3)]
    ax_b[2].set_xlim([time[4600],time[5101]])
    ax_b[2].set_xticks([9.2,9.4,9.6,9.8,10.0,10.2])
    ax_b[2].set_xticklabels(['9.2', '9.4', '9.6', '9.8', '10.0','10.2'])
    ax_b[0].set_ylabel(r'$z(m)$')
    ax_b[0].yaxis.set_label_coords(-0.163, 0.5)
    ax_b[1].set_ylabel(r'$\theta_{r}(rad)$')
    ax_b[1].yaxis.set_label_coords(-0.163, 0.5)
    ax_b[2].set_ylabel(r'$\theta_{p}(rad)$')
    ax_b[2].yaxis.set_label_coords(-0.163, 0.5)
    ax_b[2].set_xlabel("Time (s)")
    pass

def plot_joint_space(fig_handler, g):
    file_name = []
    file_name.append("Exp_Raw_Data\\info_trot_f1.csv")
    file_name.append("Exp_Raw_Data\\info_trot_m1.csv")
    file_name.append("Exp_Raw_Data\\info_trot_v1.csv")
    file_name.append("Exp_Raw_Data\\info_trot_m65_v25_f10.csv")
    robot = [RobotInfo(file) for file in file_name]
    
    num = 1
    q = [ro.q[-num*100-1:,:] for ro in robot]
    dq = [ro.dq[-num*100-1:,:] for ro in robot]
    
    df = pd.read_csv("Exp_Raw_Data\\trot_ref_.csv", sep=" ")
    temp_q = [df["q"+str(i)].to_numpy() for i in range(12)]
    temp_dq = [df["dq"+str(i)].to_numpy() for i in range(12)]
    q.append((np.asarray(temp_q).T)[-num*100-1:,:])
    dq.append((np.asarray(temp_dq).T)[-num*100-1:,:])
    del temp_q,temp_dq,df

    ax = [fig_handler.add_subplot(g[i]) for i in range(4)]
    # [ax[i].sharex(ax[0]) for i in range(1,4)]
    # [ax[i].sharey(ax[0]) for i in range(1,4)]
    [ax[i].set_yticklabels([]) for i in range(1,4)]
    [ax[i].set_ylim([-35,45]) for i in range(4)]
    ax[0].set_xlim([-1.5,0.5])
    ax[1].set_xlim([-1.5,0.5])
    ax[2].set_xlim([0.5,2.5])
    ax[3].set_xlim([0.5,2.5])
    ax[0].set_ylabel(r"$\dot{\theta}\ (\mathrm{rad/s})$")
    ax[0].yaxis.set_label_coords(-0.12, 0.5)
    ax[0].set_xlabel(r"$\mathrm{HR}\ {\theta_{\mathrm{hip}}}\ (\mathrm{rad})$")
    ax[1].set_xlabel(r"$\mathrm{HL}\ {\theta_{\mathrm{hip}}}\ (\mathrm{rad})$")
    ax[2].set_xlabel(r"$\mathrm{HR}\ {\theta_{\mathrm{knee}}}\ (\mathrm{rad})$")
    ax[3].set_xlabel(r"$\mathrm{HL}\ {\theta_{\mathrm{knee}}}\ (\mathrm{rad})$")


    color = ['C0','C3','C1','gold','dimgray']
    leg_id = [7,10,8,11]
    # leg_id = [1,4,2,5]
    vis_id = [0,1,4]

    for i,l_id in enumerate(leg_id):
        alpha=1
        for j in vis_id:
            alpha -= 0.1
            ax[i].plot(q[j][:,l_id],dq[j][:,l_id],lw=2,color=color[j],alpha=alpha)
            pass
        pass

    pass

def plot_reward(fig_handler, g, flag_normalized=True):
    from mpltern.ternary.datasets import get_triangular_grid
    
    # plot reward function disturbation in the parameter space
    g_sub = g.subgridspec(2,4,wspace=0.3,hspace=0.15)
    ax = []
    ax.append(fig_handler.add_subplot(g_sub[0:2,0:2],projection='ternary'))
    ax.append(fig_handler.add_subplot(g_sub[0,2],projection='ternary'))
    ax.append(fig_handler.add_subplot(g_sub[0,3],projection='ternary'))
    ax.append(fig_handler.add_subplot(g_sub[1,2],projection='ternary'))
    ax.append(fig_handler.add_subplot(g_sub[1,3],projection='ternary'))
    # ax = [fig_handler.add_subplot(gi,projection='ternary') for gi in g]

    # process data
    # data_path = "src\\data\\total_reward.txt"
    data_path = "Exp_Raw_Data\\total_reward_2.txt"
    df = pd.read_csv(data_path, sep=" ");
    w=[]  # coordinate for ternary
    w0 = df['w0'].to_numpy()
    w1 = df['w1'].to_numpy()
    w2 = 1.0 - w1 - w0
    w.append(w1)
    w.append(w2)
    w.append(w0)
    w = np.asarray(w).T
    origin_reward = np.empty([w0.shape[0], 5])  # order: v,m,b,t
    origin_reward[:,1] = 0.5*df['cmd_linear'].to_numpy() + 0.5*df['cmd_angular'].to_numpy()
    origin_reward[:,2] = 0.25*df['mimic_q'].to_numpy() + 0.75*df['mimic_dq'].to_numpy()
    origin_reward[:,3] = 0.5*df['height_keep'].to_numpy() + 0.5*df['balance_keep'].to_numpy()
    origin_reward[:,4] = 0.5*df['torque'].to_numpy() + 0.5*df['torque_d'].to_numpy()

    ratio = np.array([0.3,0.1,0.3,0.3])
    origin_reward[:,0] = np.dot(origin_reward[:,1:], ratio)

    if(flag_normalized):
        vmin = np.zeros(5)
        vmax = np.ones(5)
        origin_reward = (origin_reward-np.min(origin_reward,axis=0))/(np.max(origin_reward,axis=0)-np.min(origin_reward,axis=0))
        pass
    else:
        vmin = np.ones(5) * np.min(origin_reward)
        vmax = np.ones(5) * np.max(origin_reward)
        pass

    cx = [ax[i].tricontourf(w[:,0],w[:,1],w[:,2],origin_reward[:,i],levels=50,cmap='magma',vmin=vmin[i],vmax=vmax[i]) for i in range(5)]


    name = [r"$r^f$",r"$r^v$",r"$r^m$",r"$r^b$",r"$r^t$"]
    for i in range(5):
        ax[i].axis("off")
        # ax[i].scatter([1],[0],[0],color="C1")
        # ax[i].scatter([0],[1],[0],color="C0")
        # ax[i].scatter([0],[0],[1],color="C3")
        ax[i].set_tlabel(r'$\mathbf{\Theta}^v$')#,fontsize=6)
        ax[i].set_llabel(r'$\mathbf{\Theta}^f$')#,fontsize=6)
        ax[i].set_rlabel(r'$\mathbf{\Theta}^m$')#,fontsize=6)
        pass
    ax[0].scatter([1],[0],[0],color="C1")
    ax[0].scatter([0],[1],[0],color="C0")
    ax[0].scatter([0],[0],[1],color="C3")
    cax = ax[0].inset_axes([0.95, 0.35, 0.05, 0.6], transform=ax[0].transAxes)
    colorbar = fig_handler.colorbar(cx[0], cax=cax,ticks=[0,1])
    colorbar.set_label('Normalized Reward', rotation=270, va='baseline')#,fontsize=6)
    # colorbar.ax.tick_params(labelsize=6)

    ax[0].axis("on")
    ax[0].taxis.label.set_color('C1')
    ax[0].laxis.label.set_color('C0')
    ax[0].raxis.label.set_color('C3')
    ax[0].taxis.set_tick_params(tick1On=True, colors='C1', grid_color='C1')#,labelsize=6)
    ax[0].laxis.set_tick_params(tick1On=True, colors='C0', grid_color='C0')#,labelsize=6)
    ax[0].raxis.set_tick_params(tick1On=True, colors='C3', grid_color='C3')#,labelsize=6)
    ax[0].plot([0.25,0.9],[0.1,0.1],[0.65,0],lw=2,ls=':',color='C0')
    ax[0].plot([0.25,0.0],[0.1,0.35],[0.65,0.65],lw=2,ls=':',color='C3')
    ax[0].plot([0.25,0.25],[0.1,0.0],[0.65,0.75],lw=2,ls=':',color='C1')
    ax[0].scatter([0.25],[0.1],[0.65],color='gold',zorder=100)
    # -------------------------------------------
    # visualization

    pass

def plot_reward3d(fig_handler, g, flag_normalized=True):
    # import thrid part
    import matplotlib.tri as tri

    # region load data
    data = [dict() for _ in range(5)]
    w=[]
    data_path = "Exp_Raw_Data\\total_reward.txt"
    df = pd.read_csv(data_path, sep=" ");
    w0 = df['w0'].to_numpy()
    w1 = df['w1'].to_numpy()
    w2 = 1.0 - w1 - w0
    # w.append(w1)
    # w.append(w0)
    # w.append(w2)
    w.append(w1)
    w.append(w2)
    w.append(w0)
    w = np.asarray(w).T
    origin_reward = np.empty([w0.shape[0], 4])  # order: v,m,b,t
    reward = np.empty([w0.shape[0], 4])  # order: v,m,b,t
    origin_reward[:,0] = 0.5*df['cmd_linear'].to_numpy() + 0.5*df['cmd_angular'].to_numpy()
    origin_reward[:,1] = 0.25*df['mimic_q'].to_numpy() + 0.75*df['mimic_dq'].to_numpy()
    origin_reward[:,2] = 0.5*df['height_keep'].to_numpy() + 0.5*df['balance_keep'].to_numpy()
    origin_reward[:,3] = 0.5*df['torque'].to_numpy() + 0.5*df['torque_d'].to_numpy()
    for i in range(4):
        reward[:,i] = (origin_reward[:,i]-np.min(origin_reward[:,i]))/(np.max(origin_reward[:,i])-np.min(origin_reward[:,i]))
        pass
    # endregion

    #region transform the coordinate of ternary into cartesian coordinates
    y = w[:,1] / 2 * sqrt(3)
    x = y / sqrt(3) + w[:,0]
    x = x - 0.5
    y = y - sqrt(3)/6
    z = [origin_reward[:,i] for i in range(4)]
    z.append(0.3*z[0]+0.3*z[1]+0.3*z[2]+0.1*z[1])
    z = [z[4],z[0],z[1],z[2],z[3]]
    #endregion

    # region start plot
    # plot reward function disturbation in the parameter space
    # g_sub = g.subgridspec(2,4,wspace=0.3,hspace=0.15)
    g_sub = g.subgridspec(2,4,wspace=-0.3,hspace=0)

    ax = []
    ax.append(fig_handler.add_subplot(g_sub[0:2,0:2],projection='3d'))
    ax.append(fig_handler.add_subplot(g_sub[0,2],projection='3d'))
    ax.append(fig_handler.add_subplot(g_sub[0,3],projection='3d'))
    ax.append(fig_handler.add_subplot(g_sub[1,2],projection='3d'))
    ax.append(fig_handler.add_subplot(g_sub[1,3],projection='3d'))

    # config the axis
    for axx in ax:
        axx._axis3don = False
        # axx.set_xlim(0, 1)
        # axx.set_ylim(0, sqrt(3)/2)
        axx.set_xlim(-0.5, 0.5)
        axx.set_ylim(-sqrt(3)/3, sqrt(3)/3)
        axx.set_zlim(0, 10000)
        axx.view_init(10, -110)
        # axx.view_init(90, -90)
        
        pass

    triang = tri.Triangulation(x, y)
    con = [ax[i].tricontour(triang, z[i], 10, zdir='z',linewidths=0.5, offset=0, cmap='magma',vmax=10000,vmin=0,zorder=-100) for i in range(5)]
    
    [ax[i].plot([-0.5,0.5], [-sqrt(3)/6, -sqrt(3)/6], lw=1, color=(0.51, 0.65, 0.55), zorder=14.1+i) for i in range(5)]
    [ax[i].plot([0.5,0.0], [-sqrt(3)/6, sqrt(3)/3], lw=1, color=(0.38, 0.53, 0.77), zorder=14.1+i) for i in range(5)]
    [ax[i].plot([0,-0.5], [sqrt(3)/3,-sqrt(3)/6], lw=1, color=(0.70,0.47,0.46), zorder=14.1+i) for i in range(5)]

    [ax[i].scatter(np.array([-0.5]),np.array([-sqrt(3)/6]),np.zeros(1),color=(0.70,0.47,0.46),zorder=200+i) for i in range(5)]
    [ax[i].scatter(np.array([0.5]),np.array([-sqrt(3)/6]),np.zeros(1),color=(0.51,0.65,0.55),zorder=200+i) for i in range(5)]
    [ax[i].scatter(np.array([0]),np.array([sqrt(3)/3]),np.zeros(1),color=(0.38,0.53,0.77),zorder=200+i) for i in range(5)]

    temp_y = 0.1/2*sqrt(3)
    temp_x = temp_y/sqrt(3)+0.25
    temp_x = temp_x - 0.5
    temp_y = temp_y - sqrt(3)/6
    ax[0].scatter([temp_x],[temp_y],[0],color='gold')
    ax[0].text(-0.7,-sqrt(3)/6-0.15,0,r'$\mathbf{\Theta}^\mathrm{m}$',color=(0.70,0.47,0.46))
    ax[0].text(0.55,-sqrt(3)/6-0.05,0,r'$\mathbf{\Theta}^\mathrm{v}$',color=(0.51,0.65,0.55))
    ax[0].text(-0.1,sqrt(3)/3+0.05,0,r'$\mathbf{\Theta}^\mathrm{f}$',color=(0.38,0.53,0.77))
    ax[0].text(temp_x+0.05,temp_y+0.05,0,r'$\mathbf{\Theta}^l$',color='gold')

    surf = [ax[i].plot_trisurf(x,y,z[i],lw=0,cmap='magma',alpha=1,antialiased=True,vmax=10000,vmin=0,zorder=2000+i) for i in range(5)]
    fig_handler.colorbar(surf[0],ax=ax[0],shrink=0.6)

    # plot xticks:
    ax[0].text(-0.25*1.2,sqrt(3)/12*1.2,0,r"$\beta$",(0.5,sqrt(3)/2,0),color=(0.70,0.47,0.46),horizontalalignment='center')
    ax[0].text(0.25*1.2,sqrt(3)/12*1.2,0,r"$\alpha$",(0.5,-sqrt(3)/2,0),color=(0.38,0.53,0.77),horizontalalignment='center')
    ax[0].text(0*1.2,-sqrt(3)/6*2.0,0,r"$\gamma$",(-1,0,0),color=(0.51,0.65,0.55),horizontalalignment='center')
    

    #endregion

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

    fig = plt.figure(figsize=(18.28/2.54, 20.32/2.54),dpi=600, constrained_layout=False)
    gs = fig.add_gridspec(4, 1,height_ratios=[1.5, 0.5, 1, 0.8], hspace=0.25)
    # gs = fig.add_gridspec(4, 1,height_ratios=[1.9, 0.1, 1, 0.8])
    g_perf = gs[2].subgridspec(1, 3, wspace=0.25)     # for plot the total performance of controller
    g_joint = gs[3].subgridspec(1, 4, wspace=0.25)    # plot hip or knee joint
    # g_reward = gs[2].subgridspec(1, 5)   # plot the reward function

    plot_performance(fig, g_perf)
    plot_joint_space(fig, g_joint)
    # plot_reward(fig, gs[0])
    # plot_reward3d(fig, gs[0])

    # add legend
    label = ["Reference",
             "Controller "+r'$\mathbf{\Theta}^\mathrm{f}$',
             "Controller "+r'$\mathbf{\Theta}^\mathrm{m}$',
             "Controller "+r'$\mathbf{\Theta}^\mathrm{v}$',
             "Controller "+r'$\mathbf{\Theta}^l$']
    color = ['dimgray','C0','C3','C1','gold']
    legend_elements = [Line2D([0],[0],color=color[i],lw=2,label=label[i]) for i in range(5)]

    l4 = fig.legend(bbox_to_anchor=(0.12,0.47,0.78,0.1), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=5, handles=legend_elements)

    # plt.tight_layout()
    # plt.draw()

    fig.savefig(os.getcwd()+'/Fig2.pdf',dpi=600,transparent=True)

    pass

def main_reward():
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
    fig2 = plt.figure(figsize=(18.28/2.54, 20.32/2.54),dpi=600, constrained_layout=False)
    gs = fig2.add_gridspec(4, 1,height_ratios=[1.9, 0.1, 1, 0.8])
    plot_reward3d(fig2, gs[0])
    fig2.savefig(os.getcwd()+'/Fig2_reward.pdf',dpi=600,transparent=True)
    pass

if __name__ == "__main__":
    main()
    # main_reward()
    pass
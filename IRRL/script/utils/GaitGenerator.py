"""
this define GaitGenerator object
which is used to receive gamepad command and produce reference joint angle for black panther
"""
import numpy as np
from math import sqrt, sin, acos, asin

L_HIP = 0.085
L_THIGH = 0.209
L_CALF = 0.2175


class GaitGenerator(object):
    """
    this class is used to produce the reference joint of BlackPanther robot
    """

    def __init__(self, config, command_filter_freq=10, phase=np.array([0.5, 0.0, 0.0, 0.5])):
        """
        param config, cfg file, load control dt, period and other information
        """
        self._current_time = 0.0  # current time
        self.time_reset()
        self._dt = config['environment']['control_dt']  # time step
        self._filter = 1.0 - command_filter_freq * self._dt  # command filter parameter
        self._period = config['environment']['period']  # gait period
        self._phase = phase  # default phase of four leg
        self._lam = config['environment']['lam']  # ratio of leg touch the ground during one phase

        self._command = np.zeros(3)  # store the command

        # ------------------------------------------------
        # gait information update
        self._Vx_max = config['environment']['Vx']  # max forward speed
        self._Vy_max = config['environment']['Vy']  # max lateral speed
        self._OmegaZ_max = config['environment']['Omega']  # max rotation speed
        self._down_height = config['environment']['down_height']  # distance of the toe to push the ground
        self._stand_height = config['environment']['stand_height']  # stand height of robot
        self._up_height = config['environment']['up_height']  # toe left height
        self._up_height_max = self._up_height
        self._Lean_middle = config['environment']['Lean_middle']  # distance of toe move to the center

        # ------------------------------------------------
        # load flag
        self._flag_HeightVariable = config['environment']['HeightVariable']

        # ------------------------------------------------
        # define robot physics size
        self._l_hip = L_HIP  # for now, this is fixed
        self._l_thigh = L_THIGH
        self._l_calf = L_CALF
        # ------------------------------------------------
        # main interface data
        self._joint = np.zeros((4, 3))  # fr, fl, hr, hl
        self._end_effector = np.zeros((4, 3))  # fr, fl, hr, hl
        pass

    def time_reset(self):
        # reset timer, prepare for new gait
        self._current_time = 0
        pass

    def update_command(self, gamepad):
        """
        param gamepad is a xbox object
        update time here
        """
        self._current_time += self._dt
        # ---------------------------
        # update command
        # temp = gamepad.axis_l.y if gamepad.axis_l.y < 0 else 0
        temp = gamepad.axis_l.y
        self._command[0] = self._command[0] * self._filter + \
                           (1 - self._filter) * (-1 * temp) * self._Vx_max
        self._command[1] = self._command[1] * self._filter + \
                           (1 - self._filter) * -1 * gamepad.axis_l.x * self._Vy_max
        self._command[2] = self._command[2] * self._filter + \
                           (1 - self._filter) * gamepad.axis_r.x * -1 * self._OmegaZ_max
        # print("command: ", self._command)
        # ----------------------------
        # update toe's position
        gait_step = self._command[0] * self._period  # step along forward
        side_step = self._command[1] * self._period  # step along lateral direction
        # rot_step = self._command[2] * self._period * 0.3  # rotation step, omega*radius
        rot_step = self._command[2] * self._period * 0.4

        for i in range(4):
            if self._flag_HeightVariable:
                # ratio = np.linalg.norm(self._command) / \
                #         sqrt((self._Vx_max * self._Vx_max +
                #               self._Vy_max * self._Vy_max +
                #               self._OmegaZ_max * self._OmegaZ_max))
                ratio = max(abs(self._command[0]) / self._Vx_max,
                            abs(self._command[1]) / self._Vy_max)
                ratio = max(ratio,
                            abs(self._command[2]) / self._OmegaZ_max)
                self._up_height = ratio * self._up_height_max if ratio < 0.3 else self._up_height_max
                pass
            real_phase = self._current_time + self._phase[i] * self._period
            real_phase = (real_phase % self._period) / self._period
            anti_flag = 1.0 if i < 2 else -1.0
            if real_phase < self._lam:
                temp_r = real_phase / self._lam
                p0 = np.array([gait_step / 2.0, side_step / 2.0 + anti_flag * rot_step / 2.0, -self._stand_height])
                pf = np.array([-gait_step / 2.0, -side_step / 2.0 + -anti_flag * rot_step / 2.0, -self._stand_height])
                toe = GaitGenerator.cubicBezier(p0, pf, temp_r)
                self._end_effector[i, 0] = toe[0]
                self._end_effector[i, 1] = toe[1]
                self._end_effector[i, 2] = toe[2]
                # self._end_effector[i, 0] = (gait_step / 2.0) * (1.0 - real_phase / self._lam) + \
                #                            (-gait_step / 2.0) * (real_phase / self._lam)
                # self._end_effector[i, 1] = (side_step / 2.0) * (1.0 - real_phase / self._lam) + \
                #                            (-side_step / 2.0) * (real_phase / self._lam) + \
                #                            (anti_flag * rot_step / 2.0) * (1.0 - real_phase / self._lam) + \
                #                            (-anti_flag * rot_step / 2.0) * (real_phase / self._lam)
                # # self._end_effector[i, 2] = -sin(real_phase / self._lam * np.pi) * self._down_height - \
                # #                            self._stand_height
                # self._end_effector[i, 2] = -GaitGenerator.gauss(real_phase / self._lam, 1.0, self._down_height) - \
                #                            self._stand_height
                pass
            else:
                temp_r = (real_phase - self._lam) / (1.0 - self._lam)
                if temp_r < 0.5:
                    p0 = np.array([-gait_step / 2.0,
                                   -side_step / 2.0 + -anti_flag * rot_step / 2.0,
                                   -self._stand_height])
                    pf = np.array([0, 0, self._up_height - self._stand_height])
                    toe = GaitGenerator.cubicBezier(p0, pf, temp_r / 0.5)
                    pass
                else:
                    p0 = np.array([0, 0, self._up_height - self._stand_height])
                    pf = np.array([gait_step / 2.0,
                                   side_step / 2.0 + anti_flag * rot_step / 2.0,
                                   -self._stand_height])
                    toe = GaitGenerator.cubicBezier(p0, pf, (temp_r - 0.5) / 0.5)
                    pass
                self._end_effector[i, 0] = toe[0]
                self._end_effector[i, 1] = toe[1]
                self._end_effector[i, 2] = toe[2]
                # temp = real_phase - self._lam
                # self._end_effector[i, 0] = (-gait_step / 2.0) * (1.0 - temp / (1.0 - self._lam)) + \
                #                            (gait_step / 2.0) * temp / (1.0 - self._lam)
                # self._end_effector[i, 1] = (-side_step / 2.0) * (1.0 - temp / (1.0 - self._lam)) + \
                #                            (side_step / 2.0) * temp / (1.0 - self._lam) + \
                #                            (-anti_flag * rot_step / 2.0) * (1.0 - temp / (1.0 - self._lam)) + \
                #                            (anti_flag * rot_step / 2.0) * temp / (1.0 - self._lam)
                # # self._end_effector[i, 2] = sin(temp / (1.0 - self._lam) * np.pi) * self._up_height - \
                # #                            self._stand_height
                # self._end_effector[i, 2] = GaitGenerator.gauss(temp / (1.0 - self._lam), 1.0, self._up_height) - \
                #                            self._stand_height
                pass
            if i == 0 or i == 2:
                self._end_effector[i, 1] = self._end_effector[i, 1] - self._l_hip + self._Lean_middle
                pass
            else:
                self._end_effector[i, 1] = self._end_effector[i, 1] + self._l_hip - self._Lean_middle
                pass
            pass
        # print("end:", self._end_effector)
        pass

    def cal_ref(self):
        for i in range(4):
            temp = GaitGenerator.ik(self._end_effector[i, 0], self._end_effector[i, 1], self._end_effector[i, 2],
                                    self._l_hip, self._l_thigh, self._l_calf, (i % 2 == 0))
            self._joint[i, 0] = temp[0]
            self._joint[i, 1] = -temp[1]
            self._joint[i, 2] = -temp[2]
            pass
        return self._joint.copy()

    def update_and_return_angle(self, gamepad):
        self.update_command(gamepad)
        self.cal_ref()
        return self._joint.reshape(-1).copy()

    def update_command2(self, command):
        self._current_time += self._dt
        # self._command = command.copy()
        self._command[0] = self._command[0] * self._filter + \
                           (1 - self._filter) * command[0]
        self._command[1] = self._command[1] * self._filter + \
                           (1 - self._filter) * command[1]
        self._command[2] = self._command[2] * self._filter + \
                           (1 - self._filter) * command[2]
        # print(self._command[0])
        # ----------------------------
        # update toe's position
        gait_step = self._command[0] * self._period  # step along forward
        side_step = self._command[1] * self._period  # step along lateral direction
        rot_step = self._command[2] * self._period * 0.4  # rotation step, omega*radius

        for i in range(4):
            real_phase = self._current_time + self._phase[i] * self._period
            real_phase = (real_phase % self._period) / self._period
            anti_flag = 1.0 if i < 2 else -1.0
            if real_phase < self._lam:
                temp_r = real_phase / self._lam
                p0 = np.array([gait_step / 2.0, side_step / 2.0 + anti_flag * rot_step / 2.0, -self._stand_height])
                pf = np.array([-gait_step / 2.0, -side_step / 2.0 + -anti_flag * rot_step / 2.0, -self._stand_height])
                toe = GaitGenerator.cubicBezier(p0, pf, temp_r)
                self._end_effector[i, 0] = toe[0]
                self._end_effector[i, 1] = toe[1]
                self._end_effector[i, 2] = toe[2]
                # self._end_effector[i, 0] = (gait_step / 2.0) * (1.0 - real_phase / self._lam) + \
                #                            (-gait_step / 2.0) * (real_phase / self._lam)
                # self._end_effector[i, 1] = (side_step / 2.0) * (1.0 - real_phase / self._lam) + \
                #                            (-side_step / 2.0) * (real_phase / self._lam) + \
                #                            (anti_flag * rot_step / 2.0) * (1.0 - real_phase / self._lam) + \
                #                            (-anti_flag * rot_step / 2.0) * (real_phase / self._lam)
                # # self._end_effector[i, 2] = -sin(real_phase / self._lam * np.pi) * self._down_height - \
                # #                            self._stand_height
                # self._end_effector[i, 2] = -GaitGenerator.gauss(real_phase / self._lam, 1.0, self._down_height) - \
                #                            self._stand_height
                pass
            else:
                temp_r = (real_phase - self._lam) / (1.0 - self._lam)
                if temp_r < 0.5:
                    p0 = np.array([-gait_step / 2.0,
                                   -side_step / 2.0 + -anti_flag * rot_step / 2.0,
                                   -self._stand_height])
                    pf = np.array([0, 0, self._up_height - self._stand_height])
                    toe = GaitGenerator.cubicBezier(p0, pf, temp_r / 0.5)
                    pass
                else:
                    p0 = np.array([0, 0, self._up_height - self._stand_height])
                    pf = np.array([gait_step / 2.0,
                                   side_step / 2.0 + anti_flag * rot_step / 2.0,
                                   -self._stand_height])
                    toe = GaitGenerator.cubicBezier(p0, pf, (temp_r - 0.5) / 0.5)
                    pass
                self._end_effector[i, 0] = toe[0]
                self._end_effector[i, 1] = toe[1]
                self._end_effector[i, 2] = toe[2]
                # temp = real_phase - self._lam
                # self._end_effector[i, 0] = (-gait_step / 2.0) * (1.0 - temp / (1.0 - self._lam)) + \
                #                            (gait_step / 2.0) * temp / (1.0 - self._lam)
                # self._end_effector[i, 1] = (-side_step / 2.0) * (1.0 - temp / (1.0 - self._lam)) + \
                #                            (side_step / 2.0) * temp / (1.0 - self._lam) + \
                #                            (-anti_flag * rot_step / 2.0) * (1.0 - temp / (1.0 - self._lam)) + \
                #                            (anti_flag * rot_step / 2.0) * temp / (1.0 - self._lam)
                # # self._end_effector[i, 2] = sin(temp / (1.0 - self._lam) * np.pi) * self._up_height - \
                # #                            self._stand_height
                # self._end_effector[i, 2] = GaitGenerator.gauss(temp / (1.0 - self._lam), 1.0, self._up_height) - \
                #                            self._stand_height
                pass
            if i == 0 or i == 2:
                self._end_effector[i, 1] = self._end_effector[i, 1] - self._l_hip + self._Lean_middle
                pass
            else:
                self._end_effector[i, 1] = self._end_effector[i, 1] + self._l_hip - self._Lean_middle
                pass
            pass
        pass

    def update_and_return_angle2(self, command):
        self.update_command2(command)
        self.cal_ref()
        return self._joint.reshape(-1).copy()

    def get_end_effector(self):
        return self._end_effector.reshape(-1).copy()

    def get_command(self):
        return self._command.copy()

    @staticmethod
    def ik(x, y, z, l_hip, l_thigh, l_calf, is_right):
        """
        calculate inverse kinematics of black panther
        """
        theta = np.zeros(3)
        max_len = sqrt(l_hip * l_hip + (l_thigh + l_calf) * (l_thigh + l_calf))
        temp, temp1, temp2 = 0, 0, 0
        if is_right:
            temp = (-z * l_hip - sqrt(y * y * (z * z + y * y - l_hip * l_hip))) / (z * z + y * y)
            if abs(temp) <= 1:
                theta[0] = asin(temp)
            else:
                # print("error1")
                pass
            pass
        else:
            temp = (z * l_hip + sqrt(y * y * (z * z + y * y - l_hip * l_hip))) / (z * z + y * y)
            if abs(temp) <= 1:
                theta[0] = asin(temp)
            else:
                # print("error1")
                pass
            pass
        lr = sqrt(x * x + y * y + z * z - l_hip * l_hip)
        lr = min(lr, l_calf + l_thigh)
        temp = (l_thigh * l_thigh + l_calf * l_calf - lr * lr) / 2 / l_thigh / l_calf + 1e-5
        if abs(temp) <= 1:
            theta[2] = -(np.pi - acos(temp))
        else:
            # print("error2")
            pass
        temp1 = x / sqrt(y * y + z * z) - 1e-10;
        temp2 = (lr * lr + l_thigh * l_thigh - l_calf * l_calf) / 2 / lr / l_thigh - 1e-5;
        if abs(temp1) <= 1 and abs(temp2) <= 1:
            theta[1] = acos(temp2) - asin(temp1)
        else:
            # print("error3")
            pass
        return theta.copy()

    @staticmethod
    def gauss(x, width, height):
        return height * np.exp(-(x - width / 2) * (x - width / 2) / (2 * (width / 6) * (width / 6)))

    @staticmethod
    def cubicBezier(p0, pf, phase):
        pDiff = pf - p0
        bezier = phase * phase * phase + 3 * (phase * phase * (1 - phase))
        return p0 + bezier * pDiff

    @staticmethod
    def kinematic(theta_abad, theta_hip, theta_knee, l_hip=L_HIP, l_thigh=L_THIGH, l_calf=L_CALF, is_right=False):
        l_hip = l_hip if is_right else -l_hip
        theta_hip = theta_hip
        theta_knee = theta_knee
        x = -l_thigh * np.sin(theta_hip) - l_calf * np.sin(theta_hip + theta_knee)
        y = -l_hip * np.cos(theta_abad) + \
            l_thigh * np.sin(theta_abad) * np.cos(theta_hip) + \
            l_calf * np.sin(theta_abad) * np.cos(theta_hip - theta_knee)
        z = -l_hip * np.sin(theta_abad) - \
            l_thigh * np.cos(theta_abad) * np.cos(theta_hip) - \
            l_calf * np.cos(theta_abad) * np.cos(theta_hip + theta_knee)
        return x, y, z


def main():
    from xbox360controller import Xbox360Controller
    from ruamel.yaml import YAML

    cfg_abs_path = './../config/bp3_test.yaml'  # config file
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    gg = GaitGenerator(cfg)
    xbox = Xbox360Controller(0, axis_threshold=0.2)
    gg.update_and_return_angle(xbox)
    pass


def main1():
    # test the filter
    from ruamel.yaml import YAML
    import matplotlib.pyplot as plt
    cfg_abs_path = './../config/bp3_test.yaml'  # config file
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    command = np.array([0.5, 0, 0])
    gg = GaitGenerator(cfg)

    angle = []
    angle_filter = []
    angle_filter2 = []

    N = 100
    for i in range(N):
        gg.update_command2(command)
        angle.append(gg.cal_ref().reshape(-1))
        if i == 0:
            angle_filter.append(angle[0])
            angle_filter2.append(angle[0])
            pass
        else:
            angle_filter.append(0.9 * angle_filter[i - 1] + 0.1 * angle[i])
            angle_filter2.append(0.75 * angle_filter2[i - 1] + 0.25 * angle[i])
            pass
        pass

    plt.style.use("seaborn-deep")
    fig, axs = plt.subplots(1, 3)
    angle = np.asarray(angle)
    angle_filter = np.asarray(angle_filter)
    angle_filter2 = np.asarray(angle_filter2)
    time = np.linspace(0, N, N) * 0.01
    axs[0].plot(time, angle[:, 3], c='C0', lw=3)
    axs[1].plot(time, angle[:, 4], c='C1', lw=3)
    axs[2].plot(time, angle[:, 5], c='C2', lw=3, label='designed')
    axs[0].plot(time, angle_filter2[:, 3], c='C0', lw=3, linestyle=':')
    axs[1].plot(time, angle_filter2[:, 4], c='C1', lw=3, linestyle=':')
    axs[2].plot(time, angle_filter2[:, 5], c='C2', lw=3, linestyle=':', label='25Hz')
    axs[0].plot(time, angle_filter[:, 3], c='C0', lw=3, linestyle='--')
    axs[1].plot(time, angle_filter[:, 4], c='C1', lw=3, linestyle='--')
    axs[2].plot(time, angle_filter[:, 5], c='C2', lw=3, linestyle='--', label='10Hz')
    plt.legend()
    plt.show()
    pass


def main2():
    from ruamel.yaml import YAML
    import matplotlib.pyplot as plt

    cfg_abs_path = './../config/bp4_test.yaml'  # config file
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    command = np.array([0, 0, 0])
    gg = GaitGenerator(cfg)

    ee_traj = []
    theta_ref = []
    ee_fake = []
    for i in range(50):
        theta_ref.append(gg.update_and_return_angle2(command))
        ee_traj.append(gg.get_end_effector())
        pass

    plt.style.use("seaborn-deep")
    fig, ax = plt.subplots()
    ee_traj = np.asarray(ee_traj)
    theta_ref = np.asarray(theta_ref)
    x, y, z = GaitGenerator.kinematic(theta_ref[:, 0], theta_ref[:, 1], theta_ref[:, 2])
    ax.plot(ee_traj[:, 0], ee_traj[:, 2])
    ax.plot(x, z)
    plt.show()
    pass


if __name__ == "__main__":
    main2()
    pass

"""
this script is used to control
control the black panther
direct put the reference joint trajectory into the observation
add contact
"""
from ruamel.yaml import YAML, dump, RoundTripDumper
from flex_gym.env.RaisimGymVecEnv import RaisimGymVecEnv as Environment
from flex_gym.env.env.BlackPanther_V55 import __BLACKPANTHER_V55_RESOURCE_DIRECTORY__ as __RSCDIR__
from flex_gym.algo.ppo2 import PPO2
from flex_gym.archi.policies import ActorCriticPolicy, LstmPolicy, MlpPolicy
from flex_gym.helper.raisim_gym_helper import ConfigurationSaver, TensorboardLauncher
from _flexible_robot import FlexibleGymEnv

import os
import sys
import math
import numpy as np
import argparse
import stable_baselines
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
import tensorflow as tf
from termcolor import colored
from math import floor

arg_parser = None


def parse_args(args):
    """
    prepare parameter
    """
    parser = argparse.ArgumentParser(
        description="Train or test control policies.")
    # train or test model
    parser.add_argument("--train", dest="train", action="store_true", default=True,
                        help="true, train model. false, test model")
    parser.add_argument("--test", dest="train",
                        action="store_false", default=True)
    parser.add_argument('--cfg', type=str,
                        default=os.path.abspath(
                            __RSCDIR__ + "/default_cfg.yaml"),
                        help='configuration file')
    parser.add_argument("--max_iter", dest="max_iter", type=int, default=200000000,
                        help='max iteration for model training, for train mode')
    parser.add_argument("--save", dest="save_flag", type=bool, default=True,
                        help="true for save trained model and other files")
    parser.add_argument("--l", dest="learn_rate", type=float, default=1e-3,
                        help="learning rate, default is 1e-3")
    parser.add_argument("--sens", dest="sens_flag", action="store_true", default=False,
                        help="open this will take sensitivity analysis during test")
    parser.add_argument("--model", dest="trained_model", type=str, default=None,
                        help="file path of the trained model, looks like *.pkl, can't be ignored")
    parser.add_argument("--cmd_traj", dest="flag_cmd_visual", action="store_true", default=False,
                        help="whether visual the cmd data and real velocity")
    parser.add_argument("--load", dest="pre_trained_model", type=str, default=None,
                        help="load pre-trained model")
    parser.add_argument("--eval", dest="flag_eval", action="store_true", default=False,
                        help="evaluate the performance of network controller")
    parser.add_argument("--joint_vis", dest="flag_joint", action="store_true", default=False,
                        help="visualize the joint state of leg0's knee joint")
    parser.add_argument("--leg", dest="leg_index", type=int, default=0,
                        help="choose which leg's data to be visualized, from  0 ~ 3")
    parser.add_argument("--state_space", dest="flag_state_space", action="store_true", default=False,
                        help="visualize the state space of each joints")
    parser.add_argument("--ss", dest="flag_state_space2", action="store_true", default=False,
                        help="visualize the state space of each joints with error curve")
    parser.add_argument("--ee", dest="flag_end_effector", action="store_true", default=False,
                        help="visualize the end effector trajectory")
    # parser.add_argument("--fix_cmd", dest="flag_fix_cmd", action="store_true", default=False,
    #                     help="fix command during test mode")
    parser.add_argument("--fix_cmd", dest="flag_fix_cmd", type=float, default=None,
                        help="fix speed command")
    parser.add_argument("--step", dest="flag_step", action="store_true", default=False,
                        help="Stepped command")
    parser.add_argument("--val", dest="flag_value", type=int, default=0,
                        help="value function visualization and stability analysis")
    parser.add_argument("--o", dest="flag_output", action="store_true", default=False,
                        help="output the control model parameter to csv file")
    parser.add_argument("--torque", dest="flag_torque", action="store_true", default=False,
                        help="visualize the torque of joint")
    parser.add_argument("--wc", dest="flag_work_condition", action="store_true", default=False,
                        help="visualize motor's work condition")
    parser.add_argument("--vel_filter", dest="vel_filter_freq", type=int, default=10000,
                        help="apply filter to velocity")
    parser.add_argument("--act_filter", dest="act_filter_freq", type=int, default=10000,
                        help="apply filter to action")
    parser.add_argument("--delay", type=int, default=1,
                        help="apply filter to action")
    parser.add_argument("--corr", dest="flag_corr_analysis", action="store_true", default=False,
                        help="take correlation analysis")
    parser.add_argument("--virba", dest="flag_virbation_analysis", action="store_true", default=False,
                        help="perform time-frequency spectrum analysis of output")
    parser.add_argument("--dir_fd", dest="flag_direction_feedback", action="store_true", default=False,
                        help="add PID control of the command to realize robot keep forward")
    parser.add_argument("--save_data", type=str, default='NoSave',
                        help="save cmd and origin state data for dynamics analysis")
    parser.add_argument("--save_energy_data", type=str, default='NoSave',
                        help="save dynamics data for energy flow analysis")
    parser.add_argument("--ref", action="store_true", default=False,
                        help="get and save the reference joint trajectory from environment")
    parser.add_argument("--vid", action="store_true", default=False,
                        help="start record video")
    parser.add_argument("--vnote", type=str, default='',
                        help='note for video which will be added into the video file name')

    arg_parser = parser.parse_args()
    return arg_parser


N_LSTM = [48, 48]


# N_LSTM = [48, 48, 48, 48]


class CustomLSTMPolicy(LstmPolicy, ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=[32, 32], reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, layer_norm=False, feature_extraction="mlp",
                 **kwargs):
        # super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
        #                                         scale=(feature_extraction == "cnn"))
        ActorCriticPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                   scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("input", reuse=True):
            self.masks_ph = tf.placeholder(
                tf.float32, [n_batch], name="masks_ph")  # mask (done t-1)
            # n_lstm * 2 dim because of the cell and hidden states of the LSTM
            # self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2], name="states_ph")  # states
            # self.states_ph = [tf.placeholder(tf.float32, [self.n_env, n_lstm * 2],
            #                                  name="states_ph{}".format(i))
            #                   for i in range(len(n_lstm))]
            self.states_ph = tf.placeholder(
                tf.float32, [self.n_env, sum(n_lstm) * 2 * 2], name="states_ph")
            # create split number
            size_splits = [k * 2 for k in (n_lstm + n_lstm)]
            hidden_cell_collection = tf.split(self.states_ph, size_splits, 1)
            # put value's hidden and cell and policy's hidden and cell into one states_ph

        with tf.variable_scope("model", reuse=reuse):
            latent_pi = tf.layers.flatten(self.processed_obs)
            latent_v = tf.layers.flatten(self.processed_obs)

            self.snew_pi = []
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            for idx, lstm_num in enumerate(n_lstm):
                input_sequence = batch_to_seq(latent_pi, self.n_env, n_steps)
                rnn_output, temp = lstm(input_sequence, masks, hidden_cell_collection[idx],
                                        'lstm_pi{}'.format(idx), n_hidden=lstm_num,
                                        layer_norm=layer_norm)
                self.snew_pi.append(temp)
                latent_pi = seq_to_batch(rnn_output)
                pass

            self.snew_v = []
            for idx, lstm_num in enumerate(n_lstm):
                input_sequence = batch_to_seq(latent_v, self.n_env, n_steps)
                rnn_output, temp = lstm(input_sequence, masks, hidden_cell_collection[idx + len(n_lstm)],
                                        'lstm_v{}'.format(idx), n_hidden=lstm_num,
                                        layer_norm=layer_norm)
                self.snew_v.append(temp)
                latent_v = seq_to_batch(rnn_output)
                pass

            self.value_fn = linear(latent_v, 'vf', 1)
            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(latent_pi, latent_v)

        self.snew = tf.concat([s for s in self.snew_pi] +
                              [s for s in self.snew_v], axis=1)
        self.initial_state = np.zeros(
            (self.n_env, sum(n_lstm) * 2 * 2), dtype=np.float32)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self._value, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        else:
            return self.sess.run([self.action, self._value, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        pass

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    pass


def main(args):
    global arg_parser
    arg_parser = parse_args(args)

    # load config file-
    cfg_abs_path = arg_parser.cfg
    cfg = YAML().load(open(cfg_abs_path, 'r'))

    # create environment from the configuration file
    env = Environment(FlexibleGymEnv(__RSCDIR__,
                                     dump(cfg['environment'],
                                          Dumper=RoundTripDumper)))

    if arg_parser.train:
        # whether to save the model
        if arg_parser.save_flag:
            rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../'
            log_dir = rsg_root + '/data'
            saver = ConfigurationSaver(log_dir=log_dir + '/black_panther_v5_test',
                                       save_items=[rsg_root + 'FlexibleRobotRaisimGym/flex_gym/env/env/BlackPanther_V55/Environment.hpp',
                                                   cfg_abs_path])

            pass
        else:
            saver = None
            pass

        log_path = saver.data_dir if arg_parser.save_flag else None

        log_path = saver.data_dir if arg_parser.save_flag else None
        if arg_parser.pre_trained_model is None:
            model = PPO2(tensorboard_log=log_path,
                         policy=CustomLSTMPolicy,
                         policy_kwargs=dict(n_lstm=N_LSTM),
                         env=env,
                         gamma=0.99,
                         n_steps=math.floor(
                             cfg['environment']['max_time'] / cfg['environment']['control_dt']),
                         ent_coef=0.000,
                         learning_rate=arg_parser.learn_rate,
                         vf_coef=0.5,
                         max_grad_norm=0.5,
                         lam=0.998,
                         nminibatches=1,
                         noptepochs=10,
                         cliprange=0.2,
                         verbose=1)
            pass
        else:
            model = PPO2.load(arg_parser.pre_trained_model)
            model.env = env
            model.tensorboard_log = log_path
            model.learning_rate = arg_parser.learn_rate
            pass
        if arg_parser.save_flag:
            TensorboardLauncher(saver.data_dir + '/PPO2_1')
            pass

        model.learn(total_timesteps=arg_parser.max_iter, eval_every_n=100,
                    log_dir=saver.data_dir, record_video=cfg['record_video'])
        # Need this line if you want to keep tensorflow alive after training
        input("Press Enter to exit... Tensorboard will be closed after exit\n")

        pass

    else:
        print('*' * 50)
        print('Make sure the Manual flag is opened!!!')
        print('Connect xbox gamepad with computer!!!')
        print('*' * 50)
        print('\n')
        if arg_parser.trained_model is None:
            print("model path can't be ignored during test mode")
            print("run -h to help")
            sys.exit()

        # import package used in model test
        from xbox360controller import Xbox360Controller
        from itertools import count
        from utils.GaitGenerator import GaitGenerator
        from utils.CustomerLstmNN import CustomerLstmNN
        from utils.GaitColorBar import GaitBar
        from utils.DelayTool import DelayTool
        from bp5_config import obs_mean, obs_std, action_mean, action_std
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, BoundaryNorm

        plt.style.use("seaborn-deep")
        gg = GaitGenerator(cfg,
                           command_filter_freq=1.0)  # GaitGenerator, receive data from gamepad and return reference joint angle
        # model = PPO2.load(arg_parser.trained_model)  # load model
        ctrl = CustomerLstmNN(arg_parser.trained_model,
                              flag_v=False, n_lstm=N_LSTM)

        if arg_parser.flag_output:
            # ctrl.save_model('/model/bp5/')
            model_name = arg_parser.trained_model
            model_name = model_name.rstrip('.pkl')
            model_name = model_name.strip('./pkl/')
            path = os.getcwd() + '/model/' + model_name
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
                pass
            else:
                pass
            ctrl.save_model('/model/' + model_name + '/')
            pass

        xbox = Xbox360Controller(0, axis_threshold=0.02)

        if arg_parser.flag_direction_feedback:
            from utils.PID import PID
            from utils.Rotation import qua2euler
            pid = PID(P=1, I=0.1, D=0, current_time=0)
            pid.setSampleTime(cfg['environment']['control_dt'])
            pid.setWindup(cfg['environment']['Omega'])
            pid_current = 0
            pass

        env.SetContactCoefficient(
            np.array([[0.8, 0.2, 0.01]], dtype=np.float32))
        obs = env.reset()
        action_total = np.zeros([env.num_envs, env.num_acts], dtype=np.float32)

        if arg_parser.vid:
            # record video
            model_name = arg_parser.trained_model
            model_name = model_name.rstrip('.pkl')
            model_name = model_name.strip('./pkl/')
            env.start_recording_video(
                '../src/video/' + model_name + arg_parser.vnote + '.mp4')
            pass

        joint = []
        joint_dot = []
        joint_ref = []
        posture = []
        omega = []
        act = []
        phase = []
        oss = []  # origin state data
        contact_oss = []
        cmd = []
        origin_obs = []
        value = []
        nn_state = []
        joint_effort = []
        sphere_info = []
        generalized_effort = []
        nonlinear_effort = []
        inverse_mass = []

        # start simulation
        current_time = 0.0
        filter_vel_alpha = 2 * np.pi * cfg['environment']['control_dt'] * arg_parser.vel_filter_freq / (
            2 * np.pi * cfg['environment']['control_dt'] * arg_parser.vel_filter_freq + 1.0)
        filter_act_alpha = 2 * np.pi * cfg['environment']['control_dt'] * arg_parser.act_filter_freq / (
            2 * np.pi * cfg['environment']['control_dt'] * arg_parser.act_filter_freq + 1.0)

        vel_his = np.zeros(35)
        act_his = np.zeros(12)

        d_tool = DelayTool(cfg['environment']['control_dt'],
                           cfg['environment']['control_dt'] * arg_parser.delay)

        for t in count():
            # obs = obs[0, :]
            obs = d_tool.input_output(obs[0, :])
            # obs[17:29] = (1 - filter_vel_alpha) * vel_his + filter_vel_alpha * obs[17:29]
            # obs[17:29] = 0
            # obs[5:29] = (1 - filter_vel_alpha) * vel_his + filter_vel_alpha * obs[5:29]
            # obs[5:35] = 0
            obs[32:35] = (1 - filter_vel_alpha) * \
                vel_his[32:35] + filter_vel_alpha * obs[32:35]
            obs[17:29] = (1 - filter_vel_alpha) * \
                vel_his[17:29] + filter_vel_alpha * obs[17:29]
            vel_his = obs[0:35]

            # print(env.GetJointEffort())

            if arg_parser.flag_value:
                origin_obs.append(obs)
                pass
            if arg_parser.flag_fix_cmd is not None:
                # joint_ref.append(gg.update_and_return_angle2(np.array([0.5, 0, 0])))
                if arg_parser.flag_step and t < 100 * 50 * 4:
                    temp_vel_cmd = (t // (100 * 50) + 1) / \
                        4 * arg_parser.flag_fix_cmd
                    # print(temp_vel_cmd)
                    joint_ref.append(gg.update_and_return_angle2(
                        np.array([temp_vel_cmd, 0.0, 0])))
                    pass
                else:
                    joint_ref.append(gg.update_and_return_angle2(
                        np.array([arg_parser.flag_fix_cmd, 0.0, 0])))
                pass
            else:
                joint_ref.append(gg.update_and_return_angle(xbox))
                pass
            if cfg['environment']['Manual']:
                cmd.append(gg.get_command())
                # obs[0:3] = cmd[0:3]
                if arg_parser.flag_direction_feedback and len(oss) != 0:
                    pid_current += cfg['environment']['control_dt']
                    _, _, yaw = qua2euler(
                        oss[-1][3], oss[-1][4], oss[-1][5], oss[-1][6])
                    pid.update(yaw, current_time=pid_current)
                    cmd[-1][2] = pid.output
                    # print("yaw: ", yaw, " pid: ", cmd[-1][2])
                    pass
                obs[0:3] = cmd[-1]
                obs[0:3] = (obs[0:3] - obs_mean[0:3]) / obs_std[0:3]
                pass
            else:
                cmd.append(obs[0:3] * obs_std[0:3] + obs_mean[0:3])
                pass
            # obs[4] = (np.sin(current_time / 0.3 * 2 * 3.1415926) - 0.5) * 2
            # obs[5] = (np.cos(current_time / 0.3 * 2 * 3.1415926) - 0.5) * 2
            # current_time = current_time + 0.002
            env.wrapper.showWindow()
            action = ctrl.predict(obs)
            # action = ctrl.predict2(obs)
            action = (1 - filter_act_alpha) * \
                act_his + filter_act_alpha * action
            act_his = action
            # temp_obs = np.zeros((200, 35))
            # temp_obs[0, :] = obs
            # action = ctrl.predict2(temp_obs)[0, :]

            if arg_parser.flag_value:
                value.append(ctrl.get_v())
                pass
            action_total[0, :] = action
            ob_Double = obs * obs_std + obs_mean
            # print(ob_Double[29:32])
            # print(env.GetInverseMassMatrix())
            # print(env.GetGeneralizedForce())
            # print(env.GetNonlinear())
            joint.append(ob_Double[5:17])
            joint_dot.append(ob_Double[17:29])
            posture.append(ob_Double[29:32])
            omega.append(ob_Double[32:35])
            phase.append(ob_Double[3:5])
            act.append(action * action_std + action_mean)
            total_real_state = env.OriginState()[0, :]
            oss.append(total_real_state[0:37])
            # sphere_info.append(env.GetSphereInfo()[0])

            if arg_parser.save_energy_data != 'NoSave':
                # record data for energy flow analysis
                generalized_effort.append(env.GetGeneralizedForce()[0, :])
                nonlinear_effort.append(env.GetNonlinear()[0, :])
                inverse_mass.append(env.GetInverseMassMatrix()[0, :])
            pass

            # !! since now, this is wrong
            joint_effort.append(env.GetJointEffort()[0, :])
            # joint_effort.append(cfg['environment']['Stiffness'] * (act[-1] - joint[-1]) -
            #                     cfg['environment']['Damping'] * joint_dot[-1])
            contact_oss.append(total_real_state[37:41])
            if arg_parser.flag_corr_analysis:
                nn_state.append(np.hstack((obs, ctrl.get_hidden_state())))
                pass
            # oss.append(env.OriginState()[0, :])
            obs, reward, done, info = env.step(action_total, visualize=True)
            if xbox.button_start.is_pressed:
                xbox.close()
                if arg_parser.vid:
                    env.stop_recording_video()
                    pass
                break
                pass
            pass

        joint = np.asarray(joint)
        joint_dot = np.asarray(joint_dot)
        joint_ref = np.asarray(joint_ref)
        act = np.asarray(act)
        time = np.arange(joint.shape[0]) * cfg['environment']['control_dt']
        phase = np.asarray(phase)
        oss = np.asarray(oss)
        cmd = np.asarray(cmd)

        if arg_parser.save_data != 'NoSave':
            np.save('./data/' + arg_parser.save_data +
                    '.npy', np.hstack((cmd, oss)))
            np.save('./data/' + arg_parser.save_data +
                    '_sphere.npy', sphere_info)
            if arg_parser.ref:
                np.save('./data/' + arg_parser.save_data +
                        'ref' + '.npy', joint_ref)
                pass
            print("data save success")
            pass

        if arg_parser.save_energy_data != 'NoSave':
            model_name = arg_parser.trained_model
            model_name = model_name.rstrip('.pkl')
            model_name = model_name.strip('./pkl/')
            np.save('./data/' + arg_parser.save_energy_data + model_name + 'cmd.npy',
                    np.asarray(cmd))
            np.save('./data/' + arg_parser.save_energy_data + model_name + 'gen_force.npy',
                    np.asarray(generalized_effort))
            # np.save(
            #     './data/' + arg_parser.save_energy_data + model_name + 'non_linear.npy',
            #     np.asarray(nonlinear_effort))
            # np.save('./data/' + arg_parser.save_energy_data + model_name + 'inv_mass.npy',
            #         np.asarray(inverse_mass))
            np.save('./data/' + arg_parser.save_energy_data +
                    model_name + 'act.npy', act)
            np.save('./data/' + arg_parser.save_energy_data +
                    model_name + 'oss.npy', oss)
            print("energy data save success")
            pass

        if arg_parser.flag_value:
            origin_obs = np.asarray(origin_obs)
            value = np.asarray(value)
            pass
        # visualization

        # state space visualization
        if arg_parser.flag_state_space:
            # assert (joint.shape[0] < 500), "too little data"

            xlim_ss = [-0.5, 0.5, -1.6, 0, 1.4, 2.2]
            ylim_ss = [-7, 7, -8, 10, -10, 15]
            name = ['FR', 'FL', 'HR', 'HL']
            joint_name = ['Abad', 'Hip', 'Knee']
            offset = np.argmax(joint_ref[-500:10:-500 + 61, 2]) / 50
            fig_ss = plt.figure(figsize=(16, 9))
            axs_ss = fig_ss.subplots(3, 4)
            for i in range(3):
                for j in range(4):
                    points = np.array([joint[-500:, j * 3 + i],
                                       joint_dot[-500:, j * 3 + i]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap='autumn', norm=plt.Normalize(0, 1))
                    lc.set_array(((time[-500:] + offset) % 0.5) * 2)
                    lc.set_alpha(0.8)
                    lc.set_linewidth(1)
                    line = axs_ss[i, j].add_collection(lc)
                    points_ref = np.array([joint_ref[-500:-400, j * 3 + i][10:61],
                                           np.convolve(joint_ref[-500:-400, j * 3 + i], np.array([1, -1]), 'same')[
                                           10:61] / 0.01])
                    points_ref = points_ref.T.reshape(-1, 1, 2)
                    segments_ref = np.concatenate(
                        [points_ref[:-1], points_ref[1:]], axis=1)
                    lc_ref = LineCollection(
                        segments_ref, cmap='coolwarm', norm=plt.Normalize(0, 1))
                    lc_ref.set_array(
                        ((time[-500 + 10:-500 + 61] + offset) % 0.5) * 2)
                    # lc_ref.set_alpha(0.8)
                    lc_ref.set_linewidth(2)
                    lc_ref.set_linestyle('--')
                    axs_ss[i, j].add_collection(lc_ref)

                    if i == 2:
                        axs_ss[i, j].set_xlabel(
                            name[j] + " joint angle (rad)", fontname="Arial", fontsize=18)
                        axins = axs_ss[i, j].inset_axes([0.6, 0.6, 0.39, 0.39])
                        lc_ref = LineCollection(
                            segments_ref, cmap='coolwarm', norm=plt.Normalize(0, 1))
                        lc_ref.set_array(
                            ((time[-500 + 10:-500 + 61] + offset) % 0.5) * 2)
                        # lc_ref.set_alpha(0.8)
                        lc_ref.set_linewidth(2)
                        lc_ref.set_linestyle('--')
                        lc = LineCollection(
                            segments, cmap='autumn', norm=plt.Normalize(0, 1))
                        lc.set_array(((time[-500:] + offset) % 0.5) * 2)
                        lc.set_alpha(0.8)
                        lc.set_linewidth(1)
                        axins.add_collection(lc)
                        axins.add_collection(lc_ref)
                        axins.set_xlim(1.41, 1.6)
                        axins.set_ylim(-3, 3)
                        axins.set_xticklabels('')
                        axins.set_yticklabels('')
                        axs_ss[i, j].indicate_inset_zoom(axins)
                        pass
                    if j == 0:
                        axs_ss[i, j].set_ylabel(
                            joint_name[i], fontname="Arial", fontsize=18)
                        pass
                    axs_ss[i, j].set_xlim(xlim_ss[2 * i], xlim_ss[2 * i + 1])
                    axs_ss[i, j].set_ylim(ylim_ss[2 * i], ylim_ss[2 * i + 1])
                    pass
                pass
            cbaxes = fig_ss.add_axes([0.93, 0.115, 0.015, 0.77])
            cb = plt.colorbar(line, cax=cbaxes)
            fig_ss.text(0.05, 0.65, 'joint angular velocity (rad/s)', fontname="Arial", fontsize=18,
                        rotation='vertical')
            fig_ss.subplots_adjust(wspace=0.20, hspace=0.15)
            pass

        if arg_parser.flag_state_space2:
            # another Dependency
            from utils.ErrorBand import error_band
            from matplotlib.patches import PathPatch

            fig_ss2 = plt.figure(figsize=(16, 9))
            axs_ss2 = fig_ss2.subplots(3, 4)
            xlim_ss = [-0.5, 0.5, -2.5, 1.5, 0.5, 2.5]
            ylim_ss = [-7, 7, -20, 35, -30, 30]
            name = ['FR', 'FL', 'HR', 'HL']
            joint_name = ['Abad', 'Hip', 'Knee']
            points_per_period = int(
                cfg['environment']['period'] / cfg['environment']['control_dt'])
            num_traj = 40
            # temp_joint = (oss[:, 7:19])[-num_traj * points_per_period:].reshape([num_traj, points_per_period, 12])
            # temp_joint_dot = (oss[:, 25:37])[-num_traj * points_per_period:].reshape([num_traj, points_per_period, 12])
            temp_joint = (oss[:, 7:19])[-num_traj * points_per_period:]
            temp_joint_dot = (oss[:, 25:37])[-num_traj * points_per_period:]
            # np.save('/home/wooden/Desktop/temp/bp5_118_inertia_2k_oss_q.npy', temp_joint)
            # np.save('/home/wooden/Desktop/temp/bp5_118_inertia_2k_oss_dq.npy', temp_joint_dot)

            color = np.arctan2(phase[-num_traj * points_per_period:, 0],
                               phase[-num_traj * points_per_period:, 1])
            color = color % (2 * np.pi) / (2 * np.pi)

            for i in range(3):
                for j in range(4):
                    idx = i + j * 3
                    start = -num_traj * points_per_period
                    points = np.array(
                        [temp_joint[:, idx], temp_joint_dot[:, idx]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap='coolwarm', norm=plt.Normalize(0, 1))
                    lc.set_array(color)
                    lc.set_alpha(0.8)
                    lc.set_linewidth(1)
                    line = axs_ss2[i, j].add_collection(lc)
                    # points_ref = np.array([joint_ref[start:start + 100, idx][10:points_per_period + 11],
                    #                        np.convolve(joint_ref[start:start + 100, idx], np.array([1, -1]),
                    #                                    'same')[10:points_per_period + 11] / 0.01])
                    # points_ref = points_ref.T.reshape(-1, 1, 2)
                    # segments_ref = np.concatenate([points_ref[:-1], points_ref[1:]], axis=1)
                    # lc_ref = LineCollection(segments_ref, cmap='coolwarm', norm=plt.Normalize(0, 1))
                    # lc_ref.set_array(color[10:11 + points_per_period])
                    # lc_ref.set_linewidth(2)
                    # lc_ref.set_linestyle('--')
                    # axs_ss2[i, j].add_collection(lc_ref)
                    if i == 2:
                        axs_ss2[i, j].set_xlabel(
                            name[j] + " joint angle (rad)", fontname="Arial", fontsize=18)
                        pass
                    if j == 0:
                        axs_ss2[i, j].set_ylabel(
                            joint_name[i], fontname="Arial", fontsize=18)
                        pass
                    axs_ss2[i, j].set_xlim(xlim_ss[2 * i], xlim_ss[2 * i + 1])
                    axs_ss2[i, j].set_ylim(ylim_ss[2 * i], ylim_ss[2 * i + 1])
                    pass
                pass
            cbaxes = fig_ss2.add_axes([0.93, 0.115, 0.015, 0.77])
            cb = plt.colorbar(line, cax=cbaxes)
            fig_ss2.text(0.05, 0.65, 'joint angular velocity (rad/s)', fontname="Arial", fontsize=18,
                         rotation='vertical')
            fig_ss2.subplots_adjust(wspace=0.20, hspace=0.15)
            pass

        if arg_parser.flag_joint:
            joint_name = ["Abad", "Hip", "Knee"]
            data_len = oss.shape[0]
            points_per_period = int(
                cfg['environment']['period'] / cfg['environment']['control_dt'])
            tt = np.arange(0, data_len, int(points_per_period / 2)
                           ) * cfg['environment']['control_dt']
            fig_joint = plt.figure(figsize=(6, 8))
            ax_joint = fig_joint.subplots(3, 1, sharex=True)
            for i in range(3):
                for j in range(floor(tt.shape[0] / 2) - 1):
                    ax_joint[i].axvspan(
                        tt[j * 2], tt[2 * j + 1], facecolor='C3', alpha=0.3)
                    ax_joint[i].axvspan(
                        tt[2 * j + 1], tt[2 * j + 2], facecolor='C4', alpha=0.3)
                    pass
                ax_joint[i].plot(
                    time, oss[:, 7 + i + 3 * arg_parser.leg_index], lw=3, color='C0', label='Now')
                # ax_joint[i].plot(time, joint_ref[:, i + 3 * arg_parser.leg_index], lw=3, color='C2', label='Ref',
                #                  alpha=0.8)
                ax_joint[i].plot(time, act[:, i + 3 * arg_parser.leg_index],
                                 lw=3, color='C1', label='Cmd', alpha=0.6)
                ax_joint[i].set_ylabel(joint_name[i] + " (rad)")
                ax_joint[i].legend(loc=3)
                pass
            ax_joint[2].set_xlabel('Time (s)')
            pass

        if arg_parser.flag_end_effector:
            fig_ee = plt.figure(figsize=(16, 4))
            axs_ee = fig_ee.subplots(1, 4)

            leg_name = ['FR', 'FL', 'RR', 'RL']
            points_per_period = int(
                cfg['environment']['period'] / cfg['environment']['control_dt'])
            num_traj = 15
            temp_joint = (oss[:, 7:19])[-num_traj * points_per_period:]
            # temp_joint = joint_ref[-num_traj * points_per_period:]
            color = np.arctan2(phase[-num_traj * points_per_period:, 0],
                               phase[-num_traj * points_per_period:, 1])
            color = color % (2 * np.pi) / (2 * np.pi)
            e_traj = []
            for i in range(4):
                ee_x, ee_y, ee_z = GaitGenerator.kinematic(temp_joint[:, 3 * i],
                                                           temp_joint[:,
                                                                      3 * i + 1],
                                                           temp_joint[:,
                                                                      3 * i + 2],
                                                           is_right=(i % 2 == 1))
                e_traj.append(ee_x)
                e_traj.append(ee_y)
                e_traj.append(ee_z)
                points = np.array([ee_x, ee_z]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='coolwarm',
                                    norm=plt.Normalize(0, 1))
                lc.set_array(color)
                lc.set_alpha(0.8)
                lc.set_linewidth(1)
                # axs_ee[i].plot(ee_x, ee_z)
                line = axs_ee[i].add_collection(lc)
                axs_ee[i].axis('equal')
                # axs_ee[i].axis('off')
                axs_ee[i].set_xticklabels('')
                axs_ee[i].set_yticklabels('')
                axs_ee[i].set_xlabel(
                    leg_name[i], fontname="Arial", fontsize=18)
                pass

            cbaxes = fig_ee.add_axes([0.93, 0.115, 0.015, 0.77])
            cb = plt.colorbar(line, cax=cbaxes)
            fig_ee.subplots_adjust(wspace=0.20, hspace=0.15)
            pass

        if arg_parser.flag_eval:
            from utils.Rotation import qua2euler, qua2matrix
            st = 1
            ed = -1
            llll = (ed - st) % (oss.shape[0])
            temp_oss = oss[st:ed, :]
            temp_time = time[st:ed:]
            roll, pitch, yaw = qua2euler(temp_oss[:, 3], temp_oss[:, 4],
                                         temp_oss[:, 5], temp_oss[:, 6])
            fig_perf = plt.figure(figsize=(15, 7))
            axs_perf = fig_perf.subplots(2, 1, sharex=True)
            # plot zero order information
            c1 = 'tab:red'
            p1 = axs_perf[0].plot(temp_time, temp_oss[:, 2], color=c1, marker=11, markevery=int((llll) / 20),
                                  label='Z     , mean: {0:.3f},std: {1:.3f}'.format(np.mean(temp_oss[:, 2]),
                                                                                    np.std(temp_oss[:, 2])))
            ax = axs_perf[0].twinx()
            c2 = 'tab:blue'
            p2 = ax.plot(temp_time, roll, color=c2, marker='o', alpha=0.6, markevery=int((llll) / 20),
                         label='Roll  , mean: {0:.3f},std: {1:.3f}'.format(np.mean(roll),
                                                                           np.std(roll)))
            p3 = ax.plot(temp_time, pitch, color=c2, marker=4, alpha=0.6, markevery=int((llll) / 20),
                         label='Pitch, mean: {0:.3f},std: {1:.3f}'.format(np.mean(pitch),
                                                                          np.std(pitch)))
            p4 = ax.plot(temp_time, yaw, color=c2, marker=5, alpha=0.6, markevery=int((llll) / 20),
                         label='Yaw, mean: {0:.3f},std: {1:.3f}'.format(np.mean(yaw),
                                                                        np.std(yaw)))
            lns = p1 + p2 + p3 + p4
            labs = [l.get_label() for l in lns]
            axs_perf[0].legend(lns, labs, loc=3)
            axs_perf[0].set_xlabel("time (s)")
            axs_perf[0].set_ylabel("Height (m)", color=c1)
            axs_perf[0].tick_params(axis='y', labelcolor=c1)
            ax.tick_params(axis='y', labelcolor=c2)
            ax.set_ylabel("Posture (rad)", color=c2)
            axs_perf[0].set_ylim([0.25, 0.32])
            ax.set_ylim([-0.2, 0.2])

            # plot velocity
            temp_cmd = cmd[st:ed, :]
            rot = qua2matrix(temp_oss[:, 3], temp_oss[:, 4],
                             temp_oss[:, 5], temp_oss[:, 6])
            body_vel = np.zeros_like(temp_cmd)
            body_omega = np.zeros_like(temp_cmd)
            for i in range(llll):
                body_vel[i, :] = np.dot(rot[i, :, :].T, temp_oss[i, 19:22])
                body_omega[i, :] = np.dot(rot[i, :, :].T, temp_oss[i, 22:25])
                pass
            if cfg['environment']['WILDCAT']:
                body_vel[:, 0] = -body_vel[:, 0]
                pass

            p1 = axs_perf[1].plot(temp_time, body_vel[:, 0], label='Forward_Vel,mean:{0:.3f},std:{1:.3f}'.format(
                np.mean(body_vel[:, 0] - cmd[st:ed, 0]), np.std(body_vel[:, 0] - cmd[st:ed, 0])), color=c1,
                markevery=int((llll) / 20))
            p2 = axs_perf[1].plot(temp_time, cmd[st:ed, 0], label='Forward_Cmd', color=c1, linestyle='--',
                                  markevery=int((llll) / 20))
            p3 = axs_perf[1].plot(temp_time, body_vel[:, 1], label='Lateral_Vel,mean:{0:.3f},std:{1:.3f}'.format(
                np.mean(body_vel[:, 1] - cmd[st:ed, 1]), np.std(body_vel[:, 1] - cmd[st:ed, 1])), color=c1, marker=11,
                markevery=int((llll) / 20))
            p4 = axs_perf[1].plot(temp_time, cmd[st:ed, 1], label='Lateral_Cmd', color=c1, linestyle='--', marker=4,
                                  markevery=int((llll) / 20))
            axs_perf[1].set_ylabel("Velocity (m/s)", color=c1)
            axs_perf[1].tick_params(axis='y', labelcolor=c1)
            ax = axs_perf[1].twinx()
            p5 = ax.plot(temp_time, body_omega[:, 2], color=c2, marker=5, alpha=0.6, markevery=int((llll) / 20),
                         label='Yaw_Vel,mean:{0:.3f},std:{1:.3f}'.format(np.mean(body_omega[:, 2] - cmd[st:ed, 2]),
                                                                         np.std(body_omega[:, 2] - cmd[st:ed, 2])))
            p6 = ax.plot(temp_time, cmd[st:ed, 2], label='Yaw_Cmd', color=c2, linestyle='--', marker=6, alpha=0.6,
                         markevery=int((llll) / 20))
            ax.tick_params(axis='y', labelcolor=c2)
            ax.set_ylabel("Omega (rad/s)", color=c2)
            axs_perf[1].set_ylim([-2.0, cfg['environment']['Vx']])
            axs_perf[1].set_xlabel("time (s)")
            ax.set_ylim([-2.2, 2.2])
            lns = p1 + p2 + p3 + p4 + p5 + p6
            labs = [l.get_label() for l in lns]
            axs_perf[1].legend(lns, labs, loc=4)
            fig_perf.subplots_adjust(wspace=0.30, hspace=0.15)

            pass

        if arg_parser.flag_value:
            # 1st, PCA is used to get 2d space
            # plot the traj in 2d space and interpolation
            # sampling points around the trajectory and remap to original space
            # get the value of the sampling points and color them in 2d space
            from sklearn.decomposition import PCA
            # assert cfg['environment']['ObsNoise'] < 0.1, "Since now, to analysis value function, obs noise must be 0"

            st = 100
            ed = -1

            pca = PCA(n_components=arg_parser.flag_value)
            pca.fit(origin_obs[st:ed, :])  # cmd is ignored
            print(pca.explained_variance_ratio_)
            state_de = pca.transform(origin_obs[st:ed, :])
            fig_v = plt.figure(figsize=(8, 8))
            ax = fig_v.subplots()
            # print(value.shape)
            cc = value[st:ed, 0]
            cc = (cc - np.min(cc)) / (np.max(cc) - np.min(cc))
            print(cc)
            sc = ax.scatter(
                state_de[:, 0], state_de[:, 1], c=cc, cmap='viridis')
            plt.colorbar(sc)
            pass

        if arg_parser.flag_torque:
            fig_tt = plt.figure(figsize=(16, 9))
            axs_tt = fig_tt.subplots(3, 4, sharex=True, sharey=True)
            # torque = cfg['environment']['Stiffness'] * (act - oss[:, 7:19]) + \
            #          cfg['environment']['Damping'] * (0 - oss[:, 25:37])
            # torque[:, [0, 3, 6, 9]] = torque[:, [0, 3, 6, 9]] * cfg['environment']['AbadRatio']
            torque = np.asarray(joint_effort)
            data_len = oss.shape[0]
            points_per_period = int(
                cfg['environment']['period'] / cfg['environment']['control_dt'])
            tt = np.arange(0, data_len, int(points_per_period / 2)
                           ) * cfg['environment']['control_dt']
            # np.save('wc.npy', np.hstack((oss[:, 19:25], oss[:, 25:37], torque)))
            name = ['FR', 'FL', 'HR', 'HL']
            joint_name = ['Abad', 'Hip', 'Knee']
            for i in range(3):
                for j in range(4):
                    for k in range(floor(tt.shape[0] / 2) - 1):
                        axs_tt[i, j].axvspan(
                            tt[k * 2], tt[2 * k + 1], facecolor='C3', alpha=0.3)
                        axs_tt[i, j].axvspan(
                            tt[2 * k + 1], tt[2 * k + 2], facecolor='C4', alpha=0.3)
                        pass
                    axs_tt[i, j].plot(
                        time, torque[:, j * 3 + i], lw=2, color='C0')
                    axs_tt[i, j].plot(
                        time, oss[:, 25 + j * 3 + i], lw=2, color='C1')

                    axs_tt[i, j].set_ylim(-40, 40)
                    axs_tt[i, j].grid(b=True, which='major',
                                      axis='y', linestyle='--')
                    if i == 2:
                        axs_tt[i, j].set_xlabel(
                            name[j] + " joint angle (rad)", fontname="Arial", fontsize=18)
                        pass
                    if j == 0:
                        axs_tt[i, j].set_ylabel(
                            joint_name[i], fontname="Arial", fontsize=18)
                        pass
                    if j == 3:
                        ax = axs_tt[i, j].twinx()
                        ax.plot(time, oss[:, 25 + j * 3 + i] * torque[:,
                                j * 3 + i], lw=2, color='C2', alpha=0.4)
                        ax.tick_params(axis='y', labelcolor='C2')
                        ax.set_ylim(-300, 300)
                        pass
                    else:
                        ax = axs_tt[i, j].twinx()
                        ax.plot(time, oss[:, 25 + j * 3 + i] * torque[:,
                                j * 3 + i], lw=2, color='C2', alpha=0.4)
                        ax.tick_params(axis='y', labelcolor='C2')
                        ax.set_ylim(-300, 300)
                        ax.set_yticklabels('')
                        pass
                    pass
                pass

            # fig_tt.text(0.05, 0.45, 'Torque (Nm)', fontname="Arial", fontsize=18, rotation='vertical', color='C0')
            # fig_tt.text(0.15, 0.45, 'Speed (rad/s)', fontname="Arial", fontsize=18, rotation='vertical', color='C1')
            # fig_tt.text(0.85, 0.45, 'Power (w)', fontname="Arial", fontsize=18, rotation='vertical', color='C2')
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='C0', lw=2),
                               Line2D([0], [0], color='C1', lw=2),
                               Line2D([0], [0], color='C2', lw=2)]
            legend_label = ['Troque(Nm)', 'Velocity(rad/s)', 'Power(w)']
            fig_tt.legend(legend_elements, legend_label, bbox_to_anchor=(
                0.5, 0.05), loc='upper center', ncol=3)
            fig_tt.subplots_adjust(wspace=0.30, hspace=0.15)
            pass

        if arg_parser.flag_work_condition:
            from matplotlib import cm
            from matplotlib.path import Path
            import matplotlib.patches as patches
            fig_motor = plt.figure(figsize=(16, 9))
            ax_motor = fig_motor.subplots(3, 4)
            name = ['FR', 'FL', 'HR', 'HL']
            joint_name = ['Abad', 'Hip', 'Knee']
            points_per_period = int(
                cfg['environment']['period'] / cfg['environment']['control_dt'])
            num_traj = 40
            # torque = cfg['environment']['Stiffness'] * (
            #         act[-num_traj * points_per_period:, :] - oss[-num_traj * points_per_period:, 7:19]) + \
            #          cfg['environment']['Damping'] * (0 - oss[-num_traj * points_per_period:, 25:37])
            torque = np.asarray(
                joint_effort)[-num_traj * points_per_period:, :]
            temp_joint_dot = (oss[:, 25:37])[-num_traj * points_per_period:]

            # np.save('/home/wooden/Desktop/temp/bp5_146_inertia_torque.npy', np.asarray(torque))
            # np.save('/home/wooden/Desktop/temp/bp5_146_inertia_dq.npy', np.asarray(temp_joint_dot))

            # torque[:, [0, 3, 6, 9]] = torque[:, [0, 3, 6, 9]] * cfg['environment']['AbadRatio']
            torque[:, [2, 5, 8, 11]] = torque[:, [2, 5, 8, 11]] / 1.55
            temp_joint_dot[:, [2, 5, 8, 11]
                           ] = temp_joint_dot[:, [2, 5, 8, 11]] * 1.55

            color = np.arctan2(phase[-num_traj * points_per_period:, 0],
                               phase[-num_traj * points_per_period:, 1])
            color = color % (2 * np.pi) / (2 * np.pi)

            motor_property = [
                np.array([18, 18, 18, 0, -18, -18, -18, 0, 18]),  # torque
                np.array([-41.67, -14.2, 14.2, 41.67, 41.67,
                         14.67, -14.67, -41.67, -41.67])  # speed
            ]

            power_speed = np.linspace(-41.67, 41.67, 200)
            power_torque = np.linspace(-18, 18, 200)
            X, Y = np.meshgrid(power_speed, power_torque)
            power = X * Y
            # power[np.where(power > 300)] = 300

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
            # mask = []
            # mask.append(patches.PathPatch(mask1, facecolor='w', lw=0))
            # mask.append(patches.PathPatch(mask2, facecolor='w', lw=0))
            for i in range(3):
                for j in range(4):
                    # plot the motor property
                    ax_motor[i, j].contourf(
                        X, Y, power, 20, alpha=0.6, cmap='summer')
                    # ax_motor[i, j].contour(X, Y, power, 40, alpha=0.4, cmap='RdYlBu')
                    ax_motor[i, j].add_patch(patches.PathPatch(
                        mask1, facecolor='white', lw=0))
                    ax_motor[i, j].add_patch(patches.PathPatch(
                        mask2, facecolor='white', lw=0))

                    ax_motor[i, j].plot(
                        motor_property[1], motor_property[0], color='k', linestyle='--', lw=3)
                    idx = i + j * 3
                    start = -num_traj * points_per_period
                    points = np.array(
                        [temp_joint_dot[:, idx], torque[:, idx]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap='coolwarm', norm=plt.Normalize(0, 1))
                    lc.set_array(color)
                    lc.set_alpha(0.8)
                    lc.set_linewidth(1)
                    line = ax_motor[i, j].add_collection(lc)
                    if i == 2:
                        ax_motor[i, j].set_xlabel(
                            name[j] + " joint speed (rad/s)", fontname="Arial", fontsize=18)
                        pass
                    if j == 0:
                        ax_motor[i, j].set_ylabel(
                            joint_name[i], fontname="Arial", fontsize=18)
                        pass
                    ax_motor[i, j].set_xlim(-50, 50)
                    ax_motor[i, j].set_ylim(-20, 20)
                    pass
                pass
            cbaxes = fig_motor.add_axes([0.91, 0.05, 0.07, 0.90])
            if cfg['environment']['GaitType'] == 0:
                GaitBar(cbaxes, N=12, phase=np.array(
                    [0, 0.5, 0.5, 0]), lam=cfg['environment']['lam'])
            elif cfg['environment']['GaitType'] == 1:
                GaitBar(cbaxes, N=12, phase=np.array(
                    [0.5, 0.5, 0, 0]), lam=cfg['environment']['lam'])
            elif cfg['environment']['GaitType'] == 2:
                # GaitBar(cbaxes, N=10, phase=np.array([0.667, 0.333, 0.333, 0]), lam=cfg['environment']['lam'])
                GaitBar(cbaxes, N=12, phase=np.array(
                    [0.0, 0.25, 0.5, 0.75]), lam=cfg['environment']['lam'])
            else:
                pass
            # cb = plt.colorbar(line, cax=cbaxes)
            fig_motor.text(0.05, 0.49, 'Torque (Nm)', fontname="Arial", fontsize=18,
                           rotation='vertical')
            fig_motor.subplots_adjust(wspace=0.20, hspace=0.15)
            pass

        if arg_parser.flag_corr_analysis:
            import pandas as pd
            from utils.VisTool import heatmap
            fig_corr = plt.figure(figsize=(16, 9))
            ax_corr = fig_corr.subplots(4, 2)
            # ax_corr = fig_corr.subplots()
            contact_oss = np.asarray(contact_oss)
            nn_state = np.asarray(nn_state)
            data = np.hstack((oss, contact_oss, nn_state))
            df = pd.DataFrame(data)
            corr_name = ['body_1', 'body_2', 'body_3',
                         'quat1', 'quat2', 'quat3', 'quat4']
            [corr_name.append('jo' + str(i)) for i in range(12)]
            [corr_name.append('body_dot' + str(i)) for i in range(3)]
            [corr_name.append('omega' + str(i)) for i in range(3)]
            [corr_name.append('jo_dot' + str(i)) for i in range(12)]
            [corr_name.append('contact' + str(i)) for i in range(4)]
            corr_name = corr_name + \
                ['cmd_x', 'cmd_y', 'yaw', 'phase1', 'phase2']
            [corr_name.append('~jo' + str(i)) for i in range(12)]
            [corr_name.append('~jo_dot' + str(i)) for i in range(12)]
            [corr_name.append('z_axis' + str(i)) for i in range(3)]
            [corr_name.append('~omega' + str(i)) for i in range(3)]
            [corr_name.append('layer1_c' + str(i)) for i in range(N_LSTM[0])]
            [corr_name.append('layer1_h' + str(i)) for i in range(N_LSTM[0])]
            [corr_name.append('layer2_c' + str(i)) for i in range(N_LSTM[1])]
            [corr_name.append('layer2_h' + str(i)) for i in range(N_LSTM[1])]
            corr = df.corr().values

            im, cbar = heatmap(corr[37:41, (41 + 35):(41 + 35 + N_LSTM[0])], corr_name[37:41],
                               corr_name[(41 + 35):(41 + 35 + N_LSTM[0])
                                         ], ax=ax_corr[0, 0],
                               cmap="RdBu", cbarlabel="Correlation")
            heatmap(corr[37:41, (41 + 35 + N_LSTM[0]):(41 + 35 + 2 * N_LSTM[0])], corr_name[37:41],
                    corr_name[(41 + 35 + N_LSTM[0]):(41 + 35 +
                                                     2 * N_LSTM[0])], ax=ax_corr[1, 0],
                    cmap="RdBu", cbarlabel="Correlation")
            heatmap(corr[37:41, (41 + 35 + N_LSTM[0] * 2):(41 + 35 + 2 * N_LSTM[0] + N_LSTM[1])], corr_name[37:41],
                    corr_name[(41 + 35 + N_LSTM[0] * 2):(41 + 35 +
                                                         2 * N_LSTM[0] + N_LSTM[1])], ax=ax_corr[2, 0],
                    cmap="RdBu", cbarlabel="Correlation")
            heatmap(corr[37:41, (41 + 35 + N_LSTM[0] * 2 + N_LSTM[1]):(41 + 35 + 2 * N_LSTM[0] + N_LSTM[1] * 2)],
                    corr_name[37:41],
                    corr_name[(41 + 35 + N_LSTM[0] * 2 + N_LSTM[1]):(41 + 35 + 2 * N_LSTM[0] + N_LSTM[1] * 2)],
                    ax=ax_corr[3, 0],
                    cmap="RdBu", cbarlabel="Correlation")
            ax_corr[0, 1].plot(data[:, 37])
            ax_corr[0, 1].plot(data[:, 82])
            # im, cbar = heatmap(corr[58:70,
            #                    (41 + 35 + N_LSTM[0] * 1):(41 + 35 + N_LSTM[0] * 2)],
            #                    corr_name[58:70],
            #                    corr_name[
            #                    (41 + 35 + N_LSTM[0] * 1):(41 + 35 + N_LSTM[0] * 2)],
            #                    ax=ax_corr,
            #                    cmap="RdBu", cbarlabel="Correlation")
            fig_corr.tight_layout()
            pass

        if arg_parser.flag_virbation_analysis:
            fig_virb = plt.figure(figsize=(16, 9), dpi=150)
            ax_virb = fig_virb.subplots(3, 3, sharex=True, sharey=True)
            virb_name = ['Abad', 'Hip', 'Knee']
            from scipy import signal
            for i in range(3):
                # f, t, Sxx = signal.spectrogram(act[:, i], 1 / cfg['environment']['control_dt'])
                # ax_virb[i].pcolormesh(t, f, Sxx)
                # ax_virb[i].set_xlabel("Time [s]" + virb_name[i])
                # ax_virb[i].set_ylim([0, 50])
                f, t, sxx = signal.spectrogram(
                    joint[:, i], 1 / cfg['environment']['control_dt'])
                ax_virb[0, i].pcolormesh(t, f, sxx)
                f, t, sxx = signal.spectrogram(
                    joint_dot[:, i], 1 / cfg['environment']['control_dt'])
                ax_virb[1, i].pcolormesh(t, f, sxx)
                f, t, sxx = signal.spectrogram(
                    act[:, i], 1 / cfg['environment']['control_dt'])
                ax_virb[2, i].pcolormesh(t, f, sxx)
                pass
            ax_virb[0, 0].set_ylim([0, 50])
            ax_virb[2, 0].set_xlabel("Time [s] Abad")
            ax_virb[2, 1].set_xlabel("Time [s] Hip")
            ax_virb[2, 2].set_xlabel("Time [s] Knee")
            ax_virb[0, 0].set_ylabel("Freq [Hz] Joint")
            ax_virb[1, 0].set_ylabel("Freq [Hz] Joint_dot")
            ax_virb[2, 0].set_ylabel("Freq [Hz] Output")
            pass

        plt.show()

        pass


if __name__ == "__main__":
    import os
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    main(sys.argv)

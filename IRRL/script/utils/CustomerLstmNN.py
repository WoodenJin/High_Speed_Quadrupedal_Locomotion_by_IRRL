from raisim_gym.algo.ppo2 import PPO2
import numpy as np
import os


class CustomerLstmNN(object):
    """
    this class is used to create an customer lstm nn by loading a trained model
    """

    def __init__(self, model_path, n_lstm=[32, 32], flag_v=False, is_LSTM=True):

        self.lstm_wx = []
        self.lstm_wh = []
        self.lstm_b = []
        self.pi_w = None
        self.pi_b = None
        self.cell_state = []
        self.hidden = []
        self.n_lstm = n_lstm

        self.flag_v = flag_v
        self.v_lstm_wx = []
        self.v_lstm_wh = []
        self.v_lstm_b = []
        self.v_w = None
        self.v_b = None
        self.v_cell_state = []
        self.v_hidden = []
        self.state_ph = np.zeros([200, sum(n_lstm) * 4])

        self.is_LSTM = is_LSTM

        model_name = (model_path.rstrip('.pkl')).strip('./pkl/')
        path = os.getcwd() + '/model/' + model_name
        isExists = os.path.exists(path)

        if not isExists:
            self._ppo_model = PPO2.load(model_path)
            self._param = self._ppo_model.get_parameters()

            if self.is_LSTM:
                for idx, lstm_num in enumerate(n_lstm):
                    self.lstm_wx.append(
                        self._param['model/lstm_pi{}/wx:0'.format(idx)])
                    self.lstm_wh.append(
                        self._param['model/lstm_pi{}/wh:0'.format(idx)])
                    self.lstm_b.append(
                        self._param['model/lstm_pi{}/b:0'.format(idx)])
                    self.cell_state.append(np.zeros(lstm_num))
                    self.hidden.append(np.zeros(lstm_num))
                    if self.flag_v:
                        self.v_lstm_wx.append(
                            self._param['model/lstm_v{}/wx:0'.format(idx)])
                        self.v_lstm_wh.append(
                            self._param['model/lstm_v{}/wh:0'.format(idx)])
                        self.v_lstm_b.append(
                            self._param['model/lstm_v{}/b:0'.format(idx)])
                        self.v_cell_state.append(np.zeros(lstm_num))
                        self.v_hidden.append(np.zeros(lstm_num))
                        pass
                    pass
                self.pi_w = self._param['model/pi/w:0']
                self.pi_b = self._param['model/pi/b:0']

                if self.flag_v:
                    self.v_w = self._param['model/vf/w:0']
                    self.v_b = self._param['model/vf/b:0']
                    pass
                pass
            else:
                pass
        else:
            print("Here!!!!")
            if self.is_LSTM:
                for idx, lstm_num in enumerate(n_lstm):
                    self.lstm_wx.append(np.loadtxt(
                        path+'/lstm_wx'+str(idx)+'.csv', delimiter=','))
                    self.lstm_wh.append(np.loadtxt(
                        path+'/lstm_wh'+str(idx)+'.csv', delimiter=','))
                    self.lstm_b.append(np.loadtxt(
                        path+'/lstm_b'+str(idx)+'.csv', delimiter=','))
                    self.cell_state.append(np.zeros(lstm_num))
                    self.hidden.append(np.zeros(lstm_num))
                    pass
                self.pi_w = np.loadtxt(path+'/pi_w.csv', delimiter=',')
                self.pi_b = np.loadtxt(path+'/pi_b.csv', delimiter=',')
                pass
            else:
                pass
            pass

        self.input = None
        self.output = None
        self.v = None

    def reset(self):
        for i in range(len(self.v_cell_state)):
            self.v_cell_state[i] = np.zeros_like(self.v_cell_state[i])
            pass
        for i in range(len(self.v_hidden)):
            self.v_hidden[i] = np.zeros_like(self.v_hidden[i])
            pass
        for i in range(len(self.cell_state)):
            self.cell_state[i] = np.zeros_like(self.cell_state[i])
            pass
        for i in range(len(self.hidden)):
            self.hidden[i] = np.zeros_like(self.hidden[i])
            pass
        pass

    def predict(self, obs):
        self.input = obs
        h = self.input
        for i, lstm_num in enumerate(self.n_lstm):
            gate = np.dot(h, self.lstm_wx[i]) + \
                np.dot(self.hidden[i], self.lstm_wh[i]) + \
                self.lstm_b[i]
            in_gate = gate[0:lstm_num]
            in_gate = CustomerLstmNN.sigmod(in_gate)
            forget_gate = gate[lstm_num:2 * lstm_num]
            forget_gate = CustomerLstmNN.sigmod(forget_gate)
            out_gate = gate[2 * lstm_num:3 * lstm_num]
            out_gate = CustomerLstmNN.sigmod(out_gate)
            cell_candidate = gate[3 * lstm_num:4 * lstm_num]
            cell_candidate = np.tanh(cell_candidate)
            self.cell_state[i] = forget_gate * \
                self.cell_state[i] + in_gate * cell_candidate
            self.hidden[i] = out_gate * np.tanh(self.cell_state[i])
            h = self.hidden[i]
            pass
        self.output = np.dot(h, self.pi_w) + self.pi_b
        self.output = np.clip(self.output, np.ones_like(
            self.output) * -1, np.ones_like(self.output))

        if self.flag_v:
            h = self.input
            for i, lstm_num in enumerate(self.n_lstm):
                gate = np.dot(h, self.v_lstm_wx[i]) + \
                    np.dot(self.v_hidden[i], self.v_lstm_wh[i]) + \
                    self.v_lstm_b[i]
                in_gate = gate[0:lstm_num]
                in_gate = CustomerLstmNN.sigmod(in_gate)
                forget_gate = gate[lstm_num:2 * lstm_num]
                forget_gate = CustomerLstmNN.sigmod(forget_gate)
                out_gate = gate[2 * lstm_num:3 * lstm_num]
                out_gate = CustomerLstmNN.sigmod(out_gate)
                cell_candidate = gate[3 * lstm_num:4 * lstm_num]
                cell_candidate = np.tanh(cell_candidate)
                self.v_cell_state[i] = forget_gate * \
                    self.v_cell_state[i] + in_gate * cell_candidate
                self.v_hidden[i] = out_gate * np.tanh(self.v_cell_state[i])
                h = self.v_hidden[i]
                pass
            self.v = np.dot(h, self.v_w) + self.v_b
            pass

        # gate = np.dot(self.input, self.lstm_wx) + \
        #        np.dot(self.hidden, self.lstm_wh) + \
        #        self.lstm_b
        # in_gate = gate[0:32]
        # in_gate = CustomerLstmNN.sigmod(in_gate)
        # forget_gate = gate[32:64]
        # forget_gate = CustomerLstmNN.sigmod(forget_gate)
        # out_gate = gate[64:96]
        # out_gate = CustomerLstmNN.sigmod(out_gate)
        # cell_candidate = gate[96:128]
        # cell_candidate = np.tanh(cell_candidate)
        #
        # self.cell_state = forget_gate * self.cell_state + in_gate * cell_candidate
        # self.hidden = out_gate * np.tanh(self.cell_state)
        # fc0 = np.tanh(np.dot(self.hidden, self.pi_fc0_w) + self.pi_fc0_b)
        # fc1 = np.tanh(np.dot(fc0, self.pi_fc1_w) + self.pi_fc1_b)
        # self.output = np.dot(fc1, self.pi_w) + self.pi_b
        return self.output.copy()

    def predict2(self, obs):
        if self.is_LSTM:
            oo, self.state_ph = self._ppo_model.predict(
                obs, state=self.state_ph, deterministic=True)
            # oo, self.state_ph = self._ppo_model.predict(obs, deterministic=True)  # for NN
            return oo
        else:
            oo, self.state_ph = self._ppo_model.predict(
                obs, deterministic=True)  # for NN
            return oo

    def get_hidden_state(self):
        return np.hstack((self.cell_state[0], self.hidden[0], self.cell_state[1], self.hidden[1]))
        pass

    def get_hidden_state2(self):
        return self.state_ph[0, 0:128]
        pass

    def get_v(self):
        if self.flag_v:
            return self.v
            pass
        else:
            return 0

    def save_model(self, path='/model/bp4/'):
        # ===================
        # since now, suppose the layer parameter of NN is fixed to 2
        assert len(
            self.n_lstm) == 2, "for now, only two layer lstm nn is supported to be written"
        np.savetxt(os.getcwd() + path + 'lstm_wh0.csv',
                   self.lstm_wh[0], delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'lstm_wh1.csv',
                   self.lstm_wh[1], delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'lstm_wx0.csv',
                   self.lstm_wx[0], delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'lstm_wx1.csv',
                   self.lstm_wx[1], delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'lstm_b0.csv',
                   self.lstm_b[0], delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'lstm_b1.csv',
                   self.lstm_b[1], delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'pi_w.csv',
                   self.pi_w, delimiter=',', fmt='%.6f')
        np.savetxt(os.getcwd() + path + 'pi_b.csv',
                   self.pi_b, delimiter=',', fmt='%.6f')
        pass

    @staticmethod
    def sigmod(x):
        return 1 / (1 + np.exp(-x))


def main():
    model_path = './../pkl/bp4_8.pkl'
    policy = CustomerLstmNN(model_path)
    input = np.array([-0.25, 0, 0, -1, 1,
                      0.0495351, -0.0115055, 0.10003,
                      0.0494637, -0.0106432, 0.0999373,
                      -0.0649202, -0.0107303, 0.110537,
                      0.064962, -0.010784, 0.110514,
                      -0.00990702, -0.158301, 0.334006,
                      0.00989274, -0.158129, 0.333987,
                      -0.012984, -0.158146, 0.336107,
                      0.0129924, -0.158157, 0.336103,
                      0.0144317, 0.00293443, -7.58682e-05,
                      0.00637404, 0.000356918, -0.00200364])
    output = policy.predict(input)
    obs_mean = np.hstack(([0.25, 0.0, 0.0],
                          [0.5, 0.5],
                          [0, -0.78, 1.57, 0, -0.78, 1.57,
                              0, -0.78, 1.57, 0, -0.78, 1.57],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]))
    obs_std = np.hstack((np.ones(3),
                         np.ones(2) * 0.5,
                         np.ones(12),
                         np.ones(12) * 5,
                         np.ones(3) * 0.7,
                         np.ones(3) * 2.0))
    action_mean = obs_mean[5:17].copy()
    action_std = np.ones(12) * 0.6
    # print(output*action_std+action_mean)
    print(output)
    pass


def main1():
    # define obs flow
    # obs_flow = np.random.random((10000, 35))
    N = 10
    period = 0.3
    period_vir = 0.3
    dt = 0.002
    ratio = 10.0
    time = np.linspace(0, period * N, int(period * N / dt))
    obs_flow = np.asarray([time for i in range(35)])
    obs_flow = obs_flow.transpose()
    # obs_flow = np.sin(obs_flow)*0.0 + np.random.random((10000, 35)) * 1.0
    obs_flow = np.sin(2 * np.pi / period_vir * obs_flow) * ratio + np.random.random((int(period * N / dt), 35)) * (
        1 - ratio)

    N_LSTM = [32, 32]
    customer = CustomerLstmNN('./../pkl/bp5_40.pkl',
                              flag_v=False, n_lstm=N_LSTM)
    # customer = CustomerLstmNN('./../pkl/bp4_33.pkl', flag_v=False, n_lstm=N_LSTM)

    temp_obs = np.zeros((200, 35))
    res1 = []
    res2 = []

    sh1 = []
    sh2 = []

    for i in range(obs_flow.shape[0]):
        res1.append(customer.predict(obs_flow[i, :]))
        temp_obs[0, :] = obs_flow[i, :]
        res2.append(customer.predict2(temp_obs)[0, :])
        sh1.append(customer.get_hidden_state())
        sh2.append(customer.get_hidden_state2())
        # print("hello")
        pass

    res1 = np.asarray(res1)
    res2 = np.asarray(res2)
    sh1 = np.asarray(sh1)
    sh2 = np.asarray(sh2)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')

    fig_ss = plt.figure(figsize=(16, 9))
    axs_ss = fig_ss.subplots(3, 4)

    for i in range(3):
        for j in range(4):
            idx = j * 3 + i
            axs_ss[i, j].plot(time, res1[:, idx] - res2[:, idx])
            axs_ss[i, j].set_ylim([-1, 1])
            pass
        pass
    # plt.title("harmonic freq: {0:.2f} ratio: {1:.2f}".format(1 / period_vir, ratio))
    fig_ss.text(0.35, 0.03, "harmonic stimulation freq: {0:.2f}hz ratio: {1:.2f}".format(1 / period_vir, ratio),
                fontsize=18)

    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.subplots()
    c = ax.pcolor(sh1 - sh2)
    temp = sh1 - sh2
    fig2.colorbar(c)
    plt.show()
    pass


def main2():
    N_LSTM = [32, 32]
    customer = CustomerLstmNN('./../pkl/bp5_47.pkl',
                              flag_v=False, n_lstm=N_LSTM)
    N = 50
    period = 0.3
    dt = 0.002
    obs_flow = np.zeros([int(N * period / dt), 35])
    obs_flow[:, 0] = -1.5
    time = np.linspace(0, N * period, int(N * period / dt))
    obs_flow[:, 3] = np.sin(2 * np.pi * time / period)
    obs_flow[:, 4] = np.cos(2 * np.pi * time / period)
    res = []
    for i in range(int(N * period / dt)):
        res.append(customer.predict(obs_flow[i, :]))
        pass
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')
    res = np.asarray(res)
    for i in range(12):
        plt.plot(time, res[:, i])
    plt.show()
    pass


if __name__ == "__main__":
    main2()

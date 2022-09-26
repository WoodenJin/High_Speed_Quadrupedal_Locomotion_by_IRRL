import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv


class RaisimGymVecEnv(VecEnv):

    def __init__(self, impl):
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf,
                                             dtype=np.float32)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1., dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs, len(self._extraInfoNames)], dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]

    def seed(self, seed=None):
        self.wrapper.seed(seed)

    def step(self, action, visualize=False):
        # print('\033[1;33;44mShape of Action is {0}\033[0m'.format(action.shape))
        # print('\033[1;33;44mShape of Observation is {0}\033[0m'.format(self._observation.shape))
        # print('\033[1;33;44mShape of done is {0}\033[0m'.format(self._done.shape))
        if not visualize:
            self.wrapper.step(action, self._observation, self._reward, self._done, self._extraInfo)
        else:
            self.wrapper.testStep(action, self._observation, self._reward, self._done, self._extraInfo)

        if len(self._extraInfoNames) is not 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j],
            }} for j in range(0, len(self._extraInfoNames)) for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), self._done.copy(), info.copy()

    def OriginState(self):
        temp = np.zeros([self.num_envs, self.wrapper.GetOriginStateDim()], dtype=np.float32)
        self.wrapper.OriginState(temp)
        return temp

    def ReferenceState(self):
        temp = np.zeros([self.num_envs, self.num_acts*2], dtype=np.float32)
        self.wrapper.ReferenceState(temp)
        return temp

    def GetJointEffort(self):
        temp = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.wrapper.GetJointEffort(temp)
        return temp

    def GetGeneralizedForce(self):
        # notice, the generalized_dim = num_acts+6 for float base model
        # if the robot is not float base, please check this
        temp = np.zeros([self.num_envs, self.num_acts + 6], dtype=np.float32)
        self.wrapper.GetGeneralizedForce(temp)
        return temp

    def GetInverseMassMatrix(self):
        temp = np.zeros([self.num_envs, (self.num_acts + 6) * (self.num_acts + 6)], dtype=np.float32)
        self.wrapper.GetInverseMassMatrix(temp)
        return temp

    def GetNonlinear(self):
        temp = np.zeros([self.num_envs, self.num_acts + 6], dtype=np.float32)
        self.wrapper.GetNonlinear(temp)
        return temp

    def GetSphereInfo(self):
        temp = np.zeros([self.num_envs, 4], dtype=np.float32)
        self.wrapper.GetSphereInfo(temp)
        return temp

    def SetContactCoefficient(self, contact_coeff):
        self.wrapper.SetContactCoefficient(contact_coeff)
        pass

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def start_recording_video(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_recording_video(self):
        self.wrapper.stopRecordingVideo()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    def show_window(self):
        self.wrapper.showWindow()

    def hide_window(self):
        self.wrapper.hideWindow()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

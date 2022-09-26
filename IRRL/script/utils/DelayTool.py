# delay class
from queue import Queue


class DelayTool(object):
    def __init__(self, dt, delay_time):
        self.num = int(delay_time/dt)
        self.Q = Queue(self.num)
        self.flag_first = True
        pass

    def input_output(self, s0):
        if self.flag_first:
            self.flag_first = False
            for i in range(self.num):
                self.Q.put(s0)
                pass
            pass
        res = self.Q.get()
        self.Q.put(s0)
        return res
        pass
    pass

#adapted from work by Abhinav Verma
import functools
from bayes_opt import BayesianOptimization
from scipy import spatial
import numpy as np

def clip_to_range(value, lw=-1, up=1):
    if value > up:
        return up
    if value < lw:
        return lw
    return value

def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval

def fold(fun, obs, init):
    return functools.reduce(fun, obs, init)

class PIDController():
    def __init__(self, pid_constants=(0.0,0.0,0.0), pid_target=0.0, pid_increment=0.0, para_condition=0.0, initial_obs=0.0, condition='False'):
        self.pid_constants = pid_constants
        self.pid_target = pid_target
        self.pid_increment = pid_increment
        self.para_condition = para_condition
        self.condition = condition
        self.final_target = pid_target
        self.observations = [initial_obs]

    def fold_pid(self, acc, lobs):
        return acc + (self.final_target - lobs)

    def pid_execute(self, obs):
        self.observations.append(obs)
        if eval(self.condition):
            self.final_target = self.pid_target + self.pid_increment
        else:
            self.final_target = self.pid_target
        #print("obs:", obs)
        #print("consts: ", self.pid_constants)
        act = self.pid_constants[0] * (self.final_target - obs) + \
              self.pid_constants[1] * fold(self.fold_pid, self.observations, 0) + self.pid_constants[2] * \
              (self.observations[-2] - obs)
        return act

    def update_parameters(self, pid_constants=(0, 0, 0), pid_target=0.0, pid_increment=0.0, para_condition=0.0):
        self.pid_constants = pid_constants
        self.pid_target = pid_target
        self.pid_increment = pid_increment
        self.para_condition = para_condition

    def pid_info(self):
        return [self.pid_constants, self.pid_target, self.pid_increment, self.para_condition]

class ParameterFinder():
    def __init__(self, positive_inputs, positive_actions, negative_inputs, negative_actions, pids):
        self.positive_inputs = positive_inputs
        self.negative_inputs = negative_inputs
        self.positive_actions = positive_actions
        self.negative_actions = negative_actions
        self.pids = pids

    def find_distance_paras(self, p1,i1,d1,p2,i2,d2,p3,i3,d3,p4,i4,d4):
        mapping = {0:(p1,i1,d1), 1:(p2,i2,d2), 2:(p3,i3,d3), 3:(p4,i4,d4)}
        print("Finding new distance parameters...")
        for i in range(4):
            self.pids[i].update_parameters(mapping[i])
        positive_actions_list = [[]] * 4
        positive_actions = []
        negative_actions_list = [[]] * 4
        negative_actions = []
        for inp in self.positive_inputs:
            #print(inp)
            for pididx in range(len(positive_actions_list)):
                positive_actions_list[pididx].append(clip_to_range(self.pids[pididx].pid_execute(inp[pididx])))
            if sum(positive_actions_list[pididx]) > 0:
                positive_actions.append(1)
            else:
                positive_actions.append(0)
        for inp in self.negative_inputs:
            for pididx in range(len(negative_actions_list)):
                negative_actions_list[pididx].append(clip_to_range(self.pids[pididx].pid_execute(inp[pididx])))
            #TODO: Choose correct method of usage for the series of PIDs
            if min(negative_actions_list[pididx]) > 0:
                negative_actions.append(1)
            else:
                negative_actions.append(0)
        loss = dual_loss(self.positive_actions, positive_actions, self.negative_actions, negative_actions)
        return loss * -1.0

    def pid_parameters(self,  pid_range_list):
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 1}  # Optimizer configuration
        print('Optimizing Controller')
        bo_pid = BayesianOptimization(self.find_distance_paras,
                                        {'p1': pid_range_list[0][0], 'i1': pid_range_list[0][1],'d1': pid_range_list[0][2],
                                         'p2': pid_range_list[1][0], 'i2': pid_range_list[1][1], 'd2': pid_range_list[1][2],
                                         'p3': pid_range_list[2][0], 'i3': pid_range_list[2][1], 'd3': pid_range_list[2][2],
                                         'p4': pid_range_list[3][0], 'i4': pid_range_list[3][1], 'd4': pid_range_list[3][2]}, verbose=0)
        bo_pid.maximize(init_points=1, n_iter=1, kappa=5, **gp_params)
        return bo_pid.res['max']

def dual_loss(pos_truth, pos_model, neg_truth, neg_model, const=1.0):
    numneg = len(neg_model)
    pos_diff = sum(np.abs(np.array(pos_truth) - np.array(pos_model)))
    neg_diff = sum(np.abs(np.array(neg_truth) - np.array(neg_model)))
    #lower bound it by zero
    return (pos_diff - neg_diff + numneg) / const
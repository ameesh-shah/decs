import gym
import numpy as np
import argparse
# import torch
# from controllers import PIDController, clip_to_range, ParameterFinder, create_interval
# from z3 import *
from sklearn.tree import DecisionTreeClassifier
from model_system import ModelSystem, Learner, Verifier
from ground_truth import GroundTruth
"""
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c import a2c
from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy
from stable_baselines import DQN, PPO2, A2C
"""
import pickle

env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda: env])

def train_expert_policy():
    model = a2c.A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000)
    model.save("a2c_cartpole")

class CartPoleGroundTruth(GroundTruth):

    def __init__(self, positive_exs, negative_exs=[]):
        super().__init__(False, positive_exs, negative_exs)

    def aggregate_negative_examples(self, negative_examples, current_model):
        #TODO: GET TRAJECTORIES ON THIS?
        for example in negative_examples:
            exampleout = current_model.predict(example)
            self.negative_examples.append((example, exampleout))


class CartPoleModelSystem(ModelSystem):

    def __init__(self, learner, verifier, groundtruthmodel):
        super().__init__(learner, verifier)
        self.groundtruthmodel = groundtruthmodel

    def train_candidate(self):
        model = self.learner.synthesize_candidate(self.groundtruthmodel)
        return model

    def check_candidate(self, candidate, yboundary):
        verification = self.verifier.verify(self.learner, yboundary)
        return verification

    def get_verifiable_decision_tree(self, max_iters, yboundary):
        for itr in range(max_iters):
            print("training next candidate")
            candidate = self.train_candidate()
            retval = self.check_candidate(candidate, yboundary)
            if retval == True:
                print("required {} verification iterations to get satisfiable solution.".format(itr))
                return candidate
            else:
                print("Adding counterexample")
                self.groundtruthmodel.aggregate_negative_examples(retval, self.learner)
        print("Exhausted max number of iterations without finding verified solution.")
        return candidate


class CartPoleDecisionTreeLearner(Learner):

    def __init__(self, max_depth=4):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def synthesize_candidate(self, positive_x, positive_y):
        self.model.fit(positive_x, positive_y)
        return self.model

class CartPolePIDLearner(Learner):

    def __init__(self, num_sensors=4):
        """
        :param initial_values: a list of tuples (or lists) that
        are the initial values for each PID controller
        """
        initial_values = [[0.0,0.0,0.0]] * num_sensors
        self.pids = []
        self.num_pids = len(initial_values)
        for initial in initial_values:
            self.pids.append(PIDController(pid_constants=initial))

    def synthesize_candidate(self, groundtruthmodel):
        positive_exs, negative_exs = groundtruthmodel.positive_examples, groundtruthmodel.negative_examples
        pinputs, poutputs = map(list, zip(*positive_exs))
        pinputs, poutputs = np.array(pinputs).squeeze(), np.array(poutputs).squeeze()
        if len(negative_exs) != 0:
            ninputs, noutputs = map(list, zip(*negative_exs))
            ninputs, noutputs = np.array(ninputs).squeeze(), np.array(noutputs).squeeze()
        else:
            ninputs = []
            noutputs = []
        paramfinder = ParameterFinder(pinputs, poutputs, ninputs, noutputs, self.pids)
        pid_ranges = []
        for i in range(len(self.pids)):
            pid_ranges.append(tuple([create_interval(self.pids[i].pid_constants[const], 0.1) for const in range(3)]))
        new_params = paramfinder.pid_parameters(pid_ranges)
        print("Found new parameters. Updating")
        for idx in range(len(self.pids)):
            templst = []
            for j in ['p', 'i', 'd']:
                templst.append(new_params['max_params'][j + str(idx + 1)])
            self.pids[idx].update_parameters(tuple(templst))
        return self.pids

    def predict(self, obs):
        actions = []
        for i in range(len(obs)):
            #print(obs[i])
            thing = self.pids[i].pid_execute(obs[i])
            #print(thing)
            val = clip_to_range(self.pids[i].pid_execute(obs[i]))
            #print(val)
            actions.append(val)
        #TODO: Choose correct method of usage for the series of PIDs
        if min(actions) > 0.0:
            return 1
        else:
            return 0




class CartPolePIDCorrectnessVerifier(Verifier):

    def __init__(self):
        #placeholder
        self.states = [0,0,0,0]


    def verify(self, candidate, yboundary, timebound=10, num_counters = 5):
        # candidate is a decision tree model
        # impose that the starting state must be in [-.05, .05]
        #impose that producing a trajectory from x of length 10 stays
        #less than yboundary
        for i in range(4):
            self.states[i] = Real('s{}'.format(i))
        self.solver = Solver()
        for i in range(4):
            self.solver.add(self.states[i] >= -.05)
            self.solver.add(self.states[i] <= .05)
        states, prev_states, actions = self.set_linear_system(candidate, timebound)
        states = np.squeeze(states)
        print(states.shape)
        print(prev_states.shape)
        #create the linear approximation of the system
        soln, residuals, rank, singular_vals = np.linalg.lstsq(prev_states, states, rcond=None)
        soln = np.transpose(soln)
        obs = env.reset()
        #print("solution matrix:", soln)
        #solution is of shape 4, 5
        #now, use the solution as a linear approximation and verify by computing next state
        nexttuples = [self.states]
        for t in range(timebound):
            newnexttuple = []
            self.solver.add(Or(soln[2][0] * nexttuples[t][0] + \
                            soln[2][1] * nexttuples[t][1] + \
                            soln[2][2] * nexttuples[t][2] + \
                            soln[2][3] * nexttuples[t][3] + \
                            soln[2][4] * actions[t] > yboundary,
                            soln[2][0] * nexttuples[t][0] + \
                            soln[2][1] * nexttuples[t][1] + \
                            soln[2][2] * nexttuples[t][2] + \
                            soln[2][3] * nexttuples[t][3] + \
                            soln[2][4] * actions[t] < (-1 * yboundary)))
            for i in range(4):
                #we impose a constraint of being GREATER than the boundary
                #so that a satisfactory assignment is actually
                #a counterexample, and "unsat" is sat

                newnexttuple.append(Real('state{}{}'.format(t + 1, i)))
                self.solver.add(newnexttuple[i] == soln[i][0] * nexttuples[t][0] + \
                                soln[i][1] * nexttuples[t][1] + \
                                soln[i][2] * nexttuples[t][2] + \
                                soln[i][3] * nexttuples[t][3] + \
                                soln[i][4] * actions[t])
            nexttuples.append(newnexttuple)
        print(self.solver.check())
        if self.solver.check() == sat:
            #add counterexamples
            counters = []
            is_sat = self.solver.check()
            while is_sat != sat and len(counters) < num_counters:
                print('counter example found:')
                mod = self.solver.model()
                #TODO: a less hacky way to convert to python values
                es0 = int('{}'.format(mod[self.states[0]].numerator())) / int('{}'.format(mod[self.states[0]].denominator()))
                es1 = int('{}'.format(mod[self.states[1]].numerator())) / int('{}'.format(mod[self.states[1]].denominator()))
                es2 = int('{}'.format(mod[self.states[2]].numerator())) / int('{}'.format(mod[self.states[2]].denominator()))
                es3 = int('{}'.format(mod[self.states[3]].numerator())) / int('{}'.format(mod[self.states[3]].denominator()))
                print(es0, es1, es2, es3)
                curr_counter = (es0, es1, es2, es3)
                counters.append(curr_counter)
                #add these as constraints to get a diff counterexample
                for i in range(4):
                    self.solver.add(self.states[i] != curr_counter[i])
                is_sat = self.solver.check()
            return counters
        else:
            print('Correctness of cartpole verified.')
            return True


    def set_linear_system(self, candidate, timebound):
        states = [] # dimension T by delta
        #A_matrix = [] # dimension delta by delta
        prev_states_augmented = [] # dimension T by delta + 1
        #B_vector = [] # dimension 1 by T
        action_rhs = [] # dimension T x 1
        obs = env.reset()
        #the third value of the observations is the starting angle of the pole,
        #so set this based on our starting value
        #states.append(obs)
        i = 0
        done = False
        while i < timebound or done:
            #action, _states = candidate.predict(obs)
            action = candidate.predict(obs[0])
            newobs = np.append(obs, action)#1 if action == 1 else -1)
            prev_states_augmented.append(newobs)
            action_rhs.append(action)#1 if action == 1 else -1)
            obs, rewards, done, info = env.step([action])
            states.append(obs)
            i += 1
        return np.array(states), np.array(prev_states_augmented), action_rhs

#train_expert_policy()
def evaluate_policy(model, expert=True):
    reward = 0
    obs = env.reset()
    dones = False
    while not dones:
        if expert:
            action, _states = model.predict(obs[0])
        else:
            action = model.predict(obs[0])
            action = [action]
        #print(action)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
    return reward

def generate_initial_dataset(model, numexs):
    positiveexamples = []
    while len(positiveexamples) < numexs:
        obs = env.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            positiveexamples.append((obs, action[0]))
            obs, rewards, dones, info = env.step(action)
            #print("Action in initial build is:", action)
    return positiveexamples






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, help='Train a model before running the rest of the script', type=bool)
    args = parser.parse_args()
    if args.train:
        train_expert_policy()
    model = A2C.load("a2c_cartpole")
    verifier = CartPolePIDCorrectnessVerifier()
    learner = CartPolePIDLearner()
    initial_positive = generate_initial_dataset(model, 150)
    groundtruth = CartPoleGroundTruth(initial_positive)
    system = CartPoleModelSystem(learner, verifier, groundtruth)
    candidate = system.get_verifiable_decision_tree(50, .15)
    print(evaluate_policy(learner, expert=False))
    candidate = system.get_verifiable_decision_tree(100, .15)
    print(evaluate_policy(candidate, expert=False))

    # load positive data from expert.py
    file = open("pos_data", "rb")
    pos_data = pickle.load(file)





if __name__ == '__main__':
    main()
# expert.train_expert(saved=True)
    #model = A2C.load("a2c_cartpole")
    #print(evaluate_policy(model))

    # obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     print("actions:", action, _states)
#     obs, rewards, dones, info = env.step(action)
#     newobs = np.append(obs, action)
#     print(obs, rewards)
#     env.render()
#     if dones:
#         break
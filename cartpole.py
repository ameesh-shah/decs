import gym
import numpy as np
import argparse
from z3 import *
from sklearn.tree import DecisionTreeClassifier
from model_system import ModelSystem, Learner, Verifier
from ground_truth import GroundTruth
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c import a2c
from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy
from stable_baselines import DQN, PPO2, A2C

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

def train_expert_policy():
    model = a2c.A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000)
    model.save("a2c_cartpole")

class CartPoleGroundTruth(GroundTruth):

    def __init__(self, model, initlen):
        self.model = model
        self.initlen = initlen
        positive_examples = self.generate_initial_dataset()
        super().__init__(True, positive_examples)

    def generate_initial_dataset(self):
        positiveexamples = []
        while len(positiveexamples) < self.initlen:
            obs = env.reset()
            dones = False
            while not dones:
                action, _states = self.model.predict(obs)
                positiveexamples.append((obs, action[0]))
                obs, rewards, dones, info = env.step(action)
                #print("Action in initial build is:", action)
        return positiveexamples

    def query(self, inp, aggregate=True):
        env.reset()
        inp = np.array([list(inp)])
        obs = inp
        dones = False
        while not dones:
            action, _states = self.model.predict(obs)
            print("result of query is :", obs, action[0])
            self.positive_examples.append((inp, action[0]))
            obs, rewards, dones, info = env.step(action)
        return action


class CartPoleModelSystem(ModelSystem):

    def __init__(self, learner, verifier, groundtruthmodel):
        super().__init__(learner, verifier)
        self.groundtruthmodel = groundtruthmodel

    def train_candidate(self):
        inputs, outputs = map(list, zip(*self.groundtruthmodel.positive_examples))
        inputs, outputs = np.array(inputs).squeeze(), np.array(outputs).squeeze()
        model = self.learner.synthesize_candidate(inputs, outputs)
        return model

    def check_candidate(self, candidate, yboundary):
        verification = self.verifier.verify(candidate, yboundary)
        return verification

    def get_verifiable_decision_tree(self, max_iters, yboundary):
        for itr in range(max_iters):
            print("training decision tree candidate")
            candidate = self.train_candidate()
            retval = self.check_candidate(candidate, yboundary)
            if retval == True:
                print("required {} verification iterations to get satisfiable solution.".format(itr))
                return candidate
            else:
                print("Adding counterexample")
                self.groundtruthmodel.query(retval)
        print("Exhausted max number of iterations without finding verified solution.")
        return candidate


class CartPoleDecisionTreeLearner(Learner):

    def __init__(self, max_depth=4):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def synthesize_candidate(self, positive_x, positive_y):
        self.model.fit(positive_x, positive_y)
        return self.model

class CartPoleDecisionTreeCorrectnessVerifier(Verifier):

    def __init__(self):
        #placeholder
        self.states = [0,0,0,0]


    def verify(self, candidate, yboundary, timebound=10):
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
            print('counter example found:')
            mod = self.solver.model()
            #TODO: a less hacky way to convert to python values
            es0 = int('{}'.format(mod[self.states[0]].numerator())) / int('{}'.format(mod[self.states[0]].denominator()))
            es1 = int('{}'.format(mod[self.states[1]].numerator())) / int('{}'.format(mod[self.states[1]].denominator()))
            es2 = int('{}'.format(mod[self.states[2]].numerator())) / int('{}'.format(mod[self.states[2]].denominator()))
            es3 = int('{}'.format(mod[self.states[3]].numerator())) / int('{}'.format(mod[self.states[3]].denominator()))
            print(es0, es1, es2, es3)
            return (es0, es1, es2, es3)
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
            action = candidate.predict(obs)
            newobs = np.append(obs, action.item())#1 if action == 1 else -1)
            prev_states_augmented.append(newobs)
            action_rhs.append(action.item())#1 if action == 1 else -1)
            obs, rewards, done, info = env.step(action)
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
            action, _states = model.predict(obs)
        else:
            action = model.predict(obs)
            action = [action.item()]
        #print(action)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
    return reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, help='Train a model before running the rest of the script', type=bool)
    args = parser.parse_args()
    if args.train:
        train_expert_policy()
    model = A2C.load("a2c_cartpole")
    verifier = CartPoleDecisionTreeCorrectnessVerifier()
    learner = CartPoleDecisionTreeLearner()
    groundtruth = CartPoleGroundTruth(model, 150)
    system = CartPoleModelSystem(learner, verifier, groundtruth)
    candidate = system.get_verifiable_decision_tree(100, .15)
    print(evaluate_policy(candidate, expert=False))

if __name__ == '__main__':
    main()
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
# OpenAI Gym Cartpole implementation.
# Using a Policy Gradient and a 2-node 2-layer network.
# By Tom Jacobs
#
# Runs on Python 3.
# Originally based on https://github.com/kvfrans/openai-cartpole
# You can submit it to the OpenAI Gym scoreboard by entering your OpenAI API key and enabling submit below.
# It will submit only if it is considered solved.

import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

# Submit it?
submit = False
api_key = ''

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [4, 2])   # Parameters
        state = tf.placeholder("float", [None, 4])              # World state
        actions = tf.placeholder("float", [None, 2])            # Actions - move left, or right
        advantages = tf.placeholder("float", [None, 1])         # Ooh, advantages
        linear = tf.matmul(state, params)                       # Combine
        probabilities = tf.nn.softmax(linear)                   # Probabilities
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss) # Learning rate 0.1, aim to minimize loss
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 4])       # World state
        newvals = tf.placeholder("float", [None, 1])
        w1 = tf.get_variable("w1", [4, 2])              # Value gradient is *w1+b1, Relu, *w2+b2. 4, 2, 1.
        b1 = tf.get_variable("b1", [2])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [2, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals                     # How different did we do from expected?
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess, render=False):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    for t in range(200):
        # Render
        if render:
            env.render()

        # Calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0, 1) < probs[0][0] else 1

        # Record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)

        # Take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        # Done?
        if done:
            break

    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # Calculate discounted Monte Carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.99
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

        # Advantage: how much better was this action than normal?
        advantages.append(future_reward - currentval)

        # Update the value function towards new return
        update_vals.append(future_reward)

    # Update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    # Update policy function
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    # Done
    return totalreward

# Go
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'cartpole', force=True)
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Learn
results = []
for i in range(200):
    reward = run_episode(env, policy_grad, value_grad, sess)
    results.append(reward)
    if reward < 200:
        print("Fail at {}".format(i))

# Run 100
print("Running 100 more.")
t = 0
for _ in range(100):
    reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
    results.append(reward)
print("Got {}".format(t / 100))

# Submit
if submit and t/100 > 195:
    # Submit to OpenAI Gym
    print("Submitting to gym...")
    gym.scoreboard.api_key = api_key
    env.close()
    gym.upload('cartpole')
else:
    # Plot
    #plt.plot(results)
    #plt.xlabel('Episode')
    #plt.ylabel('Rewards')
    #plt.title('Rewards over time')
    #plt.show()

    # Show ten
    print("Showing 3")
    for _ in range(3):
        reward = run_episode(env, policy_grad, value_grad, sess, True)
import gym
import math
import numpy as np
from random import random

learning_rate = 0.7
discount_factor = 0.95
env = gym.make("MountainCar-v0")
initial_state = env.reset()


def choose_action(state,weight_vector, epsilon):
    greedy = random() > epsilon
    if greedy:
        curr_pos = state[0]
        curr_vel = state[1]
        actions_state_value = [np.dot(state_to_feature_function(curr_pos, curr_vel, i), weight_vector) for i in
                               [0, 1, 2]]
        max = actions_state_value[0]
        chosen_action = 0
        for index, a in enumerate(actions_state_value):
            if a > max:
                max = a
                chosen_action = index
    else:
        chosen_action = int(random() * 3)

    return chosen_action


def choose_greedy_action(state, weight_vector):
    curr_pos = state[0]
    curr_vel = state[1]
    actions_state_value = [np.dot(state_to_feature_function(curr_pos, curr_vel, i), weight_vector) for i in
                           [0, 1, 2]]
    max = actions_state_value[0]
    chosen_action = 0
    for index, a in enumerate(actions_state_value):
        if a > max:
            max = a
            chosen_action = index

    return (max, chosen_action)


def state_to_feature_function(position, velocity, action):
    if position == 0.6:
        position = 0.59
    if velocity == 0.07:
        velocity = 0.069
    if velocity == -0.07:
        velocity = -0.069
    position_index = math.floor(((math.floor(position * 10) / 10) + 1.2) * 10)
    velocity_index = math.floor(((math.floor(velocity * 100) / 100) + 0.07) * 100) + 18
    action_index = action + 32
    features = [0] * 35
    features[position_index] = 1
    features[velocity_index] = 1
    features[action_index] = 1

    # print("intial state is: {} , vel_index = {} action_index = {} , position_index = {}".
    #       format(initial_state, velocity_index, action_index, position_index))
    return features

def dec_epsilon(epsilon_var):
    return epsilon_var-0.1 if epsilon_var > 0.1 else epsilon_var


def qlearning(weight_vector):
    """Q Learning"""
    epsilon = 0.5
    done = False
    action = 0, 1, 2
    number_of_episodes = 0
    while not done:
        initial_state = env.reset()
        # env.render()
        curr_state = initial_state
        episode_done = False
        num_of_steps = 0
        while not episode_done:
            chosen_action = choose_action(curr_state,weight_vector, epsilon)  # chosen
            curr_feature_vector = state_to_feature_function(curr_state[0], curr_state[1], chosen_action)
            if curr_feature_vector[17] == 1:
                episode_done = True
            # curr_feature_vector = X(current state,chosen action)
            curr_as_value = np.dot(curr_feature_vector, weight_vector)
            # curr_as_value = q(current state, chosen action, w)
            next_state, curr_reward, dont_care, addition_inf = env.step(chosen_action)
            # env.render()
            ga_value, ga_index = choose_greedy_action(next_state,weight_vector)
            # ga_value = greedy action value = q(next state, optimal action, w)
            # ga_index = index of greedy action chosen
            delta_w = np.dot(curr_feature_vector,
                             learning_rate * (curr_reward + discount_factor * ga_value - curr_as_value))
            weight_vector = np.add(delta_w, weight_vector)
            curr_state = next_state
            num_of_steps += 1
            if num_of_steps == 1000:
                episode_done = True

        if number_of_episodes % 100 == 0:
            print ("run number= ", number_of_episodes , "num of steps = ", num_of_steps)
        if number_of_episodes % 100 == 0:
            epsilon = dec_epsilon(epsilon)
        number_of_episodes += 1
        if number_of_episodes == 100000:
            done = True
            env.close()



qlearning([0] * 35)
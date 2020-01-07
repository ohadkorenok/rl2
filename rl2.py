import gym
import math
import numpy as np
from random import random

learning_rate = 0.7
discount_factor = 0.95
env = gym.make("MountainCar-v0")
initial_state = env.reset()


def choose_action(state, weight_vector, epsilon):
    greedy = random() > epsilon
    if greedy:
        curr_pos = state[0]
        curr_vel = state[1]
        left = np.dot(state_to_feature_function(curr_pos, curr_vel, 0), weight_vector)
        stay = np.dot(state_to_feature_function(curr_pos, curr_vel, 1), weight_vector)
        right = np.dot(state_to_feature_function(curr_pos, curr_vel, 2), weight_vector)
        actions = [left, stay, right]

        chosen_action = np.argmax(actions)
    else:
        chosen_action = int(random() * 3)
        if chosen_action == 3:
            chosen_action = 2

    return chosen_action


def choose_greedy_action(state, weight_vector):
    curr_pos = state[0]
    curr_vel = state[1]
    left = np.dot(state_to_feature_function(curr_pos, curr_vel, 0), weight_vector)
    stay = np.dot(state_to_feature_function(curr_pos, curr_vel, 1), weight_vector)
    right = np.dot(state_to_feature_function(curr_pos, curr_vel, 2), weight_vector)
    actions = [left, stay, right]
    max_value = max(actions)
    max_index = np.argmax(actions)

    return (max_value, max_index)


def state_to_feature_function(position, velocity, action):

    position_index = math.floor(position * 10) + 12
    velocity_index = math.floor(velocity * 100) + 7
    if position >= 0.6:
        position_index = 17
    if velocity <= -0.07:
        velocity_index = 0
    if velocity >= 0.07:
        velocity_index = 13

    table_index = position_index * 42 + velocity_index*3 + action
    features = [0] * 756
    features[table_index] = 1
    return features


def dec_epsilon(epsilon_var):
    new_epsilon = epsilon_var - 0.01
    if new_epsilon > 0.01:
        return new_epsilon
    else:
        return 0.01

def qlearning(weight_vector):
    """Q Learning"""
    means = []
    temp_arr = []
    epsilon = 0.1
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
            chosen_action = choose_action(curr_state, weight_vector, epsilon)  # chosen action = 0 -> left 1-> stay 2->right
            curr_feature_vector = state_to_feature_function(curr_state[0], curr_state[1], chosen_action)
            # curr_feature_vector = X(current state,chosen action)

            if curr_state[0] >= 0.5:
                episode_done = True

            curr_as_value = np.dot(curr_feature_vector, weight_vector)
            # curr_as_value = q(current state, chosen action, w)
            next_state, curr_reward, dont_care, addition_inf = env.step(chosen_action)
            # env.render()
            ga_value, ga_index = choose_greedy_action(next_state, weight_vector)
            # ga_value = greedy action value = q(next state, optimal action, w)
            # ga_index = index of greedy action chosen
            delta_w = np.dot(curr_feature_vector,
                             learning_rate * (curr_reward + discount_factor * ga_value - curr_as_value))
            weight_vector = np.add(delta_w, weight_vector)
            curr_state = next_state
            num_of_steps += 1
            if num_of_steps == 1000:
                episode_done = True

        if number_of_episodes % 75 == 0:
            epsilon = dec_epsilon(epsilon)
        if number_of_episodes % 10 == 0:
            temp_arr.append(num_of_steps)
        if number_of_episodes % 100 == 0:
            means.append(np.mean(temp_arr))
            temp_arr.clear()
        if number_of_episodes % 10 == 0:
            print("\nrun number= ", number_of_episodes, "num of steps = ", num_of_steps , "\nweights: ",weight_vector)
        number_of_episodes += 1
        if number_of_episodes == 15000:
            print(means)
            done = True
            # env.close()

qlearning([0] * 756)
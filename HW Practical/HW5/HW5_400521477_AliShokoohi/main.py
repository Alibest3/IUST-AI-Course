import utils
import flappy_bird_gym
import random
import time
import numpy as np


class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 1
        self.epsilon = 0.9  # change to proper value
        self.alpha = 0.9  # change to proper value
        self.states = set()
        self.iterations = iterations
        
        
        self.bins = 10
        self.discrete_states = [
            np.linspace(0, 1.7, num=(self.bins + 1))[1:-1],
            np.linspace(-0.4, 0.5, num=(self.bins + 1))[1:-1],
        ]
        
        self.actions_number = 2
        states_number = self.bins ** len(self.discrete_states)
        
        self.Qtable = np.zeros(shape=(states_number, self.actions_number))

    def policy(self, state):
        # implement the policy
        if (1-self.epsilon) <= np.random.uniform():
            action = np.random.randint(0, self.actions_number)
            
        else:
            action = self.max_arg(state)
            
        return action

    @staticmethod
    def get_all_actions():
        return [0, 1]

    def convert_continuous_to_discrete(self, state):
        discrete = 0
        for i, feature in enumerate(state):
            bin_values = self.discrete_states[i]
            bin_index = np.digitize(feature, bin_values)
            discrete += bin_index * (self.bins ** i)

        self.states.add(discrete)
        return discrete

    def compute_reward(self, prev_info, new_info, done, observation):
        # implement the best way to compute reward base on observation and score
        if done:
            return -1000
        
        if observation[0]<0.3:
            if abs(observation[1])<0.1:
                return 5
            else:
                return -1
            
        if prev_info['score']<new_info['score']:
            return 10
        
        return 0.5
    
    def get_action(self, state):
        # implement the best way to get action base on current state
        next_action = self.policy(state)
        
        return next_action

    def maxQ(self, state):
        # return max Q value of a state
        return np.max(self.Qtable[state, :])

    def max_arg(self, state):
        # return argument of the max q of a state
        return np.argmax(self.Qtable[self.convert_continuous_to_discrete(state)])

    def update(self, reward, state, action, next_state):
        # update q table
        discrete = self.convert_continuous_to_discrete(state)
        discrete_next_state = self.convert_continuous_to_discrete(next_state)
        #self.Qtable[discrete, action] = (1 - self.alpha) * (reward + self.landa * self.maxQ(discrete_next_state) - self.Qtable[discrete, action])
        self.Qtable[discrete, action] = (1-self.alpha) * self.Qtable[discrete, action] + self.alpha * (reward + self.landa *
                                                                                                       self.maxQ(discrete_next_state))
    def update_epsilon_alpha(self):
        # update epsilon and alpha base on iterations
        
        self.epsilon *= (1 - 3e-3) 
        self.alpha = max(1e-5, self.alpha * (1 - 1e-3)) 
        

    def run_with_policy(self, landa):
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        for _ in range(self.iterations):
            observation = env.reset()
            done = False
            print(_)
            while not done:
                action = self.get_action(observation)  # policy affects here
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                reward = self.compute_reward(prev_info, info, done, observation)
                self.update(reward, this_state, action, observation)
                
            self.update_epsilon_alpha()
              
        env.close()


    def run_with_no_policy(self, landa):
        self.landa = landa
        # no policy test
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        done = False
        while not done:
            action = self.max_arg(observation)
            
            observation, reward, done, info = env.step(action)
            
            env.render()
            time.sleep(1 / 30)  # FPS
        
        env.close()

    def run(self):
        self.run_with_policy(1)
        self.run_with_no_policy(1)


program = SmartFlappyBird(iterations=3000)
program.run()



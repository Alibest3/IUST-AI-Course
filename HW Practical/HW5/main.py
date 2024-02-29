import utils
import flappy_bird_gym
import random
import time


class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 1
        self.epsilon = 0  # change to proper value
        self.alpha = 0  # change to proper value
        self.iterations = iterations

    def policy(self, state):
        # implement the policy
        return NotImplemented

    @staticmethod
    def get_all_actions():
        return [0, 1]

    @staticmethod
    def convert_continuous_to_discrete(state):
        # implement the best way to convert continuous distance values to discrete values
        return NotImplemented

    def compute_reward(self, prev_info, new_info, done, observation):
        # implement the best way to compute reward base on observation and score
        return NotImplemented

    def get_action(self, state):
        # implement the best way to get action base on current state
        # you can use `utils.flip_coin` and `random.choices`
        return NotImplemented

    def maxQ(self, state):
        # return max Q value of a state
        return NotImplemented

    def max_arg(self, state):
        # return argument of the max q of a state
        return NotImplemented

    def update(self, reward, state, action, next_state):
        # update q table
        return NotImplemented

    def update_epsilon_alpha(self):
        # update epsilon and alpha base on iterations
        return NotImplemented

    def run_with_policy(self, landa):
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        for _ in range(self.iterations):
            while True:
                action = self.get_action(observation)  # policy affects here
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                reward = self.compute_reward(prev_info, info, done, observation)
                self.update(reward, this_state, action, observation)
                self.update_epsilon_alpha()
                if done:
                    observation = env.reset()
                    break
        env.close()


    def run_with_no_policy(self, landa):
        self.landa = landa
        # no policy test
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        while True:
            action = self.get_action(observation)
            prev_info = info
            observation, reward, done, info = env.step(action)
            reward = self.compute_reward(prev_info, info, done, observation)
            env.render()
            time.sleep(1 / 30)  # FPS
            if done:
                break
        env.close()

    def run(self):
        self.run_with_policy(1)
        self.run_with_no_policy(1)


program = SmartFlappyBird(iterations=100)
program.run()



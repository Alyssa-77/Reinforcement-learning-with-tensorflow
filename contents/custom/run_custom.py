import gym

env = gym.make('GridWorld-v1')
env.reset()
for i in range(200):
    env.render()
env.close()
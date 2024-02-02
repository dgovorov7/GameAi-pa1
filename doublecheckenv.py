from snakeenv import SnekEnv

env = SnekEnv()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done: #not done:
		random_action = env.action_space.sample()
		print(f"Episode {episode}: action",random_action)
		obs, reward, done, trucated, info = env.step(random_action)
		print(f"Episode {episode}: reward",reward)
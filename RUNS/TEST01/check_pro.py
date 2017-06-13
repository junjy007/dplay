import json
import numpy as np
import matplotlib.pyplot as plt

with open ('rec.json', 'r') as f:
	l = json.load(f)
rewards = np.asarray(l['episode_reward_history'])
n = rewards.shape[0]
print "#. Records", n
epv = rewards.reshape((n/1000, 1000)).mean(axis=1)
plt.plot(epv)
plt.show()



	

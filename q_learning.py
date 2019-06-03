import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colors

rewards =[
			[-1, -1, -1, -100, -100, 1000, -1],
			[-1, -1, -1, -1, -100, -1, -100],
			[-1, -100, -100, -1, -100, -1, -1],
			[-1, -1, -100, -1, -1, -100, -1],
			[-1, -1, -100, -1, -1, -1, -1]
		]

start_state = (3,0)

n_row_max = 5;
n_col_max = 7;
n_actions = 8;

def td_policy_evaluation(policy, n_episodes, discount_factor, alpha):
	""" TD(0) policy evaluation
	this function will return both V(s), and Q(s,a) for each grid cell
	"""
	Q = defaultdict(lambda: np.zeros(8))
	V = defaultdict(lambda: 0)
	for i in range(n_episodes):
		state = start_state;
		is_done = False
		
		# pick action
		action = np.random.choice(n_actions, p=policy)
		while not is_done:
			
			# observe next state and reward
			next_state = get_next_state(state, action)
			reward = rewards[next_state[0]][next_state[1]]
			if reward == -100 or reward == 1000:
				is_done = True
			
			# pick next action
			next_action = np.random.choice(n_actions, p=policy)
			
			# V(s)
			target = reward + discount_factor * V[next_state]
			delta = target - V[state]
			V[state] += alpha * delta
			
			# Q(s, a)
			target_Q = reward + discount_factor * Q[next_state][next_action]
			delta_Q = target_Q - Q[state][action]
			Q[state][action] += alpha * delta_Q
			
			state = next_state
			action = next_action

	return V, Q
	
def q_learning(n_episodes, discount_factor, alpha, epsilon, Q, V):
	""" Q-learning, non-deterministic
	this function will return both V(s), and Q(s,a) for each grid cell
	"""
	if Q is None or V is None: # if initial Q is not setted from parameters
		Q = defaultdict(lambda: np.zeros(8))
		V = defaultdict(lambda: 0)
		
	epsilon_greedy_policy = create_epsilon_greedy_policy(Q, epsilon, n_actions)
	for i in range(n_episodes):
		state = start_state;
		is_done = False
		while not is_done:
			# pick action
			policy = epsilon_greedy_policy(state)
			action = np.random.choice(n_actions, p=policy)
			
			# observe next state and reward
			next_state = get_next_state(state, action)
			reward = rewards[next_state[0]][next_state[1]]
			if reward == -100 or reward == 1000:
				is_done = True
			
			# pick best next action from next_state
			next_action = np.argmax(Q[next_state])
			
			# Q(s, a)
			target = reward + discount_factor * Q[next_state][next_action]
			delta = target - Q[state][action]
			Q[state][action] += alpha * delta
			
			# V(s)
			target_V = reward + discount_factor * V[next_state]
			delta_V = target_V - V[state]
			V[state] += alpha * delta_V
			
			state = next_state

	return V,Q
	
def create_epsilon_greedy_policy(Q, epsilon, n_actions):
	def policy_function(state):
		policy = np.ones(n_actions, dtype=float)*epsilon / n_actions
		best_action = np.argmax(Q[state])
		policy[best_action] += (1 - epsilon)
		return policy
	return policy_function

def get_next_state(state, action):

	#non-deterministic
	deviation_probs = [0.2, 0.6, 0.2]
	deviation = np.random.choice(np.arange(-1, 2), p=deviation_probs)
	action += deviation
	if action < 0:
		action = n_actions - 1
	elif action > n_actions - 1:
		action = 0
		
	if action == 0: # left
		row = state[0]
		col = state[1]-1 if  state[1] > 0 else 0
	elif action == 1: # up-left
		row = state[0]-1 if state[0] > 0 else 0
		col = state[1]-1 if state[1] > 0 else 0
	elif action == 2: # up
		row = state[0]-1 if state[0] > 0 else 0
		col = state[1]
	elif action == 3: # up-right
		row = state[0]-1 if state[0] > 0 else 0
		col = state[1]+1 if state[1] < n_col_max - 1 else n_col_max - 1
	elif action == 4: # right
		row = state[0]
		col = state[1]+1 if state[1] < n_col_max - 1 else n_col_max - 1
	elif action == 5: # down-right
		row = state[0]+1 if state[0] < n_row_max - 1 else n_row_max - 1
		col = state[1]+1 if state[1] < n_col_max - 1 else n_col_max - 1
	elif action == 6: # down
		row = state[0]+1 if state[0] < n_row_max - 1 else n_row_max - 1
		col = state[1]
	elif action == 7: # down-left
		row = state[0]+1 if state[0] < n_row_max - 1 else n_row_max - 1
		col = state[1]-1 if state[1] > 0 else 0
	return (row, col)
	
def display_value_function(V):
	""" display value function to the console
	"""
	for i in range(n_row_max):
		for j in range(n_col_max):
			print(V[i,j] , end=" ")
		print()

		
def display_arrow_policy(Q):
	""" display policy using matplotlib
	"""
	labels_int = [[0 for i in range(n_col_max)] for j in range(n_row_max)]
	for i in range(n_row_max):
		for j in range(n_col_max):
			if rewards[i][j] == -100:
				a = 10
			elif rewards[i][j] == 1000:
				a = 9
			else:
				a = np.argmax(Q[i,j])
			labels_int[i][j] = a
		print()
	cmap = colors.ListedColormap(['white'])

	fig, ax = plt.subplots()
	ax.imshow(labels_int, cmap=cmap)
	ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	ax.set_xticks(np.arange(-.5, 10, 1));
	ax.set_xticklabels([])
	ax.set_yticks(np.arange(-.5, 10, 1));
	ax.set_yticklabels([])
	ax.axis('image')

	for (j,i),label in np.ndenumerate(labels_int):
		ax.text(i,j,get_symbol(label),ha='center',va='center')

	plt.show()
					
def get_symbol(a):
	""" get symbol from a given integer
	"""
	if (a == 0):
		symbol = '←'
	elif (a == 1):
		symbol = '↖  '
	elif (a == 2):
		symbol = '↑'
	elif (a == 3):
		symbol = '↗ '
	elif (a == 4):
		symbol = '→'
	elif (a == 5):
		symbol = '↘ '
	elif (a == 6):
		symbol = '↓'
	elif (a == 7):
		symbol = '↙'
	elif (a == 8):
		symbol = 'S'
	elif (a == 9):
		symbol = 'G'
	else:
		symbol = 'W'
	return symbol

	
		
# Policy: move to the right with probability 0.5. Move up with probability 0.25, move down with probability 0.25.
policy = [0, 0, 0.25, 0, 0.5, 0, 0.25, 0]
V,Q = td_policy_evaluation(policy, 1000, 1, 0.01)
print("-------------------------------------------------------")
print ("state value V(s) for each grid cell. episodes=1000, discount_factor=1.0, alpha=0.1")
print("-------------------------------------------------------")
display_value_function(V)

V,Q = q_learning(10000, 1, 0.1, 0.1, Q, V)
print("-------------------------------------------------------")
print ("state value V(s) for each grid cell. episodes=10000, discount_factor=1.0, alpha=0.1, epsilon=0.1")
print("-------------------------------------------------------")
display_value_function (V)
display_arrow_policy(Q)
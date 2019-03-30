from MDP import MDP
import numpy as np

class BellmanDPSolver(object):
	def __init__(self):
		self.MDP = MDP()
		# Set the size of our grid
		self.domain_size = (5, 5)

		# TODO: I picked the threshold value out of thin air
		self.threshold = 0.9

		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]

		self.initVs()

	def initVs(self):
		# Initialise all our state values to 0
		self.state_values = dict(((i, j), 0)
								for i in range(self.domain_size[0])
									for j in range(self.domain_size[1]))
		self.state_values['GOAL'] = 0
		self.state_values['OUT'] = 0

	def BellmanUpdate(self):
		actions = {}
		# Order of actions
		order_actions = ["DRIBBLE_UP", "DRIBBLE_DOWN", "DRIBBLE_LEFT",  "DRIBBLE_RIGHT", "SHOOT"]

		# Loop over all our states
		for state, value in self.state_values.items():
			summed_action_results = {}
			# Loop over all possible actions
			for action in self.actions:

				# Get a dictionary of potential next states and their probability
				nextStatesProbs = self.MDP.probNextStates(state, action)

				# Calculate the rewards and values for each potential next state
				nextStateRewards = {}
				nextStateValues = {}
				for nextState in nextStatesProbs:
					nextStateRewards[nextState] = self.MDP.getRewards(state, action, nextState)
					nextStateValues[nextState] = self.state_values[nextState]

				# Calculate the return for this action in this state
				action_result = []
				for nextState in nextStatesProbs:
					prob = nextStatesProbs[nextState]
					reward = nextStateRewards[nextState]
					value = nextStateValues[nextState]
					action_result.append(prob*(reward + self.threshold*value))

				summed_action_result = np.sum(action_result)

				# Store the summed potential returns for this action in this state
				summed_action_results[action] = summed_action_result

			# Pick the action with the max return in this state
			maxReturn = max(summed_action_results.values())
			bestActions = [k for k, v in summed_action_results.items() if v == maxReturn]
	
			orderedBestActions = sorted(list(bestActions), key=lambda x: order_actions.index(x))

			actions[state] = orderedBestActions

			self.state_values[state] = maxReturn

		return self.state_values, actions

if __name__ == '__main__':
	solution = BellmanDPSolver()
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

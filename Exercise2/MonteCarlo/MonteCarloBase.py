#!/usr/bin/env python3
# encoding utf-8
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent

import random
import operator

import numpy as np

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.epsilon = epsilon

		# Initialise the size of the environment
		self.domain_size = (5, 5)
		# Initialise the allowed actions
		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]
		# Initialise the Q-value state action pairs to be 0
		self.q_values = dict(((i, j), dict(action, 0))
                         for i in range(self.domain_size[0])
                             for j in range(self.domain_size[1])
                                for action in self.actions)
		# Initialise an empty list for all the "Returns" values
		self.returns = dict()
		# Set the random seed
		random.seed(123)
		self.state = (0, 0)

		self.total_reward = 0

	def learn(self):
		''' TODO:
		1. Figure out what G is: it's the "return following the first occurence of state s"
		2. Get the reward for time t+1
		3. Check if the pair of state and action has occurred before
		4. If it hasn't then append G to Returns(State, Action)
		5. Add the return for this state and action to the Q values
		6. Find the action that results in the largest Q value
		'''


	def toStateRepresentation(self, state):
		return state

	def setExperience(self, state, action, reward, status, nextState):
		'''
		This method should update the rewards and state action pairs from the
		reward and returned status etc. 
		'''
		# TODO: If the status is complete then do what you need to do on completion
		self.status = status

		self.reward = reward
		self.action = action

		self.total_reward += reward

		self.state_ = nextState

		# Set the state/action pair to have the passed reward
		# self.q_values[state][action] += reward

		# Set the current state to be the nextState variable passed in
		self.state = nextState

	def setState(self, state):
		self.state = state

	def reset(self):
		raise NotImplementedError

	def act(self):
		'''
		This should return the action that this should take based on the state
		'''
		if np.random.rand() < self.epsilon:
			# Pick the best action
			action = max(self.q_values[self.state].keys(), key=(lambda k: self.q_values[self.state][k]))
		else:
			# Pick a random action
			action = self.actions[random.randint(0, len(self.actions)-1)]

		return action

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon


if __name__ == '__main__':
	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = 1, numTeammates = 0, agentId = 'tom')
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = 10
	numTakenActions = 0

	# Run training Monte Carlo Method
	for episode in range(numEpisodes):
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = 0.9 #agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()

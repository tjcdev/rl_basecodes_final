#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent

import numpy as np

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon

		self.domain_size = (6, 5)
		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]

		# Initialise the Q-value state action pairs to be 0
		self.q_values = dict(((i, j), dict(action, 0))
                         for i in range(self.domain_size[0])
                             for j in range(self.domain_size[1])
                                for action in self.actions)
		
		self.action_ = ''
		self.state_ = (0, 0)

		self.total_reward = 0
		self.reward = 0

		self.status = ''

	def learn(self):
		q = self.q_values[self.state][self.action]
		q_ = self.q_values[self.state_][self.action_]

		# TODO: I'm not sure if I have to actually update the value
		# or do I just return it
		self.q_values[self.state][self.action] = q + self.learningRate*(self.reward + self.discountFactor*q_ - q)

		return self.q_values[self.state][self.action]

	def act(self):
		'''
		This should return the action that this agent should take based on the state
		'''
		if np.random.rand() < self.epsilon:
			# Pick the best action
			action = max(self.q_values[self.state].keys(), key=(lambda k: self.q_values[self.state][k]))
		else:
			action = self.actions[np.random.randint(0, len(self.actions))]

		return action

	def setState(self, state):
		self.state = state

	def setExperience(self, state, action, reward, status, nextState):
		# Set the current state to the next state
		self.state_ = nextState
		self.action_ = max(self.q_values[nextState].keys(), key=(lambda k: self.q_values[nextState][k]))

		# The action that was taken
		self.action = action

		# The reward that was returned
		self.reward = reward

		self.total_reward += reward

		# TODO: Handle the scenario where the status is an end 
		self.status = status
	def toStateRepresentation(self, state):
		return state

	def reset(self):
		raise NotImplementedError

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':
	
	numEpisode = 10
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(self, numTakenActions, episode)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	

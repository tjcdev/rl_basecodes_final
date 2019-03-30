#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent

import numpy as np

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.state = (0,0)
		self.learningRate = 0.9
		self.epsilon = 0.1
		self.discountFactor = discountFactor

		self.total_reward = 0
		self.reward

		self.action_ = "DRIBBLE_UP"
		self.state_ = ()

		self.status = 0

		# Initialise the size of the environment
		self.domain_size = (5, 5)
		
		# Initialise the allowed actions
		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]
		
		# Initialise the Q-value state action pairs to be 0
		q_values = {}
		for i in range(5):
			for j in range(5):
				actions = {}
				for action in self.actions:
					actions[action] = 0
				q_values[(i, j)] = actions

	def learn(self):
		q = self.q_values[self.state][self.action]
		q_ = self.q_values[self.state_][self.action_]
		
		# Set the update q_value
		self.q_values[self.state][self.action] = q + self.learningRate*(self.reward + self.discountFactor*q_  - q) 

		return self.q_values[self.state][action]

	def act(self):
		if np.random.rand() < epsilon:
			action = max(self.q_values[self.state].keys(), key=(lambda k: self.q_values[self.state][k]))
		else:
			action = self.actions[np.random.randint(0, len(self.actions))]
		return action

	def toStateRepresentation(self, state):
		return state

	def setState(self, state):
		self.state = state

	def setExperience(self, state, action, reward, status, nextState):
		# Reward
		self.reward = reward
		# The action that was taken
		self.action = action
		# Set the status
		# TODO: handle the point where the status is an end state
		self.status = status

		# Set the total reward
		self.total_reward += reward	

		# Set the state and action primes
		self.action_ = max(self.q_values[nextState].keys(), key=(lambda k: self.q_values[nextState][k]))
		self.state_ = nextState



	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def reset(self):
		self.state = (0,0)
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# TODO: decay these things as I should but for now just leave it
		return (self.learningRate, self.epsilon)

if __name__ == '__main__':
	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(self, numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	

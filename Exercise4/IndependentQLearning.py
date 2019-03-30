#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
import numpy as np
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.state = (0,0)
		self.action = ''

		self.learningRate = learningRate
		self.epsilon = epsilon
		self.discountFactor = discountFactor

		self.total_reward = 0

		# Initialise the size of the environment
		self.domain_size = (5, 5)
		
		# Initialise the allowed actions
		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]
		
		# Initialise the Q-value state action pairs to be 0
		self.q_values = {}
		for i in range(5):
			for j in range(5):
				actions = {}
				for action in self.actions:
					actions[action] = 0
				self.q_values[(i, j)] = actions

	def setExperience(self, state, action, reward, status, nextState):
		# Reward
		self.reward = reward
		
		# Set the state and action primes
		self.action_ = max(self.q_values[nextState].keys(), key=(lambda k: self.q_values[nextState][k]))
		
		self.state_ = (0,0)

		# Set the total reward
		self.total_reward += reward

		self.state = state			

		# Set the status
		self.status = status

		# The action taken
		self.action = action
	
	def learn(self):		
		q = self.q_values[self.state][self.action]
		q_ = self.q_values[self.state_][self.action_]
		
		# Set the update q_value
		self.q_values[self.state][self.action] = q + self.learningRate*(self.reward + self.discountFactor*q_  - q) 

	def act(self):
		if np.random.rand() < epsilon:
			action = max(self.q_values[self.state].keys(), key=(lambda k: self.q_values[self.state][k]))
		else:
			action = self.actions[np.random.randint(0, len(self.actions))]
		return action

	def toStateRepresentation(self, state):
		print(state)
		return state

	def setState(self, state):
		self.state = state

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# TODO: decay these things as I should but for now just leave it
		return (self.learningRate, self.epsilon)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=1)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args = parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = 50000
	numTakenActions = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies, nextStateCopies = [], []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
				

#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools

import numpy as np
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()

		self.learingRate = learningRate
		self.epsilon = epsilon
		self.discountFactor = discountFactor
		self.numTeamates = numTeammates

		self.oppoActions = ''

		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]

		# The number of times we have visited each state
		self.num_visits = {}
		for i in range(5):
			for j in range(5):
				self.num_visits[(i, j)] = 0

		self.C_values = {}
		for i in range(5):
			for j in range(5):
				actions = {}
				for action in self.actions:
					actions[action] = 1
				self.C_values[(i, j)] = actions
	
		# Initialise the Q-value state action pairs to be 0
		self.q_values = {}
		for i in range(5):
			for j in range(5):
				actions = {}
				for action in self.actions:
					for action_ in self.actions:
						actions[(action, action_)] = 0
				self.q_values[(i, j)] = actions

		self.state = (0,0)

		self.total_reward = 0
		self.reward = 0

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		# Reward
		self.reward = reward
		
		# Set the state and action primes
		self.state_ = nextState

		# Set the total reward
		self.total_reward += reward

		# Set the status
		self.status = status

		# The action taken
		self.action = action

		# Update the C_values based on the oppoActions
		self.C_values[state][oppoActions] += 1

		# Count the number of times we've visited a particular state
		self.num_visits[state] += 1

		self.oppoActions = oppoActions

	# This method is just for the Q-value equation	
	def learn(self):
		V_ = 0

		sums = {}
		# Find the value of V
		for action in self.actions:
			for oppoAction in self.actions:
				sum += (self.C_values[self.nextState][oppoAction] / self.num_visits[self.nextState])*self.q_values[self.nextState][(self.action, oppoAction)]
			sums[action] = sum
		V_ = max(sums)

		# Set the newest value of Q
		q = self.q_values[self.state][(self.action, self.oppoActions)]
		# TODO: check that the learning rates are in the correct place (since the equation just says alpha)
		self.q_values[self.state][(self.action, self.oppoActions)] = (1-self.learningRate)*q + self.learingRate*(self.reward + self.discountFactor*V_)

	def act(self):
		if np.random.rand() < epsilon:
			sums = {}
			# Find the value of V
			for action in self.actions:
				for oppoAction in self.actions:
					sum += (self.C_values[self.nextState][oppoAction] / self.num_visits[self.nextState])*self.q_values[self.nextState][(self.action, oppoAction)]
				sums[action] = sum
			# Pick the action that maximises the actions
			action = max(sums.keys(), key=(lambda k: sums[k]))
		else:
			action = self.actions[np.random.randint(0, len(self.actions))]
		return action

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self.learingRate = learningRate

	def setState(self, state):
		self.state = state

	def toStateRepresentation(self, rawState):
		return rawState
		
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
	
	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents, seed=randomSeed)
	agents = []
	numAgents = 2
	numEpisodes = 4000
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0

	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], nextObservation[agentIdx])
				agents[agentIdx].learn()
				
			observation = nextObservation

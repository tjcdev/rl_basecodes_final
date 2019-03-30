#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, initVals, winDelta, loseDelta):
		super(WolfPHCAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.winDelta = winDelta
		self.loseDelta = loseDelta

		self.state = (0,0)
		self.action = ''
		self.reward = 0
		self.status = ''
		self.nextState = (0,0)

		self.actions = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]

		self.average_policy = {}
		for i in range(5):
			for j in range(5):
				actions = {}
				for action in self.actions:
					actions[action] = 0
				self.average_policy[(i, j)] = actions

		self.policy = {}
		for i in range(5):
			for j in range(5):
				actions = {}
				for action in self.actions:
					actions[action] = 0
				self.policy[(i, j)] = actions

		self.C = {}
		for i in range(5):
			for j in range(5):
				self.C[(i, j)] = 0

	def setExperience(self, state, action, reward, status, nextState):
		self.state - state
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState

		self.C[self.state] += 1

	def learn(self):
		q = ((1-self.learningRate)*self.q_values[self.state][self.action])
		return q

	def act(self):
		bestAction = max(self.policy[self.nextState].keys(), key=(lambda k: self.policy[self.nextState][k]))
		return bestAction

	def calculateAveragePolicyUpdate(self):
		for action in self.actions:
			self.average_policy[self.state][action] = self.average_policy[self.state][action] + ((1.0/self.C[self.state]) * (self.policy[self.state][action]-self.average_policy[self.state][action]))
		return self.average_policy[self.state]

	def calculatePolicyUpdate(self):
		bestAction = max(self.q_values[self.nextState].keys(), key=(lambda k: self.q_values[self.nextState][k]))
		for action in self.actions:
			if (action==bestAction):
				self.policy[self.state][action] = min(1.0,self.policy[self.state][action] + self.winDelta)
			else:
				self.policy[self.state][action] = max(0.0,self.policy[self.state][action] + self.loseDelta)
		
		return self.policy

	def toStateRepresentation(self, state):
		return state

	def setState(self, state):
		self.state = state

	def setWinDelta(self, winDelta):
		self.winDelta = winDelta

	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.loseDelta, self.winDelta, self.learningRate

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=1)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args = parser.parse_args()

	numOpponents = 1
	numAgents = 2
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99)
		agents.append(agent)

	numEpisodes = 4000
	numTakenActions = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation

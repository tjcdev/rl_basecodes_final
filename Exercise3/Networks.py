import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Define your neural networks in this class. 
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
	def __init__(self):
		super(ValueNetwork, self).__init__()

		self.input_size = 4
		self.action_space_size = 2
		self.hidden_size = 64

		# TODO: should I set the weights and biases
		# TODO: like here https://github.com/raharrasy/DeepRLLectures/blob/master/DeepRLLecture-2/DDPG/Networks.py

		self.linear1 = nn.Linear(self.input_size, self.hidden_size)
		self.layerNorm = nn.LayerNorm(self.hidden_size)

		self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.layerNorm2 = nn.LayerNorm(self.hidden_size)

		self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
		self.layerNorm3 = nn.LayerNorm(self.hidden_size)

		self.acts = nn.Linear(self.hidden_size, self.action_space_size)
		

	def forward(self, inputs) :
		out = self.linear1(inputs)
		out = self.layerNorm(out)
		out = F.relu(out)
		out = self.linear2(out)
		out = self.layerNorm2(out)
		out = F.relu(out)
		out = self.linear3(out)
		out = self.layerNorm3(out)
		out = F.relu(out)
		out = self.acts(out)
		
		return out

import math
import random

class NeuralNet:

	class neuron:
		def __init__ (self, weights): #first weight is c in ax+by+c
			self.weights = weights
			
		def output (self,inputs):
			out = 0
			for i in range(len(inputs)):
				out += inputs[i]*self.weights[i+1]
				
			out += self.weights[0]
			return 1 / (1 + math.exp(-out)) # activation function, squishes output to 0-1
			
		def addInput (self, val):
			self.weights.append(val)


	def __init__ (self, inputNum, outputNum):
		self.inputs = inputNum
		self.outputs = outputNum
		self.hiddenLayNum = math.floor((self.inputs + self.outputs)/2)
		
		self.network = [[], [], []] # inputs, outputs, hidden#1
		for i in range(self.inputs): #create inputs
			self.network[0].append(self.neuron([random.gauss(0,1), random.gauss(0,1)]))
			#using gauss allows large numbers but keeps most around 0
		
		for i in range(self.hiddenLayNum): #create hidden layer 1
			self.network[2].append(self.neuron([random.gauss(0,1)]))
			for j in range(self.inputs):
				self.network[2][len(self.network[2])-1].addInput(random.gauss(0,1))
		
		for i in range(self.outputs): #create outputs
			self.network[1].append(self.neuron([random.gauss(0,1)]))
			for j in range(self.hiddenLayNum):
				self.network[1][len(self.network[1])-1].addInput(random.gauss(0,1))
				
	def run(self, data):
		outputs = [[]]
		for i in range(len(self.network[0])):
			outputs[0].append(self.network[0][i].output([data[i]]))
			
		for i in range(2, len(self.network)):
			outputs.append([])
			for j in range(len(self.network[i])):
				outputs[i-1].append(self.network[i][j].output(outputs[i-2]))
				
		outputs.append([])
		for i in range(len(self.network[1])):
			outputs[len(outputs)-1].append(self.network[1][i].output(outputs[len(outputs)-2]))
		
		return outputs[len(outputs)-1]
	

'''
d:
cd chickentech/Programming/python/"genetic algorithms"

import neuralNet
from neuralNet import NeuralNet
neuron = NeuralNet.neuron([2,1])
out = neuron.output([1])
out
'''

'''
start with two layer, fully connected network (one input is a bias node that always outputs 1).

evolve by adding a new connection between unconnected nodes (connection can jump layers), adding a new node to the middle of a connection pass (set preceding connection weight to 1 and forward connection to original), or changing weights.

connection/disable rules: target node can't be in input history. Must leave one input and output enabled
'''
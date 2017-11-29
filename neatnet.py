import math
import random
from copy import deepcopy


#possible improvements:
#	find equations to replace every constant
#	allow every mutation instead of one at a time (add one for enable/disable)




class NeatNet:

	def __init__ (self, inputNum, outputNum):
		self.noProgressLimit = 15 #number of times a network can fail to improve before dying, default 15
		self.genSize = 150 #number of genomes per generation
		self.goodGeneMultiplier = 3 #number of times more likely a good gene will be picked over the next best



		self.inputNum = inputNum
		self.outputNum = outputNum

		self.geneIdCounter = self.inputNum + self.outputNum + 1
		self.innovNumCounter = self.inputNum + 1
		self.generationInnovs = {'newNodes': [], 'newConnections': []}
		#newNodes : {'geneId': , 'input':inputNodeId, 'output':outputNodeId}
		#newConnections: {'innovNum': , 'input':inputNodeId, 'output':outputNodeId}

		self.generationCounter = 1

		self.resetGeneration()

		self.speciateGeneration()
		self.specLifeCounter = [[0,0] for i in range(len(self.speciesList))] #[counter,lastFitness]

	def resetGeneration (self):
		self.speciesList = [[i for i in range(self.genSize)]]
		# gives index of every genome in generation for that species.
		# starts with all genomes in one species
		self.specLifeCounter = [[0,0] for i in range(len(self.speciesList))]
		self.generation = []
		for counter in range(self.genSize):
			nodes = []
			connections = []
			for i in range(self.inputNum):
				nodes.append([i, 0]) #[geneId, nodeType]  nodeTyp: 0=input, 1=bias, 2=hidden, 3=out

			nodes.append([len(nodes), 1])

			for i in range(self.outputNum):
				nodes.append([len(nodes), 3])

			innov = 0
			for i in range(self.inputNum + 1):
				for o in range(self.outputNum):
					weight = random.uniform(-3,3) #weights
					connections.append([i, o+self.inputNum+1, weight, innov, True])
						# [inNode, outNode, weight, innovation#, enabled]
						# innovation increase for a given mutation only once a generation
						# innovation will always increase between generations
					innov += 1

			nodeValue = [0 for i in range(len(nodes))] #output value of every node
			nodeValue[self.inputNum] = 1 #bias node always outputs 1

			self.generation.append({'nodes':nodes, 'connections':connections, \
				'nodeValue':nodeValue})


	def run (self, inputs, genMembNum):
		if len(inputs) != self.inputNum:
			print("network needs " + repr(self.inputNum) + " inputs")
			exit(0)

		#normalize inputs
		sum = 0
		for i in inputs:
			sum += i
		for index,item in enumerate(inputs):
			if sum != 0:
				inputs[index] = item/sum
			else:
				inputs[index] = 0
			
		i=0
		for index,item in enumerate(self.generation[genMembNum]['nodes']):
			if item[1] == 0:
				self.generation[genMembNum]['nodeValue'][index] = inputs[i]
				i += 1

		nodeSequence = [[i[0]] for i in self.generation[genMembNum]['nodes']]
		#nodes in same order and index as self.nodes and self.nodeValue

		for i in range(len(self.generation[genMembNum]['connections'])): #put each connection input&weight into nodeSequence array
			if self.generation[genMembNum]['connections'][i][4]: #check if connection enabled
				inputNode = self.generation[genMembNum]['connections'][i][0]
				outputNode = self.generation[genMembNum]['connections'][i][1]
				weight = self.generation[genMembNum]['connections'][i][2]

				nodeIndex = 0
				for index,item in enumerate(nodeSequence):
					if item[0] == outputNode:
						nodeIndex = index
						break
				nodeSequence[nodeIndex].append([inputNode, weight])
		# end up with array of
		# nodeSequence[x] = [nodeID, [input1NodeID,weight1], [in2NodeID,weight2], ...]

		answer = []
		for index,node in enumerate(nodeSequence):
			if len(node) > 1: #exlude all input and offset nodes (inputs don't have input weight data)
				sumNodeValue = 0
				for connection in node[1:]: #loops through every connection found on a node, calc node val

					for index2,item2 in enumerate(self.generation[genMembNum]['nodes']): #find input node index to get output value from nodeValue
						if item2[0] == connection[0]:
							sumNodeValue += self.generation[genMembNum]['nodeValue'][index2] * connection[1]

				sumNodeValue = 1 / (1 + math.exp(-4.9*sumNodeValue)) # activation function, squishes output to 0-1, called a sigmoid transfer function (modified with 4.9)
				if self.generation[genMembNum]['nodes'][index][1] == 3:
					answer.append(sumNodeValue)
				else:
					self.generation[genMembNum]['nodeValue'][index] = sumNodeValue

		answerIndex = 0
		for index, item in enumerate(self.generation[genMembNum]['nodes']):
			#writes output nodes results to nodeValue last to ensure there's no recursion on outputs
			if item[1] == 3:
				self.generation[genMembNum]['nodeValue'][index] = answer[answerIndex]
				answerIndex +=1
		return answer


	def findFitness(self, inputs, solutions, genMembNum):
		totalDist = 0
		roundDist = 0
		for index, item in enumerate(inputs):
			testAnswer = self.run(item, genMembNum)

			totalDist += abs(testAnswer[0]-solutions[index][0])
			roundDist += abs(round(testAnswer[0])-solutions[index][0])

		fitness = ((4 - totalDist)/4)**2

		if roundDist == 0:
			print(fitness, "testing")
			fitness = 1

		self.resetNodeValues(genMembNum)
		return fitness


	def breedNet (self, parent1, parent2, equalFitness = False):
	#parent1 assumed to be most fit

	#always include the excess and disjoints of the fittest genome. Randomly select weights for matched genes between the two parents.

		genomeComparison = self.compareGenomes(parent1, parent2)
		# add breeding logic here. All mutation stuff comes after they've bred.

		connections = []
		nodes = deepcopy(parent1['nodes'])


		if parent1 == parent2:
			child = deepcopy(parent1)
			child['nodeValue'] = None
		else:
			if equalFitness:
				for index1, node1 in enumerate(genomeComparison['parent2']['uniqueNodes']):
					for i in range(len(parent2['nodes'])):
						if parent2['nodes'][i][0] == node1[0]:
							n = 1
							foundInputNode = False
							while not foundInputNode:
								inputNode = parent2['nodes'][i-n][0]
								break
								for index2, node2 in enumerate(nodes):
									if node2[0] == inputNode:
										nodes.insert(index2+1, node1)
										foundInputNode = True
										break
								n = n + 1
							break


			matchedConnections = []
			for index,item in enumerate(genomeComparison['parent1']['match']):
				matchedConnections.append([item])
			for index,item in enumerate(genomeComparison['parent2']['match']):
				matchedConnections[index].append(item)

			for i in range(len(matchedConnections)):
				geneParent = random.randint(0,1)
				connections.append(matchedConnections[i][geneParent])
				if matchedConnections[i][0][4] == False or matchedConnections[i][1][4] == False:
					disableChance = random.randint(1,100)
					if disableChance >= 75: #75% inherited gene disabled if disabled in either parent
						connections[len(connections)-1][4] = False
					else:
						connections[len(connections)-1][4] = True

			connections = connections + genomeComparison['parent1']['disjoint']
			connections = connections + genomeComparison['parent1']['excess']



			child = {'nodes':nodes, 'connections':connections, 'nodeValue':None} #nodeValue only used in solving

		#80% connection weight mutation
			#90% uniformly perturbed
			#10% new random value
		mutationSelect = random.randint(1,100)
		if mutationSelect <= 80:
			connectionSelect = random.randint(0,len(child['connections'])-1)
			child['connections'][connectionSelect][4] = True
			localitySelect = random.randint(1,100)
			if localitySelect <=10: #new random value
				child['connections'][connectionSelect][2] = random.uniform(-3,3) #weights
			else: #localy perturbed
				weight = child['connections'][connectionSelect][2]
				child['connections'][connectionSelect][2] = random.uniform(-.1,.1) * weight + weight
				#local purturbation of weight is +- 10%
				if weight == 0:
					child['connections'][connectionSelect][2] = random.uniform(-.01,.01)
		#3% adding new node
		elif 81 <= mutationSelect <= 83:
			randConnection = random.randint(0,len(child['connections'])-1)
			gene = child['connections'][randConnection]
			inputNode = gene[0] #geneId of node
			outputNode = gene[1]

			# All new nodes and connection made in the same place in the same generation should have the same generID and innovations
			nodeId = 0
			for index,item in enumerate(self.generationInnovs['newNodes']):
				if item['input'] == inputNode and item['output'] == outputNode:
					nodeId = item['geneId']
					break;
			if nodeId == 0:
				self.geneIdCounter += 1
				nodeId = self.geneIdCounter
				self.generationInnovs['newNodes'].append({'input':inputNode, 'output':outputNode, 'geneId':nodeId})

			innovNumInput = 0
			innovNumOutput = 0
			for index,item in enumerate(self.generationInnovs['newConnections']):
				if item['input'] == inputNode and item['output'] == nodeId:
					innovNumInput = item['innovNum']
				if item['input'] == nodeId and item['output'] == outputNode:
					innovNumOutput = item['innovNum']
			if innovNumInput == 0:
				self.innovNumCounter += 1
				innovNumInput = self.innovNumCounter
				self.generationInnovs['newConnections'].append({'input':inputNode, 'output':nodeId, 'innovNum':innovNumInput})
			if innovNumOutput == 0:
				self.innovNumCounter += 1
				innovNumOutput = self.innovNumCounter
				self.generationInnovs['newConnections'].append({'input':nodeId, 'output':outputNode, 'innovNum':innovNumOutput})

			#create new node and disable connection. Add two new connections. connection into new node gets weight of 1, connection out gets weight of original connection
			child['connections'][randConnection][4] = False
			newConnection1 = [inputNode, nodeId, 1, innovNumInput, True]
			child['connections'].append(newConnection1)

			weight = child['connections'][randConnection][2]
			newConnection2 = [nodeId, outputNode, weight, innovNumOutput, True]
			child['connections'].append(newConnection2)

			for index,item in enumerate(child['nodes']):
				if item[0] == inputNode:
					child['nodes'].insert(index+1, [nodeId, 2])
					break

		#5% new connection (30% for larger population)
		elif 84 <= mutationSelect <= 89:
			randNodeInput = random.randint(0,len(child['nodes'])-1)
			while True:
				randNodeOutput = random.randint(0,len(child['nodes'])-1)
				if child['nodes'][randNodeOutput][1] <= 2:
					break

			innovNum = 0
			for index,item in enumerate(self.generationInnovs['newConnections']):
				if item['input'] == randNodeInput and item['output'] == randNodeOutput:
					innovNum = item['innovNum']
					break
			if innovNum == 0:
				self.innovNumCounter += 1
				innovNum = self.innovNumCounter
				self.generationInnovs['newConnections'].append({'input':randNodeInput, 'output':randNodeOutput, 'innovNum':innovNum})

			connectionExists = False
			for index,item in enumerate(child['connections']):
				if randNodeInput == item[0] and randNodeOutput == item[1]:
					connectionExists = True
					self.innovNumCounter -= 1
					break

			if not connectionExists:
				weight = random.uniform(-3,3) #weights
				child['connections'].append([randNodeInput, randNodeOutput, weight, innovNum, True])

		child['nodeValue'] = [0 for i in range(len(child['nodes']))] #output value of every node
		for index, item in enumerate(child['nodes']):
			if item[1] == 1:
				child['nodeValue'][index] = 1
				break

		return child


	def createGeneration(self, speciesFitnessData):
		self.generationInnovs = {'newNodes': [], 'newConnections': []}

		#species that don't improve after 15 generations should die
		openPositions = 0
		deleteList = []
		for index,item in enumerate(speciesFitnessData):
			if item[-1][1] <= self.specLifeCounter[index][1]:
				self.specLifeCounter[index][0] += 1
				if self.specLifeCounter[index][0] >= self.noProgressLimit:
					openPositions += len(self.speciesList[index])
					deleteList.append(index)
			else:
				self.specLifeCounter[index][0] = 0
				self.specLifeCounter[index][1] = item[-1][1]

		for index,item in enumerate(deleteList):
			del self.speciesList[item-index]
			del self.specLifeCounter[item-index]
			del speciesFitnessData[item-index]

		#if last species is deleted generate new random network
		if openPositions == len(self.generation):
			print('reset')
			self.resetGeneration()
			return None

		if openPositions !=0:
			speciesNum = len(self.speciesList)
			addMembers = [math.floor(openPositions/speciesNum) for i in range(speciesNum)]
			for i in range(openPositions%speciesNum):
				addMembers[0] += 1

		#each species should inter breed a proportional amount of the population (should make as many children as species currently has)
		#use explicit fitness sharing to determine proportion of offsping each member should have of species
		newGeneration = []
		for specIndex,specItem in enumerate(speciesFitnessData):
			m=0
			if openPositions !=0:
				m = addMembers[specIndex]
			for i in range(len(specItem)+m):
				#champion of 5>= species is copied to new generation unchanged
				if len(specItem) >= 5 and i == 0:
					newGeneration.append(self.generation[specItem[-1][0]])
					continue

				#between species mating rate .1%
				x = self.goodGeneMultiplier #each fittest genome has x times the chance of being picked as the next
				if random.randint(1,1000) == 1 and len(speciesFitnessData) > 1:
					while True:
						specIndex2 = random.randint(0,len(speciesFitnessData)-1)
						if specIndex2 != specIndex:
							break
					parentIndex2 = round(math.log(random.randint(1,x**(len(speciesFitnessData[specIndex2])-1)),x))
					parent2 = self.generation[speciesFitnessData[specIndex2][parentIndex2][0]]
					parent2Fitness = speciesFitnessData[specIndex2][parentIndex2][1]

				else:
					parentIndex2 = round(math.log(random.randint(1,x**(len(specItem)-1)),x))
					parent2 = self.generation[specItem[parentIndex2][0]]
					parent2Fitness = specItem[parentIndex2][1]

				parentIndex1 = round(math.log(random.randint(1,x**(len(specItem)-1)),x))
				#math ceil and log2 decode random number into parentIndex1 = genome index
				parent1 = self.generation[specItem[parentIndex1][0]]

				if specItem[parentIndex1][1]==parent2Fitness:
					newGeneration.append(self.breedNet(parent1,parent2,True))
				elif specItem[parentIndex1][1]>=parent2Fitness:
					newGeneration.append(self.breedNet(parent1,parent2))
				else:
					newGeneration.append(self.breedNet(parent2,parent1))

		self.generation = newGeneration
		#25% offspring from mutation w/o crossover


	def resetNodeValues(self, genomeNum):
		arraySize = len(self.generation[genomeNum]['nodeValue'])
		self.generation[genomeNum]['nodeValue'] = [0 for i in range(arraySize)]

		for index, item in enumerate(self.generation[genomeNum]['nodes']):
			if item[1] == 1:
				self.generation[genomeNum]['nodeValue'][index] = 1
				break

	def speciateGeneration(self):
		threshold = 3.0# change to tune network, default 3
		N =1 #For # of connections < 20 in largest genome N=1, N is # of genes in largest genome
		c1 = 1
		c2 = 1
		c3 = .4 #0.4

		speciesReps = [];
		for index, array in enumerate(self.speciesList): #grabs one from each old species
			randIndex = array[random.randint(0,len(array)-1)]
			speciesReps.append(self.generation[randIndex])

		self.speciesList = [[] for i in range(len(speciesReps))]
		for index,item in enumerate(self.generation):
			for index2, item2 in enumerate(speciesReps):
				#find # of disjoint nodes, genes, and average weight differences between rep and new gene

				compared = self.compareGenomes(item, item2)
				distance = c1*compared['excess']/N + c2*compared['disjoint']/N \
				+ c3*compared['avgWeight']

				# if distance is less than threshold add item to self.speciesList[index2] and break.
				if distance < threshold:
					self.speciesList[index2].append(index)
					break
				# if no match make new species
				if index2 == len(speciesReps)-1:
					self.speciesList.append([index])
					self.specLifeCounter.append([0,0])

		#remove any empty species
		deleteList = []
		for index,item in enumerate(self.speciesList):
			if len(item) == 0:
				deleteList.append(index)
		for index,item in enumerate(deleteList):
			del self.speciesList[item-index]


	def compareGenomes(self, genome1, genome2):

		#find highest inov # for genome with lowest max inov
		max1 = 0
		for index, item in enumerate(genome1['connections']):
			if item[3] > max1:
				max1 = item[3]
		max2 = 0
		for index, item in enumerate(genome2['connections']):
			if item[3] > max2:
				max2 = item[3]
		excessInovLimit = max1 if max1 < max2 else max2 #Last inov # that isn't an excess inov

		parent1 = {'match':[], 'disjoint':[], 'excess':[], 'uniqueNodes':[]}
		parent2 = {'match':[], 'disjoint':[], 'excess':[], 'uniqueNodes':[]} # first 3 are connections

		#find similar, disjoint, and excess connections
		similar = []
		disjoint = 0
		excess = 0;
		for index, item in enumerate(genome1['connections']):
			hasMatch = False
			for index2, item2 in enumerate(genome2['connections']):
				if item[3] == item2[3]:
					similar.append([item[2], item2[2]])
					hasMatch = True
					parent1['match'].append(item)
					parent2['match'].append(item2)
					break
			if not hasMatch and item[3] <= excessInovLimit:
				parent1['disjoint'].append(item)
				disjoint += 1
			elif not hasMatch and item[3] > excessInovLimit:
				parent1['excess'].append(item)
				excess += 1

		for index, item in enumerate(genome2['connections']):
			hasMatch = False
			for index2, item2 in enumerate(genome1['connections']):
				if item[3] == item2[3]:
					hasMatch = True
					break
			if not hasMatch and item[3] <= excessInovLimit:
				parent2['disjoint'].append(item)
				disjoint += 1
			elif not hasMatch and item[3] > excessInovLimit:
				parent2['excess'].append(item)
				excess += 1

		localAvg = []
		for index, item in enumerate(similar):
			localAvg.append(abs(item[0]-item[1]))
		avg = sum(localAvg) / float(len(localAvg))

		#find unique nodes
		for item in enumerate(genome1['nodes']):
			hasMatch = False
			for item2 in enumerate(genome2['nodes']):
				if item[0] == item2[0]:
					hasMatch = True
					break
			if hasMatch == False:
				parent1['uniqueNodes'].append(item)

		for item in enumerate(genome2['nodes']):
			hasMatch = False
			for item2 in enumerate(genome1['nodes']):
				if item[0] == item2[0]:
					hasMatch = True
					break
			if hasMatch == False:
				parent2['uniqueNodes'].append(item)

		return {'avgWeight': avg, 'disjoint': disjoint, 'excess':excess, \
		'parent1':parent1, 'parent2':parent2}
		# {'avgWeight': weight average, 'disjoint': number of disjoints,
		# 'excess': number of excess, 'parent1':parent1, 'parent2':parent2}
		#		parent = {'match':[], 'disjoint':[], 'excess':[], 'uniqueNodes':[]}



	def evolveNet(self, trainingData, trainingAnswers, minFitnessNeeded):
		while True:

			#find fitness of every member, first parent in breedNet is fitest:
			#loop that uses self.breednet to create self.genSize # of new genomes for new gen.
			fitness = 0
			speciesFitnessData = []
			for specIndex,specItem in enumerate(self.speciesList):
				speciesFitnessData.append([])
				for index,item in enumerate(specItem):
					fitness = self.findFitness(trainingData, trainingAnswers, item)
					speciesFitnessData[specIndex].append([item,fitness])
			#store new gen locally until end when self.generation and self.speciesList are over written

			#use explicit fitness sharing to determine proportion of offsping each member should have of species
			adjustedFitnessData = deepcopy(speciesFitnessData)
			for specIndex,specItem in enumerate(adjustedFitnessData):
				for index,item in enumerate(specItem):
					fitnessSharingAdjust = item[1]/len(specItem)
					adjustedFitnessData[specIndex][index][1] = fitnessSharingAdjust

			#sort from weakest to fittest genome in species
			for specIndex in range(len(adjustedFitnessData)):
				for index in range(len(adjustedFitnessData[specIndex])-1):
					weakestGenome = deepcopy(adjustedFitnessData[specIndex][index])
					replaceIndex = index
					for genoIndex in range(index+1,len(adjustedFitnessData[specIndex])):
						if adjustedFitnessData[specIndex][genoIndex][1] <= weakestGenome[1]:
							weakestGenome = deepcopy(adjustedFitnessData[specIndex][genoIndex])
							replaceIndex = genoIndex
					adjustedFitnessData[specIndex][replaceIndex] = adjustedFitnessData[specIndex][index]
					adjustedFitnessData[specIndex][index] = weakestGenome

			#test each member of generation with other set of data/answers
			#find highest fitness of new generation
			#compare fitness to desired fitness
			maxFitness = 0
			fittestIndex = 0

			for specIndex,specItem in enumerate(speciesFitnessData):
				for index,item in enumerate(specItem):
					if item[1] > maxFitness:
						maxFitness = item[1]
						fittestIndex = item[0]


			print(maxFitness, " ", self.generationCounter, len(adjustedFitnessData))
			if maxFitness >= minFitnessNeeded:
				fitness = self.findFitness(trainingData, trainingAnswers, fittestIndex)
				print(fitness)
				return [self.generation[fittestIndex],fittestIndex, self.generationCounter, maxFitness]


			#if not reached loop this function

			self.createGeneration(adjustedFitnessData)
			self.speciateGeneration()
			#increase generation counter
			self.generationCounter += 1

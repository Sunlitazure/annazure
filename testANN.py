import neatnet

testData = [[0,0],[0,1],[1,0],[1,1]]
testAnswers = [[0],[1],[1],[0]]

# network = neatnet.NeatNet(2,1)
	# winner, winnerIndex, iterations, fittness = network.evolveNet(testData, testAnswers, .9)
	# print("end")
	# answers1 = network.run(testData[0], winnerIndex)
	# print(answers1[0], testData[0])
	# answers2 = network.run(testData[1], winnerIndex)
	# print(answers2[0], testData[1])
	# answers3 = network.run(testData[2], winnerIndex)
	# print(answers3[0], testData[2])
	# answers4 = network.run(testData[3], winnerIndex)
	# print(answers4[0], testData[3])


sum = 0
tests =20
for i in range(tests):
	network = neatnet.NeatNet(2,1)
	winner, winnerIndex, iterations, fittness = network.evolveNet(testData, testAnswers, 1)
	print("end")
	sum += iterations
	print("Iterations = ", i+1)

avgIterations = sum/tests
print("average number of iterations: ", avgIterations)


#answer = network.run([1,0], 0)

#compared = network.compareGenomes(network.generation[0], network.generation[1])

#fitness = network.findFitness([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], 0)
#print(fitness)

#network.speciateGeneration(network.generation)

'''
network = neuralNet.NeuralNet(2,100)
answer = network.run([1,0])
print(answer)
'''

#xor
#0,0 = 0
#0,1 = 1
#1,0 = 1
#1,1 = 0

'''
d:
cd chickentech/Programming/python/"genetic algorithms"
'''

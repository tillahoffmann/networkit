import stopwatch
import csv
import os

from NetworKit import *


def getFileList(directory):
	""" Get list of graph files in directory"""
	ls = []
	for (root, _, filenames) in os.walk(directory):
		for filename in filenames:
			ls.append(os.path.join(root, filename))
	return ls

def communityDetectionBenchmark(graphPaths, algorithms, outPath, repeat=1):
	"""
		Evaluate community detection algorithms on a collection of graphs and save benchmark data in .csv format
		:param	graphPaths	paths to graph files
		:param 	algorithms	list of algorithms
	"""

	# write results
	with open(outPath, 'w') as outFile:
		writer = csv.writer(outFile, delimiter='\t')
		header = ["graph", "algo", "time", "modularity"]
		writer.writerow(header)
		for graphPath in graphPaths:
			print("reading graph: {0}".format(graphPath))
			G = graphio.readGraph(graphPath)
			graphName = os.path.basename(graphPath).split(".")[0]
			(n, m) = properties.size(G)
			for algo in algorithms:
				algoName = algo.toString()
				for i in range(repeat):
					print("evaluating {0} on {1}".format(algoName, graphName))
					timer = stopwatch.Timer()
					zeta = algo.run(G)
					timer.stop()
					time = timer.elapsed

					mod = community.Modularity().getQuality(zeta, G)
					# nc = zeta.numberOfSubsets()

					row = [graphName, algoName, time, mod]
					writer.writerow(row)
					print(row)


def testPLPThreshold(graphPaths, thresholdFactors, outPath, repeat=1):
	from NetworKit import community

	with open(outPath, 'w') as outFile:
		writer = csv.writer(outFile, delimiter='\t')
		header = ["graph", "algorithm","factor", "threshold", "time", "modularity"]
		writer.writerow(header)
		for graphPath in graphPaths:
			print("reading graph: {0}".format(graphPath))
			G = graphio.readGraph(graphPath)
			(n, m) = properties.size(G)
			graphName = os.path.basename(graphPath).split(".")[0]
			for factor in thresholdFactors:
				theta = int(n * factor)
				for i in range(repeat):
					timer = stopwatch.Timer()
					algo = community.PLP(updateThreshold=theta)
					algoName = algo.toString()
					zeta = algo.run(G)
					timer.stop()
					time = timer.elapsed

					mod = community.Modularity().getQuality(zeta, G)
					row = [graphName, "PLP", factor, theta,  time, mod]
					writer.writerow(row)
					print(row)



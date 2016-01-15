package bayesianNetwork;
import java.util.LinkedList;
import java.util.Set;

import org.apache.commons.collections15.Transformer;

import edu.uci.ics.jung.algorithms.shortestpath.PrimMinimumSpanningTree;
import edu.uci.ics.jung.graph.DelegateTree;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;

public class ChowLiuTree {
	private Set<BNSample> samples;
	private double[][] mutualInfo;
	private DelegateTree<Integer, UndirectedEdge> chowLiuTree;

	public ChowLiuTree(Set<BNSample> trainingSamples) {
		samples = trainingSamples;
		int n = BNSample.numberOfAttributes;
		mutualInfo = new double[n][n];
	}

	public void train() {
		// calculate mutual information matrix
		for (int i = 0; i < BNSample.numberOfAttributes; i++) {
			for (int j = i; j < BNSample.numberOfAttributes; j++) {
				mutualInfo[i][j] = getMutualInfo(i + 1, j + 1);
				mutualInfo[j][i] = mutualInfo[i][j];
			}
		}

		// construct a complete graph and find the maximum spanning tree
		UndirectedSparseGraph<Integer, UndirectedEdge> cGraph = new UndirectedSparseGraph<>();
		for (int i = 1; i <= BNSample.numberOfAttributes; i++) {
			cGraph.addVertex(i);
		}

		// anonymous class for transforming edge weights
		Transformer<UndirectedEdge, Double> cWeights = new Transformer<UndirectedEdge, Double>() {
			@Override
			public Double transform(UndirectedEdge edge) {
				// return the negative value of the edge weight
				// turn a Prim minimum spanning tree into a maximum spanning
				// tree
				return (-1) * edge.getWeight();
			}
		};

		for (int i = 0; i < mutualInfo.length; i++) {
			// j start with i+1 means no self loop is allowed in the graph
			for (int j = i + 1; j < mutualInfo.length; j++) {
				UndirectedEdge edge = new UndirectedEdge(mutualInfo[i][j]);
				cGraph.addEdge(edge, (i + 1), (j + 1));
			}
		}

		PrimMinimumSpanningTree<Integer, UndirectedEdge> pmst = new PrimMinimumSpanningTree<Integer, UndirectedEdge>(
				UndirectedSparseGraph.<Integer, UndirectedEdge> getFactory(),
				cWeights);
		Graph<Integer, UndirectedEdge> mstGraph = pmst.transform(cGraph);

		chowLiuTree = new DelegateTree<Integer, UndirectedEdge>();

		for (Integer node : mstGraph.getVertices()) {
			if (BNSample.attributeNameMap.get(node).equals(BNSample.LABEL)) {
				chowLiuTree.addVertex(node);
			}
		}

		LinkedList<Integer> nodesQueue = new LinkedList<Integer>();
		nodesQueue.add(chowLiuTree.getRoot());

		while (!nodesQueue.isEmpty()) {
			Integer current = nodesQueue.poll();
			for (Integer node : mstGraph.getNeighbors(current)) {
				if (!chowLiuTree.containsVertex(node)) {
					chowLiuTree.addChild(new UndirectedEdge(
							mutualInfo[current - 1][node - 1]), current, node);
				}

				if (mstGraph.getNeighborCount(node) != chowLiuTree
						.getNeighborCount(node)) {
					nodesQueue.add(node);
				}
			}
		}
	}

	public void printChowLiuTree() {
		Utility.printArray(mutualInfo);
		Utility.visualizeGraph(chowLiuTree, "Learnt Chow-Liu Tree");
	}

	private double getMutualInfo(int i, int j) {
		if (i < 1 || i > BNSample.numberOfAttributes || j < 1
				|| j > BNSample.numberOfAttributes) {
			System.out
					.println("error creating new sample - index of sample attribute out of bound.");
		}

		double result = 0;
		for (String xi : BNSample.attributeValueMap.get(i)) {
			for (String xj : BNSample.attributeValueMap.get(j)) {
				// count how many sample with xi
				int Ni = singleCount(i, xi);
				// count how many sample with xj
				int Nj = singleCount(j, xj);
				// count how many sample with xj and xj
				int Nij = jointCount(i, xi, j, xj);

				// calculate P(xi, xj) * log(P(xi, xj) / (P(xi) * P(xj)))
				if (Nij != 0) {
					result += ((double) Nij / samples.size())
							* Math.log((double) Nij * samples.size()
									/ (Ni * Nj));
				}

				// System.out.println("F" + i + ": " + xi + ", F" + j + ": " +
				// xj
				// + "; Ni=" + Ni + ", Nj=" + Nj + ", Nij=" + Nij
				// + "; subMutual=" + ((double) Nij / samples.size())
				// * Math.log((double) Nij * samples.size() / (Ni * Nj))
				// + "; mutual=" + result);
			}
		}
		return result;
	}

	private int singleCount(int i, String xi) {
		int count = 0;
		for (BNSample s : samples) {
			if (s.getData(i).equals(xi)) {
				count++;
			}
		}
		return count;
	}

	private int jointCount(int i, String xi, int j, String xj) {
		int count = 0;
		for (BNSample s : samples) {
			if (s.getData(i).equals(xi) && s.getData(j).equals(xj)) {
				count++;
			}
		}
		return count;
	}

}

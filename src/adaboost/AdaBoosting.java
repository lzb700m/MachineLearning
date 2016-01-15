package adaboost;

import java.awt.Color;
import java.awt.Paint;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Stack;

import javax.swing.JFrame;

import org.apache.commons.collections15.Transformer;

import decisionTree.DTNode;
import decisionTree.LeafNode;
import decisionTree.NonLeafNode;
import decisionTree.Sample;
import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.algorithms.layout.TreeLayout;
import edu.uci.ics.jung.graph.DelegateTree;
import edu.uci.ics.jung.visualization.BasicVisualizationServer;
import edu.uci.ics.jung.visualization.decorators.ToStringLabeller;
import edu.uci.ics.jung.visualization.renderers.Renderer.VertexLabel.Position;

public class AdaBoosting {
	public static Map<Double, DTNode> train(HashSet<Sample> samples,
			HashMap<Integer, ArrayList<String>> description, int M) {
		Map<Double, DTNode> classifier = new HashMap<Double, DTNode>();
		double[] epsilon = new double[M];
		double[] alpha = new double[M];

		// initialize sample weights
		for (Sample s : samples) {
			s.setWeight((double) 1 / samples.size());
		}

		// boosting
		for (int m = 0; m < M; m++) {
			DTNode decisionStump = findDepth1Tree(samples, description);
			if (m < 5) {
				display(decisionStump, m);
			}
			epsilon[m] = getError(samples, decisionStump);
			alpha[m] = 0.5 * Math.log((1 - epsilon[m]) / epsilon[m]);

			classifier.put(alpha[m], decisionStump);
			// update sample weight
			for (Sample sample : samples) {
				if (sample.getLabel().equals(predict(sample, decisionStump))) {
					double newWeight = sample.getWeight()
							* (Math.exp(-1 * alpha[m]) / (2 * Math
									.sqrt(epsilon[m] * (1 - epsilon[m]))));
					sample.setWeight(newWeight);
				} else {
					double newWeight = sample.getWeight()
							* (Math.exp(1 * alpha[m]) / (2 * Math
									.sqrt(epsilon[m] * (1 - epsilon[m]))));
					sample.setWeight(newWeight);
				}
			}
			System.out.println("m = " + (m + 1) + ", error = " + epsilon[m]
					+ ", alpha = " + alpha[m]);
		}
		return classifier;
	}

	public static DTNode findDepth1Tree(HashSet<Sample> samples,
			HashMap<Integer, ArrayList<String>> description) {
		double error = 1;
		NonLeafNode classifier = new NonLeafNode();

		for (int i = 0; i < description.size(); i++) {

			double weightedError = 0;

			HashMap<String, HashSet<Sample>> split = splitSamples(samples, i,
					description.get(i));

			NonLeafNode tempRoot = new NonLeafNode();
			tempRoot.setParent(null);
			tempRoot.setSplittingAttribute(i);
			tempRoot.setPosCount(numOfPos(samples));
			tempRoot.setNegCount(numOfNeg(samples));

			Iterator<Entry<String, HashSet<Sample>>> it = split.entrySet()
					.iterator();

			while (it.hasNext()) {
				Entry<String, HashSet<Sample>> entry = it.next();

				LeafNode leaf = new LeafNode(tempRoot);
				tempRoot.addChildren(leaf);
				leaf.setAttributeValue(entry.getKey());
				leaf.setPosCount(numOfPos(entry.getValue()));
				leaf.setNegCount(numOfNeg(entry.getValue()));
				// use weighted majority vote
				leaf.setPrediction(weightedMajorityVote(entry.getValue()));

			}

			// calculate error
			weightedError = getError(samples, tempRoot);

			if (error > weightedError) {
				error = weightedError;
				classifier = tempRoot;
			}

		}

		return classifier;
	}

	public static DTNode findDepth2Tree(HashSet<Sample> samples,
			HashMap<Integer, ArrayList<String>> description) {

		double error = 1;
		NonLeafNode classifier = new NonLeafNode();

		for (int i = 0; i < description.size(); i++) {

			HashMap<String, HashSet<Sample>> split = splitSamples(samples, i,
					description.get(i));

			// the code below only handle the heart data set, it's a bad
			// programming style
			HashSet<Sample> sampleSet1 = split.get("0");
			HashSet<Sample> sampleSet2 = split.get("1");

			for (int j = 0; j < description.size(); j++) {
				if (j == i) {
					continue;
				}
				for (int k = 0; k < description.size(); k++) {
					if (k == i) {
						continue;
					}
					double weightedError = 0;

					NonLeafNode tempRoot = new NonLeafNode();
					tempRoot.setParent(null);
					tempRoot.setSplittingAttribute(i);
					tempRoot.setPosCount(numOfPos(samples));
					tempRoot.setNegCount(numOfNeg(samples));
					if (tempRoot.getPosCount() >= tempRoot.getNegCount()) {
						tempRoot.setMajorityVote(AdaBoostingMain.POS_LABEL);
					} else {
						tempRoot.setMajorityVote(AdaBoostingMain.NEG_LABEL);
					}

					NonLeafNode level2Node1 = new NonLeafNode();
					level2Node1.setParent(tempRoot);
					tempRoot.addChildren(level2Node1);
					level2Node1.setAttributeValue("0");
					level2Node1.setSplittingAttribute(j);
					level2Node1.setPosCount(numOfPos(sampleSet1));
					level2Node1.setNegCount(numOfNeg(sampleSet1));
					if (level2Node1.getPosCount() >= level2Node1.getNegCount()) {
						level2Node1.setMajorityVote(AdaBoostingMain.POS_LABEL);
					} else {
						level2Node1.setMajorityVote(AdaBoostingMain.NEG_LABEL);
					}
					HashMap<String, HashSet<Sample>> split1 = splitSamples(
							sampleSet1, j, description.get(j));
					Iterator<Entry<String, HashSet<Sample>>> it1 = split1
							.entrySet().iterator();
					while (it1.hasNext()) {
						Entry<String, HashSet<Sample>> entry = it1.next();
						LeafNode leaf = new LeafNode(level2Node1);
						level2Node1.addChildren(leaf);
						leaf.setAttributeValue(entry.getKey());
						leaf.setPosCount(numOfPos(entry.getValue()));
						leaf.setNegCount(numOfNeg(entry.getValue()));
						// use weighted majority vote
						leaf.setPrediction(weightedMajorityVote(entry
								.getValue()));
						// if (leaf.getPosCount() > leaf.getNegCount()) {
						// leaf.setPrediction(AdaBoostingMain.POS_LABEL);
						// } else if (leaf.getPosCount() < leaf.getNegCount()) {
						// leaf.setPrediction(AdaBoostingMain.NEG_LABEL);
						// } else {
						// leaf.setPrediction(level2Node1.getmajorityVote());
						// }
					}

					// repeated code - again, very bad
					NonLeafNode level2Node2 = new NonLeafNode();
					level2Node2.setParent(tempRoot);
					tempRoot.addChildren(level2Node2);
					level2Node2.setAttributeValue("1");
					level2Node2.setSplittingAttribute(k);
					level2Node2.setPosCount(numOfPos(sampleSet2));
					level2Node2.setNegCount(numOfNeg(sampleSet2));
					if (level2Node2.getPosCount() >= level2Node2.getNegCount()) {
						level2Node2.setMajorityVote(AdaBoostingMain.POS_LABEL);
					} else {
						level2Node2.setMajorityVote(AdaBoostingMain.NEG_LABEL);
					}
					HashMap<String, HashSet<Sample>> split2 = splitSamples(
							sampleSet2, k, description.get(k));
					Iterator<Entry<String, HashSet<Sample>>> it2 = split2
							.entrySet().iterator();
					while (it2.hasNext()) {
						Entry<String, HashSet<Sample>> entry = it2.next();
						LeafNode leaf = new LeafNode(level2Node2);
						level2Node2.addChildren(leaf);
						leaf.setAttributeValue(entry.getKey());
						leaf.setPosCount(numOfPos(entry.getValue()));
						leaf.setNegCount(numOfNeg(entry.getValue()));
						// use weighted majority vote
						leaf.setPrediction(weightedMajorityVote(entry
								.getValue()));

						// if (leaf.getPosCount() > leaf.getNegCount()) {
						// leaf.setPrediction(AdaBoostingMain.POS_LABEL);
						// } else if (leaf.getPosCount() < leaf.getNegCount()) {
						// leaf.setPrediction(AdaBoostingMain.NEG_LABEL);
						// } else {
						// leaf.setPrediction(level2Node2.getmajorityVote());
						// }
					}

					// calculate error
					weightedError = getError(samples, tempRoot);

					if (error > weightedError) {
						error = weightedError;
						classifier = tempRoot;
					}
				}
			}
		}
		return classifier;
	}

	public static double test(HashSet<Sample> samples,
			Map<Double, DTNode> classifier) {
		int countCorrect = 0;
		for (Sample s : samples) {
			if (s.getLabel().equals(boostingPredict(s, classifier))) {
				countCorrect++;
			}
		}
		return (double) countCorrect / samples.size();
	}

	// count # of positive samples
	private static int numOfPos(HashSet<Sample> samples) {
		int count = 0;
		for (Sample s : samples) {
			if (s.getLabel().equals(AdaBoostingMain.POS_LABEL)) {
				count++;
			}
		}
		return count;
	}

	// count # of negative samples
	private static int numOfNeg(HashSet<Sample> samples) {
		int count = 0;
		for (Sample s : samples) {
			if (s.getLabel().equals(AdaBoostingMain.NEG_LABEL)) {
				count++;
			}
		}
		return count;
	}

	// split samples given a attribute index and attribute values
	private static HashMap<String, HashSet<Sample>> splitSamples(
			HashSet<Sample> samples, int index, ArrayList<String> values) {

		HashMap<String, HashSet<Sample>> result = new HashMap<String, HashSet<Sample>>();

		for (String s : values) {
			result.put(s, new HashSet<Sample>());
		}

		for (Sample sample : samples) {
			String value = sample.getAttribute(index);
			result.get(value).add(sample);
		}
		return result;
	}

	private static double getError(HashSet<Sample> samples, DTNode classifier) {
		double error = 0;
		for (Sample sample : samples) {
			String prediction = predict(sample, classifier);
			if (!sample.getLabel().equals(prediction)) {
				error += sample.getWeight();
			}
		}
		return error;
	}

	// predict a sample
	private static String predict(Sample s, DTNode classifier) {
		DTNode currentNode = classifier;
		while (currentNode instanceof NonLeafNode) {
			int index = ((NonLeafNode) currentNode).getSplittingAttribute();
			String value = s.getAttribute(index);
			ArrayList<DTNode> children = ((NonLeafNode) currentNode)
					.getChildren();
			for (DTNode child : children) {
				if (child.getAttributeValue().equals(value)) {
					currentNode = child;
				}
			}
		}
		return ((LeafNode) currentNode).getPrediction();
	}

	private static String boostingPredict(Sample s,
			Map<Double, DTNode> classifier) {
		Iterator<Entry<Double, DTNode>> it = classifier.entrySet().iterator();
		double result = 0;

		while (it.hasNext()) {
			Entry<Double, DTNode> entry = it.next();
			if (AdaBoostingMain.POS_LABEL.equals(predict(s, entry.getValue()))) {
				result += entry.getKey();
			} else {
				result -= entry.getKey();
			}
		}
		if (result >= 0) {
			return AdaBoostingMain.POS_LABEL;
		} else {
			return AdaBoostingMain.NEG_LABEL;
		}
	}

	// display decision tree using JUNG2
	public static void display(DTNode root, int m) {
		DelegateTree<DTNode, String> decisionTree = buildTree(root);

		Layout<DTNode, String> layout = new TreeLayout<DTNode, String>(
				decisionTree);

		BasicVisualizationServer<DTNode, String> vv = new BasicVisualizationServer<DTNode, String>(
				layout);
		Transformer<DTNode, Paint> vertexPaint = new Transformer<DTNode, Paint>() {
			public Paint transform(DTNode i) {
				if (i instanceof NonLeafNode) {
					return Color.GREEN;
				} else {
					return Color.YELLOW;
				}
			}
		};

		Transformer<DTNode, Shape> vertexShape = new Transformer<DTNode, Shape>() {
			public Shape transform(DTNode i) {
				if (i instanceof NonLeafNode) {
					Ellipse2D e = new Ellipse2D.Double(-15, -15, 40, 30);
					return AffineTransform.getScaleInstance(2, 2)
							.createTransformedShape(e);
				} else {
					Rectangle r = new Rectangle(22, 20);
					return AffineTransform.getScaleInstance(2, 2)
							.createTransformedShape(r);
				}
			}
		};

		vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
		vv.getRenderContext().setVertexShapeTransformer(vertexShape);
		vv.getRenderContext().setVertexLabelTransformer(
				new ToStringLabeller<DTNode>());
		vv.getRenderer().getVertexLabelRenderer().setPosition(Position.CNTR);

		JFrame frame = new JFrame("AdaBoosting M = " + (m + 1));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(vv);
		frame.pack();
		frame.setVisible(true);
	}

	// build a tree structure for visualization from calculated decision tree
	private static DelegateTree<DTNode, String> buildTree(DTNode root) {
		DelegateTree<DTNode, String> result = new DelegateTree<DTNode, String>();
		DTNode currentNode = root;
		Stack<NonLeafNode> expandableNodes = new Stack<NonLeafNode>();

		if (currentNode instanceof NonLeafNode) {
			expandableNodes.push((NonLeafNode) currentNode);
			result.addVertex(currentNode);
			while (!expandableNodes.isEmpty()) {
				NonLeafNode parentNode = expandableNodes.pop();

				ArrayList<DTNode> childreNodes = ((NonLeafNode) parentNode)
						.getChildren();
				for (DTNode childNode : childreNodes) {
					result.addChild(
							parentNode.toString() + childNode.toString(),
							parentNode, childNode);
					if (childNode instanceof NonLeafNode) {
						expandableNodes.push((NonLeafNode) childNode);
					}
				}
			}
		} else {
			result.addVertex(currentNode);
		}
		return result;
	}

	private static String weightedMajorityVote(HashSet<Sample> samples) {
		double weightedVote = 0;
		for (Sample s : samples) {
			if (s.getLabel().equals(AdaBoostingMain.POS_LABEL)) {
				weightedVote += s.getWeight();
			} else if (s.getLabel().equals(AdaBoostingMain.NEG_LABEL)) {
				weightedVote -= s.getWeight();
			}
		}
		if (weightedVote >= 0) {
			return AdaBoostingMain.POS_LABEL;
		} else {
			return AdaBoostingMain.NEG_LABEL;
		}
	}
}

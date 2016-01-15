package decisionTree;
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
import java.util.Map.Entry;
import java.util.Stack;

import javax.swing.JFrame;

import org.apache.commons.collections15.Transformer;

import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.algorithms.layout.TreeLayout;
import edu.uci.ics.jung.graph.DelegateTree;
import edu.uci.ics.jung.visualization.BasicVisualizationServer;
import edu.uci.ics.jung.visualization.decorators.ToStringLabeller;
import edu.uci.ics.jung.visualization.renderers.Renderer.VertexLabel.Position;

public class DecisionTree {

	public static DTNode train(HashSet<Sample> samples,
			HashMap<Integer, ArrayList<String>> description,
			NonLeafNode parent, String attributeValue) {

		int posSampleCount = numOfPos(samples);
		int negSampleCount = numOfNeg(samples);

		// base case 1: sample space is empty
		if (samples == null || samples.size() == 0) {
			// return a leaf node with classification = parent.majorityVote
			if (parent == null || attributeValue == null) {
				return null;
			}
			LeafNode node = new LeafNode(parent);
			node.setAttributeValue(attributeValue);
			node.setPrediction(parent.getmajorityVote());
			node.setPosCount(posSampleCount);
			node.setNegCount(negSampleCount);
			return node;
		}

		// base case 2: sample space contains only 1 label
		if (posSampleCount == 0) {
			// return a leaf node with classification = NEG_LABEL
			LeafNode node = new LeafNode(parent);
			node.setAttributeValue(attributeValue);
			node.setPrediction(DecisionTreeTest.NEG_LABEL);
			node.setPosCount(posSampleCount);
			node.setNegCount(negSampleCount);
			return node;
		} else if (negSampleCount == 0) {
			// return a leaf node with classification = POS_LABEL
			LeafNode node = new LeafNode(parent);
			node.setAttributeValue(attributeValue);
			node.setPrediction(DecisionTreeTest.POS_LABEL);
			node.setPosCount(posSampleCount);
			node.setNegCount(negSampleCount);
			return node;
		}

		// base case 3: all samples have identical attributes value
		if (sameAttribute(samples)) {
			LeafNode node = new LeafNode(parent);
			node.setAttributeValue(attributeValue);
			node.setPosCount(posSampleCount);
			node.setNegCount(negSampleCount);
			if (posSampleCount >= negSampleCount) {
				node.setPrediction(DecisionTreeTest.POS_LABEL);
			} else {
				node.setPrediction(DecisionTreeTest.NEG_LABEL);
			}
			return node;
		}

		// find attribute with highest IG
		// create non-leaf node with children
		// recurse

		// calculate entropy and information gain on every possible split
		float entropy = Utility.getEntropy(posSampleCount, negSampleCount);
		float ig = 0;
		int targetIndex = 0;

		for (int i = 0; i < description.size(); i++) {
			ArrayList<String> valueList = description.get(i);
			HashMap<String, HashSet<Sample>> splittedSamples = splitSamples(
					samples, i, valueList);

			float conditionalEntropy = 0;
			Iterator<Entry<String, HashSet<Sample>>> it = splittedSamples
					.entrySet().iterator();
			while (it.hasNext()) {
				Entry<String, HashSet<Sample>> entry = it.next();
				int pos = numOfPos(entry.getValue());
				int neg = numOfNeg(entry.getValue());
				float prob = (float) entry.getValue().size() / samples.size();
				conditionalEntropy += prob * Utility.getEntropy(pos, neg);
			}

			if ((entropy - conditionalEntropy) >= ig) {
				ig = entropy - conditionalEntropy;
				targetIndex = i;
			}
		}

		NonLeafNode root = new NonLeafNode();
		root.setParent(parent);
		if (posSampleCount >= negSampleCount) {
			root.setMajorityVote(DecisionTreeTest.POS_LABEL);
		} else {
			root.setMajorityVote(DecisionTreeTest.NEG_LABEL);
		}
		root.setPosCount(posSampleCount);
		root.setNegCount(negSampleCount);
		root.setAttributeValue(attributeValue);
		root.setSplittingAttribute(targetIndex);
		root.setInfoGain(ig);

		// create child node and recurse
		HashMap<String, HashSet<Sample>> split = splitSamples(samples,
				targetIndex, description.get(targetIndex));
		Iterator<Entry<String, HashSet<Sample>>> it = split.entrySet()
				.iterator();
		while (it.hasNext()) {
			Entry<String, HashSet<Sample>> entry = it.next();
			root.addChildren(train(entry.getValue(), description, root,
					entry.getKey()));
		}
		return root;
	}

	// test a sample set
	public static float test(HashSet<Sample> samples, DTNode root) {
		int countCorrect = 0;
		for (Sample s : samples) {
			String prediction = predict(s, root);
			if (prediction.equals(s.getLabel())) {
				countCorrect++;
			}
		}
		return (float) countCorrect / samples.size();
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

	// calculate decision tree size - # of nodes
	public static int size(DTNode root) {
		if (root instanceof LeafNode) {
			return 1;
		} else {
			ArrayList<DTNode> children = ((NonLeafNode) root).getChildren();
			int count = 1;
			for (DTNode n : children) {
				count = count + size(n);
			}
			return count;
		}
	}

	// calculate decision tree depth
	public static int depth(DTNode root) {
		if (root instanceof LeafNode) {
			return 0;
		} else {
			ArrayList<DTNode> children = ((NonLeafNode) root).getChildren();
			int depth = 1;
			for (DTNode n : children) {
				if (depth(n) >= depth) {
					depth += depth(n);
				}
			}
			return depth;
		}
	}

	// display decision tree using JUNG2
	public static void display(DTNode root) {
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

		JFrame frame = new JFrame("Decision Tree");
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

	// count # of positive samples
	private static int numOfPos(HashSet<Sample> samples) {
		int count = 0;
		for (Sample s : samples) {
			if (s.getLabel().equals(DecisionTreeTest.POS_LABEL)) {
				count++;
			}
		}
		return count;
	}

	// count # of negative samples
	private static int numOfNeg(HashSet<Sample> samples) {
		int count = 0;
		for (Sample s : samples) {
			if (s.getLabel().equals(DecisionTreeTest.NEG_LABEL)) {
				count++;
			}
		}
		return count;
	}

	// compare attributes of given sample set
	private static boolean sameAttribute(HashSet<Sample> samples) {
		for (Sample s : samples) {
			for (Sample other : samples) {
				if (!s.compareAttribute(other)) {
					return false;
				}
			}
		}
		return true;
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
}

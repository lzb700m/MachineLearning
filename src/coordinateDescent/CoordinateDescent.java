package coordinateDescent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import adaboost.AdaBoostingMain;
import decisionTree.DTNode;
import decisionTree.LeafNode;
import decisionTree.NonLeafNode;
import decisionTree.Sample;

public class CoordinateDescent {

	private static final double CUT_OFF_VALUE = 0.001;
	public ArrayList<DTNode> stumpList = new ArrayList<DTNode>();
	public ArrayList<Double> alpha = new ArrayList<Double>();

	double loss;

	public CoordinateDescent(HashMap<Integer, ArrayList<String>> description) {
		for (int i = 0; i < description.size(); i++) {
			ArrayList<String> valueList = description.get(i);
			for (int j = 0; j < valueList.size(); j++) {
				for (int k = 0; k < valueList.size(); k++) {
					NonLeafNode tempRoot = new NonLeafNode();
					tempRoot.setParent(null);
					tempRoot.setSplittingAttribute(i);

					LeafNode leftChild = new LeafNode(tempRoot);
					tempRoot.addChildren(leftChild);
					leftChild
							.setAttributeValue(CoordinateDescentTest.POS_LABEL);
					leftChild.setPrediction(valueList.get(j));

					LeafNode rightChild = new LeafNode(tempRoot);
					tempRoot.addChildren(rightChild);
					rightChild
							.setAttributeValue(CoordinateDescentTest.NEG_LABEL);
					rightChild.setPrediction(valueList.get(k));

					stumpList.add(tempRoot);
					alpha.add((double) 0);
				}
			}
		}
	}

	// coordinate descent, update alpha value until converge
	public void update(HashSet<Sample> samples) {
		boolean hasConverged = false;

		while (!hasConverged) {
			hasConverged = true;
			for (int tPrime = 0; tPrime < stumpList.size(); tPrime++) {
				double alphaTPrime;
				HashSet<Sample> correctClassified = new HashSet<Sample>();
				HashSet<Sample> misClassified = new HashSet<Sample>();

				double numerator = 0;
				double denominator = 0;

				// classify all samples using given Tth classifier
				for (Sample s : samples) {
					String prediction = predict(s, stumpList.get(tPrime));
					if (prediction.equals(s.getLabel())) {
						correctClassified.add(s);
					} else {
						misClassified.add(s);
					}
				}

				for (Sample s : correctClassified) {
					double innerTotal = 0;
					double y;
					if (s.getLabel().equals(AdaBoostingMain.POS_LABEL)) {
						y = (double) 1;
					} else {
						y = (double) -1;
					}

					for (int t = 0; t < stumpList.size(); t++) {
						if (t != tPrime) {
							String prediction = predict(s, stumpList.get(t));
							if (prediction.equals(AdaBoostingMain.POS_LABEL)) {
								innerTotal += alpha.get(t);
							} else {
								innerTotal -= alpha.get(t);
							}
						}
					}
					numerator += Math.exp(-1 * y * innerTotal);
				}

				for (Sample s : misClassified) {
					double innerTotal = 0;
					double y;
					if (s.getLabel().equals(AdaBoostingMain.POS_LABEL)) {
						y = (double) 1;
					} else {
						y = (double) -1;
					}

					for (int t = 0; t < stumpList.size(); t++) {
						if (t != tPrime) {
							String prediction = predict(s, stumpList.get(t));
							if (prediction.equals(AdaBoostingMain.POS_LABEL)) {
								innerTotal += alpha.get(t);
							} else {
								innerTotal -= alpha.get(t);
							}
						}
					}
					denominator += Math.exp(-1 * y * innerTotal);
				}

				alphaTPrime = 0.5 * Math.log(numerator / denominator);
				if (Math.abs(alphaTPrime - alpha.get(tPrime)) > CUT_OFF_VALUE) {
					hasConverged = false;
				}
				alpha.set(tPrime, alphaTPrime);
				setLoss(samples);
				System.out.println("Loss: " + loss);
			}
		}

	}

	// calculate loss function
	public void setLoss(HashSet<Sample> samples) {
		double result = 0;
		for (Sample s : samples) {
			double y;
			double innerTotal = 0;
			for (int t = 0; t < stumpList.size(); t++) {
				String prediction = predict(s, stumpList.get(t));
				if (prediction.equals(AdaBoostingMain.POS_LABEL)) {
					innerTotal += alpha.get(t);
				} else {
					innerTotal -= alpha.get(t);
				}
			}
			if (s.getLabel().equals(AdaBoostingMain.POS_LABEL)) {
				y = (double) 1;
			} else {
				y = (double) -1;
			}

			result += Math.exp(-1 * y * innerTotal);
		}
		loss = result;
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

}

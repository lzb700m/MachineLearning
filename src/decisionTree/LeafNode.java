package decisionTree;
public class LeafNode extends DTNode {

	private String prediction;

	public LeafNode(NonLeafNode parent) {
		this.parent = parent;
	}

	public String getPrediction() {
		return prediction;
	}

	public void setPrediction(String s) {
		prediction = s;
	}

	public String toString() {
		String s = ("<html>" + attributeValue + ":" + prediction + "<br>"
				+ positiveSampleCount + ":" + negativeSampleCount + "</html>");
		return s;
	}
}

package decisionTree;
public class DTNode {
	protected String attributeValue;
	protected NonLeafNode parent;
	protected int positiveSampleCount;
	protected int negativeSampleCount;

	public DTNode() {
		parent = null;
	}

	public String getAttributeValue() {
		return attributeValue;
	}

	public DTNode getParent() {
		return parent;
	}

	public int getPosCount() {
		return positiveSampleCount;
	}

	public int getNegCount() {
		return negativeSampleCount;
	}

	public void setAttributeValue(String s) {
		attributeValue = s;
	}

	public void setParent(NonLeafNode n) {
		parent = n;
	}

	public void setPosCount(int i) {
		positiveSampleCount = i;
	}

	public void setNegCount(int i) {
		negativeSampleCount = i;
	}
}

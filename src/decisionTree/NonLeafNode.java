package decisionTree;
import java.util.ArrayList;

public class NonLeafNode extends DTNode {
	private int splittingAttribute;
	private ArrayList<DTNode> children;
	private String majorityVote;
	private float infoGain;

	public NonLeafNode() {
		children = new ArrayList<DTNode>();
	}

	public int getSplittingAttribute() {
		return splittingAttribute;
	}

	public void setSplittingAttribute(int i) {
		splittingAttribute = i;
	}

	public ArrayList<DTNode> getChildren() {
		return children;
	}

	public String getmajorityVote() {
		return majorityVote;
	}

	public float getInfoGain() {
		return infoGain;
	}

	public void addChildren(DTNode n) {
		children.add(n);
	}

	public void clearChildren() {
		children = new ArrayList<DTNode>();
	}

	public void setMajorityVote(String s) {
		majorityVote = s;
	}

	public void setInfoGain(float f) {
		infoGain = f;
	}

	public String toString() {
		String s = ("<html>" + attributeValue + ":" + (splittingAttribute + 1)
				+ "<br>" + positiveSampleCount + ":" + negativeSampleCount + "</html>");
		return s;
	}
}

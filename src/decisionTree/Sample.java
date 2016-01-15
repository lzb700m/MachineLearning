package decisionTree;
import java.util.ArrayList;
import java.util.List;

public class Sample {

	private ArrayList<String> attributes;
	private String label;
	private double weight;

	public Sample(ArrayList<String> attributes, String label) {
		this.attributes = attributes;
		this.label = label;
	}

	public Sample(ArrayList<String> attributes) {
		this.attributes = attributes;
		label = new String();
	}

	public Sample() {
		attributes = new ArrayList<String>();
		label = new String();
	}

	public List<String> getAttributeList() {
		return attributes;
	}

	public String getAttribute(int index) {
		if ((index < 0) || (index >= attributes.size())) {
			throw new IllegalArgumentException(
					"Index number out of attributes range.");
		} else {
			return attributes.get(index);
		}
	}

	public String getLabel() {
		return label;
	}

	public double getWeight() {
		return weight;
	}

	public boolean compareAttribute(Sample other) {
		for (int i = 0; i < attributes.size(); i++) {
			if (!attributes.get(i).equals(other.attributes.get(i))) {
				return false;
			}
		}
		return true;
	}

	public void setWeight(double w) {
		weight = w;
	}

	public String toString() {
		String s1 = "label [" + label + "]; ";
		String s2 = attributes.toString();
		return s1 + s2;
	}
}

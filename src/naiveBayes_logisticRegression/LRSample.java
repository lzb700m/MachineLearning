package naiveBayes_logisticRegression;
/**
 * Sample for Logistic Regression allowed
 * 
 * @author LiP
 *
 */
public class LRSample {
	public static final String POS = "democrat";
	public static final String NEG = "republican";
	public static final String aPOS = "y";
	public static final String aNEG = "n";
	private int label;
	private int[] attribute;

	public LRSample() {

	}

	public void setLable(int l) {
		label = l;
	}

	public int getLabel() {
		return label;
	}

	public void setAttribute(int[] a) {
		attribute = a;
	}

	public int[] getAttribute() {
		return attribute;
	}

	public double dot(double[] weight) {
		if (attribute.length != weight.length) {
			System.out.println("sample size and weight vector does not agree.");
			return 0;
		}
		double result = 0;
		for (int i = 0; i < attribute.length; i++) {
			result += attribute[i] * weight[i];
		}
		return result;
	}

	@Override
	public String toString() {
		String s = "Label: " + label + ", Attirbute: ";
		for (int i = 0; i < attribute.length; i++) {
			s = s + attribute[i] + " ";
		}
		return s;
	}

}

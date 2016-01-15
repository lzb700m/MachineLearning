package naiveBayes_logisticRegression;
import java.util.ArrayList;

/**
 * Sample for Naive Bayes with missing data allowed
 * 
 * @author LiP
 *
 */
public class Sample {
	public static final String MISSING_DATA = "?";
	public static int numberOfAttribute;
	private ArrayList<String> data;

	public Sample(ArrayList<String> s) {
		numberOfAttribute = s.size() - 1;
		this.data = s;
	}

	public Sample() {
		data = new ArrayList<String>();
	}

	// set the ith attribute value to s
	public void setAttributeValue(int i, String s) {
		this.data.set(i, s);
	}

	public String getAttributeValue(int i) {
		if (i < 1) {
			System.out.println("attirbute index start with 1");
			return null;
		} else {
			return this.data.get(i);
		}
	}

	public String getLabel() {
		return this.data.get(0);
	}

	// if sample contains missing data
	public boolean isComplete() {
		for (String s : data) {
			if (s.equals(MISSING_DATA)) {
				return false;
			}
		}
		return true;
	}

	@Override
	public String toString() {
		String s = data.toString();
		return s;
	}
}

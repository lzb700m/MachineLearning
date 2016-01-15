package bayesianNetwork;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BNSample {
	public static final String LABEL = "type";
	public static final String MISSING = "?";
	public static final String POS = "1";
	public static final String NEG = "0";
	public static int numberOfAttributes;
	public static Map<Integer, String> attributeNameMap;
	public static Map<Integer, List<String>> attributeValueMap;

	private ArrayList<String> data;
	private double weight;

	public BNSample(List<String> str, double w) {
		if (str == null || str.size() != numberOfAttributes) {
			System.out
					.println("error creating new sample - data dimension does not agree.");
		} else {
			data = new ArrayList<String>(str);
			weight = w;
		}
	}

	@Deprecated
	public void setData(int i, String s) {
		if (i < 1 || i > numberOfAttributes) {
			System.out.println("index of sample attribute out of bound.");
		} else {
			data.set(i - 1, s);
		}
	}

	public List<String> getData() {
		return data;
	}

	public String getData(int i) {
		if (i < 1 || i > numberOfAttributes) {
			System.out.println("index of sample attribute out of bound.");
			return null;
		} else {
			return data.get(i - 1);
		}
	}

	public void setWeight(double w) {
		if (w < 0 || w > 1) {
			System.out.println("invalid weight value for samples.");
		} else {
			weight = w;
		}
	}

	public double getWeight() {
		return weight;
	}

	public boolean hasMissing() {
		for (String s : data) {
			if (s.equals(MISSING)) {
				return true;
			}
		}
		return false;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for (int i = 1; i <= numberOfAttributes; i++) {
			sb.append(attributeNameMap.get(i) + ": " + data.get(i - 1) + "; ");
		}
		sb.append("weight= " + weight + "]");
		return sb.toString();
	}
}

package bayesianNetwork;
import java.util.HashMap;
import java.util.Map;

public class CPTEntry implements Comparable<CPTEntry> {
	Integer targetIndex;
	String targetValue;
	Map<Integer, String> conditions;

	public CPTEntry(Integer i, String v, Map<Integer, String> c) {
		targetIndex = i;
		targetValue = v;
		conditions = (c == null) ? new HashMap<Integer, String>() : c;
	}

	@Override
	public int compareTo(CPTEntry other) {
		if (this.targetIndex != (other.targetIndex)) {
			return (this.targetIndex > other.targetIndex) ? 1 : -1;
		}

		if (!this.targetValue.equals(other.targetValue)) {
			return 1;
		}

		for (Integer i : this.conditions.keySet()) {
			if (!other.conditions.containsKey(i)) {
				return 1;
			} else if (!this.conditions.get(i).equals(other.conditions.get(i))) {
				return 1;
			}
		}
		return 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("P(");
		sb.append(BNSample.attributeNameMap.get(targetIndex) + "="
				+ targetValue);
		if (conditions != null && conditions.size() != 0) {
			sb.append(" | ");
			for (Integer i : conditions.keySet()) {
				sb.append(BNSample.attributeNameMap.get(i) + "="
						+ conditions.get(i) + ", ");
			}
			sb.delete(sb.length() - 2, sb.length());
		}

		sb.append(")");

		return sb.toString();
	}
}
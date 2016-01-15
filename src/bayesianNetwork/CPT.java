package bayesianNetwork;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class CPT {
	// conditional probability table class used for Bayesian Network
	private Map<CPTEntry, Double> cpt;

	public CPT() {
		cpt = new HashMap<CPTEntry, Double>();
	}

	public void addEntry(CPTEntry e, Double p) {
		cpt.put(e, p);
	}

	public double getValue(CPTEntry e) {
		for (CPTEntry c : cpt.keySet()) {
			if (c.compareTo(e) == 0) {
				return cpt.get(c);
			}
		}
		return 0;
	}

	@Override
	public String toString() {

		StringBuilder sb = new StringBuilder();
		sb.append("Conditional Probability Table:\n");
		List<CPTEntry> cptEntryList = new LinkedList<CPTEntry>(cpt.keySet());
		Collections.sort(cptEntryList);
		for (CPTEntry entry : cptEntryList) {
			sb.append(entry.toString());
			sb.append(" = ");
			sb.append(Utility.DF.format(cpt.get(entry)));
			sb.append("\n");
		}
		sb.deleteCharAt(sb.length() - 1);

		return sb.toString();
	}
}

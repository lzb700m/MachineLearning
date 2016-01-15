package bayesianNetwork;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ParameterLearningTest {
	private static final String BN_TRNINING = "./data/bn.data";
	private static final String BN_DESCRIPTION = "./data/bn.description";

	public static void main(String[] args) throws IOException {
		Utility.initializeSample(BN_DESCRIPTION);
		// System.out.println(BNSample.numberOfAttributes);
		// System.out.println(BNSample.attributeNameMap);
		// System.out.println(BNSample.attributeValueMap);

		Set<BNSample> trainingSet = Utility.createSampleSet(BN_TRNINING);
		// for (BNSample s : trainingSet) {
		// System.out.print(s);
		// System.out.println(s.hasMissing());
		// }
		// System.out.println(trainingSet.size());
		//
		// Map<Integer, String> given = new HashMap<Integer, String>();
		// given.put(2, BNSample.NEG);

		Map<Integer, List<Integer>> structure = new HashMap<Integer, List<Integer>>();
		List<Integer> condition1 = new LinkedList<Integer>();
		List<Integer> condition2 = new LinkedList<Integer>();
		List<Integer> condition3 = new LinkedList<Integer>();
		List<Integer> condition4 = new LinkedList<Integer>();
		condition2.add(1);
		condition3.add(1);
		condition4.add(2);
		condition4.add(3);
		structure.put(1, condition1);
		structure.put(2, condition2);
		structure.put(3, condition3);
		structure.put(4, condition4);

		BayesianNet homework5_2 = new BayesianNet(trainingSet, structure);
		homework5_2.train();

	}
}

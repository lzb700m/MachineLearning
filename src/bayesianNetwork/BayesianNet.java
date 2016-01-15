package bayesianNetwork;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class BayesianNet {
	private final static double CUT_OFF = 10E-4;
	private final Set<BNSample> trainingSamples;
	private final Set<BNSample> cleanSamples;
	// describe the structure of given Bayesian Net
	// Key Integer - a node in Bayesian Net
	// Value Integer set - parent of Key Integer node
	private Map<Integer, List<Integer>> structure;
	private CPT cpt;
	private double logLikeliHood;
	private Set<BNSample> imputatedSamples;
	private Map<Integer, Double> missingProbability;

	public BayesianNet(Set<BNSample> samples,
			Map<Integer, List<Integer>> structure) {
		this.trainingSamples = new HashSet<BNSample>(samples);
		this.cleanSamples = new HashSet<BNSample>();
		this.structure = new HashMap<Integer, List<Integer>>(structure);
		this.cpt = new CPT();
		logLikeliHood = 0;

		for (BNSample s : trainingSamples) {
			if (!s.hasMissing()) {
				cleanSamples.add(new BNSample(s.getData(), s.getWeight()));
			}
		}
		imputatedSamples = new HashSet<BNSample>();
		missingProbability = new HashMap<Integer, Double>();
	}

	public void train() {
		print();
		// MLE to learn probability of an attribute is missing
		for (Integer node : BNSample.attributeNameMap.keySet()) {
			int missingCount = 0;
			for (BNSample s : trainingSamples) {
				if (s.getData(node).equals(BNSample.MISSING)) {
					missingCount++;
				}
			}
			missingProbability.put(node, (double) missingCount
					/ trainingSamples.size());
		}

		// initialize network parameter cpt, should also try different
		// initialization
		// clean samples - get rid of samples with missing data
		// cpt = generateCPT(cleanSamples, structure);
		cpt = generateCPTRandom(structure);
		// cpt = generateCPTEqual(structure);
		logLikeliHood = calculateLikeliHood(trainingSamples, cpt, structure);
		print();

		double tempLikelihood = 0;
		// carry out Expectation Maximization
		while (Math.abs(logLikeliHood - tempLikelihood) > CUT_OFF) {
			// E Step: imputate missing data
			imputateSamples();
			print();
			// M Step: update cpt
			cpt = generateCPT(imputatedSamples, structure);
			// Recalculate log likelihood
			tempLikelihood = logLikeliHood;
			logLikeliHood = calculateLikeliHood(trainingSamples, cpt, structure);
		}
	}

	public static CPT generateCPTEqual(Map<Integer, List<Integer>> structure) {
		if (structure == null) {
			return null;
		}

		CPT result = new CPT();
		for (Integer node : structure.keySet()) {
			if (structure.get(node) == null || structure.get(node).size() == 0) {
				ArrayList<String> values = (ArrayList<String>) BNSample.attributeValueMap
						.get(node);
				for (int i = 0; i < values.size(); i++) {
					CPTEntry entry = new CPTEntry(node, values.get(i),
							new HashMap<Integer, String>());
					result.addEntry(entry, (double) 1
							/ BNSample.attributeValueMap.get(node).size());
				}
			} else {
				List<Map<Integer, String>> conditionList = generateConditionMaps(structure
						.get(node));

				for (Map<Integer, String> condition : conditionList) {
					ArrayList<String> values = (ArrayList<String>) BNSample.attributeValueMap
							.get(node);
					for (int i = 0; i < values.size(); i++) {
						CPTEntry entry = new CPTEntry(node, values.get(i),
								condition);
						result.addEntry(entry, (double) 1
								/ BNSample.attributeValueMap.get(node).size());
					}
				}
			}
		}

		return result;
	}

	public static CPT generateCPTRandom(Map<Integer, List<Integer>> structure) {
		if (structure == null) {
			return null;
		}

		CPT result = new CPT();
		for (Integer node : structure.keySet()) {
			if (structure.get(node) == null || structure.get(node).size() == 0) {
				double remainingWeight = 1;
				ArrayList<String> values = (ArrayList<String>) BNSample.attributeValueMap
						.get(node);
				for (int i = 0; i < values.size() - 1; i++) {
					double probablity = Math.random() * remainingWeight;
					CPTEntry entry = new CPTEntry(node, values.get(i),
							new HashMap<Integer, String>());
					result.addEntry(entry, probablity);
					remainingWeight -= probablity;
				}
				CPTEntry lastEntry = new CPTEntry(node, values.get(values
						.size() - 1), new HashMap<Integer, String>());
				result.addEntry(lastEntry, remainingWeight);
			} else {
				List<Map<Integer, String>> conditionList = generateConditionMaps(structure
						.get(node));

				for (Map<Integer, String> condition : conditionList) {
					double remainingWeight = 1;
					ArrayList<String> values = (ArrayList<String>) BNSample.attributeValueMap
							.get(node);
					for (int i = 0; i < values.size() - 1; i++) {
						double probability = Math.random() * remainingWeight;
						CPTEntry entry = new CPTEntry(node, values.get(i),
								condition);
						result.addEntry(entry, probability);
						remainingWeight -= probability;
					}
					CPTEntry lastEntry = new CPTEntry(node, values.get(values
							.size() - 1), condition);
					result.addEntry(lastEntry, remainingWeight);
				}
			}
		}

		return result;
	}

	// generate conditional probability table given samples without missing data
	// and Bayesian net structure
	public static CPT generateCPT(Set<BNSample> samples,
			Map<Integer, List<Integer>> structure) {
		// input check
		if (samples == null || structure == null) {
			return null;
		}

		for (BNSample s : samples) {
			if (s.hasMissing()) {
				return null;
			}
		}

		CPT result = new CPT();
		for (Integer index : structure.keySet()) {
			if (structure.get(index) == null
					|| structure.get(index).size() == 0) {
				// node index has no parent, calculate probability in samples
				// directly
				for (String value : BNSample.attributeValueMap.get(index)) {
					// for all possible values on node index
					CPTEntry cptEntry = new CPTEntry(index, value,
							new HashMap<Integer, String>());
					double probability = probability(index, value, samples);
					result.addEntry(cptEntry, probability);
				}
			} else {
				// node index has parent(s), calculate conditional probability

				// generate all combination of condition
				List<Map<Integer, String>> conditionList = generateConditionMaps(structure
						.get(index));
				// calculate all combinations of conditional probability
				for (Map<Integer, String> condition : conditionList) {
					for (String value : BNSample.attributeValueMap.get(index)) {
						CPTEntry cptEntry = new CPTEntry(index, value,
								condition);
						double probability = conditonalProbability(index,
								value, condition, samples);
						result.addEntry(cptEntry, probability);
					}
				}
			}
		}
		return result;
	}

	public static double conditonalProbability(int index, String value,
			Map<Integer, String> given, Set<BNSample> samples) {
		// filter the samples set
		Set<BNSample> conditionalSamples = filterBnSamples(given, samples);

		// calculate conditional probability
		return probability(index, value, conditionalSamples);
	}

	private static double probability(int index, String value,
			Set<BNSample> samples) {
		double count = 0;
		double total = 0;
		for (BNSample s : samples) {
			if (!s.hasMissing()) {
				total = total + s.getWeight();
				if (s.getData(index).equals(value)) {
					count = count + s.getWeight();
				}
			}
		}
		return count / total;
	}

	private static Set<BNSample> filterBnSamples(Map<Integer, String> given,
			Set<BNSample> samples) {
		Set<BNSample> result = new HashSet<BNSample>();

		for (BNSample s : samples) {
			if (s.hasMissing()) {
				continue;
			}

			boolean agree = true;
			for (Integer index : given.keySet()) {
				if (!s.getData(index).equals(given.get(index))) {
					agree = false;
				}
			}

			if (agree) {
				result.add(new BNSample(s.getData(), s.getWeight()));
			}
		}
		return result;
	}

	public static List<Map<Integer, String>> generateConditionMaps(
			List<Integer> condition) {
		List<Map<Integer, String>> result = new LinkedList<Map<Integer, String>>();
		Map<Integer, String> map = new HashMap<Integer, String>();
		generateConditionMapsHelper(condition, map, result, 0);

		return result;
	}

	private static void generateConditionMapsHelper(List<Integer> condition,
			Map<Integer, String> map, List<Map<Integer, String>> result,
			int index) {
		if (map.size() == condition.size()) {
			result.add(new HashMap<Integer, String>(map));
			return;
		}

		if (!map.containsKey(condition.get(index))) {
			for (String s : BNSample.attributeValueMap
					.get(condition.get(index))) {
				map.put(condition.get(index), s);
				generateConditionMapsHelper(condition, map, result, index + 1);
				map.remove(condition.get(index));
			}
		}
	}

	private static double calculateLikeliHood(Set<BNSample> samples, CPT cpt,
			Map<Integer, List<Integer>> structure) {
		double result = 0;
		for (BNSample s : samples) {
			if (!s.hasMissing()) {
				result += Math.log(jointProbability(s, cpt, structure));
			} else {
				double logSum = 0;
				List<BNSample> splitedSamples = splitSample(s, cpt, structure);
				for (BNSample sMissing : splitedSamples) {
					logSum += jointProbability(sMissing, cpt, structure);
				}
				result += Math.log(logSum);
			}
		}
		return result;
	}

	// private static double calculateLikeliHood(Set<BNSample> samples, CPT cpt,
	// Map<Integer, List<Integer>> structure,
	// Map<Integer, Double> missingProbability) {
	// double result = 0;
	// for (BNSample s : samples) {
	// double missingLikelihood = 1;
	// for (Integer node : BNSample.attributeNameMap.keySet()) {
	// if (s.getData(node).equals(BNSample.MISSING)) {
	// missingLikelihood *= missingProbability.get(node);
	// } else {
	// missingLikelihood *= (1 - missingProbability.get(node));
	// }
	// }
	// if (!s.hasMissing()) {
	// result += Math.log(jointProbability(s, cpt, structure)
	// * missingLikelihood);
	// } else {
	// double logSum = 0;
	// List<BNSample> splitedSamples = splitSample(s, cpt, structure);
	// for (BNSample sMissing : splitedSamples) {
	// logSum += jointProbability(sMissing, cpt, structure);
	// }
	// result += Math.log(logSum * missingLikelihood);
	// }
	// }
	// return result;
	// }

	private void imputateSamples() {
		Set<BNSample> result = new HashSet<BNSample>();

		for (BNSample s : trainingSamples) {
			if (!s.hasMissing()) {
				result.add(new BNSample(s.getData(), s.getWeight()));
			} else {
				result.addAll(new HashSet<BNSample>(splitSample(s, cpt,
						structure)));
			}
		}
		imputatedSamples = new HashSet<BNSample>(result);
	}

	private static List<BNSample> splitSample(BNSample sample, CPT cpt,
			Map<Integer, List<Integer>> structure) {
		List<BNSample> result = new LinkedList<BNSample>();
		List<String> data = sample.getData();
		List<List<String>> allData = new ArrayList<List<String>>();
		List<String> list = new ArrayList<String>();

		splitSampleHelper(data, allData, list, 1);

		for (List<String> splitData : allData) {
			result.add(new BNSample(splitData, 1));
		}

		// udpate weight of splitted samples
		double totalWeight = 0;
		for (BNSample s : result) {
			totalWeight += jointProbability(s, cpt, structure);
		}

		for (BNSample s : result) {
			s.setWeight(jointProbability(s, cpt, structure) / totalWeight);
		}

		return result;
	}

	private static void splitSampleHelper(List<String> data,
			List<List<String>> allData, List<String> list, int index) {
		if (list.size() == BNSample.numberOfAttributes) {
			allData.add(new ArrayList<String>(list));
			return;
		}

		if (!data.get(index - 1).equals(BNSample.MISSING)) {
			list.add(new String(data.get(index - 1)));
			splitSampleHelper(data, allData, list, index + 1);
			list.remove(list.size() - 1);
		} else {
			for (String s : BNSample.attributeValueMap.get(index)) {
				list.add(new String(s));
				splitSampleHelper(data, allData, list, index + 1);
				list.remove(list.size() - 1);
			}
		}
	}

	private static double jointProbability(BNSample s, CPT cpt,
			Map<Integer, List<Integer>> structure) {
		double result = 1;

		for (Integer node : structure.keySet()) {
			String value = s.getData(node);
			Map<Integer, String> condition = new HashMap<Integer, String>();
			for (Integer cNode : structure.get(node)) {
				String cValue = s.getData(cNode);
				condition.put(cNode, cValue);
			}
			CPTEntry entry = new CPTEntry(node, value, condition);
			result = result * cpt.getValue(entry);
		}

		return result;
	}

	public void print() {
		System.out.println(structure);
		System.out.println("CPT: " + cpt);
		System.out.println("logLikeliHood" + logLikeliHood);
		System.out.println("Missing probability" + missingProbability);
		System.out.println();
	}
}

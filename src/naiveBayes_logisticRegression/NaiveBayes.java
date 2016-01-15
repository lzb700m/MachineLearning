package naiveBayes_logisticRegression;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Naive Bayes learning algorithm for discrete value samples
 * 
 * @author LiP
 *
 */
public class NaiveBayes {
	// splited sample by sample lable
	Map<String, HashSet<Sample>> splitSamples;
	// probability of label
	Map<String, Double> pLabel;
	// probability of Xi given label
	Map<String, List<Map<String, Double>>> pConditional;
	// uniform Dirichlet prior over all labels
	Integer dPrior;

	public NaiveBayes() {
		pLabel = new HashMap<String, Double>();
		pConditional = new HashMap<String, List<Map<String, Double>>>();
		dPrior = 0;
	}

	/*
	 * training a Naive Bayes, calculate all conditional probabilities
	 */
	public void train(Set<Sample> samples) {
		int sampleSize = samples.size();
		// split training samples by lables
		splitSamples = new HashMap<String, HashSet<Sample>>();
		for (Sample s : samples) {
			if (!splitSamples.containsKey(s.getLabel())) {
				splitSamples.put(s.getLabel(), new HashSet<Sample>());
			}
			splitSamples.get(s.getLabel()).add(s);
		}

		// update probabilities
		Iterator<Entry<String, HashSet<Sample>>> it = splitSamples.entrySet()
				.iterator();
		while (it.hasNext()) {
			Map.Entry<String, HashSet<Sample>> entry = it.next();
			// update P(Y)
			pLabel.put(entry.getKey(), (double) entry.getValue().size()
					/ sampleSize);
			// update P(X|Y)
			List<Map<String, Double>> labelAttributeList = new ArrayList<Map<String, Double>>();
			HashSet<Sample> sSample = entry.getValue();
			for (int i = 0; i < Sample.numberOfAttribute; i++) {
				Map<String, Double> attributePro = getAttributePro(sSample,
						i + 1);
				labelAttributeList.add(attributePro);
			}
			pConditional.put(entry.getKey(), labelAttributeList);
		}
	}

	/*
	 * calculate accuracy given a testing set
	 */
	public double accuracy(Set<Sample> samples) {
		int correctCount = 0;
		int totalCount = samples.size();
		for (Sample s : samples) {
			if (s.getLabel().equals(predict(s))) {
				correctCount++;
			}
		}
		return (double) correctCount / totalCount;
	}

	/*
	 * printout training statistics
	 */
	public void printNB() {
		// print P(Y)
		Iterator<Entry<String, Double>> itPLabel = pLabel.entrySet().iterator();
		while (itPLabel.hasNext()) {
			Map.Entry<String, Double> entry = itPLabel.next();
			System.out.println("P(" + entry.getKey() + ") = "
					+ entry.getValue());
		}

		// print P(X|Y)
		Iterator<Entry<String, List<Map<String, Double>>>> itCP = pConditional
				.entrySet().iterator();
		while (itCP.hasNext()) {
			Map.Entry<String, List<Map<String, Double>>> entry = itCP.next();
			String label = entry.getKey();
			List<Map<String, Double>> list = entry.getValue();
			for (int i = 0; i < Sample.numberOfAttribute; i++) {
				Map<String, Double> attributeEntryMap = list.get(i);
				Iterator<Entry<String, Double>> itAttribute = attributeEntryMap
						.entrySet().iterator();
				while (itAttribute.hasNext()) {
					Map.Entry<String, Double> attributeEntry = itAttribute
							.next();
					System.out.println("P(F" + (i + 1) + "="
							+ attributeEntry.getKey() + "|" + label + ") = "
							+ attributeEntry.getValue());
				}
			}
		}
	}

	/*
	 * calculate probability of a given attribute index, assume that input
	 * sample set have the same label
	 */
	private Map<String, Double> getAttributePro(Set<Sample> samples, int index) {
		Map<String, Double> result = new HashMap<String, Double>();
		int size = 0;
		// count samples that does not have missing value
		for (Sample s : samples) {
			String attributeValue = s.getAttributeValue(index);
			if (attributeValue.equals(Sample.MISSING_DATA)) {
				continue;
			}
			size++;
		}

		for (Sample s : samples) {
			String attributeValue = s.getAttributeValue(index);
			if (attributeValue.equals(Sample.MISSING_DATA)) {
				continue;
			}
			if (!result.containsKey(attributeValue)) {
				result.put(attributeValue, (double) 0);
			}

			result.put(attributeValue, result.get(attributeValue) + (double) 1
					/ size);
		}
		return result;
	}

	/*
	 * predict a data sample using learned Naive Bayes
	 */
	private String predict(Sample s) {
		double ml = 0;
		String mlLabel = null;

		Iterator<Entry<String, Double>> itPLabel = pLabel.entrySet().iterator();
		while (itPLabel.hasNext()) {
			Map.Entry<String, Double> entry = itPLabel.next();
			String label = entry.getKey();
			double likelihood = entry.getValue();
			for (int i = 0; i < Sample.numberOfAttribute; i++) {
				likelihood = likelihood
						* getConPro(label, i + 1, s.getAttributeValue(i + 1));
			}
			if (likelihood > ml) {
				ml = likelihood;
				mlLabel = label;
			}
		}
		return mlLabel;
	}

	/*
	 * return the trained conditional probability of given label, given
	 * attribute index and given attribute value
	 */
	private double getConPro(String label, Integer index, String AttributeValue) {
		if (pConditional.get(label).get(index - 1).containsKey(AttributeValue)) {
			return pConditional.get(label).get(index - 1).get(AttributeValue);
		} else {
			return 0;
		}
	}
}

package naiveBayes_logisticRegression;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class Utility {
	/*
	 * read sample data from file (for Naive Bayes)
	 */
	public static HashSet<Sample> readFromFile(String fileLoc)
			throws IOException {
		HashSet<Sample> result = new HashSet<Sample>();

		FileReader fin = new FileReader(fileLoc);
		BufferedReader bin = new BufferedReader(fin);

		while (true) {
			String line = bin.readLine();
			if (line == null) {
				break;
			}
			String[] record = line.split(",");
			ArrayList<String> data = new ArrayList<String>();
			for (int i = 0; i < record.length; i++) {
				data.add(record[i]);
			}
			result.add(new Sample(data));
		}
		fin.close();
		return result;
	}

	/*
	 * convert sample data from Sample object to LRSample object for easier
	 * Logistic Regression handling
	 */
	public static Set<LRSample> convertLR(Set<Sample> sNB) {
		HashSet<LRSample> result = new HashSet<LRSample>();

		for (Sample s : sNB) {
			LRSample sample = new LRSample();
			if (s.getLabel().equals(LRSample.POS)) {
				sample.setLable(1);
			} else {
				sample.setLable(0);
			}
			int[] attributeArray = new int[Sample.numberOfAttribute + 1];
			// used for b(w0)
			attributeArray[0] = 1;
			for (int i = 1; i < attributeArray.length; i++) {
				if (s.getAttributeValue(i).equals(LRSample.aPOS)) {
					attributeArray[i] = 1;
				} else if (s.getAttributeValue(i).equals(LRSample.aNEG)) {
					attributeArray[i] = 0;
				}
			}
			sample.setAttribute(attributeArray);
			result.add(sample);
		}
		return result;
	}

	/*
	 * perform listwise deletion for samples containing missing data return
	 * sample set that contains no missing data
	 */
	public static Set<Sample> listwiseDeletion(Set<Sample> samples) {
		Set<Sample> result = new HashSet<Sample>();
		for (Sample s : samples) {
			if (s.isComplete()) {
				result.add(s);
			}
		}
		return result;
	}

	/*
	 * perform data imputation (with mode) for samples containing missing data
	 * return sample set that contains no missing data
	 */
	public static Set<Sample> imputation(Set<Sample> samples) {
		// split training samples by lables
		HashMap<String, HashSet<Sample>> splitSamples = new HashMap<String, HashSet<Sample>>();
		for (Sample s : samples) {
			if (!splitSamples.containsKey(s.getLabel())) {
				splitSamples.put(s.getLabel(), new HashSet<Sample>());
			}
			splitSamples.get(s.getLabel()).add(s);
		}

		Set<Sample> result = new HashSet<Sample>();
		Iterator<Entry<String, HashSet<Sample>>> it = splitSamples.entrySet()
				.iterator();
		while (it.hasNext()) {
			result.addAll(imputationPerLable(it.next().getValue()));
		}
		return result;
	}

	/*
	 * perform imputation for missing data, assume all samples have the same
	 * label
	 */
	private static Set<Sample> imputationPerLable(Set<Sample> samples) {

		Set<Sample> result = new HashSet<Sample>();
		List<Map<String, Integer>> modeCount = new ArrayList<Map<String, Integer>>();
		List<String> mode = new ArrayList<String>();

		// find most occurence from training data
		for (int i = 0; i < Sample.numberOfAttribute; i++) {
			Map<String, Integer> attributeCount = new HashMap<String, Integer>();
			for (Sample s : samples) {
				String value = s.getAttributeValue(i + 1);
				if (!value.equals(Sample.MISSING_DATA)) {
					if (!attributeCount.containsKey(value)) {
						attributeCount.put(value, 0);
					}
					attributeCount.put(value, attributeCount.get(value) + 1);
				}
			}
			modeCount.add(attributeCount);
		}

		for (int i = 0; i < Sample.numberOfAttribute; i++) {
			int count = 0;
			String modeValue = null;
			Map<String, Integer> modeEntry = modeCount.get(i);
			Iterator<Entry<String, Integer>> modeEntryIt = modeEntry.entrySet()
					.iterator();
			while (modeEntryIt.hasNext()) {
				Map.Entry<String, Integer> modeEntryItEntry = modeEntryIt
						.next();
				if (modeEntryItEntry.getValue() > count) {
					modeValue = modeEntryItEntry.getKey();
					count = modeEntryItEntry.getValue();
				}
			}
			mode.add(modeValue);
		}

		for (Sample s : samples) {
			if (s.isComplete()) {
				result.add(s);
			} else {
				ArrayList<String> modifiedData = new ArrayList<String>();
				modifiedData.add(s.getLabel());
				for (int i = 0; i < Sample.numberOfAttribute; i++) {
					if (s.getAttributeValue(i + 1).equals(Sample.MISSING_DATA)) {
						modifiedData.add(mode.get(i));
					} else {
						modifiedData.add(s.getAttributeValue(i + 1));
					}
				}
				result.add(new Sample(modifiedData));
			}
		}
		return result;
	}

}

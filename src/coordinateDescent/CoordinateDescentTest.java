package coordinateDescent;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import adaboost.AdaBoosting;
import decisionTree.DTNode;
import decisionTree.Sample;
import decisionTree.Utility;

public class CoordinateDescentTest {

	private static final String TRAIN = "data/heart_train.data";
	private static final String TEST = "data/heart_test.data";
	private static final String DECSRIPTION_FILE = "data/heart.description";
	public static final String POS_LABEL = "1";
	public static final String NEG_LABEL = "0";

	public static void main(String[] args) {
		HashSet<Sample> trainingSamples = null;
		HashSet<Sample> testingSamples = null;
		HashMap<Integer, ArrayList<String>> sampleDescription = null;

		// read training samples from file
		try {
			trainingSamples = Utility.createSampleFromFile(TRAIN);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		// read testing samples from file
		try {
			testingSamples = Utility.createSampleFromFile(TEST);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		// read sample description from file
		try {
			sampleDescription = Utility
					.createSampleDescriptionFromFile(DECSRIPTION_FILE);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		System.out.println(trainingSamples.size() + ", "
				+ testingSamples.size() + ", " + sampleDescription.size());

		CoordinateDescent cd = new CoordinateDescent(sampleDescription);
		cd.update(trainingSamples);
		System.out.println(cd.alpha);

		Map<Double, DTNode> classifier = new HashMap<Double, DTNode>();
		for (int i = 0; i < cd.stumpList.size(); i++) {
			classifier.put(cd.alpha.get(i), cd.stumpList.get(i));
		}

		double trainingError = AdaBoosting.test(trainingSamples, classifier);
		double testingError = AdaBoosting.test(testingSamples, classifier);

		System.out.println("Accuracy: training - " + trainingError
				+ ", testing - " + testingError);

	}
}

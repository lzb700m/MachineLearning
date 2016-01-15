package adaboost;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import decisionTree.DTNode;
import decisionTree.Sample;
import decisionTree.Utility;

public class AdaBoostingMain {
	// mushroom data set configuration
	// private static final String TRAIN = "data/mush_train.data";
	// private static final String TEST = "data/mush_test.data";
	// private static final String DECSRIPTION_FILE = "data/mush.description";
	// public static final String POS_LABEL = "e";
	// public static final String NEG_LABEL = "p";

	// heart data set configuration
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

		Map<Double, DTNode> classifier = AdaBoosting.train(trainingSamples,
				sampleDescription, 20);
		double trainingAccuracy = AdaBoosting.test(trainingSamples, classifier);
		double testingAccuracy = AdaBoosting.test(testingSamples, classifier);

		System.out.println("Accuracy: training - " + trainingAccuracy * 100
				+ "%, testing - " + testingAccuracy * 100 + "%");

	}
}

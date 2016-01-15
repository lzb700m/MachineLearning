package naiveBayes_logisticRegression;
import java.io.IOException;
import java.util.Set;

/**
 * Driver function for Machine Learning Problem Set 4 - Problem 2
 * 
 * Implement the following learning functions: 
 * - Naive Bayes with pair-wise deletion
 * - Naive Bayes with list-wise deletion
 * - Naive Bayes with imputation
 * - Logistic Regression with list-wise deletion
 * - Logistic Regression with imputaion
 * 
 * Test data are processed using list-wise deletion
 * 
 * @author LiP
 *
 */

public class NBLRTest {

	private static final String TRAINING_DATA = "./data/voting_train.data";
	private static final String TESTING_DATA = "./data/voting_test.data";

	public static void main(String[] args) throws IOException {
		Set<Sample> trainOriginal = Utility.readFromFile(TRAINING_DATA);
		Set<Sample> testOriginal = Utility.readFromFile(TESTING_DATA);

		/*
		 * Naive Bayes Handle missing data with Pairwise Deletion
		 */
		Set<Sample> trainPWD = trainOriginal;
		Set<Sample> testPWD = Utility.listwiseDeletion(testOriginal);
		NaiveBayes pwd = new NaiveBayes();
		pwd.train(trainPWD);
		System.out
				.println("Naive Bayes - Pairwise Deletion on missing traning data");
		pwd.printNB();
		System.out.println("Naive Bayes - Pairwise Deletion - Accuracy: "
				+ pwd.accuracy(testPWD) + "\n");

		/*
		 * Naive Bayes Handle missing data with Listwise Deletion
		 */
		Set<Sample> trainLWD = Utility.listwiseDeletion(trainOriginal);
		Set<Sample> testLWD = Utility.listwiseDeletion(testOriginal);
		NaiveBayes lwd = new NaiveBayes();
		lwd.train(trainLWD);
		System.out
				.println("Naive Bayes - Listwise Deletion on missing traning data");
		lwd.printNB();
		System.out.println("Naive Bayes - Listwise Deletion - Accuracy: "
				+ lwd.accuracy(testLWD) + "\n");

		/*
		 * Naive Bayes Handle missing data with imputation
		 */
		Set<Sample> trainIMP = Utility.imputation(trainOriginal);
		Set<Sample> testIMP = Utility.listwiseDeletion(testOriginal);
		NaiveBayes imp = new NaiveBayes();
		imp.train(trainIMP);
		System.out.println("Naive Bayes - Imputation on missing traning data");
		imp.printNB();
		System.out.println("Naive Bayes - Imputation - Accuracy: "
				+ imp.accuracy(testIMP) + "\n");

		/*
		 * Logistic Regression Handle missing data with Listwise Deletion
		 */
		Set<LRSample> trainLWDLR = Utility.convertLR(Utility
				.listwiseDeletion(trainOriginal));
		Set<LRSample> testLWDLR = Utility.convertLR(Utility
				.listwiseDeletion(testOriginal));

		LogisticRegression lwdLR = new LogisticRegression();
		lwdLR.train(trainLWDLR);
		System.out
				.println("Logistic Regression - Listwise Deletion on missing traning data");
		lwdLR.printLR();
		System.out
				.println("Logistic Regression - Listwise Deletion - Accuracy: "
						+ lwdLR.accuracy(testLWDLR) + "\n");

		/*
		 * Logistic Regression Handle missing data with Imputation Deletion
		 */
		Set<LRSample> trainIMPLR = Utility.convertLR(Utility
				.imputation(trainOriginal));
		Set<LRSample> testIMPLR = Utility.convertLR(Utility
				.listwiseDeletion(testOriginal));

		LogisticRegression impLR = new LogisticRegression();
		impLR.train(trainIMPLR);
		System.out
				.println("Logistic Regression - Imputation on missing traning data");
		impLR.printLR();
		System.out
				.println("Logistic Regression - Listwise Deletion - Accuracy: "
						+ impLR.accuracy(testIMPLR) + "\n");
	}
}

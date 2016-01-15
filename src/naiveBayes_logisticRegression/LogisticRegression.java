package naiveBayes_logisticRegression;
import java.util.Set;

/**
 * Logistic Regression learning algorithm for discrete value samples
 * 
 * @author LiP
 *
 */
public class LogisticRegression {
	private static final double CUT_OFF = 0.01;
	private static final double STEP = 0.001;

	private double[] weight = new double[Sample.numberOfAttribute + 1];
	double logLikelihood;

	public double[] getWeight() {
		return weight;
	}

	public void train(Set<LRSample> samples) {
		// initialize weight
		for (int i = 0; i < weight.length; i++) {
			weight[i] = 0;
		}
		logLikelihood = 0;

		boolean hasConverged = false;
		while (!hasConverged) {
			double updatedLLH = updateLogLikelihood(samples);
			if (Math.abs(logLikelihood - updatedLLH) < CUT_OFF) {
				hasConverged = true;
			}
			logLikelihood = updatedLLH;
			updateWeight(samples);
		}
	}

	public double accuracy(Set<LRSample> samples) {
		int correctCount = 0;
		int totalCount = samples.size();
		for (LRSample s : samples) {
			if (s.getLabel() == predict(s)) {
				correctCount++;
			}
		}
		return (double) correctCount / totalCount;
	}

	public void printLR() {
		System.out.print("Log likelihood: " + logLikelihood);
		String s = "";
		for (int i = 0; i < weight.length; i++) {
			s = s + weight[i] + " ";
		}
		System.out.println(", Weight: " + s);
	}

	private double updateLogLikelihood(Set<LRSample> samples) {
		double result = 0;
		for (LRSample s : samples) {
			int y = s.getLabel();
			result += (y * s.dot(weight))
					- Math.log(1 + Math.exp(s.dot(weight)));
		}
		return result;
	}

	private void updateWeight(Set<LRSample> samples) {
		double[] updatedWeight = new double[weight.length];
		for (int i = 0; i < weight.length; i++) {
			updatedWeight[i] = 0;
			double increment = 0;
			for (LRSample s : samples) {
				int prediction = (p0(s) >= p1(s)) ? 0 : 1;
				increment += s.getAttribute()[i] * (s.getLabel() - prediction);
			}
			updatedWeight[i] = weight[i] + STEP * increment;
		}
		weight = updatedWeight;
	}

	private double p0(LRSample s) {
		double result = 1 / (1 + Math.exp(s.dot(weight)));
		return result;
	}

	private double p1(LRSample s) {
		double result = Math.exp(s.dot(weight)) / (1 + Math.exp(s.dot(weight)));
		return result;
	}

	private int predict(LRSample s) {
		return (p0(s) >= p1(s)) ? 0 : 1;
	}
}

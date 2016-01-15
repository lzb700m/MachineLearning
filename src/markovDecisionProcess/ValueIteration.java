package markovDecisionProcess;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ValueIteration {
	private static final double[][] REWARD = new double[][] { { 0, 1, 0.25 },
			{ 0.25, 0, 1 }, { -1, 0.25, 0 } };
	private static final double GAMMA = 0.9;
	private static final double EPSILON = 10E-9;

	public static void main(String[] args) {

		// initialize the value function to zeros
		ArrayList<Double[]> valueFunction = new ArrayList<Double[]>();
		Double[] initialValue = new Double[] { (double) 0, (double) 0,
				(double) 0 };
		valueFunction.add(initialValue);
		boolean continueIteration = true;

		// value iteration
		while (continueIteration) {
			continueIteration = false;
			Double[] Vt = valueFunction.get(valueFunction.size() - 1);
			Double[] Vtplus1 = new Double[Vt.length];

			for (int i = 0; i < Vtplus1.length; i++) {
				double v = (double) (Integer.MIN_VALUE);
				double[] reward = REWARD[i];

				for (int j = 0; j < reward.length; j++) {
					double newV = reward[j] + GAMMA * Vt[j];
					if (newV > v) {
						v = newV;
					}
				}
				Vtplus1[i] = v;
			}
			valueFunction.add(Vtplus1);

			for (int i = 0; i < Vt.length; i++) {
				if (Math.abs(Vt[i] - Vtplus1[i]) > EPSILON) {
					continueIteration = true;
				}
			}
		}

		// print value function
		for (int i = 0; i < valueFunction.size(); i++) {
			Double[] v = valueFunction.get(i);
			for (int j = 0; j < v.length; j++) {
				System.out.printf("%8.3f", v[j]);
			}
			System.out.println();
		}

		// find optimal policy
		Double[] valueOptimal = valueFunction.get(valueFunction.size() - 1);
		Map<Integer, Integer> optimalPolicy = new HashMap<Integer, Integer>();

		for (int i = 0; i < REWARD.length; i++) {
			double[] reward = REWARD[i];
			double finalValue = (double) Integer.MIN_VALUE;
			int finalAction = 0;
			for (int j = 0; j < reward.length; j++) {
				if ((reward[j] + valueOptimal[j]) > finalValue) {
					finalValue = reward[j] + GAMMA * valueOptimal[j];
					finalAction = j;
				}
			}

			optimalPolicy.put(i + 1, finalAction + 1);
		}

		System.out.println(optimalPolicy);

	}
}

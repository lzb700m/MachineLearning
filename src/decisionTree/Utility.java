package decisionTree;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Utility {

	public static HashSet<Sample> createSampleFromFile(String fileLoc)
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
			String label = record[0];
			ArrayList<String> attributes = new ArrayList<String>();
			for (int i = 1; i < record.length; i++) {
				attributes.add(record[i]);
			}
			result.add(new Sample(attributes, label));
		}
		fin.close();
		return result;
	}

	public static HashMap<Integer, ArrayList<String>> createSampleDescriptionFromFile(
			String fileLoc) throws IOException {

		HashMap<Integer, ArrayList<String>> result = new HashMap<Integer, ArrayList<String>>();

		FileReader fin = new FileReader(fileLoc);
		BufferedReader bin = new BufferedReader(fin);

		while (true) {
			String line = bin.readLine();
			if (line == null) {
				break;
			}
			String[] record = line.split(",");
			int index = Integer.parseInt(record[0]) - 1;
			ArrayList<String> attributes = new ArrayList<String>();
			for (int i = 2; i < record.length; i++) {
				attributes.add(record[i]);
			}
			result.put(index, attributes);
		}
		fin.close();
		return result;
	}

	public static float getEntropy(int posCount, int negCount) {

		if (posCount == 0 && negCount == 0) {
			return 0;
		} else {
			float posProb = (float) posCount / (posCount + negCount);
			float negProb = (float) negCount / (posCount + negCount);

			double entropy;
			if (posProb == 0) {
				entropy = -(negProb) * (Math.log(negProb) / Math.log(2));
			} else if (negProb == 0) {
				entropy = -(posProb) * (Math.log(posProb) / Math.log(2));
			} else {
				entropy = -(posProb) * (Math.log(posProb) / Math.log(2))
						- (negProb) * (Math.log(negProb) / Math.log(2));
			}
			return (float) entropy;
		}
	}

}

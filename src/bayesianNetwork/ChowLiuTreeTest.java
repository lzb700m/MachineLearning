package bayesianNetwork;
import java.io.IOException;
import java.util.Set;

public class ChowLiuTreeTest {
	private static final String ZOO_TRNINING = "./data/zoo.data";
	private static final String ZOO_DESCRIPTION = "./data/zoo.description";

	public static void main(String[] args) throws IOException {
		Utility.initializeSample(ZOO_DESCRIPTION);

		Set<BNSample> trainingSet = Utility.createSampleSet(ZOO_TRNINING);

		ChowLiuTree homework5_1 = new ChowLiuTree(trainingSet);

		homework5_1.train();
		homework5_1.printChowLiuTree();
	}
}

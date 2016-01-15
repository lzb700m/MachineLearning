package bayesianNetwork;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JFrame;

import org.apache.commons.collections15.Transformer;

import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.algorithms.layout.TreeLayout;
import edu.uci.ics.jung.graph.DelegateTree;
import edu.uci.ics.jung.visualization.BasicVisualizationServer;
import edu.uci.ics.jung.visualization.renderers.Renderer.VertexLabel.Position;

public class Utility {

	public final static DecimalFormat DF = new DecimalFormat("0.00");

	public static void initializeSample(String fileLoc) throws IOException {
		FileReader fin = new FileReader(fileLoc);
		BufferedReader bin = new BufferedReader(fin);

		Map<Integer, String> nameMap = new HashMap<Integer, String>();
		Map<Integer, List<String>> valueMap = new HashMap<Integer, List<String>>();

		while (true) {
			String line = bin.readLine();
			if (line == null) {
				break;
			}

			String[] record = line.split(",");
			/*
			 * description file structure: for each record: record[0] index of
			 * attribute, start from 1 record[1] name of attribute record[>1}
			 * possible values of attribute
			 */
			nameMap.put(Integer.valueOf(record[0]), record[1]);
			List<String> valueList = new ArrayList<String>();
			for (int i = 2; i < record.length; i++) {
				valueList.add(record[i]);
			}
			valueMap.put(Integer.valueOf(record[0]), valueList);
		}

		BNSample.attributeNameMap = nameMap;
		BNSample.attributeValueMap = valueMap;
		BNSample.numberOfAttributes = nameMap.size();
		fin.close();
	}

	/*
	 * read sample data from file (for Naive Bayes)
	 */
	public static Set<BNSample> createSampleSet(String fileLoc)
			throws IOException {
		Set<BNSample> result = new HashSet<BNSample>();

		FileReader fin = new FileReader(fileLoc);
		BufferedReader bin = new BufferedReader(fin);

		while (true) {
			String line = bin.readLine();
			if (line == null) {
				break;
			}
			String[] record = line.split(",");
			ArrayList<String> data = new ArrayList<String>();
			for (String s : record) {
				data.add(s);
			}
			result.add(new BNSample(data, 1));
		}
		fin.close();
		return result;
	}

	public static void printArray(double matrix[][]) {
		for (double[] row : matrix) {
			for (double num : row) {
				System.out.printf(DF.format(num) + " ");
			}
			System.out.println();
		}
	}

	/*
	 * Utility method - for graph visualization
	 */
	public static void visualizeGraph(
			DelegateTree<Integer, UndirectedEdge> graph, String text) {

		final DecimalFormat df = new DecimalFormat("0.000");
		Layout<Integer, UndirectedEdge> layout = new TreeLayout<Integer, UndirectedEdge>(
				graph, 100, 120);

		BasicVisualizationServer<Integer, UndirectedEdge> vv = new BasicVisualizationServer<Integer, UndirectedEdge>(
				layout);
		Transformer<Integer, Paint> vertexPaint = new Transformer<Integer, Paint>() {
			public Paint transform(Integer i) {
				if (BNSample.attributeNameMap.get(i).equals(BNSample.LABEL)) {
					return Color.GREEN;
				} else {
					return Color.YELLOW;
				}
			}
		};

		Transformer<Integer, Shape> vertexShape = new Transformer<Integer, Shape>() {
			public Shape transform(Integer i) {
				Ellipse2D e = new Ellipse2D.Double(-15, -15, 40, 30);
				return AffineTransform.getScaleInstance(2, 2)
						.createTransformedShape(e);
			}
		};

		vv.setPreferredSize(new Dimension(1400, 800));
		vv.getRenderContext().setVertexShapeTransformer(vertexShape);
		vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
		vv.getRenderContext().setVertexLabelTransformer(
				new Transformer<Integer, String>() {
					@Override
					public String transform(Integer index) {
						return index + " : "
								+ BNSample.attributeNameMap.get(index);
					}
				});
		vv.getRenderContext().setEdgeLabelTransformer(
				new Transformer<UndirectedEdge, String>() {
					@Override
					public String transform(UndirectedEdge edge) {
						return String.valueOf(df.format(edge.getWeight()));
					}
				});
		vv.getRenderer().getVertexLabelRenderer().setPosition(Position.CNTR);

		JFrame frame = new JFrame(text);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(vv);
		frame.pack();
		frame.setVisible(true);
	}

}

package aprendizaje.automatico;

import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class App {
    public static void main(String[] args) {
        try {
            // Load the dataset
            DataSource source = new DataSource("../../ev/data/weather.nominal.arff");
            Instances data = source.getDataSet();

            // Set the class index
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Create a new Id3 classifier
            Id3 id3 = new Id3();

            // Build the classifier
            id3.buildClassifier(data);

            // Print the tree
            System.out.println(id3);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
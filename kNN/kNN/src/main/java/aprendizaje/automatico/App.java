package aprendizaje.automatico;

import weka.classifiers.lazy.IBk;
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

            // Create a new IBk classifier
            IBk knn = new IBk();
            knn.setKNN(3); // Set k to 3

            // Build the classifier
            knn.buildClassifier(data);

            // Print the classifier
            System.out.println(knn);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
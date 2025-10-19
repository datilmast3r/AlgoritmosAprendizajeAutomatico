package aprendizaje.automatico;

import weka.classifiers.rules.JRip;
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

            // Create a new JRip classifier
            JRip jrip = new JRip();

            // Build the classifier
            jrip.buildClassifier(data);

            // Print the rules
            System.out.println(jrip);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
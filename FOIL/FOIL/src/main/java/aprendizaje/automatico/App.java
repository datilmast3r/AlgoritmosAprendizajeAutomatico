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

            // Create a new Foil classifier
            Foil foil = new Foil();

            // Build the classifier
            foil.buildClassifier(data);

            // Print the rules
            System.out.println(foil);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
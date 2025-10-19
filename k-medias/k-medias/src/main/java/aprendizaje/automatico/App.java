package aprendizaje.automatico;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import we.core.converters.ConverterUtils.DataSource;

public class App {
    public static void main(String[] args) {
        try {
            // Load the dataset
            DataSource source = new DataSource("../../ev/data/weather.nominal.arff");
            Instances data = source.getDataSet();

            // Create a new k-means clusterer
            SimpleKMeans kMeans = new SimpleKMeans();

            // Set the number of clusters
            kMeans.setNumClusters(2);

            // Build the clusterer
            kMeans.buildClusterer(data);

            // Print the cluster assignments
            System.out.println(kMeans);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
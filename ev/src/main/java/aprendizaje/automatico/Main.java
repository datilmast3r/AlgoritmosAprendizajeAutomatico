package aprendizaje.automatico;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.core.Instance;

import java.io.File;

public class Main {

    public static void main(String[] args) {
        System.out.println("--- Ejecutando Algoritmo Candidate-Elimination en Weka ---");

        try {
            // 1. Cargar el conjunto de datos (Reemplaza la ruta si es necesario)
            // Se asume que 'weather.nominal.arff' (o 'weather.arff') está disponible. 
            // Candidate-Elimination funciona mejor con atributos nominales.
            
            // **IMPORTANTE:** Necesitas el archivo ARFF. Si no lo tienes, usa un PATH absoluto 
            // o descárgalo de los ejemplos de Weka.
            String dataPath = "data/weather.nominal2.arff"; // Ejemplo de ruta local
            
            // Usar la clase DataSource de Weka para cargar el archivo
            DataSource source = new DataSource(dataPath);
            Instances data = source.getDataSet();

            // 2. Establecer el atributo de clase
            // Por ejemplo, el último atributo (índice data.numAttributes() - 1)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // 3. Inicializar y Entrenar el Clasificador
            // Si CandidateElimination está en el paquete 'aprendizaje.automatico', úsalo directamente:
            CandidateElimination ceClassifier = new CandidateElimination();
            
            System.out.println("\nEntrenando clasificador...");
            
            // El método buildClassifier() realiza el entrenamiento
            ceClassifier.buildClassifier(data);

            System.out.println("Entrenamiento completado.");
            System.out.println("-----------------------------------");
            
            // 4. Mostrar el espacio de versiones aprendido
            System.out.println("Espacio de Versiones Final (S y G):");
            System.out.println(ceClassifier.toString());
            System.out.println("-----------------------------------");

            // 5. Probar el Clasificador en una Instancia
            // Tomamos el primer ejemplo como ejemplo de prueba
            Instance testInstance = data.instance(0); 
            
            // El valor real de la clase del primer ejemplo
            double actualClassValue = testInstance.classValue();
            String actualClassLabel = data.classAttribute().value((int) actualClassValue);

            // Clasificar la instancia de prueba
            double predictedClassValue = ceClassifier.classifyInstance(testInstance);
            String predictedClassLabel = data.classAttribute().value((int) predictedClassValue);
            
            System.out.println("Predicción para el primer ejemplo:");
            System.out.println("  Instancia: " + testInstance.toString());
            System.out.println("  Clase Real:      " + actualClassLabel);
            System.out.println("  Clase Predicha:  " + predictedClassLabel);

            // También puedes usar la clase Evaluation de Weka para hacer un test más formal 
            // (p. ej., validación cruzada)

        } catch (Exception e) {
            System.err.println("Ocurrió un error durante la ejecución del clasificador:");
            e.printStackTrace();
        }
    }
}
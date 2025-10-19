package aprendizaje.automatico;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Utils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Implementación Conceptual del Algoritmo de Candidatos-Eliminación (Candidate-Elimination) 
 * para WEKA.
 */
public class CandidateElimination extends AbstractClassifier {

    // Usaremos "\emptyset" para representar la hipótesis más específica (no cubre nada)
    private static final String MOST_SPECIFIC_PLACEHOLDER = "∅"; 

    private List<String[]> S_boundary; // Conjunto de hipótesis más Específicas
    private List<String[]> G_boundary; // Conjunto de hipótesis más Generales
    private int numAttributes;
    private Instances m_data; // Guardar referencia a los datos para acceder a los atributos

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // 1. Inicialización
        data = new Instances(data); 
        data.deleteWithMissingClass(); 

        if (data.classIndex() < 0) {
            throw new Exception("La clase de la instancia no está configurada.");
        }
        m_data = data; 
        // Excluir el atributo de clase, asumiendo que el índice de clase es el último.
        numAttributes = data.classIndex(); 

        // Inicializar S y G
        initializeBoundaries();

        // 2. Procesar Ejemplos de Entrenamiento
        for (int i = 0; i < data.numInstances(); i++) {
            System.out.println("\n=========================================================");
            System.out.println("--- Iteración " + (i + 1) + "/" + data.numInstances() + " ---");

            Instance instance = data.instance(i);
            String[] instanceArray = instanceToHypothesis(instance);
            boolean isPositive = isPositiveExample(instance);

            System.out.println("Instancia: " + instance);
            System.out.println("Clase: " + (isPositive ? "Positiva ('yes')" : "Negativa ('no')"));
            printBoundaries("Estado ANTES de la actualización:");

            if (S_boundary.isEmpty() || G_boundary.isEmpty()) {
                System.out.println("Acción: El Espacio de Versiones ha colapsado. S=" + S_boundary.size() + ", G=" + G_boundary.size());
                break; // Salir si el espacio de versiones colapsa
            }

            if (isPositive) {
                // Si es POSITIVO: Generalizar S, Especializar G
                System.out.println("Acción: Ejemplo POSITIVO.");
                // 2.1.1 Eliminar de G las inconsistentes (las que NO cubren d+)
                pruneGeneralBoundary(instanceArray); 
                // 2.1.2 Actualizar S (generalizar las inconsistentes y verificar contra G)
                updateSpecificBoundary(instanceArray);
            } else {
                // Si es NEGATIVO: Especializar G, Podar S
                System.out.println("Acción: Ejemplo NEGATIVO.");
                // 2.2.1 Eliminar de S las inconsistentes (las que SÍ cubren d-)
                pruneSpecificBoundary(instanceArray);
                // 2.2.2 Actualizar G (especializar las inconsistentes y verificar contra S)
                updateGeneralBoundary(instanceArray);
            }
            printBoundaries("Estado DESPUÉS de la actualización:");
        }
    }

    /**
     * Inicializa S y G.
     * S0: {\emptyset} (hipótesis más específica).
     * G0: {?, ?, ?, ?} (hipótesis más general).
     */
    private void initializeBoundaries() {
        S_boundary = new ArrayList<>();
        G_boundary = new ArrayList<>();

        // Inicializar G con la hipótesis más general: [?,?,?,...]
        String[] mostGeneral = new String[numAttributes];
        Arrays.fill(mostGeneral, "?");
        G_boundary.add(mostGeneral);

        // Inicializar S con la hipótesis más específica: [\emptyset, \emptyset, \emptyset, \emptyset]
        String[] mostSpecific = new String[numAttributes];
        Arrays.fill(mostSpecific, MOST_SPECIFIC_PLACEHOLDER);
        S_boundary.add(mostSpecific);
    }

    // ... (instanceToHypothesis y isPositiveExample son correctos) ...

    /**
     * Convierte una instancia de Weka en una hipótesis (array de Strings)
     */
    private String[] instanceToHypothesis(Instance instance) {
        String[] hypothesis = new String[numAttributes];
        for (int i = 0; i < numAttributes; i++) {
            Attribute attr = instance.attribute(i);
            hypothesis[i] = instance.stringValue(i);
        }
        return hypothesis;
    }

    /**
     * Comprueba si la instancia es un ejemplo positivo.
     * Asumimos que la primera clase (índice 0, 'yes') es la positiva.
     */
    private boolean isPositiveExample(Instance instance) {
        return instance.classValue() == m_data.classAttribute().indexOfValue("yes"); 
    }

    /**
     * Implementa la lógica para generalizar S con un ejemplo positivo.
     */
    private void updateSpecificBoundary(String[] positiveExample) {
        List<String[]> hypothesesToRemove = new ArrayList<>();
        List<String[]> hypothesesToAdd = new ArrayList<>();

        for (String[] s : S_boundary) {
            // Caso especial: si S0 sigue en S, debe ser reemplazado por el primer positivo.
            if (s[0].equals(MOST_SPECIFIC_PLACEHOLDER)) {
                hypothesesToRemove.add(s);
                hypothesesToAdd.add(positiveExample.clone());
            } 
            // Si S es inconsistente (no cubre d+) y NO es el placeholder inicial
            else if (!covers(s, positiveExample)) { 
                hypothesesToRemove.add(s);
                
                // Generalización mínima: h
                String[] h = generalize(s, positiveExample);
                
                // h debe ser consistente con G (más específica que alguna g en G)
                if (isConsistentWithG(h)) {
                    // Evitar añadir duplicados
                    if (!listContains(hypothesesToAdd, h) && !listContains(S_boundary, h)) {
                         hypothesesToAdd.add(h);
                    }
                }
            }
        }
        S_boundary.removeAll(hypothesesToRemove);
        S_boundary.addAll(hypothesesToAdd);
        
        // Eliminar redundancias (hipótesis más generales que otras en S)
        S_boundary = removeMoreGeneralHypotheses(S_boundary);
    }

    /**
     * Elimina de G las hipótesis que son inconsistentes con un ejemplo positivo.
     * (es decir, que no cubren el ejemplo positivo).
     */
    private void pruneGeneralBoundary(String[] positiveExample) {
        Iterator<String[]> iter = G_boundary.iterator();
        while (iter.hasNext()) {
            if (!covers(iter.next(), positiveExample)) {
                iter.remove();
            }
        }
    }

    /**
     * Elimina de S las hipótesis que son inconsistentes con un ejemplo negativo.
     * (es decir, que cubren el ejemplo negativo).
     */
    private void pruneSpecificBoundary(String[] negativeExample) {
        Iterator<String[]> iter = S_boundary.iterator();
        while (iter.hasNext()) {
            String[] s = iter.next();
            // El placeholder inicial NUNCA cubre un ejemplo real.
            if (!s[0].equals(MOST_SPECIFIC_PLACEHOLDER) && covers(s, negativeExample)) {
                iter.remove();
            }
        }
    }
    
    /**
     * Implementa la lógica para especializar G con un ejemplo negativo.
     */
    private void updateGeneralBoundary(String[] negativeExample) {
        List<String[]> hypothesesToRemove = new ArrayList<>();
        List<String[]> hypothesesToAdd = new ArrayList<>();

        for (String[] g : G_boundary) {
            // Si G es inconsistente (cubre el ejemplo negativo)
            if (covers(g, negativeExample)) { 
                hypothesesToRemove.add(g);
                
                // Generar especializaciones mínimas
                List<String[]> specializations = generateMinimalSpecializations(g, negativeExample);
                
                for (String[] h : specializations) {
                    // h debe ser consistente con S (más general que algún miembro de S)
                    if (isConsistentWithS(h)) {
                         // Evitar añadir duplicados
                        if (!listContains(hypothesesToAdd, h) && !listContains(G_boundary, h)) {
                            hypothesesToAdd.add(h);
                        }
                    }
                }
            }
        }
        G_boundary.removeAll(hypothesesToRemove);
        G_boundary.addAll(hypothesesToAdd);
        
        // Eliminar redundancias (hipótesis más específicas que otras en G)
        G_boundary = removeSubsumed(G_boundary);
    }
    
    // --- Métodos Auxiliares Corregidos/Revisados ---
    
    /**
     * Comprueba si la hipótesis h1 (general) es más general o igual que h2 (específica).
     * También se usa para 'covers(hipótesis, ejemplo)'.
     */
    private boolean covers(String[] h1, String[] h2) {
        for (int i = 0; i < numAttributes; i++) {
            // h1 es la hipótesis. Si es el placeholder, no cubre nada real.
            if (h1[i].equals(MOST_SPECIFIC_PLACEHOLDER)) { 
                return false;
            }
            // Si h1 es un valor concreto y no coincide con h2 (ejemplo/hipótesis específica)
            if (!h1[i].equals("?") && !h1[i].equals(h2[i])) { 
                return false; 
            }
        }
        return true;
    }

    /**
     * Implementación CORREGIDA. Genera especializaciones mínimas para una hipótesis g 
     * que cubre un ejemplo negativo d-.
     * La especialización se realiza cambiando un '?' en g por un valor de dominio V 
     * tal que V != d-[i].
     */
   private List<String[]> generateMinimalSpecializations(String[] h_g, String[] negativeExample) {
        List<String[]> result = new ArrayList<>();
        
        // Para cada atributo i
        for (int i = 0; i < numAttributes; i++) {
            // Solo especializar si el atributo en G es general ('?') y el valor del ejemplo negativo es concreto
            if (h_g[i].equals("?")) {
                Attribute attr = m_data.attribute(i);
                
                // Iterar sobre todos los valores posibles del dominio
                for (int j = 0; j < attr.numValues(); j++) {
                    String domainValue = attr.value(j);
                    
                    // La especialización es minimal y NO cubre negativeExample si:
                    // 1. Reemplazamos '?' por un valor CONCRETO. (Ya hecho al iterar 'domainValue')
                    // 2. Ese valor CONCRETO es DIFERENTE del valor del ejemplo negativo en esa posición.
                    if (!domainValue.equals(negativeExample[i])) {
                        String[] h_new = h_g.clone();
                        h_new[i] = domainValue;
                        
                        // Agregar solo si no es un duplicado
                        if (!listContains(result, h_new)) {
                            result.add(h_new);
                        }
                    }
                }
            }
        }
        return result;
    }

    // ... (isConsistentWithS, isConsistentWithG, generalize, removeMoreGeneralHypotheses, removeSubsumed son correctos o adaptables) ...

    /**
     * Comprueba si una hipótesis h es más general o igual que *alguna* hipótesis en S.
     */
    private boolean isConsistentWithS(String[] h) {
        // SOLUCIÓN 1: Si S solo contiene el placeholder, cualquier especialización de G es válida por ahora.
        if (S_boundary.size() == 1 && S_boundary.get(0)[0].equals(MOST_SPECIFIC_PLACEHOLDER)) {
            return true;
        }

        for (String[] s : S_boundary) {
            // Ignorar el placeholder inicial de S si aún está por error (aunque el primer positivo lo elimina)
            if (s[0].equals(MOST_SPECIFIC_PLACEHOLDER)) continue;
            
            // h es consistente con S si cubre/es más general que s
            if (covers(h, s)) {
                return true;
            }
        }
        return false; 
    }

    /**
     * Comprueba si una hipótesis h es más específica o igual que *alguna* hipótesis en G.
     */
    private boolean isConsistentWithG(String[] h) {
        for (String[] g : G_boundary) {
            // h es consistente con G si g cubre/es más general que h
            if (covers(g, h)) {
                return true; 
            }
        }
        return false;
    }

    /**
     * Generaliza una hipótesis h_s para que cubra un ejemplo positivo.
     */
    private String[] generalize(String[] h_s, String[] positiveExample) {
        String[] h_new = h_s.clone();
        for (int j = 0; j < numAttributes; j++) {
            // Si el valor no es general ('?') y no coincide con el ejemplo, generalizarlo a '?'
            if (!h_s[j].equals("?") && !h_s[j].equals(positiveExample[j])) {
                h_new[j] = "?";
            }
        }
        return h_new;
    }

    // ... (removeMoreGeneralHypotheses, removeSubsumed y listContains son correctos) ...
    
    private List<String[]> removeMoreGeneralHypotheses(List<String[]> boundary) {
        // ... (lógica interna es correcta) ...
        List<String[]> result = new ArrayList<>();
        for (String[] h1 : boundary) {
            boolean isMoreGeneral = false;
            for (String[] h2 : boundary) {
                if (h1 != h2 && !Arrays.equals(h1, h2) && covers(h1, h2) && !covers(h2, h1)) {
                    isMoreGeneral = true;
                    break;
                }
            }
            if (!isMoreGeneral) {
                result.add(h1);
            }
        }
        return result;
    }
    
    private List<String[]> removeSubsumed(List<String[]> boundary) {
        // ... (lógica interna es correcta) ...
        List<String[]> result = new ArrayList<>();
        for (String[] h1 : boundary) {
            boolean subsumed = false;
            for (String[] h2 : boundary) {
                // Si h2 es estrictamente más general que h1
                if (h1 != h2 && !Arrays.equals(h1, h2) && covers(h2, h1) && !covers(h1, h2)) { 
                    subsumed = true;
                    break;
                }
            }
            if (!subsumed) {
                result.add(h1);
            }
        }
        return result;
    }
    
    private boolean listContains(List<String[]> list, String[] array) {
        for (String[] item : list) {
            if (Arrays.equals(item, array)) return true;
        }
        return false;
    }
    
    // ... (classifyInstance y toString son correctos) ...
    @Override
    public double classifyInstance(Instance instance) {
        String[] instanceArray = instanceToHypothesis(instance);

        if (S_boundary.size() == 1 && G_boundary.size() == 1 && Arrays.equals(S_boundary.get(0), G_boundary.get(0))) {
            String[] finalHypothesis = S_boundary.get(0);
            
            if (covers(finalHypothesis, instanceArray)) {
                return m_data.classAttribute().indexOfValue("yes");
            } else {
                return m_data.classAttribute().indexOfValue("no"); 
            }
        }
        return m_data.classAttribute().indexOfValue("no"); // No hay consenso, se predice la clase negativa
    }
    
    private void printBoundaries(String title) {
        System.out.println(title);
        System.out.print("  S: ");
        if (S_boundary.isEmpty()) {
            System.out.println("[Vacío]");
        } else {
            S_boundary.forEach(h -> System.out.print(Arrays.toString(h).replace(MOST_SPECIFIC_PLACEHOLDER, "$\\emptyset$") + " "));
            System.out.println();
        }
        System.out.print("  G: ");
        if (G_boundary.isEmpty()) {
            System.out.println("[Vacío]");
        } else {
            G_boundary.forEach(h -> System.out.print(Arrays.toString(h) + " "));
            System.out.println();
        }
    }
    
    @Override
    public String toString() {
        // ... (la implementación de toString es correcta) ...
        if (S_boundary == null || S_boundary.isEmpty() || G_boundary.isEmpty()) {
            return "Candidate-Elimination: El espacio de versiones es vacío (inconsistente).";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("Algoritmo de Candidatos-Eliminación\n");
        sb.append("-----------------------------------\n");
        sb.append("Hipótesis S (más específicas):\n");
        S_boundary.forEach(h -> sb.append("\t").append(Arrays.toString(h).replace(MOST_SPECIFIC_PLACEHOLDER, "$\\emptyset$")).append("\n"));
        sb.append("Hipótesis G (más generales):\n");
        G_boundary.forEach(h -> sb.append("\t").append(Arrays.toString(h)).append("\n"));
        
        return sb.toString();
    }
}
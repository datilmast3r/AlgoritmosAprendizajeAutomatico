package aprendizaje.automatico;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import weka.core.Attribute;
import weka.core.Instance;

public class Foil extends AbstractClassifier {

    private List<Rule> rules;
    private int m_DefaultClass;

    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] dist = new double[instance.numClasses()];
        for (Rule rule : rules) {
            if (rule.covers(instance)) {
                dist[rule.classValue] = 1.0;
                return dist;
            }
        }
        dist[m_DefaultClass] = 1.0;
        return dist;
    }

    private class Rule {
        private List<Literal> literals;
        private int classValue;

        public Rule(int classValue) {
            this.literals = new ArrayList<>();
            this.classValue = classValue;
        }

        public void addLiteral(Literal literal) {
            literals.add(literal);
        }

        public boolean covers(Instance instance) {
            for (Literal literal : literals) {
                if (!literal.covers(instance)) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < literals.size(); i++) {
                sb.append(literals.get(i).toString());
                if (i < literals.size() - 1) {
                    sb.append(" AND ");
                }
            }
            sb.append(" => class=" + classValue);
            return sb.toString();
        }
    }

    private class Literal {
        private int attributeIndex;
        private int attributeValue;

        public Literal(int attributeIndex, int attributeValue) {
            this.attributeIndex = attributeIndex;
            this.attributeValue = attributeValue;
        }

        public boolean covers(Instance instance) {
            return (int) instance.value(attributeIndex) == attributeValue;
        }

        @Override
        public String toString() {
            return "att" + attributeIndex + " = " + attributeValue;
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        rules = new ArrayList<>();
        Instances trainingData = new Instances(instances);
        int classIndex = trainingData.classIndex();
        Attribute classAttribute = trainingData.attribute(classIndex);

        for (int i = 0; i < classAttribute.numValues(); i++) {
            Instances positiveInstances = getInstancesByClass(trainingData, i);
            Instances negativeInstances = getInstancesByClassNot(trainingData, i);

            while (!positiveInstances.isEmpty()) {
                Rule rule = buildRule(positiveInstances, negativeInstances, i);
                rules.add(rule);
                trainingData = removeCoveredInstances(trainingData, rule);
                positiveInstances = removeCoveredInstances(positiveInstances, rule);
            }
        }

        double[] classCounts = new double[trainingData.numClasses()];
        for (Instance instance : trainingData) {
            classCounts[(int) instance.classValue()]++;
        }
        int maxIndex = 0;
        for (int i = 1; i < classCounts.length; i++) {
            if (classCounts[i] > classCounts[maxIndex]) {
                maxIndex = i;
            }
        }
        m_DefaultClass = maxIndex;
    }

    private Rule buildRule(Instances positiveInstances, Instances negativeInstances, int classValue) {
        Rule rule = new Rule(classValue);
        Instances coveredPositive = new Instances(positiveInstances);
        Instances coveredNegative = new Instances(negativeInstances);

        while (!coveredNegative.isEmpty()) {
            Literal bestLiteral = findBestLiteral(coveredPositive, coveredNegative);
            if (bestLiteral == null) {
                break;
            }
            rule.addLiteral(bestLiteral);
            coveredPositive = getCoveredInstances(coveredPositive, bestLiteral);
            coveredNegative = getCoveredInstances(coveredNegative, bestLiteral);
        }
        return rule;
    }

    private Literal findBestLiteral(Instances positiveInstances, Instances negativeInstances) {
        Literal bestLiteral = null;
        double maxGain = -1;

        for (int i = 0; i < positiveInstances.numAttributes(); i++) {
            if (i == positiveInstances.classIndex()) {
                continue;
            }
            Attribute attribute = positiveInstances.attribute(i);
            for (int j = 0; j < attribute.numValues(); j++) {
                Literal literal = new Literal(i, j);
                double gain = calculateGain(positiveInstances, negativeInstances, literal);
                if (gain > maxGain) {
                    maxGain = gain;
                    bestLiteral = literal;
                }
            }
        }
        return bestLiteral;
    }

    private double calculateGain(Instances positiveInstances, Instances negativeInstances, Literal literal) {
        int p = positiveInstances.size();
        int n = negativeInstances.size();

        Instances coveredPositive = getCoveredInstances(positiveInstances, literal);
        Instances coveredNegative = getCoveredInstances(negativeInstances, literal);

        int p_prime = coveredPositive.size();
        int n_prime = coveredNegative.size();

        if (p_prime == 0) {
            return -1;
        }

        double initialInfo = Math.log(p + 1) / Math.log(2) - Math.log(p + n + 1) / Math.log(2);
        double newInfo = Math.log(p_prime + 1) / Math.log(2) - Math.log(p_prime + n_prime + 1) / Math.log(2);

        return p_prime * (newInfo - initialInfo);
    }

    private Instances getInstancesByClass(Instances instances, int classValue) {
        Instances result = new Instances(instances, 0);
        for (Instance instance : instances) {
            if ((int) instance.classValue() == classValue) {
                result.add(instance);
            }
        }
        return result;
    }

    private Instances getInstancesByClassNot(Instances instances, int classValue) {
        Instances result = new Instances(instances, 0);
        for (Instance instance : instances) {
            if ((int) instance.classValue() != classValue) {
                result.add(instance);
            }
        }
        return result;
    }

    private Instances getCoveredInstances(Instances instances, Literal literal) {
        Instances result = new Instances(instances, 0);
        for (Instance instance : instances) {
            if (literal.covers(instance)) {
                result.add(instance);
            }
        }
        return result;
    }

    private Instances removeCoveredInstances(Instances instances, Rule rule) {
        Instances result = new Instances(instances, 0);
        for (Instance instance : instances) {
            if (!rule.covers(instance)) {
                result.add(instance);
            }
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("FOIL rules:\n");
        for (Rule rule : rules) {
            sb.append(rule.toString()).append("\n");
        }
        return sb.toString();
    }
}

/*
 * The MIT License
 *
 * Copyright 2015 tegarnization.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package ml.ann;

import java.io.Serializable;
import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 *
 * @author tegarnization
 */
public class BackPropagation implements Classifier, OptionHandler, WeightedInstancesHandler, Randomizable, Serializable {

    private static final long serialVersionUID = -5990607817048210779L;
    private static final int DEFAULT_INIT_WEIGHT = 0;

    private NeuronTopology neuronTopology;

    private ZeroR zeroR;
    private boolean useDefaultModel;
    private NominalToBinary nominalToBinaryFilter;
    private double[] attributeRanges;
    private double[] attributeBases;
    private final boolean normalizeAttributes;
    private double validationSize;

    public void instanceToInput(int index) {
        System.arraycopy(neuronTopology.dummyInstances[index], 0, neuronTopology.input, 0, index);
    }

    public void initNumNeuronEachLayer() {
        int firstLayerIdx = 0;
        int lastLayerIdx = neuronTopology.numLayers - 1;
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            if (i == lastLayerIdx) {
                neuronTopology.numNeuronEachLayer[i] = neuronTopology.numClasses;
            } else if (i == firstLayerIdx) {
                neuronTopology.numNeuronEachLayer[i] = neuronTopology.numAttribute;
            } else {
                neuronTopology.numNeuronEachLayer[i] = (neuronTopology.numAttribute + neuronTopology.numClasses) / 2;
            }
        }
    }

    public BackPropagation(int epoch, double terminateMSE, boolean hasBias, boolean _normalizeAttributes, double learningRate) {
        neuronTopology = new NeuronTopology();
        neuronTopology.epoch = epoch;
        neuronTopology.terminateMSE = terminateMSE;
        neuronTopology.hasBias = hasBias;
        normalizeAttributes = _normalizeAttributes;
        neuronTopology.learningRate = learningRate;
    }

    public void printOutput() {
        System.out.println("----------Output-----------");
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                System.out.print(neuronTopology.output[i][j] + " ");
            }
            System.out.println("");
        }
    }

    public void printTarget() {
        System.out.println("Target = " + neuronTopology.target);
    }

    public void printInstances() {
        for (int i = 0; i < neuronTopology.dummyInstances.length; i++) {
            for (int j = 0; j < neuronTopology.dummyInstances[i].length; j++) {
                System.out.println("Instances[" + i + "][" + j + "] =" + neuronTopology.dummyInstances[i][j]);
            }
        }
    }

    /**
     * Assuming each neuron in layers are (numAttributes+numClasses) / 2
     */
    public void setupNetwork() {

        int idxLastLayer = neuronTopology.numLayers - 1;
        int idxFirstLayer = 0;
        neuronTopology.numAttribute = neuronTopology.instances.numAttributes();
        neuronTopology.numClasses = neuronTopology.instances.numClasses();
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            if (i == idxFirstLayer) {
                neuronTopology.input = new double[i][neuronTopology.numAttribute];
                neuronTopology.output = new double[i][neuronTopology.numAttribute];
                neuronTopology.numNeuronEachLayer[i] = neuronTopology.numAttribute;
            } else if (i == idxLastLayer) {
                neuronTopology.input = new double[i][neuronTopology.numClasses];
                neuronTopology.numNeuronEachLayer[i] = neuronTopology.numClasses;
                neuronTopology.output = new double[i][neuronTopology.numClasses];
                neuronTopology.weight = new double[i][neuronTopology.numNeuronEachLayer[i - 1]][neuronTopology.numNeuronEachLayer[i]];
            } else { //hidden layers
                int numNeuronHiddenLayer = (neuronTopology.numAttribute + neuronTopology.numClasses) / 2;
                neuronTopology.numNeuronEachLayer[i] = numNeuronHiddenLayer;
                neuronTopology.output = new double[i][numNeuronHiddenLayer];
                neuronTopology.weight = new double[i][neuronTopology.numNeuronEachLayer[i - 1]][neuronTopology.numNeuronEachLayer[i]];
            }
        }
    }

    public void initNetwork() {
        int defaultWeightValue = DEFAULT_INIT_WEIGHT;
        int firstLayerIdx = 0;
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                if (i == firstLayerIdx) {
                    neuronTopology.input[i][j] = neuronTopology.instances.instance(1).value(j);
                } else {
                    neuronTopology.input[i][j] = Double.NaN;
                }
                neuronTopology.output[i][j] = Double.NaN;
            }

            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i - 1]; j++) {
                for (int k = 0; k < neuronTopology.numNeuronEachLayer[i]; i++) {
                    neuronTopology.weight[i][j][k] = defaultWeightValue;
                }
            }
        }
    }

    public double sigmoidFunction(int idxLayer, int idxCurrentNeuron) {
        int currentLayer = idxLayer;
        int previousLayer = idxLayer - 1;
        int firstLayer = 0;
        double result = 0;
        if (currentLayer == firstLayer) {
            return neuronTopology.input[currentLayer][idxCurrentNeuron] * 1; //assuming weight in first layer is 1
        } else {
            int numNeuronPreviousLayer = neuronTopology.output[previousLayer].length;
            for (int i = 0; i < numNeuronPreviousLayer; i++) {
                result += neuronTopology.output[previousLayer][i]
                        * neuronTopology.weight[previousLayer][i][idxCurrentNeuron];
            }
        }
        return 1 / (1 + Math.exp(-1 * result));
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        neuronTopology.instances = new Instances(data);
        neuronTopology.instances.deleteWithMissingClass();

        zeroR = new weka.classifiers.rules.ZeroR();
        zeroR.buildClassifier(neuronTopology.instances);
        // only class? -> use ZeroR model
        if (neuronTopology.instances.numAttributes() == 1) {
            System.err.println("Cannot build model (only class attribute present in data!), "
                    + "using ZeroR model instead!");
            useDefaultModel = true;
            return;
        } else {
            useDefaultModel = false;
        }

        nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(neuronTopology.instances);
        neuronTopology.instances = Filter.useFilter(neuronTopology.instances, nominalToBinaryFilter);

        System.out.println("Instances = " + neuronTopology.instances.toSummaryString());

        int numAttributesWithoutClassAttribute = neuronTopology.instances.numAttributes() - 1;
        neuronTopology.numAttribute = numAttributesWithoutClassAttribute;
        neuronTopology.numClasses = neuronTopology.instances.numClasses();

        normalization();

        Instances validationSet = null;
        int numValidationSet = (int) (validationSize / 100.0 * neuronTopology.instances.numInstances());
        if (validationSize > 0) {
            if (numValidationSet == 0) {
                numValidationSet = 1;
            }
            validationSet = new Instances(neuronTopology.instances, 0, numValidationSet);
        }

        setupNetwork();

        initNetwork();

        for (int epoch = 0; epoch < neuronTopology.epoch; epoch++) {
            for (int inst = 0; inst < neuronTopology.instances.numInstances(); inst++) {

                initInputAndTarget(inst);

                for (int layers = 0; layers < neuronTopology.numLayers; layers++) {
                    for (int neu = 0; neu < neuronTopology.numNeuronEachLayer[layers]; neu++) {
                        neuronTopology.output[layers][neu] = sigmoidFunction(layers, neu);
                    }
                }
                for (int layer = neuronTopology.numLayers - 1; layer >= 0; layer--) {
                    for (int neu = 0; neu < neuronTopology.numNeuronEachLayer[layer]; neu++) {
                        neuronTopology.error[layer][neu] = computeError(layer, neu);
                        for (int preneu = 0; preneu < neuronTopology.numNeuronEachLayer[layer - 1]; preneu++) {
                            neuronTopology.weight[layer - 1][preneu][neu] = updateWeight(layer-1, preneu, neu);
                        }
                    }
                }
            }
        }
        printOutput();
    }

    private double computeError(int idxLayer, int idxNeuron) {
        double currentOutput = neuronTopology.output[idxLayer][idxNeuron];
        return currentOutput * (1 - currentOutput) * neuronTopology.target - currentOutput;
    }

    private double updateWeight(int idxLayer, int idxCurrentNeuron, int idxNextNeuron) {
        return neuronTopology.weight[idxLayer][idxCurrentNeuron][idxNextNeuron] + (neuronTopology.learningRate*neuronTopology.error[idxLayer][idxCurrentNeuron]*neuronTopology.output[idxLayer][idxCurrentNeuron]);
    }

    private void initInputAndTarget(int instancesIdx) {
        int classAttributeIdx = neuronTopology.instances.classIndex();
        neuronTopology.target = neuronTopology.instances.instance(instancesIdx).value(classAttributeIdx);
        for (int i = 0; i < neuronTopology.instances.numAttributes(); i++) {
            neuronTopology.input[0][i] = neuronTopology.instances.instance(instancesIdx).value(i);
        }
    }

    private void normalization() {
        if (neuronTopology.instances != null) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            double value;
            attributeRanges = new double[neuronTopology.instances.numAttributes()];
            attributeBases = new double[neuronTopology.instances.numAttributes()];
            for (int noa = 0; noa < neuronTopology.instances.numAttributes(); noa++) {
                min = Double.POSITIVE_INFINITY;
                max = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < neuronTopology.instances.numInstances(); i++) {
                    if (!neuronTopology.instances.instance(i).isMissing(noa)) {
                        value = neuronTopology.instances.instance(i).value(noa);
                        if (value < min) {
                            min = value;
                        }
                        if (value > max) {
                            max = value;
                        }
                    }
                }

                attributeRanges[noa] = (max - min) / 2;
                attributeBases[noa] = (max + min) / 2;
                if (noa != neuronTopology.instances.classIndex() && normalizeAttributes) {
                    for (int i = 0; i < neuronTopology.instances.numInstances(); i++) {
                        if (attributeRanges[noa] != 0) {
                            neuronTopology.instances.instance(i).setValue(
                                    noa,
                                    (neuronTopology.instances.instance(i).value(noa) - attributeBases[noa])
                                    / attributeRanges[noa]);
                        } else {
                            neuronTopology.instances.instance(i).setValue(noa,
                                    neuronTopology.instances.instance(i).value(noa) - attributeBases[noa]);
                        }
                    }
                }
            }

        }
    }

    public void resetNetwork() {
        int lastLayerIdx = neuronTopology.numLayers - 1;
        for (int i = 0; i < neuronTopology.numClasses; i++) {
//            neuronTopology.output[lastLayerIdx][i] = 0;
        }
    }

    @Override
    public void setSeed(int seed) {
        throw new UnsupportedOperationException("setSeed Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int getSeed() {
        throw new UnsupportedOperationException("getSeed Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Enumeration<Option> listOptions() {
        throw new UnsupportedOperationException("listOptions Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        throw new UnsupportedOperationException("setOptions Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String[] getOptions() {
        throw new UnsupportedOperationException("getOptions Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("classifyInstance Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        Instance currentInstance;
        boolean hasNominalAttribute = false;
        // default model?
        if (useDefaultModel) {
            return zeroR.distributionForInstance(instance);
        }

        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.classIndex() != i) {
                if (instance.attribute(i).isNominal()) {
                    hasNominalAttribute = true;
                    break;
                }
            }
        }
        if (hasNominalAttribute) {
            nominalToBinaryFilter.input(instance);
            currentInstance = nominalToBinaryFilter.output();
        } else {
            currentInstance = instance;
        }

        // Make a copy of the instance so that it isn't modified
        currentInstance = (Instance) currentInstance.copy();

        for (int noa = 0; noa < neuronTopology.instances.numAttributes(); noa++) {
            if (noa != neuronTopology.instances.classIndex()) {
                if (attributeRanges[noa] != 0) {
                    currentInstance.setValue(noa,
                            (currentInstance.value(noa) - attributeBases[noa])
                            / attributeRanges[noa]);
                } else {
                    currentInstance.setValue(noa, currentInstance.value(noa)
                            - attributeBases[noa]);
                }
            }
        }
        resetNetwork();

        // since all the output values are needed.
        // They are calculated manually here and the values collected.
        double[] theArray = new double[neuronTopology.numClasses];
        for (int noa = 0; noa < neuronTopology.numClasses; noa++) {
//            theArray[noa] = neuronTopology.outputs[noa].outputValue(true);
        }
        if (neuronTopology.instances.classAttribute().isNumeric()) {
            return theArray;
        }

        // now normalize the array
        double count = 0;
        for (int noa = 0; noa < neuronTopology.numClasses; noa++) {
            count += theArray[noa];
        }
        if (count <= 0) {
//            return zeroR.distributionForInstance(i);
        }
        for (int noa = 0; noa < neuronTopology.numClasses; noa++) {
            theArray[noa] /= count;
        }
        return theArray;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("getCapabilities Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}

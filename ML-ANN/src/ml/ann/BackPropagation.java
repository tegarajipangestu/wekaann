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
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 *
 * @author tegarnization
 */
public class BackPropagation implements Classifier, OptionHandler, WeightedInstancesHandler, Randomizable, Serializable {

    private static final long serialVersionUID = -5990607817048210779L;

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

    public BackPropagation(int epoch, double terminateMSE, boolean hasBias, boolean _normalizeAttributes, double learningRate, int numHiddenLayer, double momentum) {
        neuronTopology = new NeuronTopology();
        neuronTopology.epoch = epoch;
        neuronTopology.terminateMSE = terminateMSE;
        neuronTopology.hasBias = hasBias;
        normalizeAttributes = _normalizeAttributes;
        neuronTopology.learningRate = learningRate;
        neuronTopology.numLayers = 2 + numHiddenLayer;
    }

    public void printInstances() {
        for (int i = 0; i < neuronTopology.dummyInstances.length; i++) {
            for (int j = 0; j < neuronTopology.dummyInstances[i].length; j++) {
                System.out.println("Instances[" + i + "][" + j + "] =" + neuronTopology.dummyInstances[i][j]);
            }
        }
    }

    public void printInput() {
        System.out.println("==============================Input==============================");
        for (int i = 0; i < neuronTopology.instances.numAttributes(); i++) {
            System.out.println("Input[" + i + "] = " + neuronTopology.input[i]);
        }
        System.out.println("==============================Input==============================");
    }

    public void printOutput() {
        System.out.println("==============================Output==============================");
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                System.out.print(neuronTopology.output[i][j] + " ");
            }
            System.out.println("");
        }
    }

    public void printWeight() {
        System.out.println("==============================Weight==============================");
        for (int i = 0; i < neuronTopology.numLayers - 1; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                for (int k = 0; k < neuronTopology.numNeuronEachLayer[i + 1]; k++) {
                    System.out.print(neuronTopology.weight[i][j][k] + " ");
                }
            }
            System.out.println("");
        }
        System.out.println("==============================Weight==============================");
    }

    public void printError() {
        System.out.println("==============================Error==============================");
        for (int i = 1; i < neuronTopology.numLayers; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                System.out.print(neuronTopology.error[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("==============================Error==============================");
    }

    public void printTarget() {
        System.out.println("==============================Target==============================");
        if (neuronTopology.instances.classAttribute().isNumeric()) {
            System.out.println(neuronTopology.target[0]);
        } else if (neuronTopology.instances.classAttribute().isNominal()) {
            for (int i = 0; i < neuronTopology.instances.numClasses(); i++) {
                System.out.println(neuronTopology.target[i] + " ");
            }
        }
        System.out.println("==============================Target==============================");
    }

    public void printStatistic() {
        printInput();
        printOutput();
        printTarget();
        printError();
        printWeight();
    }

    public void setupNetwork() {

        int idxLastLayer = neuronTopology.numLayers - 1;
        int idxFirstLayer = 0;
        neuronTopology.numAttribute = neuronTopology.instances.numAttributes();
        neuronTopology.numClasses = neuronTopology.instances.numClasses();
        neuronTopology.numNeuronEachLayer = new int[neuronTopology.numLayers];
        neuronTopology.output = new double[neuronTopology.numLayers][];
        neuronTopology.error = new double[neuronTopology.numLayers][];
        neuronTopology.weight = new double[neuronTopology.numLayers][][];
        neuronTopology.previousDeltaWeight = new double[neuronTopology.numLayers][][];
        neuronTopology.deltaWeight = new double[neuronTopology.numLayers][][];
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            if (i == idxFirstLayer) {
                neuronTopology.numNeuronEachLayer[i] = neuronTopology.numAttribute + 1;
            } else if (i == idxLastLayer) {
                neuronTopology.numNeuronEachLayer[i] = neuronTopology.numClasses;
            } else {
                neuronTopology.numNeuronEachLayer[i] = (neuronTopology.numAttribute + neuronTopology.numClasses) / 2;
            }
        }

        for (int i = 0; i < neuronTopology.numLayers; i++) {
            if (i == idxFirstLayer) {
                neuronTopology.input = new double[neuronTopology.numNeuronEachLayer[i]];
                neuronTopology.output[i] = new double[neuronTopology.numNeuronEachLayer[i]];
                neuronTopology.weight[i] = new double[neuronTopology.numNeuronEachLayer[i]][];
                neuronTopology.previousDeltaWeight[i] = new double[neuronTopology.numNeuronEachLayer[i]][];
                neuronTopology.deltaWeight[i] = new double[neuronTopology.numNeuronEachLayer[i]][];
                neuronTopology.error[i] = new double[neuronTopology.numAttribute];
                for (int j = 0; j < neuronTopology.weight[i].length; j++) {
                    neuronTopology.weight[i][j] = new double[neuronTopology.numNeuronEachLayer[i + 1]];
                    neuronTopology.previousDeltaWeight[i][j] = new double[neuronTopology.numNeuronEachLayer[i + 1]];
                    neuronTopology.deltaWeight[i][j] = new double[neuronTopology.numNeuronEachLayer[i + 1]];
                }
            } else if (i == idxLastLayer) {
                neuronTopology.output[i] = new double[neuronTopology.numClasses];
                neuronTopology.error[i] = new double[neuronTopology.numClasses];
            } else { //hidden layers
                int numNeuronHiddenLayer = (neuronTopology.numAttribute + neuronTopology.numClasses) / 2;
                neuronTopology.output[i] = new double[numNeuronHiddenLayer];
                neuronTopology.error[i] = new double[numNeuronHiddenLayer];
                neuronTopology.weight[i] = new double[neuronTopology.numNeuronEachLayer[i]][neuronTopology.numNeuronEachLayer[i + 1]];
                neuronTopology.previousDeltaWeight[i] = new double[neuronTopology.numNeuronEachLayer[i]][neuronTopology.numNeuronEachLayer[i + 1]];
                neuronTopology.deltaWeight[i] = new double[neuronTopology.numNeuronEachLayer[i]][neuronTopology.numNeuronEachLayer[i + 1]];
            }
        }
    }

    public void initNetwork() {
        Random random = new Random();
//        double defaultWeightValue = 0;
        double defaultWeightValue = random.nextDouble() - 0.5;
        int firstLayerIdx = 0;
        int lastLayerIdx = neuronTopology.numLayers - 1;
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                if (i == firstLayerIdx) {

                } else {
                    neuronTopology.input[j] = Double.NaN;
                }
                neuronTopology.output[i][j] = Double.NaN;
            }

            if (i != lastLayerIdx) {
                for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                    for (int k = 0; k < neuronTopology.numNeuronEachLayer[i + 1]; k++) {
                        neuronTopology.weight[i][j][k] = 0;
                        neuronTopology.previousDeltaWeight[i][j][k] = 0;
                        neuronTopology.deltaWeight[i][j][k] = 0;
                    }
                }
            }
        }
        if (neuronTopology.instances.classAttribute().isNumeric()) {
            neuronTopology.target = new double[0];
        } else if (neuronTopology.instances.classAttribute().isNominal()) {
            neuronTopology.target = new double[neuronTopology.instances.numClasses()];
        }
    }

    public double sigmoidFunction(int idxLayer, int idxCurrentNeuron) {
        int currentLayer = idxLayer;
        int previousLayer = idxLayer - 1;
        int firstLayer = 0;
        double result = 0;
        if (currentLayer == firstLayer) {
            return neuronTopology.input[idxCurrentNeuron] * 1; //assuming weight in first layer is 1
        } else {
            int numNeuronPreviousLayer = neuronTopology.numNeuronEachLayer[currentLayer - 1];
            for (int i = 0; i < numNeuronPreviousLayer; i++) {
                result += neuronTopology.output[previousLayer][i]
                        * neuronTopology.weight[previousLayer][i][idxCurrentNeuron];
            }
        }
        return 1 / (1 + Math.exp(-1 * result));
    }

    private void removeMissingValue() {
        for (int j = 0; j < neuronTopology.instances.numAttributes(); j++) {
            neuronTopology.instances.deleteWithMissing(j);
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        double epochError = 0;
        neuronTopology.originInstances = data;
        neuronTopology.instances = new Instances(data);
        neuronTopology.instances.deleteWithMissingClass();

        removeMissingValue();

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

        neuronTopology.numAttribute = neuronTopology.instances.numAttributes();
        neuronTopology.numClasses = neuronTopology.instances.numClasses();

        nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(neuronTopology.instances);
        neuronTopology.instances = Filter.useFilter(neuronTopology.instances, nominalToBinaryFilter);
//        System.out.println("Instances = " + neuronTopology.instances.toSummaryString());

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
            epochError = 0;
            int idxLastLayer = neuronTopology.numLayers - 1;
            for (int inst = 0; inst < neuronTopology.instances.numInstances(); inst++) {

//                System.out.println("inst and epoch = " + inst + " , " + epoch);
                initInputAndTarget(inst);

//                System.out.println("Initial");
//                printStatistic();
                for (int layers = 0; layers < neuronTopology.numLayers; layers++) {
                    for (int neu = 0; neu < neuronTopology.numNeuronEachLayer[layers]; neu++) {
                        neuronTopology.output[layers][neu] = sigmoidFunction(layers, neu);
                    }
                }
                for (int layer = idxLastLayer; layer > 0; layer--) {
                    for (int neu = 0; neu < neuronTopology.numNeuronEachLayer[layer]; neu++) {
                        neuronTopology.error[layer][neu] = computeError(layer, neu);
                        for (int preneu = 0; preneu < neuronTopology.numNeuronEachLayer[layer - 1]; preneu++) {
                            neuronTopology.weight[layer - 1][preneu][neu] = updateWeight(layer - 1, preneu, neu);
                        }
                    }
                }
                for (int i = 0; i < neuronTopology.numNeuronEachLayer[idxLastLayer]; i++) {
//                    if (Utils.maxIndex(neuronTopology.output[idxLastLayer]) == Utils.maxIndex(neuronTopology.target)) {
//                        epochError = 0;
//                    } else {
//                        epochError += computeMSE(neuronTopology.target[i], neuronTopology.output[idxLastLayer][i]);
//                    }
                    epochError += computeMSE(neuronTopology.target[i], neuronTopology.output[idxLastLayer][i]);
                }

//                System.out.println("AfterLearning");
//                printStatistic();
            }
            if (epochError <= neuronTopology.terminateMSE) {
                System.out.println("Terminate karena mse cuk");
                break;
            }
        }
//        System.out.println("Epoch error = " + epochError);
    }

    private double computeMSE(double target, double output) {
        return Math.pow(target - output, 2);
    }

    private double computeError(int idxLayer, int idxNeuron) throws Exception {
        int lastLayerIdx = neuronTopology.numLayers - 1;
        int firstLayerIdx = 0;
        double currentOutput = neuronTopology.output[idxLayer][idxNeuron];
        double maudyAyunda = 0;
        if (idxLayer == lastLayerIdx) {
            return currentOutput * (1 - currentOutput) * (neuronTopology.target[idxNeuron] - currentOutput);
        } else if (idxLayer != firstLayerIdx) {
            for (int i = 0; i < neuronTopology.numNeuronEachLayer[idxLayer + 1]; i++) {
                maudyAyunda = maudyAyunda + (neuronTopology.error[idxLayer + 1][i] * neuronTopology.weight[idxLayer][idxNeuron][i]);
            }
            return currentOutput * (1 - currentOutput) * maudyAyunda;
        } else {
            throw new Exception("Jancuk");
        }
    }

    private double computeMomentumCuk(int idxLayer, int idxCurrentNeuron, int idxNextNeuron) {
        return neuronTopology.momentum * neuronTopology.previousDeltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron];
    }

    private double updateWeight(int idxLayer, int idxCurrentNeuron, int idxNextNeuron) throws Exception {
        int lastLayerIdx = neuronTopology.numLayers - 1;
        int idxNextLayer = idxLayer + 1;

        if (idxLayer + 1 == lastLayerIdx) {
            neuronTopology.previousDeltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron]
                    = neuronTopology.deltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron];
            neuronTopology.deltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron] = (neuronTopology.learningRate
                    * neuronTopology.error[idxLayer + 1][idxNextNeuron]
                    * neuronTopology.output[idxLayer][idxCurrentNeuron])
                    + computeMomentumCuk(idxLayer, idxCurrentNeuron, idxNextNeuron);
            return neuronTopology.weight[idxLayer - 1][idxCurrentNeuron][idxNextNeuron]
                    + neuronTopology.deltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron];
//        } 
//        else if (idxLayer == 0) {
//            throw new Exception("Jancuk gak jalan");
//            return neuronTopology.weight[idxLayer][idxCurrentNeuron][idxNextNeuron]
//                    + (neuronTopology.learningRate * neuronTopology.error[idxLayer + 1][idxNextNeuron]
//                    * neuronTopology.input[idxCurrentNeuron]);
        } else {
            neuronTopology.previousDeltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron]
                    = neuronTopology.deltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron];
            neuronTopology.deltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron] = (neuronTopology.learningRate
                    * neuronTopology.error[idxLayer + 1][idxNextNeuron]
                    * neuronTopology.output[idxLayer][idxCurrentNeuron])
                    + computeMomentumCuk(idxLayer, idxCurrentNeuron, idxNextNeuron);
//            System.out.println("neuronTopology.error[" + (idxLayer + 1) + "][" + idxNextNeuron + "] = " + neuronTopology.error[idxLayer + 1][idxNextNeuron]);
            return neuronTopology.weight[idxLayer][idxCurrentNeuron][idxNextNeuron]
                    + neuronTopology.deltaWeight[idxLayer][idxCurrentNeuron][idxNextNeuron];
        }
    }

    private void initInputAndTarget(int instancesIdx) {
        int classAttributeIdx = neuronTopology.instances.classIndex();
        if (neuronTopology.instances.classAttribute().isNumeric()) {
            neuronTopology.target = new double[1];
            neuronTopology.target[0] = neuronTopology.instances.instance(instancesIdx).value(classAttributeIdx);
        } else if (neuronTopology.instances.classAttribute().isNominal()) {
            neuronTopology.target = new double[neuronTopology.instances.numClasses()];
            for (int i = 0; i < neuronTopology.instances.numClasses(); i++) {
                neuronTopology.target[i] = 0;
            }
            int idxClassValue = (int) neuronTopology.instances.instance(instancesIdx).classValue();
            neuronTopology.target[idxClassValue] = 1;
        }
        for (int i = 0; i < neuronTopology.instances.numAttributes(); i++) {
            neuronTopology.input[i] = 0;
        }
        for (int i = 0; i < neuronTopology.instances.numAttributes(); i++) {
            int iter = 0;
            if (i == 0) {
                neuronTopology.input[i] = 1;
                neuronTopology.output[0][i] = 1;
                iter++;
            } else {
                neuronTopology.input[i] = neuronTopology.instances.instance(instancesIdx).value(i - 1);
                neuronTopology.output[0][i] = neuronTopology.instances.instance(instancesIdx).value(i - 1);
            }
        }
//        System.out.println(neuronTopology.originInstances.instance(instancesIdx).toString());
    }

    private void initInputAndTarget(Instance instance) {

        int offset = 1;

        try {
            int classAttributeIdx = neuronTopology.instances.classIndex();
            if (neuronTopology.instances.classAttribute().isNumeric()) {
                neuronTopology.target = new double[1];
                neuronTopology.target[0] = instance.value(classAttributeIdx);
            } else if (neuronTopology.instances.classAttribute().isNominal()) {
                neuronTopology.target = new double[instance.numClasses()];
                for (int i = 0; i < instance.numClasses(); i++) {
                    neuronTopology.target[i] = 0;
                }
                int idxClassValue = (int) instance.classValue();
                neuronTopology.target[idxClassValue] = 1;
            }
            neuronTopology.input[0] = 1;
            neuronTopology.output[0][0] = 1;
            for (int i = 0; i < instance.numAttributes() - 1; i++) {
                if (instance.attribute(i).isNominal()) {
                    if ("TRUE".equals(instance.stringValue(i))) {
                        neuronTopology.input[0 + offset] = 1;
                        neuronTopology.output[0][0 + offset] = 1;
                        offset++;
                    } else if ("FALSE".equals(instance.stringValue(i))) {
                        neuronTopology.input[offset] = 1;
                        neuronTopology.output[0][offset] = 0;
                        offset++;
                    } else {
                        neuronTopology.input[instance.attribute(i).index() + offset] = 1;
                        neuronTopology.output[0][instance.attribute(i).index() + offset] = 1;
                        offset += instance.attribute(i).numValues();
                    }
                } else if (instance.attribute(i).isNumeric()) {
                    neuronTopology.input[offset] = instance.value(i);
                    neuronTopology.output[0][offset] = instance.value(i);
                    offset++;
                }
            }
//        System.out.println(neuronTopology.originInstances.instance(instancesIdx).toString());
//            printInput();
        } catch (Exception e) {
            System.err.println(neuronTopology.input.length);
            System.err.println(instance.toString());
            System.err.println("offset = " + offset);
            e.printStackTrace();
        }
    }

    private void evaluation() {

    }

    private double computeMSE() {
        return 0.0;
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
        for (int i = 0; i < neuronTopology.numLayers; i++) {
            for (int j = 0; j < neuronTopology.numNeuronEachLayer[i]; j++) {
                neuronTopology.output[i][j] = Double.NaN;
            }
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

//        initInputAndTarget(instance);
        initInputAndTarget(instance);

//        System.exit(0);
        int lastLayerIdx = neuronTopology.instances.numClasses();
        for (int layers = 0; layers < neuronTopology.numLayers; layers++) {
            for (int neu = 0; neu < neuronTopology.numNeuronEachLayer[layers]; neu++) {
                neuronTopology.output[layers][neu] = sigmoidFunction(layers, neu);
            }
        }
        return Utils.maxIndex(neuronTopology.output[lastLayerIdx]);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        Instance currentInstance;
        // default model?
        if (useDefaultModel) {
            return zeroR.distributionForInstance(instance);
        }

        // Make a copy of the instance so that it isn't modified
        currentInstance = (Instance) instance.copy();
        initInputAndTarget(currentInstance);

        for (int noa = 0; noa < currentInstance.numAttributes(); noa++) {
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
//        resetNetwork();

        int lastLayerIdx = neuronTopology.numLayers - 1;
        for (int layers = 0; layers < neuronTopology.numLayers; layers++) {
            for (int neu = 0; neu < neuronTopology.numNeuronEachLayer[layers]; neu++) {
                neuronTopology.output[layers][neu] = sigmoidFunction(layers, neu);
            }
        }

        double[] jancuk = new double[neuronTopology.numNeuronEachLayer[lastLayerIdx]];

        for (int i = 0; i < neuronTopology.numNeuronEachLayer[lastLayerIdx]; i++) {
            jancuk[i] = neuronTopology.output[lastLayerIdx][i];
        }

        // now normalize the array
        double count = 0;
        for (int noa = 0; noa < neuronTopology.numClasses; noa++) {
            count += jancuk[noa];
        }
        if (count <= 0) {
//            return zeroR.distributionForInstance(i);
        }
        for (int noa = 0; noa < neuronTopology.numClasses; noa++) {
            jancuk[noa] /= count;
        }
//        System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Jancuk~~~~~~~~~~~~~~~~~~~~~~~~~");
//        for (int i = 0; i < jancuk.length; i++) {
//            System.out.print(jancuk[i] + " ");
//        }
//        System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Jancuk~~~~~~~~~~~~~~~~~~~~~~~~~");
        return jancuk;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("getCapabilities Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}

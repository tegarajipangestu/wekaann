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

import static java.lang.Math.exp;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 *
 * @author tegarnization
 */
public class NeuralNetwork {

	private double[][][] weight; // [layer sumber][neuron sumber][neuron tujuan]
	private double[] input;
	private double[] target;
	private double[][] instances; //[n-instance][n-attribute]
	private double[][] output; // [layer sumber][neuron sumber]
	private double learningRate;
	private double momentum;
	private int numLayers;
	private int numHiddenLayers;
	private int numNeuron;
	private int epoch;

	public NeuralNetwork() {
		this.seeding();
		/*Epoch iteration*/
		for (int i = 0; i < getEpoch(); i++) {
			instanceToInput(i);
			for (int j = 0; j < input.length; i++) {

			}
		}
	}
	public NeuralNetwork (int hiddenLayers, int inputs, int neuronPerLayers){
		if (inputs > neuronPerLayers)
			weight = new double[hiddenLayers + 1][inputs][neuronPerLayers];
		else 
			weight = new double[hiddenLayers + 1][neuronPerLayers][neuronPerLayers];
		
	}
	public NeuralNetwork(double[][][] weight, double[] input, double[][] output, double learningRate, double momentum) {
		this.weight = weight;
		this.input = input;
		this.output = output;
		this.learningRate = learningRate;
		this.momentum = momentum;
	}

	public int getEpoch() {
		return epoch;
	}

	public void setEpoch(int epoch) {
		this.epoch = epoch;
	}

	public void instanceToInput(int index) {
		for (int i = 0; i < index; i++) {
			input[i] = instances[index][i];
		}
	}

	public void initWeight(double newWeight) {

	}

	/* Untuk weight random
	 **/
	public void initWeight() {
		initWeight(Math.random());
	}

	public double[][][] getWeight() {
		return weight;
	}

	public double[] getTarget() {
		return target;
	}

	public void setTarget(double[] target) {
		this.target = target;
	}

	public double[][] getInstances() {
		return instances;
	}

	public void setInstances(double[][] instances) {
		this.instances = instances;
	}

	public int getNumLayers() {
		return numLayers;
	}

	public void setNumLayers(int numLayers) {
		this.numLayers = numLayers;
	}

	private void seeding() {
		instances = new double[3][3];
		target = new double[3];

		instances[0][0] = 1;
		instances[0][1] = 0;
		instances[0][2] = 1;
		instances[1][0] = 0;
		instances[1][1] = -1;
		instances[1][2] = -1;
		instances[2][0] = -1;
		instances[2][1] = -0.5;
		instances[2][2] = -1;

		target[0] = -1;
		target[1] = 1;
		target[2] = 1;

		setEpoch(instances.length);
	}

	public void printTarget() {
		for (int i = 0; i < target.length; i++) {
			System.out.println("Target[" + i + "] =" + target[i]);
		}
	}

	public void printInstances() {
		for (int i = 0; i < instances.length; i++) {
			for (int j = 0; j < instances[i].length; j++) {
				System.out.println("Instances[" + i + "][" + j + "] =" + instances[i][j]);
			}
		}
	}

	public void setWeight(double[][][] weight) {
		this.weight = weight;
	}

	public double[] getInput() {
		return input;
	}

	public void setInput(double[] input) {
		this.input = input;
	}

	public double[][] getOutput() {
		return output;
	}

	public void setOutput(double[][] output) {
		this.output = output;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public double errorCount(double target, double output) {
		return output * (1 - output) * (target - output);
	}

	public double updateWeight(double weight, double output) {
		return weight + (learningRate * errorCount(weight, output) * output);
	}

    //Kerjaan feli
	/**
	 * I.S: Weight dan input tidak boleh kosong
     *
	 */
	public void forwardChaining() {
		for (int layerIndex = 0; layerIndex < weight.length; layerIndex++) {
			for (int nodeIndex = 0; nodeIndex < weight[layerIndex].length; nodeIndex++) {
				//TODO
			}
		}
	}

	public void backPropagation() {
		//double d = output[output.length-1][];

		for (int i = numLayers - 1; i >= 0; i--) {
			for (int j = 0; j < output[i].length; j++) {
				weight[i][j - 1][j] = updateWeight(weight[i][j - 1][j], output[i][j]);
			}
		}
	}

	//Kerjaan feli
	public double activationFunction(String function, double input) {
		double res = 0;
		switch (function.toLowerCase()) {
			case "sigmoid":
				res = 1 / (1 + exp(input));
				break;
			case "linear":
				res = input;
				break;
			case "step":
				break;
			default:
				res = 1 / (1 + exp(input));
				break;
		}
		return res;
	}

	public void numericToBinary() {

	}

	public Instances nominalToBinary(Instances C) {
		NominalToBinary NtB = new NominalToBinary();
		try {
			NtB.setInputFormat(C);
			C = new Instances(Filter.useFilter(C, NtB));
			return C;
		} catch (Exception E) {
			E.printStackTrace();
		}
		return C;
	}

	public void mapping(Instances C) {
		C = nominalToBinary(C);
		Instance temp = C.get(1);
		weight = new double[this.numHiddenLayers + 2][this.numNeuron][];
		for (Instance I : C) {
			weight = new double[I.numValues()][this.numHiddenLayers + 2][this.numLayers]; //
			//TODO
			
		}
	}
}

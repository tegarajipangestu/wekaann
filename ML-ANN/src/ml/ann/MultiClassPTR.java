/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml.ann;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Scanner;
import static ml.ann.MainPTR.m_nominalToBinaryFilter;
import static ml.ann.MainPTR.m_normalize;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;

/**
 *
 * @author Ivana Clairine
 */
public class MultiClassPTR extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler, Randomizable, Serializable {

	public int numInstance; //jumlah instance
	public int numInput; //jumlah node input, belum termasuk bias
	public int numOutput; //jumlah node output
	public double[][] weight; //weight dari input ke neuron
	public double[][] inputInstances; //jumlah masukan
	public double[][] targetInstances; //dimensi pertama: nomor instance, dimensi kedua: nomor neuron output
	public int maxEpoch;
	public double learningRate;
	public double threshold;
	public double momentum;
	public int algo;
	public boolean randomWeight;

	public int actFunc = 0;
	transient Scanner funcScan = new Scanner(System.in);

	public MultiClassPTR(int algo, boolean randomWeight, double learning_rate, int max_epoch, double error_thres) {
		this.algo = algo;
		this.randomWeight = randomWeight;
		this.learningRate = learning_rate;
		this.maxEpoch = max_epoch;
		this.threshold = error_thres;
	}

	public MultiClassPTR(int num_instance, int num_input, int num_output, double[][] input, double[][] weight,
		double[][] target, int max_epoch, double learning_rate, double threshold, int algo, double momentum, boolean isRandomWeight) {
		this.numInstance = num_instance;
		this.numInput = num_input;
		this.numOutput = num_output;
		this.inputInstances = new double[num_instance][num_input + 1];
		this.inputInstances = input;
		this.weight = new double[num_input + 1][num_output];
//		this.weight = weight;

		//copy first array of weight
//		System.out.println("This.Weight.length: " + this.weight.length);
//		System.out.println("Weight.length: " + weight.length);
//		System.out.println("This.Weight[0].length: " + this.weight[0].length);
//		System.out.println("Weight[0].length: " + weight[0].length);
		for (int copier = 0; copier < weight[0].length; copier++) {
			this.weight[0][copier] = weight[0][copier];
			System.out.println("this.weight[0][" + copier + "]:" + this.weight[0][copier]);
		}

		this.momentum = momentum;
		this.targetInstances = target;
		this.maxEpoch = max_epoch;
		this.learningRate = learning_rate;
		this.threshold = threshold;
	}

	public void initAttributes(Instances instances) {
		numInstance = instances.numInstances();
		numInput = instances.numAttributes(); // Including bias
		
		inputInstances = new double[numInstance][numInput];// add values of attributes including bias value
		Attribute classAttribute = instances.classAttribute();
		if (classAttribute.isNominal()) {
			numOutput = classAttribute.numValues();
		} else {
			numOutput = 1;
		}
		targetInstances = new double[numInstance][numOutput];
		weight = new double[numInput][numOutput];
		
		
		double rangeMin = 0.0;
		double rangeMax = 1.0;
		for (int i = 0; i < numInput; ++i) {
			for (int j = 0; j < numOutput; ++j) {
				weight[i][j] = Math.random() * (rangeMax - rangeMin) + rangeMin;
			}
		}
	}

	public void buildClassifier() {
		double[][] deltaWeight = new double[numInput][numOutput];
		for (int it = 0; it < maxEpoch; ++it) {
			for (int instanceIdx = 0; instanceIdx < numInstance; ++instanceIdx) {
				double[] input = inputInstances[instanceIdx];
				
				double[] targets = targetInstances[instanceIdx];
				for (int outputIdx = 0; outputIdx < numOutput; ++outputIdx) {
					double target = targets[outputIdx];
					
					double sigmaInputWeight = 0.0;
					for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
						sigmaInputWeight += input[inputIdx] * weight[inputIdx][outputIdx];
					}
					double output;
					output = actFunction(sigmaInputWeight);
					
					double error = target-output;
										
					for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
						sigmaInputWeight += input[inputIdx] * weight[inputIdx][outputIdx];
						deltaWeight[inputIdx][outputIdx] = (learningRate * error * input[inputIdx]) + (momentum * deltaWeight[inputIdx][outputIdx]);
						weight[inputIdx][outputIdx] += deltaWeight[inputIdx][outputIdx];
					}
					
				}
				
			}
			
			double totalError = 0.0;
			for (int instanceIdx = 0; instanceIdx < numInstance; ++instanceIdx) {
				double[] input = inputInstances[instanceIdx];
				
				double[] targets = targetInstances[instanceIdx];
				for (int outputIdx = 0; outputIdx < numOutput; ++outputIdx) {
					double target = targets[outputIdx];
					
					double sigmaInputWeight = 0.0;
					for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
						sigmaInputWeight += input[inputIdx] * weight[inputIdx][outputIdx];
					}
					double output;
					output = actFunction(sigmaInputWeight);
					
					double error = target-output;
					totalError += Math.pow(error, 2);
				}
			}
			if (0.5 * totalError < threshold) {
				break;
			}
		}
	}

	public void buildClassifierBatch() {
		double[][] deltaWeight = new double[numInput][numOutput];
		for (int it = 0; it < maxEpoch; ++it) {
			for (int instanceIdx = 0; instanceIdx < numInstance; ++instanceIdx) {
				double[] input = inputInstances[instanceIdx];
				
				double[] targets = targetInstances[instanceIdx];
				for (int outputIdx = 0; outputIdx < numOutput; ++outputIdx) {
					double target = targets[outputIdx];
					
					double sigmaInputWeight = 0.0;
					for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
						sigmaInputWeight += input[inputIdx] * weight[inputIdx][outputIdx];
					}
					double output;
					output = actFunction(sigmaInputWeight);
					
					double error = target-output;
										
					for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
						sigmaInputWeight += input[inputIdx] * weight[inputIdx][outputIdx];
						deltaWeight[inputIdx][outputIdx] = (learningRate * error * input[inputIdx]) + (momentum * deltaWeight[inputIdx][outputIdx]);
					}
					
				}
				
			}
			
			for (int outputIdx = 0; outputIdx < numOutput; ++outputIdx) {
				for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
					weight[inputIdx][outputIdx] += deltaWeight[inputIdx][outputIdx];
				}
			}
			double totalError = 0.0;
			for (int instanceIdx = 0; instanceIdx < numInstance; ++instanceIdx) {
				double[] input = inputInstances[instanceIdx];
				
				double[] targets = targetInstances[instanceIdx];
				for (int outputIdx = 0; outputIdx < numOutput; ++outputIdx) {
					double target = targets[outputIdx];
					
					double sigmaInputWeight = 0.0;
					for (int inputIdx = 0; inputIdx < numInput; ++inputIdx) {
						sigmaInputWeight += input[inputIdx] * weight[inputIdx][outputIdx];
					}
					double output;
					output = actFunction(sigmaInputWeight);
					
					double error = target-output;
					totalError += Math.pow(error, 2);
				}
			}
			if (0.5 * totalError < threshold) {
				break;
			}
		}
	}

	public double actFunction(double input) {
		if (actFunc == 1) {
			if (input >= 0) {
				return 1;
			} else {
				return 0;
			}
		} else if (actFunc == 2) {
			if (input >= 0) {
				return 1;
			} else {
				return -1;
			}
		} else if (actFunc == 3) {
			return (1 / (1 + Math.exp(-input)));
		}
		return input;
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		initAttributes(instances);
		
		// REMEMBER: only works if class index is in the last position
		for (int instanceIdx = 0; instanceIdx < instances.numInstances(); instanceIdx++) {
			Instance instance = instances.get(instanceIdx);
			double[] inputInstance = inputInstances[instanceIdx];
			inputInstance[0] = 1.0; // initialize bias value
			for (int attrIdx = 0; attrIdx < instance.numAttributes() - 1; attrIdx++) {
				inputInstance[attrIdx + 1] = instance.value(attrIdx); // the first index of input instance is for bias
			}
		}

		// Initialize target values
		if (instances.classAttribute().isNominal()) {
			for (int instanceIdx = 0; instanceIdx < instances.numInstances(); instanceIdx++) {
				Instance instance = instances.instance(instanceIdx);
				for (int classIdx = 0; classIdx < instances.numClasses(); classIdx++) {
					targetInstances[instanceIdx][classIdx] = 0.0;
				}
				targetInstances[instanceIdx][(int) instance.classValue()] = 1.0;
			}
		} else {
			for (int instanceIdx = 0; instanceIdx < instances.numInstances(); instanceIdx++) {
				Instance instance = instances.instance(instanceIdx);
				targetInstances[instanceIdx][0] = instance.classValue();
			}
		}

		if (algo == 1) {
			setActFunction();
			buildClassifier();
		} else if (algo == 2) {
			buildClassifier();
		} else if (algo == 3) {
			buildClassifierBatch();
		}
	}

	public void setActFunction() {
		System.out.println("Masukkan fungsi aktivasi: ");
		System.out.println("1. Step");
		System.out.println("2. Sign");
		System.out.println("3. Sigmoid");
		System.out.println("0. Linear");

		actFunc = funcScan.nextInt();
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] input = new double[numInput]; // remember the first index use for bias input
		input[0] = 1.0;
		for (int attrIdx = 0; attrIdx < instance.numAttributes() - 1; attrIdx++) { // we ignore the class value
			input[attrIdx+1] = instance.value(attrIdx);
		}

		double[] output = new double[numOutput];
		double totalVal = 0.0;
		System.out.println("=========================");
		for (int outputIdx = 0; outputIdx < numOutput; outputIdx++) {
			output[outputIdx] = 0.0;
			for (int inputIdx = 0; inputIdx < numInput; inputIdx++) {
				output[outputIdx] += input[inputIdx] * weight[inputIdx][outputIdx];
			}
			output[outputIdx] = actFunction(output[outputIdx]);
			System.out.print(output[outputIdx] + " ");
			totalVal += Math.exp(output[outputIdx]);
		}
		for (int out = 0; out < numOutput; out++) {
			output[out] = Math.exp(output[out]) / totalVal;
		}
		return output;
	}

	@Override
	public Capabilities getCapabilities() {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public Enumeration<Option> listOptions() {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void setOptions(String[] strings) throws Exception {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public String[] getOptions() {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public void setSeed(int i) {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

	@Override
	public int getSeed() {
		throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
	}

}

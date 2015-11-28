/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml.ann;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

/**
 *
 * @author Ivana Clairine
 */
public class MainPTR {

	public static NominalToBinary m_nominalToBinaryFilter;

	public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
		boolean randomWeight;
		double weightawal = 0.0;
		double learningRate = 0.1;
		double threshold = 0.01;
		double momentum = 0.0;
		int maxEpoch = 10;

		m_nominalToBinaryFilter = new NominalToBinary();
		Scanner in = new Scanner(System.in);
		System.out.println("Lokasi file: ");

		String filepath = in.nextLine();
		filepath = "C:\\Program Files (x86)\\Weka-3-7\\data\\iris.2D.arff";
		System.out.println("--- Algoritma ---");
		System.out.println("1. Perceptron Training Rule");
		System.out.println("2. Delta Rule Gradient");
		System.out.println("3. Delta Batch Rule");
		System.out.println("Pilihan Algoritma (1/2/3) : ");
		int choice = in.nextInt();
		String temp = in.nextLine();

		System.out.println("Apakah Anda ingin memasukkan nilai weight awal? (YES/NO)");
		String isRandom = in.nextLine();

		System.out.println("Apakah Anda ingin memasukkan konfigurasi? (YES/NO)");
		int config = in.nextInt();
		
		if (isRandom.equals("YES")) {
			randomWeight = false;
		} else {
			randomWeight = true;
		}

		if (randomWeight == false) {
			System.out.println("Masukkan nilai weight awal: ");
			weightawal = in.nextDouble();
		}

		//print config
		if (isRandom.equalsIgnoreCase("yes")) {
			System.out.print("isRandom | ");
		} else {
			System.out.print("Weight " + weightawal + " | ");
		}

		System.out.print("L.rate " + learningRate + " | ");
		System.out.print("Max Epoch " + maxEpoch + " | ");
		System.out.print("Threshold " + threshold + " | ");
		System.out.print("Momentum " + momentum + " | ");
		System.out.println();

		FileReader trainreader = new FileReader(filepath);
		Instances train = new Instances(trainreader);
		train.setClassIndex(train.numAttributes() - 1);
		m_nominalToBinaryFilter.setInputFormat(train);
		train = Filter.useFilter(train, m_nominalToBinaryFilter);
		double[][] input;
		input = new double[train.numInstances()][train.numAttributes()];
		for (int i = 0; i < train.numInstances(); i++) {
			for (int j = 1; j < train.numAttributes(); j++) {
				System.out.println(train.attribute(j - 1));
				input[i][j] = train.instance(i).value(j - 1);
				System.out.println("input[" + i + "][" + j + "]: " + input[i][j]);
			}
		}
		double[][] weight = new double[train.numAttributes()][1];
		for (int i = 0; i < train.numAttributes(); i++) {
			weight[i][0] = weightawal;
		}
		System.out.println("N.Class: " + train.get(1).numClasses());
		if (train.get(1).numClasses() == 1) {
			//init weight for numeric
			double[] target = new double[train.numInstances()];

			for (int i = 0; i < train.numInstances(); i++) {
				target[i] = train.instance(i).classValue();
//			for (int j = 0; j < train.instance(i).numAttributes(); j++)
//				System.out.print(train.instance(i).stringValue(j) + " ");
				System.out.println("target[" + i + "]: " + target[i]);
			}

				SinglePTR testrun;
				testrun = new SinglePTR(train.numInstances(), train.numAttributes() - 1, maxEpoch, learningRate, threshold, input, target, weight, 3, momentum, randomWeight);
				
		} else {
			//init weight for multiclass
			System.out.println(train.numClasses());
			double[][] target = new double[train.numInstances()][train.numClasses()];
			for (int i = 0; i < train.numInstances(); i++){
				for (int j = 0; j < train.numClasses(); j++){
					target[i][j] = 0;
				}
				System.out.println(train.get(i).classValue());
				target[i][(int) train.get(i).classValue()] = 1;
				
				
			}
			
				MultiClassPTR testrun;
				testrun = new MultiClassPTR(train.numInstances(), train.numAttributes() - 1, train.get(1).numClasses(), input, weight, target, maxEpoch, learningRate, threshold, choice, momentum, randomWeight);
				
				double[] testClassify;
				testClassify = new double[train.numAttributes() - 1];
				testClassify[0] = 1.4;
				testClassify[1] = 0.2;
				
				testrun.classifyInstance(testClassify);
		}

	}

}

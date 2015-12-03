/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml.ann;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.Scanner;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Ivana Clairine
 */
public class MainPTR implements Serializable{

	public static NominalToBinary m_nominalToBinaryFilter;
	public static Normalize m_normalize;

	public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
		boolean randomWeight;
		double weightawal = 0.0;
		double learningRate = 0.0001;
		double threshold = 0.00;
		double momentum = 0.00;
		int maxEpoch = 100000;
		int nCrossValidate = 2;
		
		m_nominalToBinaryFilter = new NominalToBinary();
		m_normalize = new Normalize();
		
		Scanner in = new Scanner(System.in);
		System.out.println("Lokasi file: ");

		String filepath = in.nextLine();
		filepath = "test-arffs/iris.arff";
		System.out.println("--- Algoritma ---");
		System.out.println("1. Perceptron Training Rule");
		System.out.println("2. Delta Rule Incremental");
		System.out.println("3. Delta Rule Batch");
		System.out.println("Pilihan Algoritma (1/2/3) : ");
		int choice = in.nextInt();
		String temp = in.nextLine();

		System.out.println("Apakah Anda ingin memasukkan nilai weight awal? (YES/NO)");
		String isRandom = in.nextLine();
		System.out.println("Apakah Anda ingin memasukkan konfigurasi? (YES/NO)");
		String config = in.nextLine();
		
		if (config.equalsIgnoreCase("yes")){
			System.out.print("Masukkan nilai learning rate: ");
			learningRate = in.nextDouble();
			System.out.print("Masukkan nilai threshold: ");
			threshold = in.nextDouble();
			System.out.print("Masukkan nilai momentum: ");
			momentum = in.nextDouble();
			System.out.print("Masukkan jumlah epoch: ");
			threshold = in.nextInt();
			System.out.print("Masukkan jumlah folds untuk crossvalidate: ");
			nCrossValidate = in.nextInt();
		}
		
		randomWeight = isRandom.equalsIgnoreCase("yes");

		if (randomWeight) {
			System.out.print("Masukkan nilai weight awal: ");
			weightawal = Double.valueOf(in.nextLine());
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
		System.out.print("Folds " + nCrossValidate + " | ");
		System.out.println();

		FileReader trainreader = new FileReader(filepath);
		Instances train = new Instances(trainreader);
		train.setClassIndex(train.numAttributes() - 1);
		
		
		m_nominalToBinaryFilter.setInputFormat(train);
		train = new Instances(Filter.useFilter(train, m_nominalToBinaryFilter));
		
		m_normalize.setInputFormat(train);
		train = new Instances(Filter.useFilter(train, m_normalize));
		
		MultiClassPTR tempMulti = new MultiClassPTR(choice, randomWeight, learningRate, maxEpoch, threshold);
		tempMulti.buildClassifier(train);
		
		Evaluation eval = new Evaluation(new Instances(train));
		eval.evaluateModel(tempMulti, train);
		System.out.println(eval.toSummaryString());	
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
	}

}

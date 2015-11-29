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
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Ivana Clairine
 */
public class MainPTR {

	public static NominalToBinary m_nominalToBinaryFilter;
	public static Normalize m_normalize;

	public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
		boolean randomWeight;
		double weightawal = 0.0;
		double learningRate = 0.1;
		double threshold = 0.01;
		double momentum = 0.0;
		int maxEpoch = 10;

		m_nominalToBinaryFilter = new NominalToBinary();
		m_normalize = new Normalize();
		
		Scanner in = new Scanner(System.in);
		System.out.println("Lokasi file: ");

		String filepath = in.nextLine();
		//filepath = "C:\\Program Files (x86)\\Weka-3-7\\data\\iris.2D.arff";
		System.out.println("--- Algoritma ---");
		System.out.println("1. Perceptron Training Rule");
		System.out.println("2. Delta Rule Gradient");
		System.out.println("3. Delta Batch Rule");
		System.out.println("Pilihan Algoritma (1/2/3) : ");
		int choice = in.nextInt();
		String temp = in.nextLine();

		System.out.println("Apakah Anda ingin memasukkan nilai weight awal? (YES/NO)");
		String isRandom = in.nextLine();

//		System.out.println("Apakah Anda ingin memasukkan konfigurasi? (YES/NO)");
//		int config = in.nextInt();
		
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
		m_normalize.setInputFormat(train);
		train = Filter.useFilter(train, m_normalize);
		
                MultiClassPTR tempMulti = new MultiClassPTR(choice, randomWeight, learningRate, maxEpoch, threshold);
                tempMulti.buildClassifier(train);
                
                System.out.println("Masukkan letak dataset: ");
                String testPath = in.nextLine();
                
                FileReader testSet = new FileReader(testPath);
                Instances test = new Instances(testSet);
                m_nominalToBinaryFilter.setInputFormat(test);
                test = Filter.useFilter(test, m_nominalToBinaryFilter);
                m_normalize.setInputFormat(test);
                test = Filter.useFilter(test, m_normalize);
                
                for(Instance i: test)
                {
                    tempMulti.classifyInstance(i);
                }
                

	}

}

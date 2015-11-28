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
    
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception{
        boolean randomWeight;
        double weightawal = 0.0;
        m_nominalToBinaryFilter = new NominalToBinary();
        Scanner in = new Scanner(System.in);
        System.out.println("Lokasi file: ");
        String filepath = in.nextLine();
        System.out.println("--- Algoritma ---");
        System.out.println("1. Perceptron Training Rule");
        System.out.println("2. Delta Rule Gradient");;
        System.out.println("3. Delta Batch Rule");
        System.out.println("Pilihan Algoritma (1/2/3) : ");
        int choice = in.nextInt();
        String temp = in.nextLine();
        
        System.out.println("Apakah Anda ingin memasukkan nilai weight awal? (YES/NO)");
        String isRandom = in.nextLine();
        
        System.out.println("isRandom: "+isRandom);
        
        if(isRandom.equals("YES"))
            randomWeight = false;
        else randomWeight = true;
        
        if(randomWeight == false)
        {
            System.out.println("Masukkan nilai weight awal: ");
            weightawal = in.nextDouble();
        }
        
        FileReader trainreader = new FileReader(filepath);
        Instances train = new Instances(trainreader);
        train.setClassIndex(train.numAttributes()-1);
        m_nominalToBinaryFilter.setInputFormat(train);
        train = Filter.useFilter(train,m_nominalToBinaryFilter);
        double[][] input;
        input = new double[train.numInstances()][train.numAttributes()];
        for(int i=0; i<train.numInstances(); i++)
        {
            for(int j = 1; j < train.numAttributes(); j++)
            {
                System.out.println(train.attribute(j-1));
                input[i][j] = train.instance(i).value(j-1);
                System.out.println("input["+i+"]["+j+"]: "+ input[i][j]);
            }
        }
        
        double[] target = new double[train.numInstances()];
        
        for(int i=0; i<train.numInstances(); i++)
        {
            target[i] = train.instance(i).classValue();
            System.out.println("target["+i+"]: "+ target[i]);
        }
        
        double[][] weight = new double[train.numAttributes()][1];
        for(int i=0; i<train.numAttributes(); i++)
        {
            weight[i][0] = weightawal;
        }
        
        if(choice == 1)
        {
            SinglePTR testrun;
            testrun = new SinglePTR(train.numInstances(), train.numAttributes()-1, 10, 0.1, 0.01, input, target, weight, 1, randomWeight);
        }
        else if(choice == 2)
        {
            SinglePTR testrun;
            testrun = new SinglePTR(train.numInstances(), train.numAttributes()-1, 10, 0.1, 0.01, input, target, weight, 2, randomWeight);
        }
        else if(choice == 3)
        {
            SinglePTR testrun;
            testrun = new SinglePTR(train.numInstances(), train.numAttributes()-1, 10, 0.1, 0.01, input, target, weight, 3, randomWeight);
        }
        
//        for(int it = 0; it<train.numInstances(); it++)
//        {
//            for(int i=0; i<=train.numAttributes(); i++)
//            {
//                System.out.println(testrun.input[it][i]);
//            }
//        }
        
//        for(int i=0; i<train.numInstances(); i++)
//        {
//            for(int j=0;j<train.numAttributes(); j++)
//            System.out.println("Instance " + i + ", input "+ j + ", value: " + testrun.input[i][j]);
//            System.out.println("Target "+i+": "+testrun.target[i]);
//        }
        
        
    }
    
}

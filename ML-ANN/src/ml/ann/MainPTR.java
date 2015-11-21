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

/**
 *
 * @author Ivana Clairine
 */
public class MainPTR {
    
    public static void main(String[] args) throws FileNotFoundException, IOException{
        Scanner in = new Scanner(System.in);
        System.out.println("Lokasi file: ");
        String filepath = in.nextLine();
        
        FileReader trainreader = new FileReader(filepath);
        Instances train = new Instances(trainreader);
        train.setClassIndex(train.numAttributes()-1);
        
        double[][] input;
        input = new double[train.numInstances()][train.numAttributes()];
        for(int i=0; i<train.numInstances(); i++)
        {
            for(int j = 0; j<train.numAttributes()-1; j++)
            {
                input[i][j+1] = train.instance(i).value(j);
            }
        }
        
        double[] target = new double[train.numInstances()];
        
        for(int i=0; i<train.numInstances(); i++)
        {
            target[i] = train.instance(i).classValue();
        }
        
//        for(int i=0; i<train.numInstances(); i++)
//        {
//            for(int j=0;j<train.numAttributes(); j++)
//                System.out.println("Instance " + i + ", input "+ j + ", value: " + input[i][j]);
//            System.out.println("Target "+i+": "+target[i]);
//        }
        
        SinglePTR testrun;
        testrun = new SinglePTR(train.numInstances(), train.numAttributes(), 10, 0.1, 0.01, input, target, true);
        
        testrun.buildclassifier();
        
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

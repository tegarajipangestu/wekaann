/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml.ann;

import static ml.ann.MainPTR.m_nominalToBinaryFilter;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.attribute.NominalToBinary;
import java.util.Scanner;

/**
 *
 * @author Ivana Clairine
 */
public class SinglePTR implements Classifier {
    public int num_instance;
    //public int instance_input; //ada berapa banyak masukan yang akan digunakan
    public int num_input; //jumlah node input
    public int max_epoch; //jika error tidak konvergen sampai max_epoch, berhenti
    public double learning_rate; //dipakai di perhitungan
    public double threshold;
    public double momentum;
    public double[][] input; //data masukan
    public double[] target; //target sebanyak instance_input
    
    public double[][] weight;//menyimpan nilai weight X ke Y (misal: input1 ke neuron1)
    //dalam kasus singlePTR, pasti selalu [input][neuron]
    //untuk sekarang 1dimensi karena output selalu 1 neuron
    
    public double[] output;
    public double[] error;
    public int algo;
    public boolean randomWeight;
    
	public int actFunc = 0;
	Scanner funcScan;
    public SinglePTR(int algo, boolean randomWeight){
        this.algo = algo;
        this.randomWeight = randomWeight;
    }
    
    public void setAttribute (int num_instance, int num_input, int max_epoch, double LR, double threshold,
                      double[][]input, double[] target, double[][] weight, int algo, double momentum, boolean isRandomWeight){
        this.num_instance = num_instance;
        this.num_input = num_input;
        this.max_epoch = max_epoch;
        this.learning_rate= LR;
        this.target = new double[num_instance];
        this.target = target;
        this.momentum = momentum;
        this.threshold = threshold;
        this.input = new double[num_instance][num_input+1];
        //this.input = input.clone();
        this.input = input;
       
        this.weight = new double[num_input+1][1];
        this.weight = weight;
        
        for(int i=0; i<num_instance; i++)
        {
            input[i][0] = 1.0;
        }        
        
    }
    
    public SinglePTR (int num_instance, int num_input, int max_epoch, double LR, double threshold,
                      double[][]input, double[] target, double[][] weight, int algo, double momentum, boolean isRandomWeight)
    {
        this.num_instance = num_instance;
        this.num_input = num_input;
        this.max_epoch = max_epoch;
        this.learning_rate= LR;
        this.target = new double[num_instance];
        this.target = target;
        this.momentum = momentum;
        this.threshold = threshold;
        this.input = new double[num_instance][num_input+1];
		//this.input = input.clone();
		this.input = input;

		this.weight = new double[num_input + 1][1];
		this.weight = weight;

		this.funcScan = new Scanner(System.in);

		for (int i = 0; i < num_instance; i++) {
			input[i][0] = 1.0;
		}

		if (algo == 1) {
			System.out.println("Pilih fungsi aktivasi: ");
			System.out.println("1. Step");
			System.out.println("2. Sign");
			System.out.println("3. Sigmoid");
			actFunc = funcScan.nextInt();
			this.buildClassifier(1, isRandomWeight);
		} else if (algo == 2) {
			this.buildClassifier(2, isRandomWeight);
		} else if (algo == 3) {
			this.buildClassifierBatch(isRandomWeight);
		}

	}

    public void setWeight() {
            double rangeMin = 0.0;
            double rangeMax = 1.0;
            for (int i = 1; i <= num_input - 1; i++) {
                    weight[i][0] = Math.random() * (rangeMax - rangeMin) + rangeMin;
            }
    }

    public void buildClassifierBatch(boolean setweight) {
            double[] outputawal = new double[num_instance];
            double[] errorawal = new double[num_instance];
            double[] outputakhir = new double[num_instance];
            double[] errorakhir = new double[num_instance];
            double[][] deltaweight = new double[num_instance][num_input + 1];
            double[] sumdelta = new double[num_input + 1];
            boolean stop = false;
            int iterator = 1;

            //pilihan untuk inisialisasi weight
            if (setweight) {
                    setWeight();
            }

            while (iterator <= max_epoch && !stop) {
                    //hitung output dan error awal
                    System.out.println("--- Iterasi" + iterator + " ---");

                    for (int i = 0; i < num_instance; i++) {
                            outputawal[i] = 0.0;
                            for (int j = 0; j <= num_input; j++) {
                                    outputawal[i] += input[i][j] * weight[j][0];
                                    outputawal[i] = this.actFunction(outputawal[i]);

                            }
                            //System.out.println("outputawal["+i+"]: "+outputawal[i]);
                            errorawal[i] = target[i] - outputawal[i];
                            //System.out.println("errorawal["+i+"]: "+errorawal[i]);
                    }

                    //hitung deltaWeight0 - deltaWeightN
                    for (int i = 0; i < num_instance; i++) {
                            for (int j = 0; j <= num_input; j++) {
                deltaweight[i][j] = learning_rate*(1-momentum)*input[i][j]*errorawal[i]+(momentum*deltaweight[i][j]);
                                    //System.out.println("deltaweight["+i+"]["+j+"]: "+deltaweight[i][j]);
                            }
                    }

                    //hitung sumdelta, untuk hitung output akhir
                    for (int i = 0; i <= num_input; i++) {
                            sumdelta[i] = 0.0;
                            for (int j = 0; j < num_instance; j++) {
                                    sumdelta[i] += deltaweight[j][i];
                            }
                            //System.out.println("sumdelta["+i+"]: "+sumdelta[i]);
                    }

                    //hitung outputakhir
                    for (int i = 0; i < num_instance; i++) {
                            outputakhir[i] = 0.0;
                            for (int j = 0; j <= num_input; j++) {
                                    outputakhir[i] += input[i][j] * (sumdelta[j] + weight[j][0]);
                                    outputakhir[i] = this.actFunction(outputakhir[i]);
                            }
                            //System.out.println("outputakhir["+i+"]: "+outputakhir[i]);
                            errorakhir[i] = target[i] - outputakhir[i];
                            //System.out.println("errorakhir["+i+"]: "+errorakhir[i]);
                    }

                    //hitung error akhir
                    double sumerror = 0.0;
                    for (int i = 0; i < num_instance; i++) {
                            sumerror += Math.pow(errorakhir[i], 2);
                    }
                    if (0.5 * sumerror < threshold) {
                            stop = true;
                    } else {
                            for (int i = 0; i <= num_input; i++) {
                                    weight[i][0] = weight[i][0] + sumdelta[i];
                            }
                    }
                    System.out.println("error total: " + 0.5 * sumerror);
                    iterator++;
            }
    }

    public double actFunction(double input) {
            if (actFunc == 1) {
                    if (input >= threshold) {
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
//			System.out.println("Passing sigmoid function;");
                    return (1 / (1 + Math.exp(-1 * input)));
            } else { //linear?
//			System.out.println("Passing linear function;");
                    return input;
            }
    }

    public void buildClassifier(int algo, boolean randomWeight) {
            double[] output = new double[num_instance];
            double[] error = new double[num_instance];
            double[] deltaweight = new double[num_input + 1];
            boolean stop = false;
            int iterator = 1;

            //set apakah mau dirandom weightnya
            if (randomWeight) {
                    setWeight();
            }

            double threshold = 0.0;
            while (iterator <= max_epoch && !stop) {
                    System.out.println("--- Iterasi " + iterator + " ---");

                    for (int i = 0; i < num_instance; i++) {
                            //hitung output
                            if (algo == 1) //kalau algoritma PTR, pake algo
                            {
                                    output[i] = 0.0;
                                    for (int j = 0; j <= num_input; j++) {
                                            output[i] += input[i][j] * weight[j][0];
                                            output[i] = this.actFunction(output[i]);
                                            System.out.print("output[" + i + "]: "+ output[i]);
                                    }

                            } else if (algo == 2) //kalau delta rule, ga pake step
                            {
                                    output[i] = 0.0;
                                    for (int j = 0; j <= num_input; j++) {
                                            output[i] += input[i][j] * weight[j][0];
                                            output[i] = this.actFunction(output[i]);
                                    }
                            }
                            //System.out.println("output awal["+i+"]: "+output[i]);
                            //hitung error
                            error[i] = target[i] - output[i];

                            //hitung deltaWeight dan setNewWeight
                            for (int j = 0; j <= num_input; j++) {
                deltaweight[j] = learning_rate*(1-momentum)*error[i]*input[i][j]+(momentum*deltaweight[j]);
                weight[j][0] = weight[j][0]+deltaweight[j];
                                    //System.out.println("deltaweight["+j+"]: "+deltaweight[j]);
                            }
                    }

                    for (int i = 0; i < num_instance; i++) {
                            //hitung output akhir
                            if (algo == 1) {
                                    output[i] = 0.0;
                                    for (int j = 0; j <= num_input; j++) {
                                            output[i] += input[i][j] * weight[j][0];
                                            output[i] = this.actFunction(output[i]);
                                    }
                            } else if (algo == 2) {
                                    output[i] = 0.0;
                                    for (int j = 0; j <= num_input; j++) {
                                            output[i] += input[i][j] * weight[j][0];
                                    }
                            }
                            System.out.println("output akhir["+i+"]: "+output[i]);
                            //hitung error akhir
                            error[i] = target[i] - output[i];

                    }

                    //hitung total error
                    double sumerror = 0.0;
                    for (int i = 0; i < num_instance; i++) {
                            sumerror += Math.pow(error[i], 2);
                    }
                    if (0.5 * sumerror < threshold) {
                            stop = true;
                    }


                    System.out.println("Total error: " + 0.5 * sumerror);
                    iterator++;
            }
    }

    public double classifyInstance(double[] input) {
            double classifyoutput = 0.0;
            for (int i = 0; i < input.length; i++) {
                    classifyoutput += input[i] * weight[i][0];
                    classifyoutput = this.actFunction(classifyoutput);
            }
            return classifyoutput;
    }

    public static void main(String[] args) {
		double LR = 0.1;
		double threshold = 0.01;
		
		//init dataset1
		int num_instance = 3;
		int num_input = 3;
		int max_epoch = 10;
		
		//init dataset2
		num_instance = 6;
		num_input = 5;
		
		double momen = 0;
		
		double[][] input = new double[num_instance][num_input + 1];
		double[] target = new double[num_instance];
		double[] error = new double[num_instance];
		double[] output = new double[num_instance];
		double[][] weight = new double[num_input + 1][1];
		//dataset 1
//		input[0][0] = 1.0;
//		input[0][1] = 1.0;
//		input[0][2] = 0.0;
//		input[0][3] = 1.0;
//		input[1][0] = 1.0;
//		input[1][1] = 0.0;
//		input[1][2] = -1.0;
//		input[1][3] = -1.0;
//		input[2][0] = 1.0;
//		input[2][1] = -1.0;
//		input[2][2] = -0.5;
//		input[2][3] = -1.0;
//		target[0] = -1;
//		target[1] = 1;
//		target[2] = 1;
//		weight[0][0] = 0.0;
//		weight[1][0] = 0.0;
//		weight[2][0] = 0.0;
//		weight[3][0] = 0.0;

		//dataset 2
		input[0][0] = 1.0;
		input[0][1] = 5.1;
		input[0][2] = 3.5;
		input[0][3] = 1.4;
		input[0][4] = 0.2;
		
		input[1][0] = 1.0;
		input[1][1] = 4.9;
		input[1][2] = 3.0;
		input[1][3] = 1.4;
		input[1][4] = 0.2;
		
		input[2][0] = 1.0;
		input[2][1] = 4.7;
		input[2][2] = 3.2;
		input[2][3] = 1.3;
		input[2][4] = 0.2;
		
		input[3][0] = 1.0;
		input[3][1] = 7.0;
		input[3][2] = 3.2;
		input[3][3] = 4.7;
		input[3][4] = 1.4;
		
		input[4][0] = 1.0;
		input[4][1] = 6.4;
		input[4][2] = 3.2;
		input[4][3] = 4.5;
		input[4][4] = 1.5;
		
		input[5][0] = 1.0;
		input[5][1] = 6.9;
		input[5][2] = 3.1;
		input[5][3] = 4.9;
		input[5][4] = 1.5;
		
		
		target[0] = 1;
		target[1] = 1;
		target[2] = 1;
		target[3] = -1;
		target[4] = -1;
		target[5] = -1;
		
		weight[0][0] = 0.0;
		weight[1][0] = 0.0;
		weight[2][0] = 0.0;
		weight[3][0] = 0.0;
		weight[4][0] = 0.0;
		
		
		SinglePTR lala = new SinglePTR(input.length, num_input, max_epoch, LR, threshold,
			input, target, weight, 1, momen, false);

	}
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        m_nominalToBinaryFilter = new NominalToBinary();//To change body of generated methods, choose Tools | Templates.
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    

    @Override
    public void buildClassifier(Instances train) throws Exception {
        double[][] input;
        double weightawal = 0.0;
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
        
        if(algo == 1)
        {
            SinglePTR testrun;
            testrun = new SinglePTR(train.numInstances(), train.numAttributes()-1, 10, 0.1, 0.01, input, target, weight, 1, momentum, randomWeight);
        }
        else if(algo == 2)
        {
            SinglePTR testrun;
            testrun = new SinglePTR(train.numInstances(), train.numAttributes()-1, 10, 0.1, 0.01, input, target, weight, 2, momentum, randomWeight);
        }
        else if(algo == 3)
        {
            SinglePTR testrun;
            testrun = new SinglePTR(train.numInstances(), train.numAttributes()-1, 10, 0.1, 0.01, input, target, weight, 3, momentum, randomWeight);
        }
    }
        
}

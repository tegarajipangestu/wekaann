/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml.ann;

/**
 *
 * @author Ivana Clairine
 */
public class MultiClassPTR {
    public int num_instance; //jumlah instance
    public int num_input; //jumlah node input, belum termasuk bias
    public int num_output; //jumlah node output
    public double[][] weight; //weight dari input ke neuron
    public double[][] input; //jumlah masukan
    public double[][] target; //dimensi pertama: nomor instance, dimensi kedua: nomor neuron output
    public int max_epoch;
    public double learning_rate;
    public double threshold;
    public double momentum;
    
    public MultiClassPTR(int num_instance, int num_input,  int num_output, double[][] input, double[][] weight,
                         double[][]target, int max_epoch, double learning_rate, double threshold, int algo, double momentum, boolean isRandomWeight){
        this.num_instance = num_instance;
        this.num_input = num_input;
        this.num_output = num_output;
        this.input = new double[num_instance][num_input+1];
        this.input = input;
        weight = new double[num_input+1][num_output];
        this.weight = weight;
        this.momentum = momentum;
        this.target = target;
        this.max_epoch = max_epoch;
        this.learning_rate = learning_rate;
        this.threshold = threshold;
        if(algo == 1)
        {
            buildClassifier(1, isRandomWeight);
        }
        else if (algo == 2)
        {
            buildClassifier(2, isRandomWeight);
        }
        else if (algo == 3)
        {
            buildClassifierBatch(isRandomWeight);
        }
    }
    
     public void setWeight(){
        double rangeMin = 0.0;
        double rangeMax = 1.0;
        for(int i = 1; i <= num_input-1; i++)
        {
            weight[i][0] = Math.random() * (rangeMax - rangeMin) + rangeMin;
        }
    }
     
     public void buildClassifier(int algo, boolean randomWeight){
        double[][] output = new double[num_instance][num_output];
        double[][] error = new double[num_instance][num_output];
        double[][] deltaweight = new double [num_input+1][num_output];
        boolean stop = false;
        int iterator = 1;
        
        //set apakah mau dirandom weightnya
        if(randomWeight)
        {    setWeight();}
        
        while(iterator<=max_epoch && !stop)
        {
            System.out.println("--- Iterasi "+ iterator + " ---");
            
            for(int out=0; out<num_output; out++)
            {
                for(int i = 0; i < num_instance; i++)
                {
                    //hitung output
                    if(algo == 1) //kalau algoritma PTR, pake step
                    {
                        output[i][out] = 0.0;
                        for(int j=0; j<=num_input; j++)
                        {
                            output[i][out] += input[i][j]*weight[j][out];
                        }
                        if(output[i][out] >= 0)
                        {
                            output[i][out] = 1;
                        }
                        else output[i][out] = -1;
                    }
                    else if(algo == 2) //kalau delta rule, ga pake step
                    {
                        output[i][out] = 0.0;
                        for(int j=0; j<=num_input; j++)
                        {
                            output[i][out]+= input[i][j]*weight[j][out];
                        }
                    }
                        //System.out.println("output awal["+i+"]: "+output[i]);
                    //hitung error
                    error[i][out] = target[i][out]-output[i][out];

                    //hitung deltaWeight dan setNewWeight
                    for(int j=0; j<=num_input; j++)
                    {
                        deltaweight[j][out] = learning_rate*(1-momentum)*error[i][out]*input[i][j]+(momentum*deltaweight[j][out]);
                        weight[j][out] = weight[j][out]+deltaweight[j][out];
                        //System.out.println("deltaweight["+j+"]: "+deltaweight[j]);
                    }
                }

                for(int i=0; i<num_instance; i ++)
                {
                    //hitung output akhir
                    if(algo == 1)
                    {
                        output[i][out] = 0.0;
                        for(int j=0; j<=num_input; j++)
                        {
                            output[i][out] += input[i][j]*weight[j][out];
                        }
                        if(output[i][out] >= 0)
                        {
                            output[i][out] = 1;
                        }
                        else output[i][out] = -1;
                    }
                    else if(algo == 2)
                    {
                        output[i][out] = 0.0;
                        for(int j=0; j<=num_input; j++)
                        {
                            output[i][out]+= input[i][j]*weight[j][out];
                        }
                    }
                    //System.out.println("output akhir["+i+"]: "+output[i]);
                    //hitung error akhir
                    error[i][out] = target[i][out]-output[i][out];
                }
            
                //hitung total error per neuron
                double sumerror = 0.0;
                for(int i=0; i<num_instance;i++)
                {
                    sumerror += Math.pow(error[i][out], 2);
                }
                if(0.5*sumerror < threshold)
                    stop = true;
                System.out.println("Total error neuron "+out+": "+0.5*sumerror);
            }
            iterator++;
        }
     }
     
     public void buildClassifierBatch(boolean setweight){
        double[][] outputawal = new double[num_instance][num_output];
        double[][] errorawal = new double[num_instance][num_output];
        double[][] outputakhir = new double[num_instance][num_output];
        double[][] errorakhir = new double[num_instance][num_output];
        
        double[][] deltaweight = new double[num_instance][num_input+1];
        double[] sumdelta = new double[num_input+1];
        boolean stop = false;
        int iterator = 1;
        
        //pilihan untuk inisialisasi weight
        if(setweight)
            setWeight();
        
        while(iterator<=max_epoch && !stop)
        {
            //hitung output dan error awal
            System.out.println("--- Iterasi"+iterator+" ---");
            
            for(int out=0; out<num_output; out++)
            {
                for(int i=0; i<num_instance; i++)
                {
                    outputawal[i][out] = 0.0;
                    for(int j=0;j<=num_input; j++)
                    {
                        outputawal[i][out]+=input[i][j]*weight[j][out];
                    }
                    //System.out.println("outputawal["+i+"]: "+outputawal[i]);
                    errorawal[i][out] = target[i][out]-outputawal[i][out];
                    //System.out.println("errorawal["+i+"]: "+errorawal[i]);
                }

                //hitung deltaWeight0 - deltaWeightN
                for(int i=0; i<num_instance;i++)
                {
                    for(int j=0; j<=num_input;j++)
                    {
                        deltaweight[i][j] = learning_rate*(1-momentum)*input[i][j]*errorawal[i][out]+(momentum*deltaweight[i][j]);
                        //System.out.println("deltaweight["+i+"]["+j+"]: "+deltaweight[i][j]);
                    }
                }

                //hitung sumdelta, untuk hitung output akhir
                for(int i = 0; i <= num_input; i++)
                {
                    sumdelta[i]=0.0;
                    for(int j = 0; j<num_instance; j++)
                    {
                        sumdelta[i] += deltaweight[j][i];
                    }
                    //System.out.println("sumdelta["+i+"]: "+sumdelta[i]);
                }

                //hitung outputakhir
                for(int i=0; i<num_instance; i++)
                {
                    outputakhir[i][out] = 0.0;
                    for(int j = 0; j<= num_input; j++)
                    {
                        outputakhir[i][out] += input[i][j]*(sumdelta[j]+weight[j][out]);
                    }
                    //System.out.println("outputakhir["+i+"]: "+outputakhir[i]);
                    errorakhir[i][out] = target[i][out]-outputakhir[i][out];
                    //System.out.println("errorakhir["+i+"]: "+errorakhir[i]);
                }

                //hitung error akhir
                double sumerror = 0.0;
                for(int i=0; i<num_instance;i++)
                {
                    sumerror+= Math.pow(errorakhir[i][out], 2);
                }
                if(0.5*sumerror < threshold)
                    stop = true;
                else
                {
                    for(int i=0; i<=num_input; i++)
                    {
                        weight[i][0] = weight[i][0]+sumdelta[i];
                    }
                }
                System.out.println("error total output ke-"+out+": " + 0.5*sumerror);
            }
            iterator++;
        }
    }
     
     public double classifyInstance(double[] input){
        double[] output = new double[num_output];
        for(int out=0; out<num_output; out++)
        {
            output[out] = 0.0;
            for(int i=0; i<input.length; i++)
            {
                output[out] += input[i]*weight[i][0];
            }
        }
        int out = 0;
        for (int i=1; i<num_output; i++)
        {
            if(output[i] > output[i-1])
                out = i;
        }
        return output[out];
    }
     
        public static void main(String[] args){
        int num_instance = 3;
        int num_input = 3;
        int max_epoch = 10;
        int num_output = 1;
        double LR = 0.1;
        double momentum = 0.9;
        double threshold = 0.01;
        double[][] input = new double[num_instance][num_input+1];
        double[][] target = new double[num_instance][num_output];
        double[] error = new double[num_instance];
        double[] output = new double[num_instance];
        double[][] weight = new double[num_input+1][1];
        input[0][0] = 1.0;
        input[0][1] = 1.0;
        input[0][2] = 0.0;
        input[0][3] = 1.0;
        input[1][0] = 1.0;
        input[1][1] = 0.0;
        input[1][2] = -1.0;
        input[1][3] = -1.0;
        input[2][0] = 1.0;
        input[2][1] = -1.0;
        input[2][2] = -0.5;
        input[2][3] = -1.0;
        target[0][0]  = -1;
        target[1][0]  = 1;
        target[2][0]  = 1;
        weight[0][0] = 0.0;
        weight[1][0] = 0.0;
        weight[2][0] = 0.0;
        weight[3][0] = 0.0;

        MultiClassPTR lala = new MultiClassPTR(input.length, num_input, num_output, input, weight, target, max_epoch, LR, threshold, 3, momentum, false);
        
    }
    
    
}

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
public class SinglePTR {
    public int num_instance;
    //public int instance_input; //ada berapa banyak masukan yang akan digunakan
    public int num_input; //jumlah node input
    public int max_epoch; //jika error tidak konvergen sampai max_epoch, berhenti
    public double learning_rate; //dipakai di perhitungan
    public double threshold;
    public double[][] input; //data masukan
    public double[] target; //target sebanyak instance_input
    
    public double[][] weight;//menyimpan nilai weight X ke Y (misal: input1 ke neuron1)
    //dalam kasus singlePTR, pasti selalu [input][neuron]
    //untuk sekarang 1dimensi karena output selalu 1 neuron
    
    public double[] output;
    public double[] error;
    
    public SinglePTR (int num_instance, int num_input, int max_epoch, double LR, double threshold,
                      double[][]input, double[] target, boolean fungsi_aktivasi)
    {
        this.num_instance = num_instance;
        this.num_input = num_input;
        this.max_epoch = max_epoch;
        this.learning_rate= LR;
        this.target = new double[num_instance];
        this.target = target;
        this.threshold = threshold;
        this.input = new double[num_instance][num_input];
        this.input = input;
        this.weight = new double[num_input][1];
        weight[0][0] = 0.0; //ini weight bias
        input[0][0] = 1.0; //ini bias, selalu 1
        this.error = new double[input.length];
        this.output = new double[input.length];
    }
    
//    public SinglePTR(){
//       //instance_input = 0;
//       num_input = 0;
//       max_epoch = 0;
//       learning_rate = 0.0;
//       target = new double[instance];
//       weight = new double[num_input+1][1];
//       input = new double[instance][num_input+1];
//       output = new double[instance];
//       weight[0][0] = 0.0; //ini weight bias
//       for(int i = 0; i < instance; i++)  //ini bias, selalu 1
//       {
//           input[i][0] = 1.0;
//       }
//    }
    
      public void setWeight(){
        double rangeMin = 0.0;
        double rangeMax = 1.0;
        for(int i = 1; i <= num_input-1; i++)
        {
            weight[i][0] = Math.random() * (rangeMax - rangeMin) + rangeMin;
        }
    }
    
    public double countOutput(boolean fungsi_aktivasi, int it)
    {
        double retval;
        double sum = 0.0;
        for(int i = 0; i <= num_input-1; i++)
        {
            sum = sum + (input[it][i] * weight[i][0]);
        }
        if(fungsi_aktivasi)
        {
            if(sum >= 0)
                retval = 1.0;
            else retval = -1.0;
        }
        else
        {
            retval = sum;
        }
        return retval;
    }

    
    public double countError(boolean fungsi_aktivasi, int it){
        double err = target[it] - output[it];
        return err;
    }
    
    public void countErrorInstaces(boolean fungsi_aktivasi)
    {
        for(int i=0; i<= input.length; i++)
        {
            error[i] = countError(fungsi_aktivasi, i);
        }
    }
    
    public double[] countDeltaWeight(boolean fungsi_aktivasi, int it)
    {
        double deltaweight[] = new double[num_input];
        double errorinstance = countError(fungsi_aktivasi, it);
//        System.out.println("num_input: "+num_input);
//        System.out.println("errorinstance "+it + ": " + errorinstance);
        
        for(int i=0; i <= num_input-1; i++)
        {
            deltaweight[i] = learning_rate*errorinstance*input[it][i];
            //System.out.println("Deltaweight instance"+ it +" , " + i + ": " + deltaweight[i]);
        }
        return deltaweight;
    }
    
    public void setNewWeight(boolean fungsi_aktivasi, int it){
        double delta[] = new double[num_input];
        delta = countDeltaWeight(fungsi_aktivasi, it);
        
        for(int i=0; i<=num_input-1; i++)
        {
            weight[i][0] = weight[i][0] + delta[i];
        }
    }
    
    
    public double countTotalError()
    {
        double sum = 0.0;
        for(int i=0; i<input.length; i++)
        {
            double temp = Math.pow(target[i]-output[i], 2);
            sum+=temp;
        }
        return 0.5*sum;
    }
    
    public boolean isConvergent()
    {
        if(countTotalError() < threshold)
            return true;
        else return false;
    }
    
   
    
    public void buildclassifier(){
//        SinglePTR test = new SinglePTR();
//        test.instance = 3;
//        
//        test.target = new double[test.instance];
//        test.error = new double[test.instance];
//        test.output = new double[test.instance];
//        test.max_epoch = 10;
//        
//        test.num_input = 3;
//        test.weight = new double[test.num_input+1][1];
//        test.input = new double[test.instance][test.num_input+1];
          setWeight();
//        test.learning_rate = 0.1;
//        test.threshold = 0.01;
//        
//        test.input[0][0] = 1.0;
//        test.input[0][1] = 1.0;
//        test.input[0][2] = 0.0;
//        test.input[0][3] = 1.0;
//        
//        test.input[1][0] = 1.0;
//        test.input[1][1] = 0.0;
//        test.input[1][2] = -1.0;
//        test.input[1][3] = -1.0;
//        
//        test.input[2][0] = 1.0;
//        test.input[2][1] = -1.0;
//        test.input[2][2] = -0.5;
//        test.input[2][3] = -1.0;
//        
//        
//        test.target[0]  = -1;
//        test.target[1]  = 1;
//        test.target[2]  = 1;
//        
//        
//        test.weight[0][0] = 1.0;
//        test.weight[1][0] = 1.0;
//        test.weight[2][0] = 1.0;
//        test.weight[3][0] = 1.0;
//        
        
//        test.input[0][0] = 1.0;
//        test.input[0][1] = 5.1;
//        test.input[0][2] = 3.5;
//        test.input[0][3] = 1.4;
//        test.input[0][4] = 0.2;
//        test.input[1][0] = 1.0;
//        test.input[1][1] = 4.9;
//        test.input[1][2] = 3.0;
//        test.input[1][3] = 1.4;
//        test.input[1][4] = 0.2;
//        test.input[2][0] = 1.0;
//        test.input[2][1] = 4.7;
//        test.input[2][2] = 3.2;
//        test.input[2][3] = 1.3;
//        test.input[2][4] = 0.2;
//        
//        test.input[3][0] = 1.0;
//        test.input[3][1] = 7.0;
//        test.input[3][2] = 3.2;
//        test.input[3][3] = 4.7;
//        test.input[3][4] = 1.4;
//        
//        test.input[4][0] = 1.0;
//        test.input[4][1] = 6.4;
//        test.input[4][2] = 3.2;
//        test.input[4][3] = 4.5;
//        test.input[4][4] = 1.5;
//        
//        test.input[5][0] = 1.0;
//        test.input[5][1] = 6.9;
//        test.input[5][2] = 3.1;
//        test.input[5][3] = 4.9;
//        test.input[5][4] = 1.5;
//        
//        
//        
//        test.target[0]  = 1;
//        test.target[1]  = 1;
//        test.target[2]  = 1;
//        test.target[3] = -1;
//        test.target[4] = -1;
//        test.target[5] = -1;
//        
//        
//        test.weight[0][0] = 0.0;
//        test.weight[1][0] = 0.0;
//        test.weight[2][0] = 0.0;
//        test.weight[3][0] = 0.0;
//        test.weight[4][0] = 0.0;
        
        //System.out.println("Output: " + test.countOutput(false));
        boolean stop = false;
        int iterator = 0;
        
        while(stop==false && iterator < max_epoch)
        {
            
            for(int i=0; i<input.length; i++)
            {
                setNewWeight(true, i);
                output[i] = countOutput(true, i);
                error[i] = countError(true, i);
            }
            if(isConvergent())
            {
                stop = true;
            }
            System.out.println("Total Error: " + countTotalError());
            
            iterator++;
        }
        
        System.out.println("Berhenti setelah " + iterator + " iterasi");
        
    }
    
    
}

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
    public double momentum;
    public double[][] input; //data masukan
    public double[] target; //target sebanyak instance_input
    
    public double[][] weight;//menyimpan nilai weight X ke Y (misal: input1 ke neuron1)
    //dalam kasus singlePTR, pasti selalu [input][neuron]
    //untuk sekarang 1dimensi karena output selalu 1 neuron
    
    public double[] output;
    public double[] error;
    
    
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
       
        this.weight = new double[num_input+1][1];
        this.weight = weight;
        
        for(int i=0; i<num_instance; i++)
        {
            input[i][0] = 1.0;
        }
        
        if(algo == 1)
            this.buildClassifier(1, isRandomWeight);
        else if (algo == 2)
            this.buildClassifier(2, isRandomWeight);
        else if (algo == 3)
        {
            this.buildClassifierBatch(isRandomWeight);
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

    
    public void buildClassifierBatch(boolean setweight){
        double[] outputawal = new double[num_instance];
        double[] errorawal = new double[num_instance];
        double[] outputakhir = new double[num_instance];
        double[] errorakhir = new double[num_instance];
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
            
           
            for(int i=0; i<num_instance; i++)
            {
                outputawal[i] = 0.0;
                for(int j=0;j<=num_input; j++)
                {
                    outputawal[i]+=input[i][j]*weight[j][0];
                }
                //System.out.println("outputawal["+i+"]: "+outputawal[i]);
                errorawal[i] = target[i]-outputawal[i];
                //System.out.println("errorawal["+i+"]: "+errorawal[i]);
            }
            
            //hitung deltaWeight0 - deltaWeightN
            for(int i=0; i<num_instance;i++)
            {
                for(int j=0; j<=num_input;j++)
                {
                    deltaweight[i][j] = learning_rate*(1-momentum)*input[i][j]*errorawal[i]+(momentum*deltaweight[i][j]);
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
                outputakhir[i] = 0.0;
                for(int j = 0; j<= num_input; j++)
                {
                    outputakhir[i] += input[i][j]*(sumdelta[j]+weight[j][0]);
                }
                //System.out.println("outputakhir["+i+"]: "+outputakhir[i]);
                errorakhir[i] = target[i]-outputakhir[i];
                //System.out.println("errorakhir["+i+"]: "+errorakhir[i]);
            }
            
            //hitung error akhir
            double sumerror = 0.0;
            for(int i=0; i<num_instance;i++)
            {
                sumerror+= Math.pow(errorakhir[i], 2);
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
            System.out.println("error total: " + 0.5*sumerror);
            iterator++;
        }
    }
    
    public void buildClassifier(int algo, boolean randomWeight){
        double[] output = new double[num_instance];
        double[] error = new double[num_instance];
        double[] deltaweight = new double [num_input+1];
        boolean stop = false;
        int iterator = 1;
        
        //set apakah mau dirandom weightnya
        if(randomWeight)
        {    setWeight();}
        
        while(iterator<=max_epoch && !stop)
        {
            System.out.println("--- Iterasi "+ iterator + " ---");
            
            for(int i = 0; i < num_instance; i++)
            {
                //hitung output
                if(algo == 1) //kalau algoritma PTR, pake step
                {
                    output[i] = 0.0;
                    for(int j=0; j<=num_input; j++)
                    {
                        output[i]+= input[i][j]*weight[j][0];
                    }
                    if(output[i] >= 0)
                    {
                        output[i] = 1;
                    }
                    else output[i] = -1;
                }
                else if(algo == 2) //kalau delta rule, ga pake step
                {
                    output[i] = 0.0;
                    for(int j=0; j<=num_input; j++)
                    {
                        output[i]+= input[i][j]*weight[j][0];
                    }
                }
                    //System.out.println("output awal["+i+"]: "+output[i]);
                //hitung error
                error[i] = target[i]-output[i];
                
                //hitung deltaWeight dan setNewWeight
                for(int j=0; j<=num_input; j++)
                {
                    deltaweight[j] = learning_rate*(1-momentum)*error[i]*input[i][j]+(momentum*deltaweight[j]);
                    weight[j][0] = weight[j][0]+deltaweight[j];
                    //System.out.println("deltaweight["+j+"]: "+deltaweight[j]);
                }
            }
            
            for(int i=0; i<num_instance; i ++)
            {
                //hitung output akhir
                if(algo == 1)
                {
                    output[i] = 0.0;
                    for(int j=0; j<=num_input; j++)
                    {
                        output[i]+= input[i][j]*weight[j][0];
                    }
                    if(output[i] >= 0)
                    {
                        output[i] = 1;
                    }
                    else output[i] = -1;
                }
                else if(algo == 2)
                {
                    output[i] = 0.0;
                    for(int j=0; j<=num_input; j++)
                    {
                        output[i]+= input[i][j]*weight[j][0];
                    }
                }
                //System.out.println("output akhir["+i+"]: "+output[i]);
                //hitung error akhir
                error[i] = target[i]-output[i];
            
            }
            
            //hitung total error
            double sumerror = 0.0;
            for(int i=0; i<num_instance;i++)
            {
                sumerror += Math.pow(error[i], 2);
            }
            if(0.5*sumerror < threshold)
                stop = true;
            System.out.println("Total error: "+0.5*sumerror);
            iterator++;
        }
    }
    
    public double classifyInstance(double[] input){
        double output = 0.0;
        for(int i=0; i<input.length; i++)
        {
            output += input[i]*weight[i][0];
        }
        return output;
    }
    
//        public static void main(String[] args){
//        int num_instance = 3;
//        int num_input = 3;
//        int max_epoch = 10;
//        double LR = 0.1;
//        double threshold = 0.01;
//        double[][] input = new double[num_instance][num_input+1];
//        double[] target = new double[num_instance];
//        double[] error = new double[num_instance];
//        double[] output = new double[num_instance];
//        double[][] weight = new double[num_input+1][1];
//        input[0][0] = 1.0;
//        input[0][1] = 1.0;
//        input[0][2] = 0.0;
//        input[0][3] = 1.0;
//        input[1][0] = 1.0;
//        input[1][1] = 0.0;
//        input[1][2] = -1.0;
//        input[1][3] = -1.0;
//        input[2][0] = 1.0;
//        input[2][1] = -1.0;
//        input[2][2] = -0.5;
//        input[2][3] = -1.0;
//        target[0]  = -1;
//        target[1]  = 1;
//        target[2]  = 1;
//        weight[0][0] = 0.0;
//        weight[1][0] = 0.0;
//        weight[2][0] = 0.0;
//        weight[3][0] = 0.0;
//
//        SinglePTR lala = new SinglePTR(input.length, num_input, max_epoch, LR, threshold,
//                      input, target, 2, false);
//        
//    }
}

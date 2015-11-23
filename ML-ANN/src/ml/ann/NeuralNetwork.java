/*
 * The MIT License
 *
 * Copyright 2015 tegarnization.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package ml.ann;

/**
 *
 * @author tegarnization
 */
public class NeuralNetwork {
    
    private double[][][] weight; // [layer sumber][neuron sumber][neuron tujuan]
    private double[] input; 
    private double[][] output; // [layer sumber][neuron sumber]
    private double learningRate;
    private double momentum;

    public double[][][] getWeight() {
        return weight;
    }

    public void setWeight(double[][][] weight) {
        this.weight = weight;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[][] getOutput() {
        return output;
    }

    public void setOutput(double[][] output) {
        this.output = output;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public NeuralNetwork(double[][][] weight, double[] input, double[][] output, double learningRate, double momentum) {
        this.weight = weight;
        this.input = input;
        this.output = output;
        this.learningRate = learningRate;
        this.momentum = momentum;
    }
    
    public void errorCount() {
        
    }
    
    public void updateWeight() {
        
    }
    
    //Kerjaan feli
    public void forwardChaining() {
        
    }
    
    public void backPropagation() {
        
    }
    
    //Kerjaan feli
    public double activationFunction() {
        return 0;
    }
    
    public void numericToBinary() {
        
    }
    
    public void nominalToBinary() {
        
    }
    
    public void binaryToNominal() {
        
    }
    
    public void binaryToNumeric() {
        
    }
}

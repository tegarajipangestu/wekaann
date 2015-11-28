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

import java.io.Serializable;
import weka.core.Instances;

/**
 *
 * @author tegarnization
 */
public class NeuronTopology implements Serializable {

    public double[][][] weight; // [layer sumber][neuron sumber][neuron tujuan]
    public double[][] input;
    public double target;
    public double[][] dummyInstances; //[n-instance][n-attribute]
    public double[][] output; // [layer sumber][neuron sumber]
    public double[][] error; // [layer sumber][neuron sumber]
    public int[] numNeuronEachLayer;
    public double learningRate;
    public double momentum;
    public int numLayers;
    public int numHiddenLayers;
    public int epoch;
    public boolean hasBias;
    public Instances instances;
    public int[] numNeuronEachHiddenLayers;
    public int numClasses;
    public int numAttribute;
    public double terminateMSE;
    
}

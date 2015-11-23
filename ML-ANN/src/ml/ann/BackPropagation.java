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

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.WeightedInstancesHandler;

/**
 *
 * @author tegarnization
 */
public class BackPropagation extends AbstractClassifier implements
        OptionHandler, WeightedInstancesHandler, Randomizable {
    
    private Classifier zeroR;
    private boolean isDefaultModel;

    @Override
    public void setSeed(int i) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public int getSeed() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(i);
        
        // remove instances with missing class
        i = new Instances(i);
        i.deleteWithMissingClass();
        
        zeroR = new weka.classifiers.rules.ZeroR();
        zeroR.buildClassifier(i);
        // only class? -> use ZeroR model
        if (i.numAttributes() == 1) {
            System.err
                    .println("Cannot build model (only class attribute present in data!), "
                            + "using ZeroR model instead!");
            isDefaultModel = true;
            return;
        } else {
            isDefaultModel = false;
        }
        
    }
    
}

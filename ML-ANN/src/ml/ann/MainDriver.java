/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml.ann;

/**
 *
 * @author Feli
 */
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MainDriver {

    /**
     * @param args the command line arguments
     */
    public static Instances data;
    public static Classifier model;
    public static Instances train;
    public static Instances test;
    public static boolean cv10 = false;

    public static void main(String[] args) throws Exception {
	  // TODO code application logic here
	  Scanner input = new Scanner(System.in);
	  System.out.print("Input complete file location: ");
	  String fileLocation = input.nextLine();
          fileLocation = "/home/tegarnization/Documents/code/repo/wekaann/ML-ANN/src/ml/ann/weather.nominal.arff";
          
//          fileLocation = "/home/tegar/Documents/code/repos/wekawekaDTL/wekaAccess/src/wekaaccess/iris.arff";

	  if (fileLocation.endsWith(".csv") || fileLocation.endsWith(".arff")) {
		try { // file arff/csv
		    System.out.println("Mengambil dataset...");
		    DataSource source = new DataSource(fileLocation);
		    data = source.getDataSet();

		    if (data.classIndex() == -1) {
			  data.setClassIndex(data.numAttributes() - 1);
		    }
		    System.out.println("Daftar atribut dari data: ");
		    int enumCounter = 1;

		    for (Enumeration<Attribute> atr = data.enumerateAttributes(); atr.hasMoreElements();) {
			  System.out.println(enumCounter + ". " + atr.nextElement());
			  enumCounter++;
		    }

		} catch (Exception E) {

		    E.printStackTrace();
		}
	  } else if (fileLocation.endsWith(".model")) { // file model
		System.out.print("Masukkan tipe model [SP-I|SP-DB|MLP]: ");
		String modelType = input.nextLine();
		System.out.println("Mengambil model dari file...");
		ModelLearning(fileLocation, modelType);

		System.out.print("Masukkan dataset testing: ");
		fileLocation = input.nextLine();
		try {
		    DataSource source = new DataSource(fileLocation);
		    data = source.getDataSet();
		    if (data.classIndex() == -1) {
			  data.setClassIndex(data.numAttributes() - 1);
		    }

		    test = new Instances(data);
		    if (test.classIndex() == -1) {
			  test.setClassIndex(test.numAttributes() - 1);
		    }
		} catch (Exception E) {
		    E.printStackTrace();
		}
	  }

	  printOptions();
	  int options = input.nextInt();
	  while (options != 0) {
		switch (options) {
		    case 1:
			  removeAttributes();
			  break;
		    case 2:
			  filterResample();
			  break;
		    case 3:
			  buildClassifier();
			  break;
		    case 4:
			  saveModel();
			  break;
		    case 5:
			  testModel();
			  break;
		    default:
			  break;

		}
		printOptions();
		options = input.nextInt();
	  }

    }

    public static void printOptions() {
	  System.out.println("#############################################");
	  System.out.println("#  Pilihan operasi:                         #");
	  System.out.println("#  1. Hapus atribut                         #");
	  System.out.println("#  2. Filter [resample]                     #");
	  System.out.println("#  3. Bulid Classifier                      #");
	  System.out.println("#  4. Simpan Model                          #");
	  System.out.println("#  5. Uji model                             #");
	  System.out.println("#  0. Keluar                                #");
	  System.out.println("#############################################");
	  System.out.print("# > ");

    }

    public static void removeAttributes() {

	  Scanner input = new Scanner(System.in);
	  System.out.print("Masukkan nomor atribut yang mau dihilangkan (0 jika tidak ingin menghilangkan atribut): ");
	  int rmvAtr = input.nextInt();
	  if (rmvAtr != 0) {
		data.deleteAttributeAt(rmvAtr - 1);
		System.out.println("Daftar atribut dari data terbaru: ");
		int enumCounter = 1;
		for (Enumeration<Attribute> atr = data.enumerateAttributes(); atr.hasMoreElements();) {
		    System.out.println(enumCounter + ". " + atr.nextElement());
		    enumCounter++;
		}
	  }
    }

    public static void filterResample() {
	  Random R = new Random();
	  data.resample(R);
    }

    public static void saveModel() {
	  Scanner in = new Scanner(System.in);
	  System.out.print("Nama file untuk penyimpanan model: ");
	  String name = in.nextLine();

	  try {
		weka.core.SerializationHelper.write(name + ".model", model);
	  } catch (Exception e) {
		e.printStackTrace();
	  }

    }

    public static void ModelLearning(String fileLocation, String ModelType) {
	  Classifier C;
	  C = null;

	  try {
		switch (ModelType) {
		    case "SP-I":
			  System.out.println("Reading SP-I Model...");
			  break;
		    case "SP-DB":
			  System.out.println("Reading SP-DB Model...");
			  break;
		    case "MLP":
			  System.out.println("Reading MLP Model...");
			  break;
		}
	  } catch (Exception E) {
		E.printStackTrace();
	  }
	  model = C;
	  //return C;
    }

    public static void buildClassifier() throws Exception {
	  Scanner in = new Scanner(System.in);
	  int learningType, split, classifierType;
	  String[] options = new String[1];

	  System.out.println("## Pilih tipe classifier: ");
	  System.out.println("## 1. Single Perceptron Training Rule");
	  System.out.println("## 2. Single Perceptron Delta Batch");
	  System.out.println("## 3. Backpropagation");
	  System.out.print("## > ");
	  classifierType = in.nextInt();

	  switch (classifierType) {
		case 1:
		    break;
		case 2:
		    break;
		case 3:
//                    model = new MultilayerPerceptron();
                    model = new BackPropagation(1000,1,false,true,0.5,4,0.5);
		    break;
	  }

	  System.out.println("");
	  System.out.println("### Pilih cara training: ");
	  System.out.println("### 1. Percentage Split");
	  System.out.println("### 2. 10-Fold Cross Validation");
	  System.out.print("### > ");
	  learningType = in.nextInt();

	  if (learningType == 1) {
		cv10 = false;
		System.out.print("Masukkan persentase split [0..100]: ");
		split = in.nextInt();

		//E:\Git\wekaAccess\iris.arff
		int trainSize = (int) Math.round(data.numInstances() * split / 100);
		int testSize = data.numInstances() - trainSize;
		//data.randomize(new Random(1));

		train = new Instances(data, 0, trainSize);
		test = new Instances(data, trainSize, testSize);

		try {
		    model.buildClassifier(train);
		} catch (Exception E) {
		    E.printStackTrace();
		}

	  } else if (learningType == 2) {

		cv10 = true;
		train = new Instances(data);
		test = new Instances(data);
	  }
    }

    public static void testModel() {
	  System.out.println("## Pilih bahan testing");
	  System.out.println("## 1. Uji dengan data dari masukan training");
	  System.out.println("## 2. Uji dengan data data masukan baru");
	  System.out.print("## > ");

	  int choice = (new Scanner(System.in)).nextInt();
	  if (choice == 1) {
		try {
		    Evaluation eval = new Evaluation(train);

		    if (cv10) {
			  eval.crossValidateModel(model, test, 10, new Random(1));
		    } else {
			  eval.evaluateModel(model, test);
		    }

		    System.out.println(eval.toSummaryString());
		    System.out.println(eval.toClassDetailsString());
		    System.out.println(eval.toMatrixString());

		} catch (Exception E) {
		    E.printStackTrace();
		}
	  } else if (choice == 2) {
		try {
		    loadTestData();
		    Evaluation eval = new Evaluation(train);
		    if (cv10) {
			  eval.crossValidateModel(model, test, 10, new Random(1));
		    } else {
			  eval.evaluateModel(model, test);
		    }

		    System.out.println(eval.toSummaryString());
		    System.out.println(eval.toClassDetailsString());
		    System.out.println(eval.toMatrixString());
		} catch (Exception E) {
		    E.printStackTrace();
		}
	  }

    }

    public static void loadTestData() {

	  System.out.println("Masukkan path file dataset:");
	  System.out.print("### > ");
	  String testDataLocation = (new Scanner(System.in)).nextLine();
          testDataLocation = "/home/tegarnization/Documents/code/repo/wekaann/ML-ANN/src/ml/ann/weather.numeric.empty.arff";

	  System.out.println("Mengambil test dataset...");
	  try {
		DataSource source = new DataSource(testDataLocation);
		test = source.getDataSet();

		if (test.classIndex() == -1) {
		    test.setClassIndex(test.numAttributes() - 1);
		}
	  } catch (Exception E) {
		E.printStackTrace();
	  }

    }
}

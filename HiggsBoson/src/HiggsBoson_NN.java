import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


public class HiggsBoson_NN {
	HiggsBosonData hb = new HiggsBosonData();
	
	
	MultilayerPerceptron mlp;
	Evaluation eval;
	
	CSV predictionTest;
	
	
	
	
	public void replaceMissingWithMean(String outputFileName){
		/*
		fc = new FilteredClassifier();
		fc.setFilter(rmv);
		fc.setClassifier(mlp);
		data.randomize(new Random(1));
		Instances trainset = data.trainCV(2,1);
		trainset.setClassIndex(trainset.classIndex()-1);
		trainset.deleteAttributeAt(trainset.classIndex()+1);
		Instances testset = data.testCV(2,1);
		trainset.setClassIndex(trainset.classIndex()-1);
		trainset.deleteAttributeAt(trainset.classIndex()+1);
		fc.buildClassifier(trainset);
		*/
		try {

			hb.readTrainingFile("d:/Google Drive/SmallProjects/Kaggle/HigsBoson/training.csv", "Label", "Weight");
			hb.removeAttribute("EventId"); //remove the eventid column
			hb.removeAttribute("Weight"); //remove the eventid column

			
			
			hb.replaceMissingWithMean();

			mlp = new MultilayerPerceptron();
			mlp.setHiddenLayers("17,4");
			//mlp.setGUI(true);
			mlp.buildClassifier(hb.crossValidateSets.get(0));

			eval = new Evaluation(hb.crossValidateSets.get(0));
			eval.evaluateModel(mlp, hb.crossValidateSets.get(0));
			System.out.println("meanAbsoluteError = "+eval.meanAbsoluteError());
			System.out.println("relativeAbsoluteError = "+eval.relativeAbsoluteError());
			System.out.println("----------end of training---------");


			predictionTest = new CSV();
			predictionTest.setBuffer(new StringBuffer());
			predictionTest.setHeader(hb.crossValidateSets.get(1));
			predictionTest.printHeader();

			Object[] obs = {predictionTest};



			eval.evaluateModel(mlp, hb.crossValidateSets.get(1), obs);

			System.out.println("meanAbsoluteError = "+eval.meanAbsoluteError());
			System.out.println("relativeAbsoluteError = "+eval.relativeAbsoluteError());


		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		hb.writePredictionsToFile(predictionTest, outputFileName);
	}
	
	
	
	
	public static void main(String[] args){
		
		
	}
}

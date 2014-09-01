import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

import utils.wekaExt.InstancesManipulations;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


public class HiggsBosonRF {
	 
	HiggsBosonData hb = new HiggsBosonData();
	RandomForest rfo;
	Evaluation eval;
	CSV predictionTest;
	
	Instances weight;
	double weightThreshold;
	boolean sGrb = false;
	
	
	Instances testset;
	
	public HiggsBosonRF(String filename, String classAtt, String weightAtt){
		
		hb.readTrainingFile(filename, classAtt, weightAtt);
		
		//hb.removeAttribute(weightAtt); //remove the weight column
	}
	
	
	
	public double crossValidateRandomForest(String resultFile){
		double auc = 0;
		
		try{
			
			rfo = new RandomForest();
			String[] options = new String[6];
			options[0] ="-I";
			options[1] ="199";
			options[2] ="-K";
			options[3] ="11";
			options[4] ="-S";
			options[5] ="1";
			
			rfo.setOptions(options);
			String[] opts = rfo.getOptions();
            
            hb.twoFoldSplit();
            
            System.out.println("------------------------------------------Begin training------------------------------------------");
            int iteration = 1;
            int folds = 2;
            int seed = 1;
            Random rand = new Random(seed++);
            
            for (int i=0; i<iteration; i++){
            	hb.data.randomize(rand);
            	System.out.println("############################################Training iteration: "+i);		
            	for (int f=0; f<folds; f++){
            		Instances train = hb.data.trainCV(folds, f);
            		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~fold "+f);
            		System.out.println("Number of instances in the training set: "+train.numInstances());
        			
            		Instances test = hb.data.testCV(folds, f);
            		System.out.println("Number of instances in the testing set: "+test.numInstances());
        			

            		Instances trueWeight = hb.extractWeight(test);
				
				
            		test.deleteAttributeAt(test.attribute(hb.weightAtt).index());
            		train.deleteAttributeAt(train.attribute(hb.weightAtt).index());

				
            		predictionTest = new CSV();
            		predictionTest.setBuffer(new StringBuffer());
            		predictionTest.setHeader(test);
            		predictionTest.printHeader();

            		Object[] obs = {predictionTest};

            		rfo.buildClassifier(train);
            		eval = new Evaluation(test);
            		eval.evaluateModel(rfo, test, obs);
            		if (resultFile!=null){
            			String s = resultFile.substring(0, resultFile.lastIndexOf("."))+"_"+f+resultFile.substring(resultFile.lastIndexOf("."));
            			hb.writePredictionsToFile(predictionTest, s);
            		}
            		auc += eval.areaUnderROC((int)hb.positiveClass);
            		System.out.println("AUC = "+eval.areaUnderROC((int)hb.positiveClass));
            		System.out.println("falseNegativeRate = "+eval.falseNegativeRate((int)hb.positiveClass));
            		System.out.println("falsePositiveRate = "+eval.falsePositiveRate((int)hb.positiveClass));

            		double[][] cm = eval.confusionMatrix();			
            		System.out.println("Confusion matrix: ");
            		for (int k=0; k<cm.length; k++){
            			for (int j=0; j<cm[0].length; j++){
            				System.out.print(cm[k][j]+", ");					
            			}
            			System.out.println("");				
            		}

            		
            		System.out.println("AMS = "+hb.getAMS(rfo, test, trueWeight));
            		
            		
            	}
            }
			
			auc = auc/hb.crossValidateSets.size();

			return auc;

		}catch (Exception e) {		
			e.printStackTrace();	
			System.exit(0);			
			return -1;
		}			
		
	}
	
	
	
	
	
	public static void main(String[] args){
		HiggsBosonRF hb = new HiggsBosonRF("C:/Users/yiqing.liang/Documents/Yiqing'sDoc/Kaggle/Higgs/training.csv", "Label", "Weight");
		hb.crossValidateRandomForest(null);
		
	}
	
	
}

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


public class HiggsBosonData {
	final String eventAtt = "EventId";
	
	Instances data;
	Instances weight;
	double weightThreshold;
	boolean sGrb = false;
	
	CSV predictionTest;
	double positiveClass;
	int numOfVar;
	FilteredClassifier fc;
	ReplaceMissingValues rmv;
	
	String classAtt;
	String weightAtt;

	
	Vector<Instances> crossValidateSets = new Vector<Instances>();
	
	double[] attsMeanS; //attributes' mean of signal samples
	double[] attsMeanB; //attributes' mean of background samples
	double[] numOfMissingS;
	double[] numOfMissingB;
	
	public boolean readTrainingFile(String filename, String classAtt, String weightAtt){
		
		this.classAtt=classAtt;
	    this.weightAtt=weightAtt;
	    
		try {
			// load CSV
		    CSVLoader loader = new CSVLoader();
		    File dataFile = new File(filename);
		    
		    loader.setSource(dataFile);
		    data = loader.getDataSet();
		    this.removeAttribute(this.eventAtt);
		    
		    data.setClassIndex(data.attribute(classAtt).index());
		    
		    
			positiveClass = data.instance(0).classValue();
			System.out.println("Positive class is: "+positiveClass);
			System.out.println("Positive class label is: "+data.instance(0).value(data.attribute(classAtt).index())
					+"/"+data.instance(0).stringValue(data.attribute(classAtt).index()));
			
			System.out.println("Number of attributes: "+data.numAttributes());
			System.out.println("Number of classes: "+data.numClasses());
			System.out.println("Number of instances: "+data.numInstances());
						
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}		
	}
	
	

	public void writePredictionsToFile(CSV predictions, String filename){				
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			writer.write(predictions.getBuffer().toString());
			writer.flush();
			writer.close();						
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	public boolean readTrainingFile(String filename, int classIndex){
		DataSource source;
		try {
			// load CSV
		    CSVLoader loader = new CSVLoader();
		    File dataFile = new File(filename);
		    
		    loader.setSource(dataFile);
		    data = loader.getDataSet();
		    data.setClassIndex(classIndex);
		    
			positiveClass = data.instance(0).classValue();
			System.out.println("Positive class is: "+positiveClass);
			System.out.println("Positive class label is: "+data.instance(0).value(classIndex));
			
			removeAttribute("EventId"); //remove the eventid column
			
			
			System.out.println("Number of attributes: "+data.numAttributes());
			System.out.println("Number of classes: "+data.numClasses());
			System.out.println("Number of instances: "+data.numInstances());
						
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}		
	}
	
	public void removeAttribute(String name){
		
		data.deleteAttributeAt(data.attribute(name).index());
		
	}
	
	public double getWeightThreshold(Instances trainset){
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		attInfo.add(new Attribute(this.weightAtt));
		double minS = 999999;
		double maxS = -999999;
		double minB = 999999;
		double maxB = -999999;
		
		int weightIndex = trainset.attribute(weightAtt).index();
		
		for (Instance sample:trainset){
			Instance inst = new DenseInstance(attInfo.size());
			
			inst.setValue(0, sample.value(weightIndex));
			weight.add(inst);
			
			if (sample.classValue()==this.positiveClass){
				
				if (minS>sample.value(weightIndex))
					minS = sample.value(weightIndex);
				if (maxS<sample.value(weightIndex))
					maxS = sample.value(weightIndex);
				
			}else{
				if (minB>sample.value(weightIndex))
					minB = sample.value(weightIndex);
				if (maxB<sample.value(weightIndex))
					maxB = sample.value(weightIndex);
							
			}						
		}
			
		if (minS>maxB){
			weightThreshold = (minS+maxB)/2;
			this.sGrb = true;
		}else if (maxS<minB){
			weightThreshold = (maxS+minB)/2;
			this.sGrb = false;
		}
		return weightThreshold;
	}
	
	
	public Instances extractSingleAtt(Instances testset, String attName){
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		attInfo.add(new Attribute(attName));
		
		Instances subset = new Instances(attName, attInfo, 1);
		
		int weightIndex = testset.attribute(attName).index();
		
		for (Instance sample:testset){
			Instance inst = new DenseInstance(attInfo.size());
			inst.setValue(0, sample.value(weightIndex));
			subset.add(inst);
									
		}
			
		return subset;
	}
	
	

	public Instances extractWeight(Instances testset){
		weight = extractSingleAtt(testset, this.weightAtt);		
		
		return weight;
	}
	
	public Vector<Instances> replaceMissingWithMean(){
		
		rmv = new ReplaceMissingValues();
		try {
			rmv.setInputFormat(data);
			data = Filter.useFilter(data, rmv);
			data.randomize(new Random(1));
			Instances trainset = data.trainCV(2,1);
			
			Instances testset = data.testCV(2,1);
			
			crossValidateSets.add(trainset);
			crossValidateSets.add(testset);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return crossValidateSets;
	}
	
	public void findMeanForAttributes(){
		int numPositive = 0;
		int numNegative = 0;
		
		for (int i=0; i<data.numInstances();i++){
			Instance sample = data.get(i);
			if (sample.classValue()==this.positiveClass){
				numPositive++;
			}else{
				numNegative++;
			}
			
			for (int j=0; j<this.numOfVar; j++){
				
				if (sample.value(j)==-999){
					data.instance(i).setMissing(j);
					sample.setMissing(j);
					if (sample.classValue()==this.positiveClass){
						numOfMissingS[j]++;
					}else{
						numOfMissingB[j]++;						
					}											
				}else{
					if (sample.classValue()==this.positiveClass){
						attsMeanS[j] +=	sample.value(j);						
					}else{
						attsMeanB[j] +=	sample.value(j);						
					}
				}			
			}			
		}
		
		for (int j=0; j<this.numOfVar; j++){
			attsMeanS[j] = attsMeanS[j]/(numPositive-numOfMissingS[j]);
			attsMeanB[j] = attsMeanB[j]/(numNegative-numOfMissingB[j]);			
		}
		System.out.println("Number of positive class: "+numPositive);
		System.out.println("Mean value for the attributes in the signal samples");
		for (int j=0; j<this.numOfVar; j++){
			System.out.print(attsMeanS[j]+",");
		}
		System.out.println();
		System.out.println("Number of Negative class: "+numNegative);
		System.out.println("Mean value for the attributes in the background samples");
		for (int j=0; j<this.numOfVar; j++){
			System.out.print(attsMeanB[j]+",");
		}
		System.out.println();
	}
	
	public double evaluation(double[] predictedClass, double[] trueClass, double[] weight){
		double br = 10;
		double s = 0;
		double b = 0;
		if (this.weight==null ||
				predictedClass==null || trueClass==null)
			return -1;
		
		if ( weight.length!=predictedClass.length ||
				weight.length!=trueClass.length)
			return -2;
		

		
		for (int i=0; i<weight.length; i++){
			if (trueClass[i]==positiveClass && predictedClass[i]==positiveClass)
				s += weight[i];
			if (trueClass[i]!=positiveClass && predictedClass[i]!=positiveClass)
				b += weight[i];
		}
							
		return Math.sqrt(2*((s+b+br)*Math.log(1+(s/(b+br)))-s));
	}
	
	public double evaluation(double[] predictedClass, Instances testset, Instances trueWeight){
		double br = 10;
		double s = 0;
		double b = 0;
		if (trueWeight==null ||
				predictedClass==null || testset==null)
			return -1;
		
		if (trueWeight.numInstances()!=predictedClass.length ||
				trueWeight.numInstances()!=testset.numInstances())
			return -2;
		

		
		for (int i=0; i<weight.numInstances(); i++){
			if (testset.instance(i).classValue()==positiveClass && predictedClass[i]==positiveClass)
				s += trueWeight.instance(i).value(0);
			if (testset.instance(i).classValue()!=positiveClass && predictedClass[i]!=positiveClass)
				b += trueWeight.instance(i).value(0);
		}
							
		return Math.sqrt(2*((s+b+br)*Math.log(1+(s/(b+br)))-s));
	}
	
	
	public double getAMS(Classifier classifier, Instances testset, Instances trueWeight){
		double br = 10;
		double s = 0;
		double b = 0;
		if (trueWeight==null ||
				testset==null)
			return -1;
		
		if (trueWeight.numInstances()!=testset.numInstances())
			return -2;
		
		
		
		for (int i=0; i<weight.numInstances(); i++){
			
			try {
				if (testset.instance(i).classValue()==positiveClass && classifier.classifyInstance(testset.instance(i))==positiveClass)
					s += trueWeight.instance(i).value(0);
				if (testset.instance(i).classValue()!=positiveClass && classifier.classifyInstance(testset.instance(i))==positiveClass)
					b += trueWeight.instance(i).value(0);
			} catch (Exception e) {
				System.out.println("Exception occured when classifying the individual instances in the testset");
				e.printStackTrace();
			}
			
		}
							
		return Math.sqrt(2*((s+b+br)*Math.log(1+(s/(b+br)))-s));
	}
	
	public double evaluation(double[] predictedClass, int[] instanceIndice){
		double br = 10;
		double s = 0;
		double b = 0;
		if (this.weight==null ||
				predictedClass==null || instanceIndice==null)
			return -1;
		
		if ( weight.numInstances()!=predictedClass.length ||
				weight.numInstances()!=instanceIndice.length)
			return -2;
		

		
		for (int i=0; i<weight.numInstances(); i++){
			if (data.instance(instanceIndice[i]).classValue()==positiveClass && predictedClass[i]==positiveClass)
				s += weight.instance(i).value(0);
			if (data.instance(instanceIndice[i]).classValue()!=positiveClass && predictedClass[i]!=positiveClass)
				b += weight.instance(i).value(0);
		}
		
		
				
		return Math.sqrt(2*((s+b+br)*Math.log(1+(s/(b+br)))-s));
	}
	

	public Vector<Instances> twoFoldSplit(){
		
		
		try {
			data.randomize(new Random(1));
			Instances trainset = data.trainCV(2,1);
			
			Instances testset = data.testCV(2,1);
			
			crossValidateSets.add(trainset);
			crossValidateSets.add(testset);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return crossValidateSets;
	}
	

	
}

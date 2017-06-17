package org.alexgdev.dl4jmodels.fer;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;

import org.datavec.api.split.FileSplit;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 *  Used to create the model after 1 epoch of training
 */
public class FERModelTraining 
{
	private static final Logger log = LoggerFactory.getLogger(FERModelTraining.class);
    public static void main( String[] args ) throws IOException 
    {
    	//Allow 6Gb to be cached
    	CudaEnvironment.getInstance().getConfiguration()
        .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
        .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
        .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
        .setMaximumHostCache(6L * 1024 * 1024 * 1024L);
    	
    	final int height = 42;
        final int width = 42;
        final int channels = 1;
        int outputNum = 7; // number of output classes
        int batchSize = 200; // batch size for each epoch
        int nEpochs = 5; // number of epochs to perform
        int nrEvalEpochs = 1; //number of epochs in whic evaluation is performed
        int lastEpochNr = 1; //last epoch number from last training
        int currentEpochNr = 0; //set during training
        int rngSeed = 488; // random number seed for reproducibility
        Random random = new Random(rngSeed);
        boolean saveUpdater = true;
        File trainingSet = new File("D:/Tools/datasets/fer2013images/AugmentedTraining");
    	File testSet = new File("D:/Tools/datasets/fer2013images/AugmentedValidation");
    	
    	
    	
    	FileSplit train = new FileSplit(trainingSet, NativeImageLoader.ALLOWED_FORMATS, random);
    	FileSplit test = new FileSplit(testSet, NativeImageLoader.ALLOWED_FORMATS, random);
    	
    	//extract parent path as image label
    	ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    	ImageRecordReader imgReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
    	ImageRecordReader imgReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
    	
    	
    	
    	//Set up training data
    	imgReaderTrain.initialize(train);
    	DataSetIterator trainDataIter = new RecordReaderDataSetIterator(imgReaderTrain, batchSize, 1,outputNum);
    	//Scale pixel values to 0-1
    	DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    	scaler.fit(trainDataIter);
    	trainDataIter.setPreProcessor(scaler);
    	
    	MultipleEpochsIterator trainIter = new MultipleEpochsIterator(nEpochs, trainDataIter, 2);
    	
    	//Set up test data
    	imgReaderTest.initialize(test);

		
    	
    	DataSetIterator testDataIter = new RecordReaderDataSetIterator(imgReaderTest, batchSize, 1,outputNum);
    	//Scale pixel values to 0-1
    	scaler.fit(testDataIter);
    	testDataIter.setPreProcessor(scaler);
    	
    	File locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout.zip");

    	MultiLayerNetwork model;

    	log.info("\n\nRestoring saved model...\n\n");
    	model = ModelSerializer.restoreMultiLayerNetwork(locationToSave, true);
			
    	log.info(model.summary());
		
    	double accuracy = 0;
    	double rate = model.conf().getLearningRateByParam("b");
    	
    	
    	log.info("Rate: "+rate);
    	log.info("Evaluate model....");
    	Evaluation eval = new Evaluation(outputNum);
    	while(testDataIter.hasNext()){
				DataSet ds = testDataIter.next();
				INDArray output = model.output(ds.getFeatureMatrix(), false);
				eval.eval(ds.getLabels(), output);
    	}
    	log.info(eval.stats());

    	testDataIter.reset();
    	for(int i=0;i<nrEvalEpochs;i++){
    		
    		log.info("Train model....");
        	model.fit(trainIter);
        	trainIter.reset();
        	currentEpochNr = (i+1) * nEpochs + lastEpochNr;
        	log.info("*** Completed epoch {} ***", currentEpochNr);
        	
        	log.info("Evaluate model....");
            eval = new Evaluation(outputNum);
            while(testDataIter.hasNext()){
                DataSet ds = testDataIter.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);

            }
            log.info(eval.stats());
            
            accuracy = eval.accuracy();
            testDataIter.reset();
            currentEpochNr = (i+1) * nEpochs + lastEpochNr;
            
            locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout_epoch"+((i+1)*nEpochs+ currentEpochNr)+".zip");
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            
            if(accuracy > 0.65){
            	locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout_065.zip");
            	ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            } else if(accuracy > 0.60){
            	locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout_060.zip");
            	ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            } else if(accuracy > 0.55){
            	locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout_055.zip");
            	ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            } else if(accuracy > 0.50){
            	locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout_050.zip");
            	ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            }
            
            //Half rate after 25 epochs
            if((currentEpochNr-1)%25 ==0){
            	rate = rate/2.0;
            	FineTuneConfiguration fineTuneConf2 = new FineTuneConfiguration.Builder()
                        .learningRate(rate)
                        .build(); 
            	model = new TransferLearning.Builder(model).fineTuneConfiguration(fineTuneConf2).build();
            } 
            locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout.zip");      
            
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
    	}
    	
    	
    	
        
    }
    
}

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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 *  Used to create the model after 1 epoch of training
 */
public class FERModelCreate {

	private static final Logger log = LoggerFactory.getLogger(FERModelCreate.class);
	public static void main(String[] args) {
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
        int nEpochs = 1; // number of epochs to perform
        int iterations = 1; //number of learning iterarions
        double rate = 0.01; // learning rate
        int rngSeed = 488; // random number seed for reproducibility
        Random random = new Random(rngSeed);
    	File trainingSet = new File("D:/Tools/datasets/fer2013images/AugmentedTraining");
    	File testSet = new File("D:/Tools/datasets/fer2013images/AugmentedValidation");
    	
    	FileSplit train = new FileSplit(trainingSet, NativeImageLoader.ALLOWED_FORMATS, random);
    	FileSplit test = new FileSplit(testSet, NativeImageLoader.ALLOWED_FORMATS, random);
    	
    	//extract parent path as image label
    	ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    	ImageRecordReader imgReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
    	ImageRecordReader imgReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
    	
    	//Set up training data
    	try {
			imgReaderTrain.initialize(train);
		} catch (IOException e) {
			
			e.printStackTrace();
		}
    	
    	DataSetIterator trainDataIter = new RecordReaderDataSetIterator(imgReaderTrain, batchSize, 1,outputNum);
    	//Scale pixel values to 0-1
    	DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    	scaler.fit(trainDataIter);
    	trainDataIter.setPreProcessor(scaler);
    	
    	MultipleEpochsIterator trainIter = new MultipleEpochsIterator(nEpochs, trainDataIter, 2);
    	
    	//Set up test data
    	try {
			imgReaderTest.initialize(test);
			//imgReader.setListeners(new LogRecordListener());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	DataSetIterator testDataIter = new RecordReaderDataSetIterator(imgReaderTest, batchSize, 1,outputNum);
    	//Scale pixel values to 0-1
    	scaler.fit(testDataIter);
    	testDataIter.setPreProcessor(scaler);
    	
    	
        
    	
    	int layer = 0;
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .iterations(iterations)
                .regularization(true).l2(0.0001)
                .learningRate(rate)
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(layer++, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(32)
                        .padding(2,2)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .build())               
                .layer(layer++, new ConvolutionLayer.Builder(4, 4)
                        .nOut(32)
                        .padding(1,1)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .build())
                .layer(layer++, new ConvolutionLayer.Builder(5, 5)
                        .nOut(64)
                        .padding(2,2)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .build())
                .layer(layer++, new DenseLayer.Builder()
                		.activation(Activation.RELU)
                        .nOut(3072)
                        .build()) 
                .layer(layer++, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(outputNum)
                        .dropOut(0.5)
                        .activation(Activation.SOFTMAX)
                        .build()) 
                .setInputType(InputType.convolutional(height,width,channels)) 
                .backprop(true).pretrain(false).build();
    	
    	MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        log.info(model.summary());
        


        log.info("Train model....");
        //model.setListeners(new ScoreIterationListener(10));
        model.fit(trainIter);
    	trainIter.reset();
        
        
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(testDataIter.hasNext()){
            DataSet ds = testDataIter.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);

        }
        log.info(eval.stats());
        testDataIter.reset();
        
        File locationToSave = new File("D:/Tools/datasets/AugmentedNetwork2_dropout/AugmentNetwork2_dropout.zip");      
        boolean saveUpdater = true;                                             
        try {
			ModelSerializer.writeModel(model, locationToSave, saveUpdater);
		} catch (IOException e) {
			
			e.printStackTrace();
		}

	}

}

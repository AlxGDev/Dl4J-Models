package org.alexgdev.dl4jmodels.fer;

import java.awt.Graphics2D;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/*
 * Used for data augementation
 * Will first perform histogram normalization on images in source path then perform 10x augmentation by cropping and writing to outputpath
 */

public class ImagePreProcessing {
	
	private static final Logger log = LoggerFactory.getLogger(ImagePreProcessing.class);

	public static void main(String[] args) throws IOException {
		String sourcepath = "D:/Tools/datasets/fer2013images/Test/";
		String destpath = "D:/Tools/datasets/fer2013images/Test_Augmented/";
		List<File> images = new ArrayList<File>();
		
		for(int i = 0;i<7;i++){
			try (Stream<Path> paths = Files.walk(Paths.get(sourcepath+i))) {
				   images =  paths
				        .filter(Files::isRegularFile)
				        .map(Path::toFile)
		                .collect(Collectors.toList());
				        
				} 
				
			batchHistogramNormalization(images);
			log.info("Finished histogram eq. of class: "+i);
		} 
		
		for(int i = 0;i<7;i++){
			try (Stream<Path> paths = Files.walk(Paths.get(sourcepath+i))) {
				   images =  paths
				        .filter(Files::isRegularFile)
				        .map(Path::toFile)
		                .collect(Collectors.toList());
				        
				} 
				
			performAugmentation(images, destpath+i, ""+i);
			log.info("Finished augment of class: "+i);
		} 
		
		
		
		

	}
	private static void performAugmentation(List<File> files, String dest, String label) throws IOException{
		
		for(int i = 0; i<files.size();i++){
			File f = files.get(i);
			BufferedImage img = ImageIO.read(f);
			BufferedImage cropCenter = img.getSubimage(3, 3, 42, 42);
			BufferedImage cropTopLeft = img.getSubimage(0, 0, 42, 42);
			BufferedImage cropTopRight = img.getSubimage(6, 0, 42, 42);
			BufferedImage cropBotLeft = img.getSubimage(0, 6, 42, 42);
			BufferedImage cropBotRight= img.getSubimage(6, 6, 42, 42);
			ImageIO.write(cropCenter, "jpg", new File(dest+"/"+label+"_"+i+"_crop1.jpg"));
			ImageIO.write(cropTopLeft, "jpg", new File(dest+"/"+label+"_"+i+"_crop2.jpg"));
			ImageIO.write(cropTopRight, "jpg", new File(dest+"/"+label+"_"+i+"_crop3.jpg"));
			ImageIO.write(cropBotLeft, "jpg", new File(dest+"/"+label+"_"+i+"_crop4.jpg"));
			ImageIO.write(cropBotRight, "jpg", new File(dest+"/"+label+"_"+i+"_crop5.jpg"));
			
			ImageIO.write(horizontalflip(cropCenter), "jpg", new File(dest+"/"+label+"_"+i+"_crop1_flip.jpg"));
			ImageIO.write(horizontalflip(cropTopLeft), "jpg", new File(dest+"/"+label+"_"+i+"_crop2_flip.jpg"));
			ImageIO.write(horizontalflip(cropTopRight), "jpg", new File(dest+"/"+label+"_"+i+"_crop3_flip.jpg"));
			ImageIO.write(horizontalflip(cropBotLeft), "jpg", new File(dest+"/"+label+"_"+i+"_crop4_flip.jpg"));
			ImageIO.write(horizontalflip(cropBotRight), "jpg", new File(dest+"/"+label+"_"+i+"_crop5_flip.jpg"));
		}
	}
	
	public static BufferedImage horizontalflip(BufferedImage img) {
		int w = img.getWidth();
		int h = img.getHeight();
		BufferedImage dimg = new BufferedImage(w, h, img.getColorModel()
		.getTransparency());
		Graphics2D g = dimg.createGraphics();
		g.drawImage(img, 0, 0, w, h, w, 0, 0, h, null);
		g.dispose();
		return dimg;
	}
	
	private static void batchHistogramNormalization(List<File> images) throws IOException{
		for(File f: images){
			log.info("Histogram equ. of File: "+f.getName());
			 BufferedImage img = ImageIO.read(f);
			 img = doHistogramNormalization(img);
			 ImageIO.write(img, "jpg", f);
		}
	}
	
	private static BufferedImage doHistogramNormalization(BufferedImage bi){
		 int width =bi.getWidth();
         int height =bi.getHeight();
         int anzpixel= width*height;
         int[] histogram = new int[256];
         int[] iarray = new int[1];
         int i =0;

         //read pixel intensities into histogram
         for (int x = 1; x < width; x++) {
             for (int y = 1; y < height; y++) {
                 int valueBefore=bi.getRaster().getPixel(x, y,iarray)[0];
                 histogram[valueBefore]++;
             }
         }

         int sum =0;
      // build a Lookup table LUT containing scale factor
         float[] lut = new float[anzpixel];
         for ( i=0; i < 255; ++i )
         {
             sum += histogram[i];
             lut[i] = sum * 255 / anzpixel;
         }

         // transform image using sum histogram as a Lookup table
         for (int x = 1; x < width; x++) {
             for (int y = 1; y < height; y++) {
                 int valueBefore=bi.getRaster().getPixel(x, y,iarray)[0];
                 int valueAfter= (int) lut[valueBefore];
                 iarray[0]=valueAfter;
                  bi.getRaster().setPixel(x, y, iarray); 
             }
         }
         
         return bi;
	}
	
	

}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package skripsi;

/**
 *
 * @author user
 */
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import java.awt.Graphics2D ;

import java.util.Arrays;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.jfree.data.category.DefaultCategoryDataset;
//import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;

public class getImage extends javax.swing.JFrame {

    private ImageIcon imageIcon;
    private BufferedImage imageLabel;
    float[][] trainingData = new float[200][900];
    BackpropNeuralNetwork backpropagationNeuralNetworks = new BackpropNeuralNetwork(900, 80, 20);
    DefaultCategoryDataset dataset = new DefaultCategoryDataset();
    
    int index = 0;
    
    final float[][] trainingResults = new float[][] {
			 new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         new float[] {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ha"
                         
                         new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
			 new float[] {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "na"
                         
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         new float[] {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ca"
                         
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         new float[] {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ra"
                         
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         new float[] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ka"
                         
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         new float[] {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "da"
                         
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         new float[] {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}, // "ta"
                         
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         new float[] {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}, // "sa"
                         
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         new float[] {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}, // "wa"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         new float[] {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}, // "la"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         new float[] {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}, // "pa"
                         
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                          new float[] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}, // "dha"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}, // "ja"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}, // "ya"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}, // "nya"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}, // "ma"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}, // "ga"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}, // "ba"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}, // "tha"
                         
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},// "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, // "nga"
                         new float[] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1} // "nga"
                         
		};

    /**
     * Creates new form getImage
     */
    
    public getImage() {
        initComponents();
        this.setLocationRelativeTo(this);
    }

//    getImage() {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
//    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();
        jButton4 = new javax.swing.JButton();
        jTextField1 = new javax.swing.JTextField();
        jTextField2 = new javax.swing.JTextField();
        a = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        jButton5 = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setBackground(new java.awt.Color(51, 0, 255));

        jButton1.setText("Browse Image");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("Training");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jButton3.setText("Browse & Testing");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        jButton4.setText("Load Training");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });

        jTextField1.setText("Terdeteksi :");
        jTextField1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField1ActionPerformed(evt);
            }
        });

        a.setBackground(new java.awt.Color(255, 255, 255));

        jLabel1.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
        jLabel1.setText("PENGENALAN AKSARA JAWA ");

        jButton5.setText("Chart");
        jButton5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton5ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(19, 19, 19)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                .addComponent(a, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(jButton3, javax.swing.GroupLayout.DEFAULT_SIZE, 152, Short.MAX_VALUE))
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                .addComponent(jTextField1, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, 151, Short.MAX_VALUE)
                                .addComponent(jTextField2, javax.swing.GroupLayout.Alignment.LEADING)))
                        .addGap(32, 32, 32)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jButton4, javax.swing.GroupLayout.PREFERRED_SIZE, 115, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 115, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 115, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jButton5, javax.swing.GroupLayout.PREFERRED_SIZE, 115, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(88, 88, 88)
                        .addComponent(jLabel1)))
                .addGap(46, 46, 46))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 23, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(a, javax.swing.GroupLayout.PREFERRED_SIZE, 142, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jButton3))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jButton1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jButton2)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jButton4)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jButton5)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(1, 1, 1)
                .addComponent(jTextField2, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    public static BufferedImage toBufferedImage(Image img)
    {
        if (img instanceof BufferedImage)
        {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }
    
    public byte[] extractBytes (BufferedImage bufferedImage){
     // open image

     // get DataBufferBytes from Raster
     WritableRaster raster = bufferedImage .getRaster();
     DataBufferByte data   = (DataBufferByte) raster.getDataBuffer();

     return ( data.getData() );
    }

    public byte[] getDataImage (Mat src){

        Mat gray = new Mat();
        Mat draw = new Mat();
        Mat wide = new Mat();
    // convert the image in gray scale
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(gray, wide, 50, 150, 3, false);        
        Mat thres = new Mat();
        Imgproc.threshold(wide, thres, 0, 100, THRESH_BINARY);
        Imgproc.dilate(thres, thres, Imgproc.getStructuringElement(Imgproc.CV_SHAPE_CROSS, new Size(3,5)));
        
        Image imgLabel = HighGui.toBufferedImage(thres);
        imageLabel = toBufferedImage(imgLabel);
    
        Mat resizeimage = new Mat();
        Size sz = new Size(30,30);
        Imgproc.resize( thres, resizeimage, sz);
        Image img = HighGui.toBufferedImage(resizeimage);
        BufferedImage test = toBufferedImage(img);
        
        byte[] data =  extractBytes(test); 

        return (data);
    }
    
    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        
            System.out.println(index);
            JFileChooser chooser = new JFileChooser();
            chooser.setMultiSelectionEnabled(true);
            
            FileNameExtensionFilter filter = new FileNameExtensionFilter(
                    "JPG, GIF, and PNG Images", "jpg", "gif", "png");
            chooser.setFileFilter(filter);

            chooser.showOpenDialog(null);
            File[] files = chooser.getSelectedFiles();
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            Mat mat = Mat.eye(30, 30, CvType.CV_8UC1);
            
//            String[] path;
//            byte[] gambar = null;
            for(int i=0;i<files.length;i++){
                String path = files[i].getAbsolutePath();
                System.out.println(path);
                Mat src = Imgcodecs.imread(path);
                if (src.empty()) {
                    System.exit(0);
                }
                byte[] gambar = getDataImage(src);
                
                for(int j=0;j<gambar.length;j++){
                    trainingData[index][j] = gambar[j];
                }
                
                index = index + 1;
            }
            
            System.out.println(index);
            
    }//GEN-LAST:event_jButton1ActionPerformed

   
    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
        
		for (int iterations = 0; iterations < NeuralNetConstants.ITERATIONS; iterations++) {
	
			for (int i = 0; i < trainingResults.length; i++) {
				backpropagationNeuralNetworks.train(trainingData[i], trainingResults[i], NeuralNetConstants.LEARNING_RATE, NeuralNetConstants.MOMENTUM);
			}
	
			if ((iterations + 1) % 100 == 0) {
				System.out.println();
				for (int i = 0; i < trainingResults.length; i++) {
					float[] data = trainingData[i];
					float[] calculatedOutput = backpropagationNeuralNetworks.run(data);
					System.out.println(calculatedOutput[0]+" "+calculatedOutput[1]+" "+calculatedOutput[2]+" "+calculatedOutput[3]+" "+calculatedOutput[4]+" "+calculatedOutput[5]+" "+calculatedOutput[6]+" "+calculatedOutput[7]+" "+calculatedOutput[8]+
                                                " "+calculatedOutput[9]+" "+calculatedOutput[10]+" "+calculatedOutput[11]+" "+calculatedOutput[12]+" "+calculatedOutput[13]+" "+calculatedOutput[14]+" "+calculatedOutput[15]+" "+calculatedOutput[16]+" "+calculatedOutput[17]+
                                                " "+calculatedOutput[18]+" "+calculatedOutput[19]);
                                        
                                        dataset.addValue(1-calculatedOutput[0], "ha", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[1], "na", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[2], "ca", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[3], "ra", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[4], "ka", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[5], "da", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[6], "ta", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[7], "sa", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[8], "wa", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[9], "la", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[10], "pa", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[11], "dha", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[12], "ja", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[13], "ya", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[14], "nya", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[15], "ma", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[16], "ga", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[17], "ba", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[18], "tha", String.valueOf(iterations));
                                        dataset.addValue(1-calculatedOutput[19], "nga", String.valueOf(iterations));
                                        
                                        
				}
			}
		}
                try{
                    FileOutputStream f = new FileOutputStream(new File("training.txt"));
                    ObjectOutputStream o = new ObjectOutputStream(f);
                    
                    o.writeObject(backpropagationNeuralNetworks);
                    
                    o.close();
                    f.close();
                
                }
                catch(Exception exc){
                    exc.printStackTrace();
                    System.out.println("File not found");
                }
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        // TODO add your handling code here:
        
            JFileChooser chooser = new JFileChooser();
            FileNameExtensionFilter filter = new FileNameExtensionFilter(
                    "JPG, GIF, and PNG Images", "jpg", "gif", "png");
            chooser.setFileFilter(filter);

            chooser.showOpenDialog(null);
            String file =  chooser.getSelectedFile().getAbsolutePath();
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            Mat mat = Mat.eye(30, 30, CvType.CV_8UC1);
            
            Mat src = Imgcodecs.imread(file);
                if (src.empty()) {
                    System.exit(0);
                }
            byte[] gambar = getDataImage(src);
//            InputStream in = new ByteArrayInputStream(gambar);
//            BufferedImage bImageFromConvert = ImageIO.read(in);
            ImageIcon labelImage = new ImageIcon(imageLabel);
             a.setIcon(ResizeImage(imageLabel));
            
            
            float[] testingGambar = new float[900];

            for(int i=0;i<gambar.length;i++){
                testingGambar[i] = gambar[i];
            }
            
            System.out.println("---------------------------");
		
            float[] calculatedOutput = backpropagationNeuralNetworks.run(testingGambar);
            
            float akurasi = 0;
            int deteksi = 0;
            for(int i = 0; i < calculatedOutput.length; i++){
               if(i == 1){
                   akurasi = calculatedOutput[i];
                   deteksi = i;
               }else if(calculatedOutput[i] >= akurasi){
                   akurasi = calculatedOutput[i];
                   deteksi = i;
               }
            }
            
            switch(deteksi){
                case 0:
                    jTextField2.setText("ha");
                    System.out.print("Ha");
                    break;
                case 1:
                    jTextField2.setText("Na");
                    System.out.print("Na");
                    break;
                case 2:
                    jTextField2.setText("Ca");
                    System.out.print("Ca");
                    break;
                case 3:
                    jTextField2.setText("Ra");
                    System.out.print("Ra");
                    break;
                case 4:
                    jTextField2.setText("Ka");
                    System.out.print("Ka");
                    break;
                case 5:
                    jTextField2.setText("Da");
                    System.out.print("Da");
                    break;
                case 6:
                    jTextField2.setText("Ta");
                    System.out.print("Ta");
                    break;
                case 7:
                    jTextField2.setText("Sa");
                    System.out.print("Sa");
                    break;
                case 8:
                    jTextField2.setText("Wa");
                    System.out.print("Wa");
                    break;
                case 9:
                    jTextField2.setText("La");
                    System.out.print("La");
                    break;
                case 10:
                    jTextField2.setText("Pa");
                    System.out.print("Pa");
                    break;
                case 11:
                    jTextField2.setText("Dha");
                    System.out.print("Dha");
                    break;
                case 12:
                    jTextField2.setText("Ja");
                    System.out.print("Ja");
                    break;
                case 13:
                    jTextField2.setText("Ya");
                    System.out.print("Ya");
                    break;
                case 14:
                    jTextField2.setText("Nya");
                    System.out.print("Nya");
                    break;
                case 15:
                    jTextField2.setText("Ma");
                    System.out.print("Ma");
                    break;
                case 16:
                    jTextField2.setText("Ga");
                    System.out.print("Ga");
                    break;
                case 17:
                    jTextField2.setText("Ba");
                    System.out.print("Ba");
                    break;
                case 18:
                    jTextField2.setText("Tha");
                    System.out.print("Tha");
                    break;
                case 19:
                    jTextField2.setText("Nga");
                    System.out.print("Nga");
                    break;
                default:
                    jTextField2.setText("Tidak Terdeteksi");
                    System.out.print("Tidak Terdeteksi");
                    break;
            } 
    }//GEN-LAST:event_jButton3ActionPerformed

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
        // TODO add your handling code here:
        try {
            FileInputStream fi = new FileInputStream(new File("training.txt"));
            ObjectInputStream oi = new ObjectInputStream(fi);
            
            backpropagationNeuralNetworks = (BackpropNeuralNetwork) oi.readObject();
            
            oi.close();
            fi.close();
            
        } catch (FileNotFoundException e) {
            System.out.println("File not found");
        } catch (IOException e) {
            System.out.println("Error initializing stream");
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(getImage.class.getName()).log(Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_jButton4ActionPerformed

    private void jTextField1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jTextField1ActionPerformed

    private void jButton5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton5ActionPerformed
        // TODO add your handling code here:
        newWindow nw = new newWindow();
        nw.newScreen();
        nw.setData(dataset);
    }//GEN-LAST:event_jButton5ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(getImage.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(getImage.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(getImage.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(getImage.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        
        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new getImage().setVisible(true);
            }
        });
    }
    
    public ImageIcon ResizeImage(Image img)
    {
        Image newImg = img.getScaledInstance(a.getWidth(), a.getHeight(), Image.SCALE_SMOOTH);
        ImageIcon image = new ImageIcon(newImg);
        return image;
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel a;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JButton jButton5;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JTextField jTextField1;
    private javax.swing.JTextField jTextField2;
    // End of variables declaration//GEN-END:variables

    private Mat imread(String srcjpg, int i) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
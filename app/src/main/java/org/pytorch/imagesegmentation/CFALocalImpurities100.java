package org.pytorch.imagesegmentation;

import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.bitwise_or;
import static org.opencv.core.Core.countNonZero;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.threshold;

import android.content.Context;
import android.util.Log;

import com.chowis.jniimagepro.CFA.JNICFAImageProCW;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class CFALocalImpurities100 {

    //
    // ---------- Impurities Detection with FA detectron2.
    //
    public static String doAnalysis(Context context,
                                  File impuritiesTFModelFile,
                                  String frontAnonymizedPPLInputImgPath,
                                  String frontPPLOriginalInputPath,
                                  String frontFullFaceMaskInputPath,
                                  String frontImpuritiesResultOutputPath,
                                  String frontImpuritiesMaskOutputPath,
                                  Map<Integer, List<Integer>> frontCoordinates,
                                    boolean capturedWithFrontCamera) {

        Mat frontOriginalImg = imread(frontAnonymizedPPLInputImgPath);
        Mat frontFullFaceMask = imread(frontFullFaceMaskInputPath, IMREAD_GRAYSCALE);

        final int originalHeight = frontOriginalImg.rows();
        final int originalWidth = frontOriginalImg.cols();

        // Impurities ROIs, respective bounding Rectangles, and mask images.
        // --- ROI mask images.
        Mat faceImpuritiesROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat chinImpuritiesROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

        // --- ROI for AI inference, cropped out of the full size original image.
        Mat faceImpuritiesROICropped = new Mat();
        Mat chinImpuritiesROICropped = new Mat();
        
        Rect faceImpuritiesROIBoundRect = new Rect();
        Rect chinImpuritiesROIBoundRect = new Rect();
        
        Mat faceImpuritiesMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat chinImpurtiesMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat finalImpuritiesMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

        // ----------------------------------- 1. Use Face Mesh to Cut ROIs for Wrinkles Detection ------------------------------
        //
        // Define lists of FaceMesh key points for ROIs
        // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
        List<Integer> faceImpuritiesROIBoundShape = Collections.unmodifiableList(Arrays.asList(245, 233, 232, 231, 230, 229, 228, 117, 50, 205, 206, 203, 48, 115, 220, 45, 4, 275, 440, 344, 279, 423, 426, 425, 280, 346, 448, 449, 450, 451, 452, 453, 465));
        List<Integer> chinImpuritiesROIBoundShape = Collections.unmodifiableList(Arrays.asList(43, 202, 210, 169, 150, 149, 176, 148, 152, 377, 400, 378, 379, 394, 430, 422, 273, 335, 406, 313, 18, 83, 182, 106));

        int x = 0, y = 0;
        List<Point> faceImpuritiesROIPoints = new ArrayList<>();
        for(Integer index : faceImpuritiesROIBoundShape) {
            x = frontCoordinates.get(index).get(0).intValue();
            y = frontCoordinates.get(index).get(1).intValue();
            faceImpuritiesROIPoints.add(new Point(x, y));
        }
        MatOfPoint faceImpuritiesROIPointMat = new MatOfPoint();
        faceImpuritiesROIPointMat.fromList(faceImpuritiesROIPoints);
        List<MatOfPoint> faceImpuritiesROIContour = new ArrayList<>();
        faceImpuritiesROIContour.add(faceImpuritiesROIPointMat);

        List<Point> chinImpuritiesROIPoints = new ArrayList<>();
        for(Integer index : chinImpuritiesROIBoundShape) {
            x = frontCoordinates.get(index).get(0).intValue();
            y = frontCoordinates.get(index).get(1).intValue();
            chinImpuritiesROIPoints.add(new Point(x, y));
        }
        MatOfPoint chinImpuritiesROIPointMat = new MatOfPoint();
        chinImpuritiesROIPointMat.fromList(chinImpuritiesROIPoints);
        List<MatOfPoint> chinImpuritiesROIContour = new ArrayList<>();
        chinImpuritiesROIContour.add(chinImpuritiesROIPointMat);

        // Get face impurities ROI in front image
        drawContours(faceImpuritiesROI, faceImpuritiesROIContour, -1, new Scalar(255), FILLED);
        faceImpuritiesROIBoundRect = boundingRect(faceImpuritiesROI);
        //rectangle(faceImpuritiesROI, faceImpuritiesROIBoundRect, new Scalar(255), 10);
        cvtColor(faceImpuritiesROI, faceImpuritiesROI, COLOR_GRAY2BGR);
        bitwise_and(faceImpuritiesROI, frontOriginalImg, faceImpuritiesROI);
        faceImpuritiesROICropped = new Mat(frontOriginalImg, faceImpuritiesROIBoundRect);

        // Get chin impurities ROi in front image.
        drawContours(chinImpuritiesROI, chinImpuritiesROIContour, -1, new Scalar(255), FILLED);
        chinImpuritiesROIBoundRect = boundingRect(chinImpuritiesROI);
        cvtColor(chinImpuritiesROI, chinImpuritiesROI, COLOR_GRAY2BGR);
        bitwise_and(chinImpuritiesROI, frontOriginalImg, chinImpuritiesROI);
        chinImpuritiesROICropped = new Mat(frontOriginalImg, chinImpuritiesROIBoundRect);

        //MyUtil.saveMatToGallery(context, "face impurities ROI.jpg", "face impurities ROI cutout", faceImpuritiesROICropped);
        //MyUtil.saveMatToGallery(context, "chin impurities.jpg", "chin impurities ROI cutout", chinImpuritiesROICropped);

        // ----------------------------------- 1. AI detections ------------------------------
        // Load AI Model
        Interpreter impuritiesAIModel = new Interpreter(impuritiesTFModelFile);

        int roiHeight = 0;
        int roiWidth = 0;
        roiHeight = faceImpuritiesROICropped.height();
        roiWidth = faceImpuritiesROICropped.width();
        Mat faceImpuritiesMaskCropped = aiInference(context, impuritiesAIModel, faceImpuritiesROICropped, roiHeight, roiWidth);

        roiHeight = chinImpuritiesROICropped.height();
        roiWidth = chinImpuritiesROICropped.width();
        Mat chinImpuritiesMaskCropped = aiInference(context, impuritiesAIModel, chinImpuritiesROICropped, roiHeight, roiWidth);

        // Process mask images for front face wrinkles.
        //
        // Front under eye wrinkles.
        Mat subMask = new Mat();
        subMask = faceImpuritiesMask.submat(faceImpuritiesROIBoundRect);
        faceImpuritiesMaskCropped.copyTo(subMask);

        subMask = chinImpurtiesMask.submat(chinImpuritiesROIBoundRect);
        chinImpuritiesMaskCropped.copyTo(subMask);

        bitwise_or(faceImpuritiesMask, chinImpurtiesMask, finalImpuritiesMask);

        //MyUtil.saveMatToGallery(context, "front impurities mask.jpg", "front impurities", finalImpuritiesMask);

        // ------------------------------------------ 3. Indexing ------------------------------------------
        //
        // Impurities scoring.

        double impuritiesRaw = 0, impuritiesScore = 0;
        double fullFaceCount = countNonZero(frontFullFaceMask);
        
        if(fullFaceCount > 0) impuritiesRaw = (double) 1000 * countNonZero(finalImpuritiesMask) / fullFaceCount;
        impuritiesScore = getCFAImpuritiesLevel(impuritiesRaw, capturedWithFrontCamera);
        
        // ------------------------------------------ 4. Prepare Output ------------------------------------------
        //
        // ----- (a) ----- Save wrinkle mask images first.
        cvtColor(faceImpuritiesMask, faceImpuritiesMask, COLOR_GRAY2BGR);
        Imgcodecs.imwrite(frontImpuritiesMaskOutputPath, finalImpuritiesMask);
        
        // ----- (b) ------ Prepare input paths for mask images.
        String frontImpuritiesMaskInputPath = frontImpuritiesMaskOutputPath;
        
        // ----- (c) ----- Save images to internal storage.
        int maskB = 0;
        int maskG = 145;
        int maskR = 255;
        int contourB = -1;
        int contourG = -1;
        int contourR = -1;
        double alpha = 0.55;

        JNICFAImageProCW myCFAImgProc = new JNICFAImageProCW();
        double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(frontPPLOriginalInputPath, frontImpuritiesMaskInputPath, frontImpuritiesResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);

        // ---------- Prepare returned values.
        //
        String returnString = String.valueOf(impuritiesScore);
        
        System.out.println("Returned String for Impurities is" + returnString);

        return returnString;
    }

    private static Mat aiInference(Context context, Interpreter impuritiesModel, Mat originalImg, int originalHeight, int originalWidth){

        // ------------------------------------------ 1. AI Model  ------------------------------------------
        //
        // Allocate input and output tensors.
        impuritiesModel.allocateTensors();

        // Prepare input tensor from OpenCV image.
        int[] inputShape = impuritiesModel.getInputTensor(0).shape(); // num, height, width, channel.
        int input_height = inputShape[1];
        int input_width = inputShape[2];

        // Resize image to our FFA UNet model input size with staging.
        int max_ = (originalHeight > originalWidth) ? originalHeight : originalWidth;

        // Stage the image on a gray background.
        Rect stagingForegroundRect = new Rect();
        if(originalHeight > originalWidth) {
            stagingForegroundRect = new Rect((max_ - originalWidth) / 2, 0, originalWidth, originalHeight);
        } else {
            stagingForegroundRect = new Rect(0, (max_ - originalHeight) / 2, originalWidth, originalHeight);
        }
        double scale_factor = (double)input_height / (double)max_;
        int staging_width = (int)((double)originalWidth * scale_factor);
        int staging_height = (int)((double)originalHeight * scale_factor);

        Mat inputImg = new Mat(new Size(max_, max_), CV_8UC3, new Scalar(128, 128, 128));
        Mat foreground = new Mat(inputImg, stagingForegroundRect);
        originalImg.copyTo(foreground);

        if (max_ * max_ < input_height * input_width) {
            resize(inputImg, inputImg, new Size(input_width, input_height), 0, 0, INTER_LINEAR);
        } else if (max_ * max_ > input_height * input_width) {
            resize(inputImg, inputImg, new Size(input_width, input_height),0, 0, INTER_AREA);
        }

        MyUtil.saveMatToGallery(context, "input image", "staged image for AI input", inputImg.clone());

        cvtColor(inputImg, inputImg, COLOR_BGR2RGB);
        //inputImg.convertTo(inputImg, CV_32F, ( 1 / 127.5), -1);
        inputImg.convertTo(inputImg, CV_32F);

        long timestamp1Start = System.currentTimeMillis();
        float[][][][] input = new float[1][input_height][input_width][3];
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                double[] pixel = inputImg.get(y, x);
                input[0][y][x][0] = (float)pixel[0];
                input[0][y][x][1] = (float)pixel[1];
                input[0][y][x][2] = (float)pixel[2];
            }
        }
        long timestamp1End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp Impurities input: ", String.valueOf(timestamp1End - timestamp1Start));

        //System.out.println("First normalized pixel value: " + input[0][0][0][0]);

        // Inference.
        int numClasses = 2;
        float[][][][] output = new float[1][input_height][input_width][numClasses];

        long timestamp2Start = System.currentTimeMillis();
        impuritiesModel.run(input, output);
        long timestamp2End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp Impurities inference: ", String.valueOf(timestamp2End - timestamp2Start));

        // Post processing.
        // Get our impurities mask
        long timestamp3Start = System.currentTimeMillis();

        Mat stagedImpuritiesMask = Mat.zeros(input_height, input_width, CV_8UC1);

        int backgroundClassID = 0;
        int wrinkleClassID = 1;
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                if(output[0][y][x][wrinkleClassID] > 0.5) stagedImpuritiesMask.put(y, x, 255);
            }
        }

        MyUtil.saveMatToGallery(context, "raw impurities mask", "mask image from AI output", stagedImpuritiesMask);

        long timestamp3End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp Impurities output: ", String.valueOf(timestamp3End - timestamp3Start));

        Rect maskForegroundRect = new Rect((int)((double)stagingForegroundRect.x * scale_factor), (int)((double)stagingForegroundRect.y * scale_factor), staging_width, staging_height);
        Mat impuritiesMask = new Mat(stagedImpuritiesMask, maskForegroundRect);

        if (originalHeight * originalWidth < staging_height * staging_width) {
            resize(impuritiesMask, impuritiesMask, new Size(originalWidth, originalHeight), 0, 0, INTER_AREA);
        } else if (originalHeight * originalWidth > staging_height * staging_width) {
            resize(impuritiesMask, impuritiesMask, new Size(originalWidth, originalHeight), 0, 0, INTER_LINEAR);
        }

        // Restore detected features after image resizing. We can't retain all 255 values in resized image.
        threshold(impuritiesMask, impuritiesMask, 10, 255, THRESH_BINARY);

        return impuritiesMask;
    }

    private static double getCFAImpuritiesLevel(double pureValue, boolean capturedWithFrontCamera) {
        double[] dbNormData = new double[0];
        int nMin = 9;
        int nMax = 9;
        int index = 9;

        if(capturedWithFrontCamera) {
            dbNormData = new double[]{0.0, 0.71, 1.39, 2.07, 2.70, 3.46, 4.35, 5.68, 7.11, 9.75, 24.59};
        }

        if(!capturedWithFrontCamera) {
            dbNormData = new double[]{0.0, 0.32, 0.61, 0.88, 1.28, 1.64, 2.29, 3.03, 4.57, 6.60, 36.57};
        }

        double nReturnValue;
        for (int i = 0; i < 10; i++) {
            if (pureValue >= dbNormData[i] && pureValue < dbNormData[i + 1]) {
                index = i;
                break;
            }
        }

        if (pureValue >= dbNormData[10]) {
            nReturnValue = 99;
        } else if (pureValue <= dbNormData[0]) {
            nReturnValue = 0;
        } else {
            nMin = index * 10;
            nMax = (index + 1) * 10;
            double dbMin = dbNormData[index];
            double dbMax = dbNormData[index + 1];
            nReturnValue = (int) (nMin + (nMax - nMin) * (pureValue - dbMin) / (dbMax - dbMin));
            if (nReturnValue > 99) {
                nReturnValue = 99;
            }
        }

        return Math.round(nReturnValue);
    }
}

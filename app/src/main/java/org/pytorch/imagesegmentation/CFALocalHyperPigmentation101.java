package org.pytorch.imagesegmentation;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.countNonZero;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.threshold;

import android.content.Context;
import android.util.Log;

import com.chowis.jniimagepro.CFA.JNICFAImageProCW;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class CFALocalHyperPigmentation101 {

    //
    // ---------- Hyper-Pigmentation Detection with FA detectron2.
    //
    public static String doAnalysis(Context context,
                                                      File hyperPigmentationTFModelFile,
                                                      String frontAnonymizedUVLInputImgPath,
                                                      String leftAnonymizedUVLInputImgPath,
                                                      String rightAnonymizedUVLInputImgPath,
                                                      String frontOriginalUVLInputPath,
                                                      String leftOriginalUVLInputPath,
                                                      String rightOriginalUVLInputPath,
                                                      String frontForeheadMaskInputPath,
                                                      String frontNoseMaskInputPath,
                                                      String frontCheekMaskInputPath,
                                                      String frontFullFaceROIMaskInputPath,
                                                      String leftForeheadMaskInputPath,
                                                      String leftNoseMaskInputPath,
                                                      String leftCheekMaskInputPath,
                                                      String leftFullFaceROIMaskInputPath,
                                                      String rightForeheadMaskInputPath,
                                                      String rightNoseMaskInputPath,
                                                      String rightCheekMaskInputPath,
                                                      String rightFullFaceROIMaskInputPath,
                                                      String frontResultOutputPath,
                                                      String leftResultOutputPath,
                                                      String rightResultOutputPath,
                                                      String frontMaskOutputPath,
                                                      String leftMaskOutputPath,
                                                      String rightMaskOutputPath,
                                                      boolean hyperPigmentationSideImageEnabled,
                                                      boolean capturedWithFrontCamera) {

        Mat frontHPforeheadMask = imread(frontForeheadMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontHPnoseMask = imread(frontNoseMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontHPcheekMask = imread(frontCheekMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontFullFaceROIMask = imread(frontFullFaceROIMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);

        Mat leftHPforeheadMask = new Mat();
        Mat leftHPnoseMask = new Mat();
        Mat leftHPcheekMask = new Mat();
        Mat leftFullFaceROIMask = new Mat();

        Mat rightHPforeheadMask = new Mat();
        Mat rightHPnoseMask = new Mat();
        Mat rightHPcheekMask = new Mat();
        Mat rightFullFaceROIMask = new Mat();

        if(hyperPigmentationSideImageEnabled) {
            leftHPforeheadMask = imread(leftForeheadMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftHPnoseMask = imread(leftNoseMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftHPcheekMask = imread(leftCheekMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftFullFaceROIMask = imread(leftFullFaceROIMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);

            rightHPforeheadMask = imread(rightForeheadMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightHPnoseMask = imread(rightNoseMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightHPcheekMask = imread(rightCheekMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightFullFaceROIMask = imread(rightFullFaceROIMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        }
        // ----------------------------------- 2. AI for Full Face Detections ------------------------------
        // Load AI model
        Interpreter hyperPigmentationModel = new Interpreter(hyperPigmentationTFModelFile);
        // Used for later indexing.
        Mat frontFullHpMask = new Mat(), leftFullHpMask = new Mat(), rightFullHpMask = new Mat();

        int numImgForAnalysis = 0;
        if (hyperPigmentationSideImageEnabled) numImgForAnalysis = 3;
        else numImgForAnalysis = 1;

        if (numImgForAnalysis > 0) {
            // ----- (1) ----- Load detectron2 optimized TorchScript model.
            Log.d("ImageSegmentation", "-- SHU LI: CFA AI model loaded.");

            for (int positionCount = 0; positionCount < numImgForAnalysis; positionCount++) {
                // position 0: front
                // position 1: left
                // position 2: right
                //
                // ------------ Prepare input tensor for tensorflow lite from opencv2 Mat image.
                Mat originalImg = new Mat();
                if (positionCount == 0) {
                    originalImg = imread(frontAnonymizedUVLInputImgPath);
                }
                if (positionCount == 1) {
                    originalImg = imread(leftAnonymizedUVLInputImgPath);
                }
                if (positionCount == 2) {
                    originalImg = imread(rightAnonymizedUVLInputImgPath);
                }

                // Resize image to our CFA UNet model input size.
                final int originalHeight = originalImg.rows();
                final int originalWidth = originalImg.cols();

                Mat hpMask = aiInference(context, hyperPigmentationModel, originalImg, originalHeight, originalWidth);

                if (positionCount == 0) {
                    // front
                    frontFullHpMask = hpMask.clone();

                    //MyUtil.saveMatToGallery(context, "front hyperpigmentation", "mask containing detected front hyperpigmentation", frontFullHpMask);
                    hpMask.release();
                }
                if (positionCount == 1) {
                    // left
                    leftFullHpMask = hpMask.clone();

                    //MyUtil.saveMatToGallery(context, "left hyperpigmentation", "mask containing detected left hyperpigmentation", leftFullHpMask);
                    hpMask.release();
                }
                if (positionCount == 2) {
                    // right
                    rightFullHpMask = hpMask.clone();

                    //MyUtil.saveMatToGallery(context, "right hyperpigmentation", "mask containing detected right hyperpigmentation", rightFullHpMask);
                    hpMask.release();
                }
                originalImg.release();
            }
        }

        // ------------------------------------------ 3. Indexing ------------------------------------------
        //
        // Obtain hyperpigmentation detection for each regional ROIs and set respecitvie mask images.

        bitwise_and(frontHPforeheadMask, frontFullHpMask, frontHPforeheadMask);
        bitwise_and(frontHPnoseMask, frontFullHpMask, frontHPnoseMask);
        bitwise_and(frontHPcheekMask, frontFullHpMask, frontHPcheekMask);

        if (hyperPigmentationSideImageEnabled) {
            bitwise_and(leftHPforeheadMask, leftFullHpMask, leftHPforeheadMask);
            bitwise_and(leftHPnoseMask, leftFullHpMask, leftHPnoseMask);
            bitwise_and(leftHPcheekMask, leftFullHpMask, leftHPcheekMask);

            bitwise_and(rightHPforeheadMask, rightFullHpMask, rightHPforeheadMask);
            bitwise_and(rightHPnoseMask, rightFullHpMask, rightHPnoseMask);
            bitwise_and(rightHPcheekMask, rightFullHpMask, rightHPcheekMask);
        }

        // Hyperpigmentation scoring.
        double frontHpForeheadScore = 0, frontHpNoseScore = 0, frontHpCheekScore = 0;
        double leftHpForeheadScore = 0, leftHpNoseScore = 0, leftHpCheekScore = 0;
        double rightHpForeheadScore = 0, rightHpNoseScore = 0, rightHpCheekScore = 0;

        double frontHpForeheadRaw = 0, frontHpNoseRaw = 0, frontHpCheekRaw = 0;
        double leftHpForeheadRaw = 0, leftHpNoseRaw = 0, leftHpCheekRaw = 0;
        double rightHpForeheadRaw = 0, rightHpNoseRaw = 0, rightHpCheekRaw = 0;

        double frontTotalRaw = 0, leftTotalRaw = 0, rightTotalRaw = 0;
        double frontHpTotalScore = 0, leftHpTotalScore = 0, rightHpTotalScore = 0;
        double allImagesHpScore = 0;

        // ------- some changes
        int frontFullFaceCount = 0, leftFullFaceCount = 0, rightFullFaceCount = 0;
        frontFullFaceCount = countNonZero(frontFullFaceROIMask);
        if (hyperPigmentationSideImageEnabled) {
            leftFullFaceCount = countNonZero(leftFullFaceROIMask);
            rightFullFaceCount = countNonZero(rightFullFaceROIMask);
        }

        if (frontFullFaceCount > 0) {
            frontHpForeheadRaw = (double) 1000 * countNonZero(frontHPforeheadMask) / frontFullFaceCount;
            frontHpNoseRaw = (double) 1000 * countNonZero(frontHPnoseMask) / frontFullFaceCount;
            frontHpCheekRaw = (double) 1000 * countNonZero(frontHPcheekMask) / frontFullFaceCount;
        }

        if (hyperPigmentationSideImageEnabled) {
            if (leftFullFaceCount > 0) {
                leftHpForeheadRaw = (double) 1000 * countNonZero(leftHPforeheadMask) / leftFullFaceCount;
                leftHpNoseRaw = (double) 1000 * countNonZero(leftHPnoseMask) / leftFullFaceCount;
                leftHpCheekRaw = (double) 1000 * countNonZero(leftHPcheekMask) / leftFullFaceCount;
            }
            if (rightFullFaceCount > 0) {
                rightHpForeheadRaw = (double) 1000 * countNonZero(rightHPforeheadMask) / rightFullFaceCount;
                rightHpNoseRaw = (double) 1000 * countNonZero(rightHPnoseMask) / rightFullFaceCount;
                rightHpCheekRaw = (double) 1000 * countNonZero(rightHPcheekMask) / rightFullFaceCount;
            }
        }

        frontHpForeheadScore = getCFAHyperPigmentationLevel(frontHpForeheadRaw, capturedWithFrontCamera,"front-forehead");
        frontHpNoseScore = getCFAHyperPigmentationLevel(frontHpNoseRaw, capturedWithFrontCamera,"front-nose");
        frontHpCheekScore = getCFAHyperPigmentationLevel(frontHpCheekRaw, capturedWithFrontCamera,"front-cheek");
        if (hyperPigmentationSideImageEnabled) {
            leftHpForeheadScore = getCFAHyperPigmentationLevel(leftHpForeheadRaw, capturedWithFrontCamera,"side-forehead");
            leftHpNoseScore = getCFAHyperPigmentationLevel(leftHpNoseRaw, capturedWithFrontCamera,"side-nose");
            leftHpCheekScore = getCFAHyperPigmentationLevel(leftHpCheekRaw, capturedWithFrontCamera,"side-cheek");

            rightHpForeheadScore = getCFAHyperPigmentationLevel(rightHpForeheadRaw, capturedWithFrontCamera,"side-forehead");
            rightHpNoseScore = getCFAHyperPigmentationLevel(rightHpNoseRaw, capturedWithFrontCamera,"side-nose");
            rightHpCheekScore = getCFAHyperPigmentationLevel(rightHpCheekRaw, capturedWithFrontCamera,"side-cheek");
        }

        if (frontFullFaceCount > 0)
            frontTotalRaw = (double) 1000 * countNonZero(frontFullHpMask) / frontFullFaceCount;
        if (hyperPigmentationSideImageEnabled) {
            if (leftFullFaceCount > 0)
                leftTotalRaw = (double) 1000 * countNonZero(leftFullHpMask) / leftFullFaceCount;
            if (rightFullFaceCount > 0)
                rightTotalRaw = (double) 1000 * countNonZero(rightFullHpMask) / rightFullFaceCount;
        }

        frontHpTotalScore = getCFAHyperPigmentationLevel(frontTotalRaw, capturedWithFrontCamera,"front-total");
        if (hyperPigmentationSideImageEnabled) {
            leftHpTotalScore = getCFAHyperPigmentationLevel(leftTotalRaw, capturedWithFrontCamera,"side-total");
            rightHpTotalScore = getCFAHyperPigmentationLevel(rightTotalRaw, capturedWithFrontCamera,"side-total");
        }

        if (hyperPigmentationSideImageEnabled)
            allImagesHpScore = Math.round(0.5 * frontHpTotalScore + 0.25 * leftHpTotalScore + 0.25 * rightHpTotalScore);
        else allImagesHpScore = frontHpTotalScore;

        // ------------------------------------------ 4. Prepare Output ------------------------------------------
        //
        for (int picCount = 0; picCount < numImgForAnalysis; picCount++) {
            Mat maskImg = new Mat();
            if (picCount == 0) {
                maskImg = frontFullHpMask;
            }
            if (picCount == 1) {
                maskImg = leftFullHpMask;
            }
            if (picCount == 2) {
                maskImg = rightFullHpMask;
            }

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(maskImg, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

            Mat bgrMaskImg = new Mat();
            cvtColor(maskImg, bgrMaskImg, COLOR_GRAY2BGR);

            List<MatOfPoint> big_HP_contours = new ArrayList<>();

            // Store big and small hyper-pigmentation spots into different masks.
            //
            // 1. Extract big hyper-pigmentation areas by using contour area.
            for (int i = 0; i < contours.size(); i++) {
                double area = Imgproc.contourArea(contours.get(i), false);

                if (area >= 18000.0) {
                    big_HP_contours.add(contours.get(i));
                }
            }
            Scalar color = new Scalar(0, 255, 0);
            Scalar fill_color = new Scalar(255, 255, 255);

            // 2-1. Draw big contour areas and fill it with solid white color in new mask image "big_HP_mask".
            Mat big_HP_mask = Mat.zeros(bgrMaskImg.rows(), bgrMaskImg.cols(), CvType.CV_8UC3);
            Imgproc.drawContours(big_HP_mask, big_HP_contours, -1, fill_color, Imgproc.FILLED, Imgproc.LINE_AA);
            Imgproc.drawContours(big_HP_mask, big_HP_contours, -1, fill_color, 1, Imgproc.LINE_AA);

            // 2-2. Draw contours of big hyper-pigmentation areas in new mask image "big_HP_contours_mask"
            Mat big_HP_contours_mask = Mat.zeros(bgrMaskImg.rows(), bgrMaskImg.cols(), CvType.CV_8UC3);
            Imgproc.drawContours(big_HP_contours_mask, big_HP_contours, -1, fill_color, 5, Imgproc.LINE_AA);

            int dilationSize = 1;
            Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * dilationSize + 1, 2 * dilationSize + 1), new Point(dilationSize, dilationSize));
            Imgproc.dilate(big_HP_contours_mask, big_HP_contours_mask, element);

            Imgproc.medianBlur(big_HP_contours_mask, big_HP_contours_mask, 5);

            // 3. Extract only small contour areas and save it to new mask image "small_HP_mask".
            Mat small_HP_mask = Mat.zeros(bgrMaskImg.rows(), bgrMaskImg.cols(), CvType.CV_8UC3);
            Core.bitwise_xor(big_HP_mask, bgrMaskImg, small_HP_mask);

            // Illustrate small hyper-pigmentation areas first.
            int maskB = 13;
            int maskG = 249;
            int maskR = 255;
            int contourB = 255;
            int contourG = 255;
            int contourR = 255;
            double alpha = 0.5;

            // Save result images to internal storage using native .so library.
            JNICFAImageProCW myCFAImgProc = new JNICFAImageProCW();

            if (picCount == 0) {
                File frontMaskFile = new File(frontMaskOutputPath);
                String frontMaskFolderPath = frontMaskFile.getParent();

                File frontSmallHPMask = new File(frontMaskFolderPath, "front small HP mask.jpg");
                String frontSmallHPMaskPath = frontSmallHPMask.getAbsolutePath();

                Imgcodecs.imwrite(frontSmallHPMaskPath, small_HP_mask);
                double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(frontOriginalUVLInputPath, frontSmallHPMaskPath, frontResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            if (picCount == 1) {
                File leftMaskFile = new File(leftMaskOutputPath);
                String leftMaskFolderPath = leftMaskFile.getParent();

                File leftSmallHPMask = new File(leftMaskFolderPath, "left small HP mask.jpg");
                String leftSmallHPMaskPath = leftSmallHPMask.getAbsolutePath();

                Imgcodecs.imwrite(leftSmallHPMaskPath, small_HP_mask);
                double saveLeftRes = myCFAImgProc.CFAGetAnalyzedImgJni(leftOriginalUVLInputPath, leftSmallHPMaskPath, leftResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            if (picCount == 2) {
                File rightMaskFile = new File(rightMaskOutputPath);
                String rightMaskFolderPath = rightMaskFile.getParent();

                File rightSmallHPMask = new File(rightMaskFolderPath, "right small HP mask.jpg");
                String rightSmallHPMaskPath = rightSmallHPMask.getAbsolutePath();

                Imgcodecs.imwrite(rightSmallHPMaskPath, small_HP_mask);
                double saveRightRes = myCFAImgProc.CFAGetAnalyzedImgJni(rightOriginalUVLInputPath, rightSmallHPMaskPath, rightResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            // Illustrate big hyper-pigmentation areas.
            maskB = 255;
            maskG = 255;
            maskR = 255;
            contourB = 255;
            contourG = 255;
            contourR = 255;
            alpha = 0.65;
            if (picCount == 0) {
                File frontMaskFile = new File(frontMaskOutputPath);
                String frontMaskFolderPath = frontMaskFile.getParent();

                File frontBigHPMask = new File(frontMaskFolderPath, "front big HP mask.jpg");
                String frontBigHPMaskPath = frontBigHPMask.getAbsolutePath();

                Imgcodecs.imwrite(frontBigHPMaskPath, big_HP_contours_mask);

                double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(frontResultOutputPath, frontBigHPMaskPath, frontResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            if (picCount == 1) {
                File leftMaskFile = new File(leftMaskOutputPath);
                String leftMaskFolderPath = leftMaskFile.getParent();

                File leftBigHPMask = new File(leftMaskFolderPath, "left big HP mask.jpg");
                String leftBigHPMaskPath = leftBigHPMask.getAbsolutePath();

                Imgcodecs.imwrite(leftBigHPMaskPath, big_HP_contours_mask);
                double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(leftResultOutputPath, leftBigHPMaskPath, leftResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            if (picCount == 2) {
                File rightMaskFile = new File(rightMaskOutputPath);
                String rightMaskFolderPath = rightMaskFile.getParent();

                File rightBigHPMask = new File(rightMaskFolderPath, "right big HP mask.jpg");
                String rightBigHPMaskPath = rightBigHPMask.getAbsolutePath();

                Imgcodecs.imwrite(rightBigHPMaskPath, big_HP_contours_mask);
                double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(rightResultOutputPath, rightBigHPMaskPath, rightResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }
        }

        // Now it's time to save full face mask images.
        if(!frontMaskOutputPath.equals("")) {
            Imgcodecs.imwrite(frontMaskOutputPath, frontFullHpMask);
        }
        if (hyperPigmentationSideImageEnabled) {
            if(!frontMaskOutputPath.equals("")) {
                Imgcodecs.imwrite(leftMaskOutputPath, leftFullHpMask);
            }
            if(!rightMaskOutputPath.equals("")) {
                Imgcodecs.imwrite(rightMaskOutputPath, rightFullHpMask);
            }
        }

        // ---------- Prepare returned values.
        //
        String returnString;
        if (hyperPigmentationSideImageEnabled) {
            returnString = allImagesHpScore + "_" + frontHpTotalScore + "_" + leftHpTotalScore + "_" + rightHpTotalScore + "_" + frontTotalRaw + "_" + leftTotalRaw + "_" + rightTotalRaw + "_" + frontHpForeheadScore + "_" + frontHpNoseScore + "_" + frontHpCheekScore + "_" + leftHpForeheadScore + "_" + leftHpNoseScore + "_" + leftHpCheekScore + "_" + rightHpForeheadScore + "_" + rightHpNoseScore + "_" + rightHpCheekScore + "_" + frontHpForeheadRaw + "_" + frontHpNoseRaw + "_" + frontHpCheekRaw + "_" + leftHpForeheadRaw + "_" + leftHpNoseRaw + "_" + leftHpCheekRaw + "_" + rightHpForeheadRaw + "_" + rightHpNoseRaw + "_" + rightHpCheekRaw;
        }
        else{
            returnString = frontHpTotalScore + "_" + frontTotalRaw + "_" + frontHpForeheadScore + "_" + frontHpNoseScore + "_" + frontHpCheekScore + "_" + frontHpForeheadRaw + "_" + frontHpNoseRaw + "_" + frontHpCheekRaw;
        }

        System.out.println("Returned String for Hyper pigmentation is: " + returnString);

        return returnString;
    }

    private static double getCFAHyperPigmentationLevel(double pureValue, boolean capturedWithFrontCamera, String roiTarget) {
        double[] dbNormData = new double[0];

        int nMin = 9;
        int nMax = 9;
        int index = 9;

        if(capturedWithFrontCamera) {
            if (roiTarget == "front-total") {
                dbNormData = new double[]{0.0, 4.33, 7.25, 9.58, 11.73, 20.91, 25.66, 29.32, 35.42, 37.54, 42.41};
            } else if (roiTarget == "front-forehead") {
                dbNormData = new double[]{0.0, 1.12, 1.56, 1.62, 4.19, 4.72, 7.42, 12.64, 15.39, 20.55, 26.68};
            } else if (roiTarget == "front-nose") {
                dbNormData = new double[]{0.0, 0.35, 0.77, 0.86, 1, 1.11, 1.98, 2, 2.75, 3, 3.85};
            } else if (roiTarget == "front-cheek") {
                dbNormData = new double[]{0.0, 1.22, 2.45, 3.13, 4.58, 5.69, 6.45, 7.1, 9.56, 11.1, 12.38};
            }

            if (roiTarget == "side-total") {
                dbNormData = new double[]{0.0, 12.821321, 13.234706, 14.81414, 16.132587, 17.217378, 20.131922, 21.786065, 22.826402, 23.613348, 25.703303};
            } else if (roiTarget == "side-forehead") {
                dbNormData = new double[]{0.0, 0.82101, 1.297657, 1.413208, 1.566513, 2.245932, 2.568084, 2.977223, 3.505659, 4.142208, 4.455294};
            } else if (roiTarget == "side-nose") {
                dbNormData = new double[]{0.0, 0.35, 0.77, 0.86, 1, 1.11, 1.98, 2, 2.75, 3, 3.85};
            } else if (roiTarget == "side-cheek") {
                dbNormData = new double[]{0.0, 3.564802, 4.14065, 5.069542, 5.544579, 5.813737, 6.33323, 7.480794, 9.127053, 10.710398, 13.368757};
            }
        }

        if(!capturedWithFrontCamera) {
            if (roiTarget == "front-total") {
                dbNormData = new double[]{0.0, 5.142558, 9.486414, 11.746017, 16.262663, 23.883841, 34.32963, 43.356302, 57.796982, 77.836579, 91.965479};
            } else if (roiTarget == "front-forehead") {
                dbNormData = new double[]{0.0, 1.237224, 1.621208, 2.390192, 3.453075, 4.195253, 4.742676, 7.424211, 12.640306, 35.640412, 52.696079};
            } else if (roiTarget == "front-nose") {
                dbNormData = new double[]{0.0, 0.358296, 0.774732, 1.305716, 1.56855, 2.31474, 2.848259, 3.778007, 4.559752, 5.267005, 7.833366};
            } else if (roiTarget == "front-cheek") {
                dbNormData = new double[]{0.0, 1.594164, 2.321267, 3.540657, 4.118747, 5.29305, 6.941183, 9.336064, 12.148387, 16.342638, 29.962019};
            }

            if (roiTarget == "side-total") {
                dbNormData = new double[]{0.0, 5.227409, 9.725433, 15.741086, 17.965239, 24.896814, 28.611048, 40.696581, 51.174925, 68.204798, 87.851381};
            } else if (roiTarget == "side-forehead") {
                dbNormData = new double[]{0.0, 0.387667, 0.581211, 1.067786, 1.799664, 3.571769, 6.955434, 10.24494, 12.414781, 21.003488, 32.765565};
            } else if (roiTarget == "side-nose") {
                dbNormData = new double[]{0.0, 0.358296, 0.774732, 1.305716, 1.56855, 2.31474, 2.848259, 3.778007, 4.559752, 5.267005, 7.833366};
            } else if (roiTarget == "side-cheek") {
                dbNormData = new double[]{0.0, 2.230656, 3.926124, 5.853364, 7.17179, 8.608412, 12.23687, 15.827294, 29.276061, 34.380094, 45.900913};
            }
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

    private static Mat aiInference(Context context, Interpreter hyperPigmentationModel, Mat originalImg, int originalHeight, int originalWidth){

        // ------------------------------------------ 1. AI inference ------------------------------------------
        //
        // Allocate input and output tensors.
        hyperPigmentationModel.allocateTensors();

        // Prepare input tensor from OpenCV image.
        int[] inputShape = hyperPigmentationModel.getInputTensor(0).shape(); // num, height, width, channel.
        int input_height = inputShape[1];
        int input_width = inputShape[2];

        // ------------ Prepare input tensor for tensorflow lite from opencv2 Mat image.
        // Resize image to our CFA UNet model input size with staging.
        int max_ = (originalHeight > originalWidth) ? originalHeight : originalWidth;

        // Stage the image on a gray background.
        Rect stagingForegroundRect = new Rect();
        if(originalHeight > originalWidth) {
            stagingForegroundRect = new Rect((max_ - originalWidth) / 2, 0, originalWidth, originalHeight);
        } else {
            stagingForegroundRect = new Rect(0, (max_ - originalHeight) / 2, originalWidth, originalHeight);
        }
        double scale_factor = (double)input_height / (double)max_;

        Mat inputImg = new Mat(new Size(max_, max_), CV_8UC3, new Scalar(128, 128, 128));
        Mat foreground = inputImg.submat(stagingForegroundRect);
        originalImg.copyTo(foreground);

        if (max_ * max_ < input_height * input_width) {
            resize(inputImg, inputImg, new Size(input_width, input_height), 0, 0, INTER_LINEAR);
        } else if (max_ * max_ > input_height * input_width) {
            resize(inputImg, inputImg, new Size (input_width, input_height), 0, 0, INTER_AREA);
        }

        cvtColor(inputImg, inputImg, COLOR_BGR2RGB);
        inputImg.convertTo(inputImg, CV_32F, ( 1 / 127.5), -1);

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
        Log.println(Log.VERBOSE, "Timestamp hyperpigmentation input: ", String.valueOf(timestamp1End - timestamp1Start));

        //System.out.println("First normalized pixel value: " + input[0][0][0][0]);

        // Inference.
        // Inference.
        int numClasses = 4;
        float[][][][] output = new float[1][input_height][input_width][numClasses];

        long timestamp2Start = System.currentTimeMillis();
        hyperPigmentationModel.run(input, output);
        long timestamp2End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp hyper pigmentation inference: ", String.valueOf(timestamp2End - timestamp2Start));

        long timestamp3Start = System.currentTimeMillis();
        Mat hpStagedMask = Mat.zeros(input_height, input_width, CV_8UC1);

        int backgroundClassID = 0;
        int spotsClassID = 1;
        int hyperPigmentationClassID = 2;
        int molesClassID = 3;
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                if(output[0][y][x][spotsClassID] > 0.8) hpStagedMask.put(y, x, 255);
                if(output[0][y][x][hyperPigmentationClassID] > 0.8) hpStagedMask.put(y, x, 255);
                if(output[0][y][x][molesClassID] > 0.8) hpStagedMask.put(y, x, 255);
            }
        }

        long timestamp3End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp hyperpigmentation output: ", String.valueOf(timestamp3End - timestamp3Start));

        if (max_ * max_ > input_height * input_width) {
            resize(hpStagedMask, hpStagedMask, new Size(max_, max_), 0, 0, INTER_LINEAR);
        } else if (max_ * max_ < input_height * input_width) {
            resize(hpStagedMask, hpStagedMask, new Size(max_, max_), 0, 0, INTER_AREA);
        }

        Mat hpMask = hpStagedMask.submat(stagingForegroundRect);
        threshold(hpMask, hpMask, 100, 255, THRESH_BINARY);

        return hpMask;
    }
}

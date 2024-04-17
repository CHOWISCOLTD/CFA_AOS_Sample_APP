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

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.util.ArrayList;
import java.util.List;

import java.io.File;

public class CFALocalPigmentationSpots102 {

    //
    // ---------- Pigmentation/Spots Detection with FA detectron2.
    //
    public static String doAnalysis(Context context,
                                               File pigmentationSpotsTFModelFile,
                                               String frontAnonymizedXPLInputImgPath,
                                               String leftAnonymizedXPLInputImgPath,
                                               String rightAnonymizedXPLInputImgPath,
                                               String frontOriginalXPLImgInputPath,
                                               String leftOriginalXPLImgInputPath,
                                               String rightOriginalXPLImgInputPath,
                                               String frontResultOutputPath,
                                               String leftResultOutputPath,
                                               String rightResultOutputPath,
                                               String frontMaskOutputPath,
                                               String leftMaskOutputPath,
                                               String rightMaskOutputPath,
                                               String frontForeheadMaskInputPath,
                                               String frontNoseMaskInputPath,
                                               String frontCheekMaskInputPath,
                                               String frontChinMaskInputPath,
                                               String frontFullFaceROIMaskInputPath,
                                               String leftForeheadMaskInputPath,
                                               String leftNoseMaskInputPath,
                                               String leftCheekMaskInputPath,
                                               String leftChinMaskInputPath,
                                               String leftFullFaceROIMaskInputPath,
                                               String rightForeheadMaskInputPath,
                                               String rightNoseMaskInputPath,
                                               String rightCheekMaskInputPath,
                                               String rightChinMaskInputPath,
                                               String rightFullFaceROIMaskInputPath,
                                               boolean sideImagesEnabled,
                                               boolean capturedWithFrontCamera) {

        Mat frontSpotsforeheadMask = imread(frontForeheadMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontSpotsnoseMask = imread(frontNoseMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontSpotscheekMask = imread(frontCheekMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontFullFaceROIMask = imread(frontFullFaceROIMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat frontSpotschinMask = imread(frontChinMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);

        Mat leftSpotsforeheadMask = new Mat();
        Mat leftSpotsnoseMask = new Mat();
        Mat leftSpotscheekMask = new Mat();
        Mat leftFullFaceROIMask = new Mat();
        Mat leftSpotschinMask = new Mat();

        Mat rightSpotsforeheadMask = new Mat();
        Mat rightSpotsnoseMask = new Mat();
        Mat rightSpotscheekMask = new Mat();
        Mat rightFullFaceROIMask = new Mat();
        Mat rightSpotschinMask = new Mat();

        if(sideImagesEnabled) {
            leftSpotsforeheadMask = imread(leftForeheadMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftSpotsnoseMask = imread(leftNoseMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftSpotscheekMask = imread(leftCheekMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftFullFaceROIMask = imread(leftFullFaceROIMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            leftSpotschinMask = imread(leftChinMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);

            rightSpotsforeheadMask = imread(rightForeheadMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightSpotsnoseMask = imread(rightNoseMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightSpotscheekMask = imread(rightCheekMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightFullFaceROIMask = imread(rightFullFaceROIMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
            rightSpotschinMask = imread(rightChinMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        }

        // ----------------------------------- 2. AI for Full Face Detections ------------------------------
        // Load AI Model
        Interpreter spotsModel = new Interpreter(pigmentationSpotsTFModelFile);

        // Used for later indexing.
        Mat frontFullSpotsMask = new Mat(), leftFullSpotsMask = new Mat(), rightFullSpotsMask = new Mat();

        int numImgForAnalysis = 0;
        if (sideImagesEnabled) numImgForAnalysis = 3;
        else numImgForAnalysis = 1;

        if (numImgForAnalysis > 0) {
            // ----- (1) ----- Load detectron2 optimized TorchScript model.
            for (int positionCount = 0; positionCount < numImgForAnalysis; positionCount++) {
                Mat originalImg = new Mat();
                if (positionCount == 0) {
                    originalImg = imread(frontAnonymizedXPLInputImgPath);
                }
                if (positionCount == 1) {
                    originalImg = imread(leftAnonymizedXPLInputImgPath);
                }
                if (positionCount == 2) {
                    originalImg = imread(rightAnonymizedXPLInputImgPath);
                }

                int originalHeight = originalImg.rows();
                int originalWidth = originalImg.cols();

                Mat hpMask = aiInference(context, spotsModel, originalImg, originalHeight, originalWidth);

                if (positionCount == 0) {
                    // front
                    frontFullSpotsMask = hpMask.clone();

                    //MyUtil.saveMatToGallery(context, "front pigmentation", "mask containing detected front pigmentation", frontFullSpotsMask);
                    hpMask.release();
                }
                if (positionCount == 1) {
                    // left
                    leftFullSpotsMask = hpMask.clone();

                    //MyUtil.saveMatToGallery(context, "left pigmentation", "mask containing detected left pigmentation", leftFullSpotsMask);
                    hpMask.release();
                }
                if (positionCount == 2) {
                    // right
                    rightFullSpotsMask = hpMask.clone();

                    //MyUtil.saveMatToGallery(context, "right pigmentation", "mask containing detected right pigmentation", rightFullSpotsMask);
                    hpMask.release();
                }
                originalImg.release();
            }
        }

        // ------------------------------------------ 3. Indexing ------------------------------------------
        //
        bitwise_and(frontSpotsforeheadMask, frontFullSpotsMask, frontSpotsforeheadMask);
        bitwise_and(frontSpotsnoseMask, frontFullSpotsMask, frontSpotsnoseMask);
        bitwise_and(frontSpotscheekMask, frontFullSpotsMask, frontSpotscheekMask);
        bitwise_and(frontSpotschinMask, frontFullSpotsMask, frontSpotschinMask);

        if (sideImagesEnabled) {
            bitwise_and(leftSpotsforeheadMask, leftFullSpotsMask, leftSpotsforeheadMask);
            bitwise_and(leftSpotsnoseMask, leftFullSpotsMask, leftSpotsnoseMask);
            bitwise_and(leftSpotscheekMask, leftFullSpotsMask, leftSpotscheekMask);
            bitwise_and(leftSpotschinMask, leftFullSpotsMask, leftSpotschinMask);

            bitwise_and(rightSpotsforeheadMask, rightFullSpotsMask, rightSpotsforeheadMask);
            bitwise_and(rightSpotsnoseMask, rightFullSpotsMask, rightSpotsnoseMask);
            bitwise_and(rightSpotscheekMask, rightFullSpotsMask, rightSpotscheekMask);
            bitwise_and(rightSpotschinMask, rightFullSpotsMask, rightSpotschinMask);
        }

        // Pigmentation scoring.
        double frontSpotsForeheadScore = 0, frontSpotsNoseScore = 0, frontSpotsCheekScore = 0, frontSpotsChinScore = 0;
        double leftSpotsForeheadScore = 0, leftSpotsNoseScore = 0, leftSpotsCheekScore = 0, leftSpotsChinScore = 0;
        double rightSpotsForeheadScore = 0, rightSpotsNoseScore = 0, rightSpotsCheekScore = 0, rightSpotsChinScore = 0;

        double frontSpotsForeheadRaw = 0, frontSpotsNoseRaw = 0, frontSpotsCheekRaw = 0, frontSpotsChinRaw = 0;
        double leftSpotsForeheadRaw = 0, leftSpotsNoseRaw = 0, leftSpotsCheekRaw = 0, leftSpotsChinRaw = 0;
        double rightSpotsForeheadRaw = 0, rightSpotsNoseRaw = 0, rightSpotsCheekRaw = 0, rightSpotsChinRaw = 0;

        double frontTotalRaw = 0, leftTotalRaw = 0, rightTotalRaw = 0;
        double frontSpotsTotalScore = 0, leftSpotsTotalScore = 0, rightSpotsTotalScore = 0;
        double allImagesSpotsScore = 0;

        // ------- some changes
        int frontFullFaceCount = 0, leftFullFaceCount = 0, rightFullFaceCount = 0;
        frontFullFaceCount = countNonZero(frontFullFaceROIMask);
        if (sideImagesEnabled) {
            leftFullFaceCount = countNonZero(leftFullFaceROIMask);
            rightFullFaceCount = countNonZero(rightFullFaceROIMask);
        }

        if(frontFullFaceCount > 0) {
            frontSpotsForeheadRaw = (double) 1000 * countNonZero(frontSpotsforeheadMask) / frontFullFaceCount;
            frontSpotsNoseRaw = (double) 1000 * countNonZero(frontSpotsnoseMask) / frontFullFaceCount;
            frontSpotsCheekRaw = (double) 1000 * countNonZero(frontSpotscheekMask) / frontFullFaceCount;
            frontSpotsChinRaw = (double) 1000 * countNonZero(frontSpotschinMask) / frontFullFaceCount;
        }

        if (sideImagesEnabled) {
            if(leftFullFaceCount > 0) {
                leftSpotsForeheadRaw = (double) 1000 * countNonZero(leftSpotsforeheadMask) / leftFullFaceCount;
                leftSpotsNoseRaw = (double) 1000 * countNonZero(leftSpotsnoseMask) / leftFullFaceCount;
                leftSpotsCheekRaw = (double) 1000 * countNonZero(leftSpotscheekMask) / leftFullFaceCount;
                leftSpotsChinRaw = (double) 1000 * countNonZero(leftSpotschinMask) / frontFullFaceCount;
            }

            if (rightFullFaceCount > 0) {
                rightSpotsForeheadRaw = (double) 1000 * countNonZero(rightSpotsforeheadMask) / rightFullFaceCount;
                rightSpotsNoseRaw = (double) 1000 * countNonZero(rightSpotsnoseMask) / rightFullFaceCount;
                rightSpotsCheekRaw = (double) 1000 * countNonZero(rightSpotscheekMask) / rightFullFaceCount;
                rightSpotsChinRaw = (double) 1000 * countNonZero(rightSpotschinMask) / frontFullFaceCount;
            }
        }

        frontSpotsForeheadScore = getCFAPigmentationSpotsLevel(frontSpotsForeheadRaw, capturedWithFrontCamera, "front-forehead");
        frontSpotsNoseScore = getCFAPigmentationSpotsLevel(frontSpotsNoseRaw, capturedWithFrontCamera, "front-nose");
        frontSpotsCheekScore = getCFAPigmentationSpotsLevel(frontSpotsCheekRaw, capturedWithFrontCamera, "front-cheek");
        frontSpotsChinScore = getCFAPigmentationSpotsLevel(frontSpotsChinRaw, capturedWithFrontCamera, "front-chin");

        if (sideImagesEnabled) {
            leftSpotsForeheadScore = getCFAPigmentationSpotsLevel(leftSpotsForeheadRaw, capturedWithFrontCamera, "side-forehead");
            leftSpotsNoseScore = getCFAPigmentationSpotsLevel(leftSpotsNoseRaw, capturedWithFrontCamera,"side-nose");
            leftSpotsCheekScore = getCFAPigmentationSpotsLevel(leftSpotsCheekRaw, capturedWithFrontCamera,"side-cheek");
            leftSpotsChinScore = getCFAPigmentationSpotsLevel(leftSpotsChinRaw, capturedWithFrontCamera,"side-chin");

            rightSpotsForeheadScore = getCFAPigmentationSpotsLevel(rightSpotsForeheadRaw, capturedWithFrontCamera,"side-forehead");
            rightSpotsNoseScore = getCFAPigmentationSpotsLevel(rightSpotsNoseRaw, capturedWithFrontCamera,"side-nose");
            rightSpotsCheekScore = getCFAPigmentationSpotsLevel(rightSpotsCheekRaw, capturedWithFrontCamera,"side-cheek");
            rightSpotsChinScore = getCFAPigmentationSpotsLevel(rightSpotsChinRaw, capturedWithFrontCamera,"side-chin");
        }

        if (frontFullFaceCount > 0) frontTotalRaw = (double) 1000 * countNonZero(frontFullSpotsMask) / frontFullFaceCount;
        if (sideImagesEnabled) {
            if (leftFullFaceCount > 0) leftTotalRaw = (double) 1000 * countNonZero(leftFullSpotsMask) / leftFullFaceCount;
            if (rightFullFaceCount > 0) rightTotalRaw = (double) 1000 * countNonZero(rightFullSpotsMask) / rightFullFaceCount;
        }

        frontSpotsTotalScore = getCFAPigmentationSpotsLevel(frontTotalRaw, capturedWithFrontCamera,"front-total");
        if (sideImagesEnabled) {
            leftSpotsTotalScore = getCFAPigmentationSpotsLevel(leftTotalRaw, capturedWithFrontCamera,"side-total");
            rightSpotsTotalScore = getCFAPigmentationSpotsLevel(rightTotalRaw, capturedWithFrontCamera,"side-total");
        }

        if (sideImagesEnabled) {
            allImagesSpotsScore = Math.round(0.5 * frontSpotsTotalScore + 0.25 * leftSpotsTotalScore + 0.25 * rightSpotsTotalScore);
        }
        else allImagesSpotsScore = frontSpotsTotalScore;

        // ------------------------------------------ 4. Prepare Output ------------------------------------------
        //
        for (int picCount = 0; picCount < numImgForAnalysis; picCount++) {
            Mat maskImg = new Mat();
            if (picCount == 0) {
                maskImg = frontFullSpotsMask;
            }
            if (picCount == 1) {
                maskImg = leftFullSpotsMask;
            }
            if (picCount == 2) {
                maskImg = rightFullSpotsMask;
            }

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(maskImg, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

            Mat bgrMaskImg = new Mat();
            cvtColor(maskImg, bgrMaskImg, COLOR_GRAY2BGR);

            Scalar color = new Scalar(0, 255, 0);
            Imgproc.drawContours(bgrMaskImg, contours, -1, color, 2, Imgproc.LINE_AA);

            int maskB = 105;
            int maskG = 251;
            int maskR = 100;
            int contourB = 10;
            int contourG = 199;
            int contourR = 68;
            double alpha = 0.55;

            // Save result images to internal storage using native .so library.
            JNICFAImageProCW myCFAImgProc = new JNICFAImageProCW();

            if (picCount == 0) {
                Imgcodecs.imwrite(frontMaskOutputPath, bgrMaskImg);
                double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(frontOriginalXPLImgInputPath, frontMaskOutputPath, frontResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            if (picCount == 1) {
                Imgcodecs.imwrite(leftMaskOutputPath, bgrMaskImg);
                double saveLeftRes = myCFAImgProc.CFAGetAnalyzedImgJni(leftOriginalXPLImgInputPath, leftMaskOutputPath, leftResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }

            if (picCount == 2) {
                Imgcodecs.imwrite(rightMaskOutputPath, bgrMaskImg);
                double saveRightRes = myCFAImgProc.CFAGetAnalyzedImgJni(rightOriginalXPLImgInputPath, rightMaskOutputPath, rightResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            }
        }

        String returnString;
        if (sideImagesEnabled) {
            returnString = String.valueOf(allImagesSpotsScore) + "_" + String.valueOf(frontSpotsTotalScore) + "_" + String.valueOf(leftSpotsTotalScore) + "_" + String.valueOf(rightSpotsTotalScore) + "_" + String.valueOf(frontTotalRaw) + "_" + String.valueOf(leftTotalRaw) + "_" + String.valueOf(rightTotalRaw) + "_" + String.valueOf(frontSpotsForeheadScore) + "_" + String.valueOf(frontSpotsNoseScore) + "_" + String.valueOf(frontSpotsCheekScore) + "_" + String.valueOf(frontSpotsChinScore) + "_" + String.valueOf(leftSpotsForeheadScore) + "_" + String.valueOf(leftSpotsNoseScore) + "_" + String.valueOf(leftSpotsCheekScore) + "_" + String.valueOf(leftSpotsChinScore) + "_" + String.valueOf(rightSpotsForeheadScore) + "_" + String.valueOf(rightSpotsNoseScore) + "_" + String.valueOf(rightSpotsCheekScore) + "_" + String.valueOf(rightSpotsChinScore) + "_" + String.valueOf(frontSpotsForeheadRaw) + "_" + String.valueOf(frontSpotsNoseRaw) + "_" + String.valueOf(frontSpotsCheekRaw) + "_" + String.valueOf(frontSpotsChinRaw) + "_" + String.valueOf(leftSpotsForeheadRaw) + "_" + String.valueOf(leftSpotsNoseRaw) + "_" + String.valueOf(leftSpotsCheekRaw) + "_" + String.valueOf(leftSpotsChinRaw) + "_" + String.valueOf(rightSpotsForeheadRaw) + "_" + String.valueOf(rightSpotsNoseRaw) + "_" + String.valueOf(rightSpotsCheekRaw) + "_" + String.valueOf(rightSpotsChinRaw);
        }
        else{
            returnString = String.valueOf(frontSpotsTotalScore) + "_" + String.valueOf(frontTotalRaw) + "_" + String.valueOf(frontSpotsForeheadScore) + "_" + String.valueOf(frontSpotsNoseScore) + "_" + String.valueOf(frontSpotsCheekScore) + "_" + String.valueOf(frontSpotsChinScore) + "_" + String.valueOf(frontSpotsForeheadRaw) + "_" + String.valueOf(frontSpotsNoseRaw) + "_" + String.valueOf(frontSpotsCheekRaw) + "_" + String.valueOf(frontSpotsChinRaw);
        }

        System.out.println("Returned String for pigmentation/spots is: " + returnString);

        return returnString;
    }

    private static double getCFAPigmentationSpotsLevel(double pureValue, boolean capturedWithFrontCamera, String roiTarget) {
        double[] dbNormData = new double[0];

        int nMin = 9;
        int nMax = 9;
        int index = 9;

        if(capturedWithFrontCamera) {
            if (roiTarget == "front-total") {
                dbNormData = new double[]{0.0, 9.81004, 11.212098, 11.850056, 13.50031, 13.846871, 17.430171, 24.115828, 28.859604, 32.308911, 36.633605};
            } else if (roiTarget == "front-forehead") {
                dbNormData = new double[]{0.0, 1.287052, 1.374578, 1.567644, 2.065558, 2.269959, 2.531744, 2.806566, 3.15691, 3.821448, 5.082582};
            } else if (roiTarget == "front-nose") {
                dbNormData = new double[]{0.0, 0.17853, 0.209303, 0.250306, 0.300659, 0.52026, 0.760867, 0.846806, 0.920014, 1.593793, 1.991217};
            } else if (roiTarget == "front-cheek") {
                dbNormData = new double[]{0.0, 3.189254, 4.401813, 4.4978, 5.259444, 5.695834, 7.92847, 12.738561, 14.688897, 17.701867, 23.825848};
            } else if (roiTarget == "front-chin") {
                dbNormData = new double[]{0.0, 0.134129, 0.251747, 0.455506, 0.596522, 0.762024, 0.840177, 0.891715, 1.051434, 1.55042, 1.799673};
            }

            if (roiTarget == "side-total") {
                dbNormData = new double[]{0.0, 5.760981, 7.384695, 8.014382, 9.271208, 9.958474, 13.115017, 14.783931, 17.197987, 19.983544, 24.269074};
            } else if (roiTarget == "side-forehead") {
                dbNormData = new double[]{0.0, 0.302925, 0.398842, 0.72376, 0.928723, 1.080273, 1.358766, 1.703198, 1.934998, 2.3904, 3.377047};
            } else if (roiTarget == "side-nose") {
                dbNormData = new double[]{0.0, 0.17853, 0.209303, 0.250306, 0.300659, 0.52026, 0.760867, 0.846806, 0.920014, 1.593793, 1.991217};
            } else if (roiTarget == "side-cheek") {
                dbNormData = new double[]{0.0, 1.640432, 2.494174, 3.09198, 3.702652, 4.341484, 5.36428, 6.305319, 8.301182, 10.467075, 15.851096};
            } else if (roiTarget == "side-chin") {
                dbNormData = new double[]{0.0, 0.134129, 0.251747, 0.455506, 0.596522, 0.762024, 0.840177, 0.891715, 1.051434, 1.55042, 1.799673};
            }
        }

        if(!capturedWithFrontCamera) {
            if (roiTarget == "front-total") {
                dbNormData = new double[]{0.0, 8.079405, 9.867466, 11.277039, 14.378839, 18.117439, 19.540324, 22.337321, 25.923478, 36.295541, 49.200543};
            } else if (roiTarget == "front-forehead") {
                dbNormData = new double[]{0.0, 1.051123, 2.112205, 2.840434, 3.223347, 4.458261, 5.250051, 5.692394, 7.438516, 0.356838, 18.663884};
            } else if (roiTarget == "front-nose") {
                dbNormData = new double[]{0.0, 1.051123, 2.112205, 2.840434, 3.223347, 4.458261, 5.25005, 5.692394, 7.438516, 9.356838, 18.663884};
            } else if (roiTarget == "front-cheek") {
                dbNormData = new double[]{0.0, 2.169072, 2.930527, 4.426592, 5.72898, 6.656355, 7.985456, 8.640588, 10.539634, 12.425758, 18.785731};
            } else if (roiTarget == "front-chin") {
                dbNormData = new double[]{0.0, 0.23, 0.35, 0.57, 0.69, 0.78, 1.12, 1.25, 1.85, 2.51, 3.65};
            }

            if (roiTarget == "side-total") {
                dbNormData = new double[]{0.0, 7.577244, 11.256, 13.024029, 14.147996, 15.108464, 17.655747, 0.113937, 23.610269, 27.430859, 42.965587};
            } else if (roiTarget == "side-forehead") {
                dbNormData = new double[]{0.0, 1.14586, 1.584476, 1.881863, 2.333233, 2.847809, 3.331612, 4.131268, 5.012135, 6.677653, 10.226537};
            } else if (roiTarget == "side-nose") {
                dbNormData = new double[]{0.0, 0.251817, 0.274772, 0.297727, 0.320682, 0.343638, 0.366593, 0.389548, 0.412503, 0.435458, 0.458414};
            } else if (roiTarget == "side-cheek") {
                dbNormData = new double[]{0.0, 2.771463, 3.94627, 5.003526, 5.748573, 6.885507, 7.872862, 8.845533, 10.004986, 12.777821, 25.681578};
            } else if (roiTarget == "side-chin") {
                dbNormData = new double[]{0.0, 0.23, 0.35, 0.57, 0.69, 0.78, 1.12, 1.25, 1.85, 2.51, 3.65};
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

    private static Mat aiInference(Context context, Interpreter spotsModel, Mat originalImg, int originalHeight, int originalWidth){

        // ------------------------------------------ 1. AI inference ------------------------------------------
        //
        // Allocate input and output tensors.
        spotsModel.allocateTensors();

        // Prepare input tensor from OpenCV image.
        int[] inputShape = spotsModel.getInputTensor(0).shape(); // num, height, width, channel.
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
        Log.println(Log.VERBOSE, "TS spots input: ", String.valueOf(timestamp1End - timestamp1Start));

        //System.out.println("First normalized pixel value: " + input[0][0][0][0]);

        // Inference.
        int numClasses = 7;
        float[][][][] output = new float[1][input_height][input_width][numClasses];

        long timestamp2Start = System.currentTimeMillis();
        spotsModel.run(input, output);
        long timestamp2End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "TS spots inference: ", String.valueOf(timestamp2End - timestamp2Start));

        long timestamp3Start = System.currentTimeMillis();
        Mat spotsStagedMask = Mat.zeros(input_height, input_width, CV_8UC1);

        int backgroundClassID = 0;
        int acneSpotsClassID = 1;
        int acneClassID = 2;
        int spotsClassID = 3;
        int molesClassID = 4;
        int scarClassID = 5;
        int skinTagsClassID = 6;
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                if(output[0][y][x][acneSpotsClassID] > 0.8) spotsStagedMask.put(y, x, 255);
                if(output[0][y][x][acneClassID] > 0.8) spotsStagedMask.put(y, x, 255);
                if(output[0][y][x][spotsClassID] > 0.8) spotsStagedMask.put(y, x, 255);
                if(output[0][y][x][molesClassID] > 0.8) spotsStagedMask.put(y, x, 255);
                if(output[0][y][x][scarClassID] > 0.8) spotsStagedMask.put(y, x, 255);
                if(output[0][y][x][skinTagsClassID] > 0.8) spotsStagedMask.put(y, x, 255);
            }
        }

        long timestamp3End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "TS spots output: ", String.valueOf(timestamp3End - timestamp3Start));

        if (max_ * max_ > input_height * input_width) {
            resize(spotsStagedMask, spotsStagedMask, new Size(max_, max_), 0, 0, INTER_LINEAR);
        } else if (max_ * max_ < input_height * input_width) {
            resize(spotsStagedMask, spotsStagedMask, new Size(max_, max_), 0, 0, INTER_AREA);
        }

        Mat spotsMask = spotsStagedMask.submat(stagingForegroundRect);
        threshold(spotsMask, spotsMask, 100, 255, THRESH_BINARY);

        return spotsMask;
    }
}

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
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.rectangle;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class CFALocalWrinkle101 {

    //
    // ---------- Wrinkles Detection with FA detectron2.
    //
    public static String doAnalysis(Context context,
                                    File wrinklesTFModelFile,
                                    String frontAnonymizedPPLInputImgPath,
                                    String leftAnonymizedPPLInputImgPath,
                                    String rightAnonymizedPPLInputImgPath,
                                    String frontOriginalInputPath,
                                    String leftOriginalInputPath,
                                    String rightOriginalInputPath,
                                    String frontFullFaceMaskInputPath,
                                    String leftFullFaceMaskInputPath,
                                    String rightFullFaceMaskInputPath,
                                    String frontForeheadMaskInputPath,
                                    String leftForeheadMaskInputPath,
                                    String rightForeheadMaskInputPath,
                                    String frontResultOutputPath,
                                    String leftResultOutputPath,
                                    String rightResultOutputPath,
                                    String frontMaskOutputPath,
                                    String leftMaskOutputPath,
                                    String rightMaskOutputPath,
                                    Map<Integer, List<Integer>> frontCoordinates,
                                    Map<Integer, List<Integer>> leftCoordinates,
                                    Map<Integer, List<Integer>> rightCoordinates,
                                    boolean capturedWithFrontCamera,
                                    boolean sideFaceImagesEnabled) {

        Mat frontFullFaceMask = Imgcodecs.imread(frontFullFaceMaskInputPath, IMREAD_GRAYSCALE);
        Mat leftFullFaceMask = new Mat();
        Mat rightFullFaceMask = new Mat();

        if(sideFaceImagesEnabled) {
            leftFullFaceMask = Imgcodecs.imread(leftFullFaceMaskInputPath, IMREAD_GRAYSCALE);
            rightFullFaceMask = Imgcodecs.imread(rightFullFaceMaskInputPath, IMREAD_GRAYSCALE);
        }

        Mat frontOriginalImg = imread(frontAnonymizedPPLInputImgPath);
        Mat leftOriginalImg = new Mat();
        Mat rightOriginalImg = new Mat();

        Mat frontForeheadMask = imread(frontForeheadMaskInputPath, IMREAD_GRAYSCALE);
        Mat leftForeheadMask = new Mat();
        Mat rightForeheadMask = new Mat();

        if(sideFaceImagesEnabled) {
            leftOriginalImg = imread(leftAnonymizedPPLInputImgPath);
            leftForeheadMask = imread(leftForeheadMaskInputPath, IMREAD_GRAYSCALE);

            rightOriginalImg = imread(rightAnonymizedPPLInputImgPath);
            rightForeheadMask = imread(rightForeheadMaskInputPath, IMREAD_GRAYSCALE);
        }

        final int originalHeight = frontFullFaceMask.rows();
        final int originalWidth = frontFullFaceMask.cols();

        // Front image wrinkles ROIs, respective bounding Rectangles, and mask images.
        // --- ROI mask images.
        Mat frontLeftUnderEyeWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontRightUnderEyeWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontLeftSmileLineWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontRightSmileLineWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontLionWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontForeheadWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontEyesROIMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontSmileLinesROIMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

        // --- ROI for AI inference, cropped out of the full size original image.
        Mat frontLeftUnderEyeWrinkleCropped = new Mat();
        Mat frontRightUnderEyeWrinkleCropped = new Mat();
        Mat frontLeftSmileLineWrinkleCropped = new Mat();
        Mat frontRightSmileLineWrinkleCropped = new Mat();
        Mat frontLionWrinkleCropped = new Mat();
        Mat frontForeheadWrinkleCropped = new Mat();

        Rect frontRightUnderEyeWrinkleBoundRect = new Rect();
        Rect frontLeftUnderEyeWrinkleBoundRect = new Rect();
        Rect frontRightSmileWrinkleBoundRect = new Rect();
        Rect frontLeftSmileWrinkleBoundRect = new Rect();
        Rect frontLionWrinkleBoundRect = new Rect();
        Rect frontForeheadWrinkleBoundRect = new Rect();

        Mat frontEyeWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontForeheadWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontLionWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontSmileWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        Mat frontWrinkleFinalMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

        // Left image wrinkles ROIs, respective bounding Rectangles, and mask images.
        // --- ROI mask images.
        Mat leftEyeCornerWrinkleROI = new Mat();
        Mat leftForeheadWrinkleROI = new Mat();
        Mat leftSmileLineROI = new Mat();

        // --- ROI for AI inference, cropped out of the full size original image.
        Mat leftEyeCornerWrinkleCropped = new Mat();
        Mat leftForeheadWrinkleCropped = new Mat();
        Mat leftSmileLineROICropped = new Mat();

        Rect leftEyeWrinkleBoundRect = new Rect();
        Rect leftForeheadWrinkleBoundRect = new Rect();
        Rect leftSmileLineROIBoundRect = new Rect();

        Mat leftEyeWrinkleMask = new Mat();
        Mat leftForeheadWrinkleMask = new Mat();
        Mat leftSmileLineMask = new Mat();
        Mat leftWrinkleFinalMask = new Mat();

        // Right image wrinkles ROIs, respective bounding Rectangles, and mask images.
        // --- ROI mask imagres.
        Mat rightEyeCornerWrinkleROI = new Mat();
        Mat rightForeheadWrinkleROI = new Mat();
        Mat rightSmileLineROI = new Mat();

        // --- ROI for AI inference, cropped out of the full size original image.
        Mat rightEyeCornerWrinkleCropped = new Mat();
        Mat rightForeheadWrinkleCropped = new Mat();
        Mat rightSmileLineROICropped = new Mat();

        Rect rightEyeWrinkleBoundRect = new Rect();
        Rect rightForeheadWrinkleBoundRect = new Rect();
        Rect rightSmileLineROIBoundRect = new Rect();

        Mat rightEyeWrinkleMask = new Mat();
        Mat rightForeheadWrinkleMask = new Mat();
        Mat rightSmileLineMask = new Mat();
        Mat rightWrinkleFinalMask = new Mat();

        if(sideFaceImagesEnabled) {
            // Left
            leftEyeCornerWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            leftForeheadWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            leftSmileLineROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

            leftEyeWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            leftForeheadWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            leftSmileLineMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            leftWrinkleFinalMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

            // Right
            rightEyeCornerWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            rightForeheadWrinkleROI = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            rightSmileLineROI =  Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);

            rightEyeWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            rightForeheadWrinkleMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            rightSmileLineMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
            rightWrinkleFinalMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
        }

        int frontImage = 0;
        int leftImage = 1;
        int rightImage = 2;

        int imgNum = 0;
        if(sideFaceImagesEnabled) imgNum = 3;
        else imgNum = 1;

        // ----------------------------------- 1. Use Face Mesh to Cut ROIs for Wrinkles Detection ------------------------------
        //
        for (int position = 0; position < imgNum; position++) {
            // position 0: front
            // position 1: left
            // position 2: right
            if(position == frontImage) {
                // Define lists of FaceMesh key points for ROIs
                // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
                List<Integer> rightUnderEyeBoundShape = Collections.unmodifiableList(Arrays.asList(112, 26, 22, 23, 24, 110, 25, 31, 117, 50, 205, 36, 142, 126, 47, 128));
                List<Integer> leftUnderEyeBoundShape = Collections.unmodifiableList(Arrays.asList(341, 256, 252, 253, 254, 339, 255, 261, 346, 280, 425, 266, 371, 355, 277, 357));
                List<Integer> rightSmileLineBoundShape = Collections.unmodifiableList(Arrays.asList(142, 36, 205, 207, 214, 210, 202, 57,  186, 92, 98));
                List<Integer> leftSmileLineBoundShape = Collections.unmodifiableList(Arrays.asList(371, 266, 425, 427, 434, 430, 422, 287, 410, 322, 327));
                List<Integer> frontLionWrinkleBoundShape = Collections.unmodifiableList(Arrays.asList(108, 107, 55, 189, 244, 245, 122, 6, 351, 465, 464, 463, 413, 285, 336, 337, 151));

                int x = 0, y = 0;
                ////////
                List<Point> rightUnderEyePoints = new ArrayList<>();
                for(Integer index : rightUnderEyeBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    rightUnderEyePoints.add(new Point(x, y));
                }
                MatOfPoint rightUnderEyePointMat = new MatOfPoint();
                rightUnderEyePointMat.fromList(rightUnderEyePoints);
                List<MatOfPoint> rightUnderEyeContour = new ArrayList<>();
                rightUnderEyeContour.add(rightUnderEyePointMat);

                ////////
                List<Point> leftUnderEyePoints = new ArrayList<>();
                for(Integer index : leftUnderEyeBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    leftUnderEyePoints.add(new Point(x, y));
                }
                MatOfPoint leftUnderEyePointMat = new MatOfPoint();
                leftUnderEyePointMat.fromList(leftUnderEyePoints);
                List<MatOfPoint> leftUnderEyeContour = new ArrayList<>();
                leftUnderEyeContour.add(leftUnderEyePointMat);

                ////////
                List<Point> rightSmileLinePoints = new ArrayList<>();
                for(Integer index : rightSmileLineBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    rightSmileLinePoints.add(new Point(x, y));
                }
                MatOfPoint rightSmileLinePointMat = new MatOfPoint();
                rightSmileLinePointMat.fromList(rightSmileLinePoints);
                List<MatOfPoint> rightSmileLineContour = new ArrayList<>();
                rightSmileLineContour.add(rightSmileLinePointMat);

                ////////
                List<Point> leftSmileLinePoints = new ArrayList<>();
                for(Integer index : leftSmileLineBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    leftSmileLinePoints.add(new Point(x, y));
                }
                MatOfPoint leftSmileLinePointMat = new MatOfPoint();
                leftSmileLinePointMat.fromList(leftSmileLinePoints);
                List<MatOfPoint> leftSmileLineContour = new ArrayList<>();
                leftSmileLineContour.add(leftSmileLinePointMat);

                ////////
                List<Point> frontLionPoints = new ArrayList<>();
                for(Integer index : frontLionWrinkleBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    frontLionPoints.add(new Point(x, y));
                }
                MatOfPoint frontLionPointMat = new MatOfPoint();
                frontLionPointMat.fromList(frontLionPoints);
                List<MatOfPoint> frontLionWrinkleContour = new ArrayList<>();
                frontLionWrinkleContour.add(frontLionPointMat);

                // Get right eye under-eye wrinkles ROI in front image
                drawContours(frontRightUnderEyeWrinkleROI, rightUnderEyeContour, -1, new Scalar(255), FILLED);
                frontRightUnderEyeWrinkleBoundRect = boundingRect(frontRightUnderEyeWrinkleROI);
                //rectangle(frontRightUnderEyeWrinkleROI, frontRightUnderEyeWrinkleBoundRect, new Scalar(255), 10);
                cvtColor(frontRightUnderEyeWrinkleROI, frontRightUnderEyeWrinkleROI, COLOR_GRAY2BGR);
                bitwise_and(frontRightUnderEyeWrinkleROI, frontOriginalImg, frontRightUnderEyeWrinkleROI);
                frontRightUnderEyeWrinkleCropped = new Mat(frontOriginalImg, frontRightUnderEyeWrinkleBoundRect);

                // Get left eye under-eye wrinkles ROI in front image
                drawContours(frontLeftUnderEyeWrinkleROI, leftUnderEyeContour, -1, new Scalar(255), FILLED);
                frontLeftUnderEyeWrinkleBoundRect = boundingRect(frontLeftUnderEyeWrinkleROI);
                //rectangle(frontLeftUnderEyeWrinkleROI, frontLeftUnderEyeWrinkleBoundRect, new Scalar(255), 10);
                cvtColor(frontLeftUnderEyeWrinkleROI, frontLeftUnderEyeWrinkleROI, COLOR_GRAY2BGR);
                bitwise_and(frontLeftUnderEyeWrinkleROI, frontOriginalImg, frontLeftUnderEyeWrinkleROI);
                frontLeftUnderEyeWrinkleCropped = new Mat(frontOriginalImg, frontLeftUnderEyeWrinkleBoundRect);

                // For later indexing
                cvtColor(frontRightUnderEyeWrinkleROI, frontRightUnderEyeWrinkleROI, COLOR_BGR2GRAY);
                cvtColor(frontLeftUnderEyeWrinkleROI, frontLeftUnderEyeWrinkleROI, COLOR_BGR2GRAY);
                bitwise_or(frontRightUnderEyeWrinkleROI, frontLeftUnderEyeWrinkleROI, frontEyesROIMask);

                // Get right side smile line ROI in front image
                drawContours(frontRightSmileLineWrinkleROI, rightSmileLineContour, -1, new Scalar(255), FILLED);
                frontRightSmileWrinkleBoundRect = boundingRect(frontRightSmileLineWrinkleROI);
                //rectangle(frontRightSmileLineWrinkleROI, frontRightSmileWrinkleBoundRect, new Scalar(255), 10);
                cvtColor(frontRightSmileLineWrinkleROI, frontRightSmileLineWrinkleROI, COLOR_GRAY2BGR);
                bitwise_and(frontRightSmileLineWrinkleROI, frontOriginalImg, frontRightSmileLineWrinkleROI);
                frontRightSmileLineWrinkleCropped = new Mat(frontOriginalImg, frontRightSmileWrinkleBoundRect);

                // Get left side smile line ROI in front image
                drawContours(frontLeftSmileLineWrinkleROI, leftSmileLineContour, -1, new Scalar(255), FILLED);
                frontLeftSmileWrinkleBoundRect = boundingRect(frontLeftSmileLineWrinkleROI);
                //rectangle(frontLeftSmileLineWrinkleROI, frontLeftSmileWrinkleBoundRect, new Scalar(255), 10);
                cvtColor(frontLeftSmileLineWrinkleROI, frontLeftSmileLineWrinkleROI, COLOR_GRAY2BGR);
                bitwise_and(frontLeftSmileLineWrinkleROI, frontOriginalImg, frontLeftSmileLineWrinkleROI);
                frontLeftSmileLineWrinkleCropped = new Mat(frontOriginalImg, frontLeftSmileWrinkleBoundRect);

                // For later indexing.
                cvtColor(frontRightSmileLineWrinkleROI, frontRightSmileLineWrinkleROI, COLOR_BGR2GRAY);
                cvtColor(frontLeftSmileLineWrinkleROI, frontLeftSmileLineWrinkleROI, COLOR_BGR2GRAY);
                bitwise_or(frontRightSmileLineWrinkleROI, frontLeftSmileLineWrinkleROI, frontSmileLinesROIMask);

                // Get front lion wrinkles ROI.
                drawContours(frontLionWrinkleROI, frontLionWrinkleContour, -1, new Scalar(255), FILLED);
                frontLionWrinkleBoundRect = boundingRect(frontLionWrinkleROI);
                //rectangle(frontLionWrinkleROI, frontLionWrinkleBoundRect, new Scalar(255), 10);
                cvtColor(frontLionWrinkleROI, frontLionWrinkleROI, COLOR_GRAY2BGR);
                bitwise_and(frontLionWrinkleROI, frontOriginalImg, frontLionWrinkleROI);
                frontLionWrinkleCropped = new Mat(frontOriginalImg, frontLionWrinkleBoundRect);

                // For later indexing.
                cvtColor(frontLionWrinkleROI, frontLionWrinkleROI, COLOR_BGR2GRAY);

                // Get front forehead wrinkles ROI.
                frontForeheadWrinkleBoundRect = boundingRect(frontForeheadMask);
                //rectangle(frontForeheadMask, frontForeheadWrinkleBoundRect, new Scalar(255), 10);
                cvtColor(frontForeheadMask, frontForeheadMask, COLOR_GRAY2BGR);
                cvtColor(frontForeheadWrinkleROI, frontForeheadWrinkleROI, COLOR_GRAY2BGR);
                bitwise_and(frontForeheadMask, frontOriginalImg, frontForeheadWrinkleROI);
                frontForeheadWrinkleCropped = new Mat(frontOriginalImg, frontForeheadWrinkleBoundRect);

                // For later indexing.
                cvtColor(frontForeheadMask, frontForeheadMask, COLOR_BGR2GRAY);

                //MyUtil.saveMatToGallery(context, "front right eye wrinkles ROI.jpg", "front right under eye wrinkles ROI cutout", frontRightUnderEyeWrinkleCropped);
                //MyUtil.saveMatToGallery(context, "front left eye wrinkles ROI.jpg", "front left under eye wrinkles ROI cutout", frontLeftUnderEyeWrinkleCropped);

                //MyUtil.saveMatToGallery(context, "front right smile wrinkles ROI.jpg", "front right smile wrinkles ROI cutout", frontRightSmileLineWrinkleCropped);
                //MyUtil.saveMatToGallery(context, "front left smile wrinkles ROI.jpg", "front left smile wrinkles ROI cutout", frontLeftSmileLineWrinkleCropped);

                //MyUtil.saveMatToGallery(context, "front lion wrinkles ROI.jpg", "front lion wrinkles ROI cutout", frontLionWrinkleCropped);
               // MyUtil.saveMatToGallery(context, "front forehead wrinkles ROI.jpg", "front forehead wrinkles ROI cutout", frontForeheadWrinkleCropped);
            }

            if(sideFaceImagesEnabled) {
                if(position == leftImage) {
                    // Define lists of FaceMesh key points for ROIs
                    // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
                    List<Integer> leftEyeCornerBoundShape = Collections.unmodifiableList(Arrays.asList(260, 467, 359, 255, 339, 254, 253, 252, 256, 453, 357, 277, 329, 330, 280, 352, 447, 389, 368, 383, 353, 445));
                    List<Integer> leftSmileLineBoundShape = Collections.unmodifiableList(Arrays.asList(279, 322, 410, 287, 422, 432, 427, 425, 266, 371));

                    int x = 0, y = 0;
                    List<Point> leftEyeCornerPoints = new ArrayList<>();
                    for(Integer index : leftEyeCornerBoundShape) {
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        leftEyeCornerPoints.add(new Point(x, y));
                    }
                    MatOfPoint leftEyeCornerPointMat = new MatOfPoint();
                    leftEyeCornerPointMat.fromList(leftEyeCornerPoints);
                    List<MatOfPoint> leftEyeCornerContour = new ArrayList<>();
                    leftEyeCornerContour.add(leftEyeCornerPointMat);

                    List<Point> leftSmileLinePoints = new ArrayList<>();
                    for(Integer index : leftSmileLineBoundShape) {
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        leftSmileLinePoints.add(new Point(x, y));
                    }
                    MatOfPoint leftSmileLinePointMat = new MatOfPoint();
                    leftSmileLinePointMat.fromList(leftSmileLinePoints);
                    List<MatOfPoint> leftSmileLineContour = new ArrayList<>();
                    leftSmileLineContour.add(leftSmileLinePointMat);

                    // Get left eye corner wrinkles ROI in left image
                    drawContours(leftEyeCornerWrinkleROI, leftEyeCornerContour, -1, new Scalar(255), FILLED);
                    leftEyeWrinkleBoundRect = boundingRect(leftEyeCornerWrinkleROI);
                    //rectangle(leftEyeCornerWrinkleROI, leftEyeWrinkleBoundRect, new Scalar(255), 10);
                    cvtColor(leftEyeCornerWrinkleROI, leftEyeCornerWrinkleROI, COLOR_GRAY2BGR);
                    bitwise_and(leftEyeCornerWrinkleROI, leftOriginalImg, leftEyeCornerWrinkleROI);
                    leftEyeCornerWrinkleCropped = new Mat(leftOriginalImg, leftEyeWrinkleBoundRect);

                    // Get forehead wrinkles ROI in left image.
                    leftForeheadWrinkleBoundRect = boundingRect(leftForeheadMask);
                    //rectangle(leftForeheadMask, leftForeheadWrinkleBoundRect, new Scalar(255), 10);
                    cvtColor(leftForeheadMask, leftForeheadMask, COLOR_GRAY2BGR);
                    cvtColor(leftForeheadWrinkleROI, leftForeheadWrinkleROI, COLOR_GRAY2BGR);
                    bitwise_and(leftForeheadMask, leftOriginalImg, leftForeheadWrinkleROI);
                    leftForeheadWrinkleCropped = new Mat(leftOriginalImg, leftForeheadWrinkleBoundRect);

                    // Get left smile line ROI in left image.
                    drawContours(leftSmileLineROI, leftSmileLineContour, -1, new Scalar(255), FILLED);
                    leftSmileLineROIBoundRect = boundingRect(leftSmileLineROI);
                    cvtColor(leftSmileLineROI, leftSmileLineROI, COLOR_GRAY2BGR);
                    bitwise_and(leftSmileLineROI, leftOriginalImg, leftSmileLineROI);
                    leftSmileLineROICropped = new Mat(leftOriginalImg, leftSmileLineROIBoundRect);

                    // For later indexing.
                    cvtColor(leftEyeCornerWrinkleROI, leftEyeCornerWrinkleROI, COLOR_BGR2GRAY);
                    cvtColor(leftForeheadMask, leftForeheadMask, COLOR_BGR2GRAY);
                    cvtColor(leftSmileLineROI, leftSmileLineROI, COLOR_BGR2GRAY);

                    //MyUtil.saveMatToGallery(context, "left right eye wrinkles ROI.jpg", "left eye wrinkles ROI cutout", leftEyeCornerWrinkleCropped);
                    //MyUtil.saveMatToGallery(context, "left forehead wrinkles ROI.jpg", "left forehead wrinkles ROI cutout", leftForeheadWrinkleCropped);
                    //MyUtil.saveMatToGallery(context, "left smile line.jpg", "left smile line", leftSmileLineROICropped);
                }

                if(position == rightImage) {
                    // Define lists of FaceMesh key points for ROIs
                    // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
                    List<Integer> rightEyeCornerBoundShape = Collections.unmodifiableList(Arrays.asList(30, 247, 226, 25, 110, 24, 23, 22, 26, 233, 128, 47, 100, 101, 50, 123, 227, 127, 162, 139, 156, 124, 225));
                    List<Integer> rightSmileLineBoundShape = Collections.unmodifiableList(Arrays.asList(49, 92, 186, 57, 202, 212, 207, 205, 36, 142));

                    int x = 0, y = 0;
                    List<Point> rightEyeCornerPoints = new ArrayList<>();
                    for(Integer index : rightEyeCornerBoundShape) {
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        rightEyeCornerPoints.add(new Point(x, y));
                    }
                    MatOfPoint rightEyeCornerPointMat = new MatOfPoint();
                    rightEyeCornerPointMat.fromList(rightEyeCornerPoints);
                    List<MatOfPoint> rightEyeCornerContour = new ArrayList<>();
                    rightEyeCornerContour.add(rightEyeCornerPointMat);

                    List<Point> rightSmileLinePoints = new ArrayList<>();
                    for(Integer index : rightSmileLineBoundShape) {
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        rightSmileLinePoints.add(new Point(x, y));
                    }
                    MatOfPoint rightSmileLinePointMat = new MatOfPoint();
                    rightSmileLinePointMat.fromList(rightSmileLinePoints);
                    List<MatOfPoint> rightSmileLineContour = new ArrayList<>();
                    rightSmileLineContour.add(rightSmileLinePointMat);

                    // Get right eye corner wrinkles ROI in right image
                    drawContours(rightEyeCornerWrinkleROI, rightEyeCornerContour, -1, new Scalar(255), FILLED);
                    rightEyeWrinkleBoundRect = boundingRect(rightEyeCornerWrinkleROI);
                    //rectangle(rightEyeCornerWrinkleROI, rightEyeWrinkleBoundRect, new Scalar(255), 10);
                    cvtColor(rightEyeCornerWrinkleROI, rightEyeCornerWrinkleROI, COLOR_GRAY2BGR);
                    bitwise_and(rightEyeCornerWrinkleROI, rightOriginalImg, rightEyeCornerWrinkleROI);
                    rightEyeCornerWrinkleCropped = new Mat(rightOriginalImg, rightEyeWrinkleBoundRect);

                    // Get forehead wrinkles ROI in right image.
                    rightForeheadWrinkleBoundRect = boundingRect(rightForeheadMask);
                    //rectangle(rightForeheadMask, rightForeheadWrinkleBoundRect, new Scalar(255), 10);
                    cvtColor(rightForeheadMask, rightForeheadMask, COLOR_GRAY2BGR);
                    cvtColor(rightForeheadWrinkleROI, rightForeheadWrinkleROI, COLOR_GRAY2BGR);
                    bitwise_and(rightForeheadMask, rightOriginalImg, rightForeheadWrinkleROI);
                    rightForeheadWrinkleCropped = new Mat(rightOriginalImg, rightForeheadWrinkleBoundRect);

                    // Get smile line ROI in right image.
                    drawContours(rightSmileLineROI, rightSmileLineContour, -1, new Scalar(255), FILLED);
                    rightSmileLineROIBoundRect = boundingRect(rightSmileLineROI);
                    cvtColor(rightSmileLineROI, rightSmileLineROI, COLOR_GRAY2BGR);
                    bitwise_and(rightSmileLineROI, rightOriginalImg, rightSmileLineROI);
                    rightSmileLineROICropped = new Mat(rightOriginalImg, rightSmileLineROIBoundRect);

                    // For later indexing.
                    cvtColor(rightEyeCornerWrinkleROI, rightEyeCornerWrinkleROI, COLOR_BGR2GRAY);
                    cvtColor(rightForeheadMask, rightForeheadMask, COLOR_BGR2GRAY);
                    cvtColor(rightSmileLineROI, rightSmileLineROI, COLOR_BGR2GRAY);

                    //MyUtil.saveMatToGallery(context, "right eye corner wrinkles ROI.jpg", "right eye corner wrinkles ROI cutout", rightEyeCornerWrinkleCropped);
                    //MyUtil.saveMatToGallery(context, "right forehead wrinkles ROI.jpg", "right forehead wrinkles ROI cutout", rightForeheadWrinkleCropped);
                    //MyUtil.saveMatToGallery(context, "right smile line ROI.jpg", "right smile line ROI", rightSmileLineROICropped);
                }
            }
        }


        // ----------------------------------- 1. AI detections ------------------------------
        //
        // Load AI model first
        Interpreter wrinklesModel = new Interpreter(wrinklesTFModelFile);

        int roiHeight = 0;
        int roiWidth = 0;
        roiHeight = frontRightUnderEyeWrinkleCropped.height();
        roiWidth = frontRightUnderEyeWrinkleCropped.width();
        if(roiHeight > roiWidth) {
            int temp = roiWidth;
            roiWidth = roiHeight;
            roiHeight = temp;
            resize(frontRightUnderEyeWrinkleCropped, frontRightUnderEyeWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
            roiHeight = frontRightUnderEyeWrinkleCropped.height();
            roiWidth = frontRightUnderEyeWrinkleCropped.width();
        }
        Mat frontRightUnderEyeWrinkleMaskCropped = aiInference(context, wrinklesModel, frontRightUnderEyeWrinkleCropped, roiHeight, roiWidth);

        roiHeight = frontLeftUnderEyeWrinkleCropped.height();
        roiWidth = frontLeftUnderEyeWrinkleCropped.width();
        if(roiHeight > roiWidth) {
            int temp = roiWidth;
            roiWidth = roiHeight;
            roiHeight = temp;
            resize(frontLeftUnderEyeWrinkleCropped, frontLeftUnderEyeWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
            roiHeight = frontLeftUnderEyeWrinkleCropped.height();
            roiWidth = frontLeftUnderEyeWrinkleCropped.width();
        }
        Mat frontLeftUnderEyeWrinkleMaskCropped = aiInference(context, wrinklesModel, frontLeftUnderEyeWrinkleCropped, roiHeight, roiWidth);

        roiHeight = frontForeheadWrinkleCropped.height();
        roiWidth = frontForeheadWrinkleCropped.width();
        if(roiHeight > roiWidth) {
            int temp = roiWidth;
            roiWidth = roiHeight;
            roiHeight = temp;
            resize(frontForeheadWrinkleCropped, frontForeheadWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
            roiHeight = frontForeheadWrinkleCropped.height();
            roiWidth = frontForeheadWrinkleCropped.width();
        }
        Mat frontForeheadWrinkleMaskCropped = aiInference(context, wrinklesModel, frontForeheadWrinkleCropped, roiHeight, roiWidth);

        roiHeight = frontLionWrinkleCropped.height();
        roiWidth = frontLionWrinkleCropped.width();
        if(roiHeight > roiWidth) {
            int temp = roiWidth;
            roiWidth = roiHeight;
            roiHeight = temp;
            resize(frontLionWrinkleCropped, frontLionWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
            roiHeight = frontLionWrinkleCropped.height();
            roiWidth = frontLionWrinkleCropped.width();
        }
        Mat frontLionWrinkleMaskCropped = aiInference(context, wrinklesModel, frontLionWrinkleCropped, roiHeight, roiWidth);

        roiHeight = frontLeftSmileLineWrinkleCropped.height();
        roiWidth = frontLeftSmileLineWrinkleCropped.width();
        Mat frontLeftSmileLineWrinkleMaskCropped = aiInference(context, wrinklesModel, frontLeftSmileLineWrinkleCropped, roiHeight, roiWidth);

        roiHeight = frontRightSmileLineWrinkleCropped.height();
        roiWidth = frontRightSmileLineWrinkleCropped.width();
        Mat frontRightSmileLineWrinkleMaskCropped = aiInference(context, wrinklesModel, frontRightSmileLineWrinkleCropped, roiHeight, roiWidth);

        // Process mask images for front face wrinkles.
        //
        // Front under eye wrinkles.
        Mat subMask = new Mat();
        for(int y = 0; y < frontRightUnderEyeWrinkleMaskCropped.rows(); y++) {
            for(int x = 0; x < frontRightUnderEyeWrinkleMaskCropped.cols(); x ++) {
                if(frontRightUnderEyeWrinkleMaskCropped.get(y, x)[0] > 100) {
                    frontEyeWrinkleMask.put(frontRightUnderEyeWrinkleBoundRect.y + y, frontRightUnderEyeWrinkleBoundRect.x + x, 255);
                }
            }
        }
        for(int y = 0; y < frontLeftUnderEyeWrinkleMaskCropped.rows(); y++) {
            for(int x = 0; x < frontLeftUnderEyeWrinkleMaskCropped.cols(); x ++) {
                if(frontLeftUnderEyeWrinkleMaskCropped.get(y, x)[0] > 100) {
                    frontEyeWrinkleMask.put(frontLeftUnderEyeWrinkleBoundRect.y + y, frontLeftUnderEyeWrinkleBoundRect.x + x, 255);
                }
            }
        }

        // Front forehead wrinkles
        for(int y = 0; y < frontForeheadWrinkleMaskCropped.rows(); y++) {
            for(int x = 0; x < frontForeheadWrinkleMaskCropped.cols(); x ++) {
                if(frontForeheadWrinkleMaskCropped.get(y, x)[0] > 100) {
                    frontForeheadWrinkleMask.put(frontForeheadWrinkleBoundRect.y + y, frontForeheadWrinkleBoundRect.x + x, 255);
                }
            }
        }

        // Remove lion wrinkles from forehead wrinkles, because forehead wrinkle ROI includes parts of lion wrinkle ROI.
        rectangle(frontForeheadMask, frontLionWrinkleBoundRect, new Scalar(0));

        // Front smile line wrinkles.
        for(int y = 0; y < frontLeftSmileLineWrinkleMaskCropped.rows(); y++) {
            for(int x = 0; x < frontLeftSmileLineWrinkleMaskCropped.cols(); x ++) {
                if(frontLeftSmileLineWrinkleMaskCropped.get(y, x)[0] > 100) {
                    frontSmileWrinkleMask.put(frontLeftSmileWrinkleBoundRect.y + y, frontLeftSmileWrinkleBoundRect.x + x, 255);
                }
            }
        }
        for(int y = 0; y < frontRightSmileLineWrinkleMaskCropped.rows(); y++) {
            for(int x = 0; x < frontRightSmileLineWrinkleMaskCropped.cols(); x ++) {
                if(frontRightSmileLineWrinkleMaskCropped.get(y, x)[0] > 100) {
                    frontSmileWrinkleMask.put(frontRightSmileWrinkleBoundRect.y + y, frontRightSmileWrinkleBoundRect.x + x, 255);
                }
            }
        }

        // Front Lion wrinkle mask
        for(int y = 0; y < frontLionWrinkleMaskCropped.rows(); y++) {
            for(int x = 0; x < frontLionWrinkleMaskCropped.cols(); x ++) {
                if(frontLionWrinkleMaskCropped.get(y, x)[0] > 100) {
                    frontLionWrinkleMask.put(frontLionWrinkleBoundRect.y + y, frontLionWrinkleBoundRect.x + x, 255);
                }
            }
        }

        // Front final/total wrinkles mask
        bitwise_or(frontEyeWrinkleMask, frontForeheadWrinkleMask, frontWrinkleFinalMask);
        bitwise_or(frontSmileWrinkleMask, frontWrinkleFinalMask, frontWrinkleFinalMask);
        bitwise_or(frontLionWrinkleMask, frontWrinkleFinalMask, frontWrinkleFinalMask);

        //MyUtil.saveMatToGallery(context, "front eye.jpg", "front eye wrinkles", frontEyeWrinkleMask);
        //MyUtil.saveMatToGallery(context, "front forehead.jpg", "front forehead wrinkles", frontForeheadWrinkleMask);
        //MyUtil.saveMatToGallery(context, "front smile.jpg", "front smile line wrinkles", frontSmileWrinkleMask);
        //MyUtil.saveMatToGallery(context, "front lion.jpg", "front lion wrinkles", frontLionWrinkleMask);

        Mat leftEyeCornerWrinkleMaskCropped = new Mat();
        Mat leftForeheadWrinkleMaskCropped = new Mat();
        Mat leftSmileLineMaskCropped = new Mat();
        Mat rightEyeCornerWrinkleMaskCropped = new Mat();
        Mat rightForeheadWrinkleMaskCropped = new Mat();
        Mat rightSmileLineMaskCropped = new Mat();
        if(sideFaceImagesEnabled) {
            roiHeight = leftEyeCornerWrinkleCropped.height();
            roiWidth = leftEyeCornerWrinkleCropped.width();
            if(roiHeight > roiWidth) {
                int temp = roiWidth;
                roiWidth = roiHeight;
                roiHeight = temp;
                resize(leftEyeCornerWrinkleCropped, leftEyeCornerWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
                roiHeight = leftEyeCornerWrinkleCropped.height();
                roiWidth = leftEyeCornerWrinkleCropped.width();
            }
            leftEyeCornerWrinkleMaskCropped = aiInference(context, wrinklesModel, leftEyeCornerWrinkleCropped, roiHeight, roiWidth);

            roiHeight = leftForeheadWrinkleCropped.height();
            roiWidth = leftForeheadWrinkleCropped.width();
            if(roiHeight > roiWidth) {
                int temp = roiWidth;
                roiWidth = roiHeight;
                roiHeight = temp;
                resize(leftForeheadWrinkleCropped, leftForeheadWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
                roiHeight = leftForeheadWrinkleCropped.height();
                roiWidth = leftForeheadWrinkleCropped.width();
            }
            leftForeheadWrinkleMaskCropped = aiInference(context, wrinklesModel, leftForeheadWrinkleCropped, roiHeight, roiWidth);

            roiHeight = leftSmileLineROICropped.height();
            roiWidth = leftSmileLineROICropped.width();
            leftSmileLineMaskCropped = aiInference(context, wrinklesModel, leftSmileLineROICropped, roiHeight, roiWidth);

            roiHeight = rightEyeCornerWrinkleCropped.height();
            roiWidth = rightEyeCornerWrinkleCropped.width();
            if(roiHeight > roiWidth) {
                int temp = roiWidth;
                roiWidth = roiHeight;
                roiHeight = temp;
                resize(rightEyeCornerWrinkleCropped, rightEyeCornerWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
                roiHeight = rightEyeCornerWrinkleCropped.height();
                roiWidth = rightEyeCornerWrinkleCropped.width();
            }
            rightEyeCornerWrinkleMaskCropped =  aiInference(context, wrinklesModel, rightEyeCornerWrinkleCropped, roiHeight, roiWidth);

            roiHeight = rightForeheadWrinkleCropped.height();
            roiWidth = rightForeheadWrinkleCropped.width();
            if(roiHeight > roiWidth) {
                int temp = roiWidth;
                roiWidth = roiHeight;
                roiHeight = temp;
                resize(rightForeheadWrinkleCropped, rightForeheadWrinkleCropped, new Size(roiWidth, roiHeight), 0, 0);
                roiHeight = rightForeheadWrinkleCropped.height();
                roiWidth = rightForeheadWrinkleCropped.width();
            }
            rightForeheadWrinkleMaskCropped = aiInference(context, wrinklesModel, rightForeheadWrinkleCropped, roiHeight, roiWidth);

            roiHeight = rightSmileLineROICropped.height();
            roiWidth = rightSmileLineROICropped.width();
            rightSmileLineMaskCropped = aiInference(context, wrinklesModel, rightSmileLineROICropped, roiHeight, roiWidth);

            // Process mask images for left & right face wrinkles.
            //
            // Left eye corner wrinkles.
            for(int y = 0; y < leftEyeCornerWrinkleMaskCropped.rows(); y++) {
                for(int x = 0; x < leftEyeCornerWrinkleMaskCropped.cols(); x ++) {
                    if(leftEyeCornerWrinkleMaskCropped.get(y, x)[0] > 100) {
                        leftEyeWrinkleMask.put(leftEyeWrinkleBoundRect.y + y, leftEyeWrinkleBoundRect.x + x, 255);
                    }
                }
            }

            // Left Forehead wrinkles.
            for(int y = 0; y < leftForeheadWrinkleMaskCropped.rows(); y++) {
                for(int x = 0; x < leftForeheadWrinkleMaskCropped.cols(); x ++) {
                    if(leftForeheadWrinkleMaskCropped.get(y, x)[0] > 100) {
                        leftForeheadWrinkleMask.put(leftForeheadWrinkleBoundRect.y + y, leftForeheadWrinkleBoundRect.x + x, 255);
                    }
                }
            }

            // Left smile line.
            for(int y = 0; y < leftSmileLineMaskCropped.rows(); y++) {
                for(int x = 0; x < leftSmileLineMaskCropped.cols(); x ++) {
                    if(leftSmileLineMaskCropped.get(y, x)[0] > 100) {
                        leftSmileLineMask.put(leftSmileLineROIBoundRect.y + y, leftSmileLineROIBoundRect.x + x, 255);
                    }
                }
            }

            // Left final/total wrinkles mask.
            bitwise_or(leftEyeWrinkleMask, leftForeheadWrinkleMask, leftWrinkleFinalMask);
            bitwise_and(leftWrinkleFinalMask, leftSmileLineMask, leftWrinkleFinalMask);

            // Right eye corner wrinkles
            for(int y = 0; y < rightEyeCornerWrinkleMaskCropped.rows(); y++) {
                for(int x = 0; x < rightEyeCornerWrinkleMaskCropped.cols(); x ++) {
                    if(rightEyeCornerWrinkleMaskCropped.get(y, x)[0] > 100) {
                        rightEyeWrinkleMask.put(rightEyeWrinkleBoundRect.y + y, rightEyeWrinkleBoundRect.x + x, 255);
                    }
                }
            }

            // Right Forehead Wrinkles.
            for(int y = 0; y < rightForeheadWrinkleMaskCropped.rows(); y++) {
                for(int x = 0; x < rightForeheadWrinkleMaskCropped.cols(); x ++) {
                    if(rightForeheadWrinkleMaskCropped.get(y, x)[0] > 100) {
                        rightForeheadWrinkleMask.put(rightForeheadWrinkleBoundRect.y + y, rightForeheadWrinkleBoundRect.x + x, 255);
                    }
                }
            }

            // Right smile line.
            for(int y = 0; y < rightSmileLineMaskCropped.rows(); y++) {
                for(int x = 0; x < rightSmileLineMaskCropped.cols(); x ++) {
                    if(rightSmileLineMaskCropped.get(y, x)[0] > 100) {
                        rightSmileLineMask.put(rightSmileLineROIBoundRect.y + y, rightSmileLineROIBoundRect.x + x, 255);
                    }
                }
            }

            // Right final/total wrinkles mask.
            bitwise_or(rightEyeWrinkleMask, rightForeheadWrinkleMask, rightWrinkleFinalMask);

            //MyUtil.saveMatToGallery(context, "left eye.jpg", "left eye wrinkles", leftEyeWrinkleMask);
            //MyUtil.saveMatToGallery(context, "left forehead.jpg", "left forehead wrinkles", leftForeheadWrinkleMask);
            //MyUtil.saveMatToGallery(context, "left smile line.jpg", "left smile wrinkle", leftSmileLineMask);
            //MyUtil.saveMatToGallery(context, "left final.jpg", "left total wrinkles", leftWrinkleFinalMask);

            //MyUtil.saveMatToGallery(context, "right eye.jpg", "right eye wrinkles", rightEyeWrinkleMask);
            //MyUtil.saveMatToGallery(context, "right forehead.jpg", "right forehead wrinkles", rightForeheadWrinkleMask);
            //MyUtil.saveMatToGallery(context, "right smile line.jpg", "right smile wrinkle", rightSmileLineMask);
            //MyUtil.saveMatToGallery(context, "right final.jpg", "right final wrinkles", rightWrinkleFinalMask);
        }

        // ------------------------------------------ 3. Indexing ------------------------------------------
        //
        // Wrinkle scoring.
        double frontEyeScore = 0, frontForeheadScore = 0, frontLionScore = 0, frontSmileScore = 0;
        double leftEyeScore = 0, leftForeheadScore = 0, leftSmileScore = 0;
        double rightEyeScore = 0, rightForeheadScore = 0, rightSmileScore = 0;

        double frontEyeRaw = 0, frontForeheadRaw = 0, frontLionRaw = 0, frontSmileRaw = 0;
        double leftEyeRaw = 0, leftForeheadRaw = 0, leftLionRaw = 0, leftSmileRaw = 0;
        double rightEyeRaw = 0, rightForeheadRaw = 0, rightLionRaw = 0, rightSmileRaw = 0;

        double frontTotalRaw = 0, leftTotalRaw = 0, rightTotalRaw = 0;
        double frontTotalScore = 0, leftTotalScore = 0, rightTotalScore = 0;
        double allImagesWrinkleScore = 0;

        int frontFullFaceCount = 0, leftFullFaceCount = 0, rightFullFaceCount = 0;
        int frontEyeCount = 0, frontForeheadCount = 0, frontLionCount = 0, frontSmileCount = 0;
        int leftEyeCount = 0, leftForeheadCount = 0, leftSmileCount = 0;
        int rightEyeCount = 0, rightForeheadCount = 0, rightSmileCount = 0;

        frontFullFaceCount = countNonZero(frontFullFaceMask);
        frontEyeCount = countNonZero(frontEyesROIMask);
        frontForeheadCount = countNonZero(frontForeheadMask);
        frontLionCount = countNonZero(frontLionWrinkleROI);
        frontSmileCount = countNonZero(frontSmileLinesROIMask);

        if(sideFaceImagesEnabled) {
            leftFullFaceCount = countNonZero(leftFullFaceMask);
            leftEyeCount = countNonZero(leftEyeCornerWrinkleROI);
            leftForeheadCount = countNonZero(leftForeheadMask);
            leftSmileCount = countNonZero(leftSmileLineROI);

            rightFullFaceCount = countNonZero(rightFullFaceMask);
            rightEyeCount = countNonZero(rightEyeCornerWrinkleROI);
            rightForeheadCount = countNonZero(rightForeheadMask);
            rightSmileCount = countNonZero(rightSmileLineROI);
        }

        if(frontEyeCount > 0)  frontEyeRaw = (double) 1000 * countNonZero(frontEyeWrinkleMask) / frontEyeCount;
        if(frontForeheadCount > 0)  frontForeheadRaw = (double) 1000 * countNonZero(frontForeheadWrinkleMask) / frontForeheadCount;
        if(frontLionCount > 0)  frontLionRaw = (double) 1000 * countNonZero(frontLionWrinkleMask) / frontLionCount;
        if(frontSmileCount > 0) frontSmileRaw = (double) 1000 * countNonZero(frontSmileWrinkleMask) / frontSmileCount;

        frontEyeScore = getCFAWrinkleLevel(frontEyeRaw, capturedWithFrontCamera, "front-eye");
        frontForeheadScore = getCFAWrinkleLevel(frontForeheadRaw, capturedWithFrontCamera, "front-forehead");
        frontLionScore = getCFAWrinkleLevel(frontLionRaw, capturedWithFrontCamera, "front-lion");
        frontSmileScore = getCFAWrinkleLevel(frontSmileRaw, capturedWithFrontCamera, "front-smileline");

        if(sideFaceImagesEnabled) {
            if (leftEyeCount > 0)
                leftEyeRaw = (double) 1000 * countNonZero(leftEyeWrinkleMask) / leftEyeCount;
            if (leftForeheadCount > 0)
                leftForeheadRaw = (double) 1000 * countNonZero(leftForeheadWrinkleMask) / leftForeheadCount;
            if(leftSmileCount > 0)  leftSmileRaw = (double) 1000 * countNonZero(leftSmileLineMask) / leftSmileCount;

            if (rightEyeCount > 0)
                rightEyeRaw = (double) 1000 * countNonZero(rightEyeWrinkleMask) / rightEyeCount;
            if (rightForeheadCount > 0)
                rightForeheadRaw = (double) 1000 * countNonZero(rightForeheadWrinkleMask) / rightForeheadCount;
            if(rightSmileCount > 0) rightSmileRaw = (double) 1000 * countNonZero(rightSmileLineMask) / rightSmileCount;

            leftEyeScore = getCFAWrinkleLevel(leftEyeRaw, capturedWithFrontCamera, "side-eye");
            leftForeheadScore = getCFAWrinkleLevel(leftForeheadRaw, capturedWithFrontCamera, "side-forehead");
            //leftSmileScore = getCFAWrinkleLevel(leftSmileRaw, capturedWithFrontCamera, "side-smileline");

            rightEyeScore = getCFAWrinkleLevel(rightEyeRaw, capturedWithFrontCamera, "side-eye");
            rightForeheadScore = getCFAWrinkleLevel(rightForeheadRaw, capturedWithFrontCamera, "side-forehead");
            //rightSmileScore = getCFAWrinkleLevel(rightSmileRaw, capturedWithFrontCamera, "side-smileline");
        }

        if(frontFullFaceCount > 0) frontTotalRaw = (double) 1000 * countNonZero(frontWrinkleFinalMask) / frontFullFaceCount;
        if(sideFaceImagesEnabled) {
            if (leftFullFaceCount > 0)
                leftTotalRaw = (double) 1000 * countNonZero(leftWrinkleFinalMask) / leftFullFaceCount;
            if (rightFullFaceCount > 0)
                rightTotalRaw = (double) 1000 * countNonZero(rightWrinkleFinalMask) / rightFullFaceCount;
        }

        frontTotalScore = getCFAWrinkleLevel(frontTotalRaw, capturedWithFrontCamera, "front-total");
        if(sideFaceImagesEnabled) {
            leftTotalScore = getCFAWrinkleLevel(leftTotalRaw, capturedWithFrontCamera, "side-total");
            rightTotalScore = getCFAWrinkleLevel(rightTotalRaw, capturedWithFrontCamera, "side-total");
        }

        if(sideFaceImagesEnabled) {
            allImagesWrinkleScore = Math.round(0.5 * frontTotalScore + 0.25 * leftTotalScore + 0.25 * rightTotalScore);
        } else {
            allImagesWrinkleScore = frontTotalScore;
        }

        // ------------------------------------------ 4. Prepare Output ------------------------------------------
        //
        // ----- (a) ----- Save wrinkle mask images first.
        cvtColor(frontWrinkleFinalMask, frontWrinkleFinalMask, COLOR_GRAY2BGR);
        if(sideFaceImagesEnabled) {
            cvtColor(leftWrinkleFinalMask, leftWrinkleFinalMask, COLOR_GRAY2BGR);
            cvtColor(rightWrinkleFinalMask, rightWrinkleFinalMask, COLOR_GRAY2BGR);
        }

        Imgcodecs.imwrite(frontMaskOutputPath, frontWrinkleFinalMask);
        if(sideFaceImagesEnabled) {
            if(leftMaskOutputPath != null) Imgcodecs.imwrite(leftMaskOutputPath, leftWrinkleFinalMask);
            if(rightMaskOutputPath != null) Imgcodecs.imwrite(rightMaskOutputPath, rightWrinkleFinalMask);
        }

        // ----- (b) ------ Prepare input paths for mask images.
        String frontWrinkleMaskInputPath = frontMaskOutputPath;
        String leftWrinkleMaskInputPath = null;
        String rightWrinkleMaskInputPath = null;
        if(sideFaceImagesEnabled) {
             leftWrinkleMaskInputPath = leftMaskOutputPath;
             rightWrinkleMaskInputPath = rightMaskOutputPath;
        }

        // ----- (c) ----- Save images to internal storage.
        int maskB = 0;
        int maskG = 145;
        int maskR = 255;
        int contourB = -1;
        int contourG = -1;
        int contourR = -1;
        double alpha = 0.55;

        com.chowis.jniimagepro.CFA.JNICFAImageProCW myCFAImgProc = new JNICFAImageProCW();
        double saveFrontRes = myCFAImgProc.CFAGetAnalyzedImgJni(frontOriginalInputPath, frontWrinkleMaskInputPath, frontResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
        if(sideFaceImagesEnabled) {
            double saveLeftRes = myCFAImgProc.CFAGetAnalyzedImgJni(leftOriginalInputPath, leftWrinkleMaskInputPath, leftResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
            double saveRightRes = myCFAImgProc.CFAGetAnalyzedImgJni(rightOriginalInputPath, rightWrinkleMaskInputPath, rightResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
        }

        // ---------- Prepare returned values.
        //
        String returnString = null;
        if(sideFaceImagesEnabled) {
            returnString = allImagesWrinkleScore + "_" + frontTotalScore + "_" + leftTotalScore + "_" + rightTotalScore + "_" + frontTotalRaw + "_" + leftTotalRaw + "_" + rightTotalRaw + "_" + frontEyeScore + "_" + frontForeheadScore + "_" + frontLionScore + "_" + frontSmileScore + "_" + leftEyeScore + "_" + leftForeheadScore + "_" + leftSmileScore + "_" + rightEyeScore + "_" + rightForeheadScore + "_" + rightSmileScore + "_" + frontEyeRaw + "_" + frontForeheadRaw + "_" + frontLionRaw + "_" + frontSmileRaw + "_" + leftEyeRaw + "_" + leftForeheadRaw + "_" + leftLionRaw + "_" + leftSmileRaw + "_" + rightEyeRaw + "_" + rightForeheadRaw + "_" + rightLionRaw + "_" + rightSmileRaw;
        }
        else {
            returnString = allImagesWrinkleScore + "_" + frontTotalScore + "_" + frontTotalRaw + "_" + frontEyeScore + "_" + frontForeheadScore + "_" + frontLionScore + "_" + frontSmileScore + "_" + frontEyeRaw + "_" + frontForeheadRaw + "_" + frontLionRaw + "_" + frontSmileRaw;
        }
        System.out.println("Returned String for Wrinkles is" + returnString);

        return returnString;
    }

    private static Mat aiInference(Context context, Interpreter wrinkleSpotsModel, Mat originalImg, int originalHeight, int originalWidth){

        // ------------------------------------------ 1. AI Model  ------------------------------------------
        //
        // Allocate input and output tensors.
        wrinkleSpotsModel.allocateTensors();

        // Prepare input tensor from OpenCV image.
        int[] inputShape = wrinkleSpotsModel.getInputTensor(0).shape(); // num, height, width, channel.
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

        MyUtil.saveMatToGallery(context, "input image", "staged image for AI input", inputImg);

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
        Log.println(Log.VERBOSE, "Timestamp wrinkles & spots input: ", String.valueOf(timestamp1End - timestamp1Start));

        //System.out.println("First normalized pixel value: " + input[0][0][0][0]);

        // Inference.
        int numClasses = 2;
        float[][][][] output = new float[1][input_height][input_width][numClasses];

        long timestamp2Start = System.currentTimeMillis();
        wrinkleSpotsModel.run(input, output);
        long timestamp2End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp wrinkles & spots inference: ", String.valueOf(timestamp2End - timestamp2Start));

        // Post processing.
        // Get our wrinkle mask
        long timestamp3Start = System.currentTimeMillis();

        Mat stagedWrinkleMask = Mat.zeros(input_height, input_width, CV_8UC1);

        int backgroundClassID = 0;
        int wrinkleClassID = 1;
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                if(output[0][y][x][wrinkleClassID] > 0.8) stagedWrinkleMask.put(y, x, 255);
            }
        }

        long timestamp3End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp wrinkles & spots output: ", String.valueOf(timestamp3End - timestamp3Start));

        Rect maskForegroundRect = new Rect((int)((double)stagingForegroundRect.x * scale_factor), (int)((double)stagingForegroundRect.y * scale_factor), staging_width, staging_height);
        Mat wrinkleMask = new Mat(stagedWrinkleMask, maskForegroundRect);

        if (originalHeight * originalWidth < staging_height * staging_width) {
            resize(wrinkleMask, wrinkleMask, new Size(originalWidth, originalHeight), 0, 0, INTER_AREA);
        } else if (originalHeight * originalWidth > staging_height * staging_width) {
            resize(wrinkleMask, wrinkleMask, new Size(originalWidth, originalHeight), 0, 0, INTER_LINEAR);
        }

        // Restore detected features after image resizing. We can't retain all 255 values in resized image.
        threshold(wrinkleMask, wrinkleMask, 10, 255, THRESH_BINARY);

        MyUtil.saveMatToGallery(context, "returned wrinkle mask ", "mask image used for result", wrinkleMask);
        System.out.println("mask feature count:" + String.valueOf(countNonZero(wrinkleMask)));

        return wrinkleMask;
    }

    private static double getCFAWrinkleLevel(double pureValue, boolean capturedWithFrontCamera, String roiTarget) {
        double[] dbNormData = new double[0];
        int nMin = 9;
        int nMax = 9;
        int index = 9;

        if(capturedWithFrontCamera) {
            if (roiTarget == "front-total") {
                dbNormData = new double[]{0.0, 0.71, 1.39, 2.07, 2.70, 3.46, 4.35, 5.68, 7.11, 9.75, 24.59};
            } else if (roiTarget == "front-forehead") {
                dbNormData = new double[]{0.0, 1.77, 2.96, 4.68, 7.02, 9.54, 13.02, 16.95, 23.68, 31.98, 42.68};
            } else if (roiTarget == "front-eye") {
                dbNormData = new double[]{0.0, 3.59, 5.66, 7.51, 9.46, 11.07, 12.25, 14.20, 16.30, 17.89, 19.06};
            } else if (roiTarget == "front-lion") {
                dbNormData = new double[]{0.0, 3.07, 3.25, 4.22, 5.31, 5.70, 6.31, 7.09, 7.89, 8.66, 9.68};
            } else if (roiTarget == "front-smileline") {
                dbNormData = new double[]{0.0, 0.15, 0.47, 1.11, 1.70, 2.65, 4.29, 6.39, 9.04, 11.13, 21.50};
            }

            if (roiTarget == "side-total") {
                dbNormData = new double[]{0.0, 0.32, 0.61, 0.88, 1.28, 1.64, 2.29, 3.03, 4.57, 6.60, 36.57};
            } else if (roiTarget == "side-forehead") {
                dbNormData = new double[]{0.0, 1.77, 2.96, 4.68, 7.02, 9.54, 13.02, 16.95, 23.68, 31.98, 42.68};
            } else if (roiTarget == "side-eye") {
                dbNormData = new double[]{0.0, 3.59, 5.66, 7.51, 9.46, 11.07, 12.25, 14.20, 16.30, 17.89, 19.06};
            } else if (roiTarget == "side-lion") {
                dbNormData = new double[]{0.0, 3.07, 3.25, 4.22, 5.31, 5.70, 6.31, 7.09, 7.89, 8.66, 9.68};
            } else if (roiTarget == "side-smileline") {
                dbNormData = new double[]{0.0, 0.15, 0.47, 1.11, 1.70, 2.65, 4.29, 6.39, 9.04, 11.13, 21.50};
            }
        }

        if(!capturedWithFrontCamera) {
            if (roiTarget == "front-total") {
                dbNormData = new double[]{0.0, 0.71, 1.39, 2.07, 2.70, 3.46, 4.35, 5.68, 7.11, 9.75, 24.59};
            } else if (roiTarget == "front-forehead") {
                dbNormData = new double[]{0.0, 1.77, 2.96, 4.68, 7.02, 9.54, 13.02, 16.95, 23.68, 31.98, 42.68};
            } else if (roiTarget == "front-eye") {
                dbNormData = new double[]{0.0, 3.59, 5.66, 7.51, 9.46, 11.07, 12.25, 14.20, 16.30, 17.89, 19.06};
            } else if (roiTarget == "front-lion") {
                dbNormData = new double[]{0.0, 3.07, 3.25, 4.22, 5.31, 5.70, 6.31, 7.09, 7.89, 8.66, 9.68};
            } else if (roiTarget == "front-smileline") {
                dbNormData = new double[]{0.0, 0.15, 0.47, 1.11, 1.70, 2.65, 4.29, 6.39, 9.04, 11.13, 21.50};
            }

            if (roiTarget == "side-total") {
                dbNormData = new double[]{0.0, 0.32, 0.61, 0.88, 1.28, 1.64, 2.29, 3.03, 4.57, 6.60, 36.57};
            } else if (roiTarget == "side-forehead") {
                dbNormData = new double[]{0.0, 1.77, 2.96, 4.68, 7.02, 9.54, 13.02, 16.95, 23.68, 31.98, 42.68};
            } else if (roiTarget == "side-eye") {
                dbNormData = new double[]{0.0, 3.59, 5.66, 7.51, 9.46, 11.07, 12.25, 14.20, 16.30, 17.89, 19.06};
            } else if (roiTarget == "side-lion") {
                dbNormData = new double[]{0.0, 3.07, 3.25, 4.22, 5.31, 5.70, 6.31, 7.09, 7.89, 8.66, 9.68};
            } else if (roiTarget == "side-smileline") {
                dbNormData = new double[]{0.0, 0.15, 0.47, 1.11, 1.70, 2.65, 4.29, 6.39, 9.04, 11.13, 21.50};
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
}

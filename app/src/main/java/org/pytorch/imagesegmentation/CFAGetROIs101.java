package org.pytorch.imagesegmentation;

import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.bitwise_or;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.circle;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.fillPoly;
import static org.opencv.imgproc.Imgproc.polylines;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.resize;

import android.content.Context;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class CFAGetROIs101 {

    //
    // ---------- Segment the Face Regional ROIs with FA detectron2.
    //
    public static void doAnalysis(Context context,
                                  String frontAnonymizedPPLInputImgPath,
                                  String leftAnonymizedPPLInputImgPath,
                                  String rightAnonymizedPPLInputImgPath,
                                  String frontAnonymizedXPLInputImgPath,
                                  String leftAnonymizedXPLInputImgPath,
                                  String rightAnonymizedXPLInputImgPath,
                                  String frontAnonymizedUVLInputImgPath,
                                  String leftAnonymizedUVLInputImgPath,
                                  String rightAnonymizedUVLInputImgPath,
                                  String frontFullFaceMaskInputPath,
                                  String leftFullFaceMaskInputPath,
                                  String rightFullFaceMaskInputPath,
                                  String frontRednessROIOutputPath,
                                  String leftRednessROIOutputPath,
                                  String rightRednessROIOutputPath,
                                  String oilinessFrontROIOutputPath,
                                  String radianceFrontROIOutputPath,
                                  String impuritiesFrontROIOutputPath,
                                  String frontForeheadMaskOutputPath,
                                  String frontNoseMaskOutputPath,
                                  String frontCheekMaskOutputPath,
                                  String frontChinMaskOutputPath,
                                  String leftForeheadMaskOutputPath,
                                  String leftNoseMaskOutputPath,
                                  String leftCheekMaskOutputPath,
                                  String leftChinMaskOutputPath,
                                  String rightForeheadMaskOutputPath,
                                  String rightNoseMaskOutputPath,
                                  String rightCheekMaskOutputPath,
                                  String rightChinMaskOutputPath,
                                  Map<Integer, List<Integer>> frontCoordinates,
                                  Map<Integer, List<Integer>> leftCoordinates,
                                  Map<Integer, List<Integer>> rightCoordinates,
                                  boolean impuritiesEnabled,
                                  boolean sideFaceImagesEnabled) {

        Mat frontOriginalPPL = new Mat();
        if (frontAnonymizedPPLInputImgPath != null) frontOriginalPPL = Imgcodecs.imread(frontAnonymizedPPLInputImgPath);

        Mat frontOriginalXPL = new Mat();
        if (frontAnonymizedXPLInputImgPath != null) frontOriginalXPL = Imgcodecs.imread(frontAnonymizedXPLInputImgPath);

        Mat frontOriginalUVL = new Mat();
        if (frontAnonymizedUVLInputImgPath != null) frontOriginalUVL = Imgcodecs.imread(frontAnonymizedUVLInputImgPath);

        Mat frontFullFaceMask = new Mat();
        if(frontFullFaceMaskInputPath != null) frontFullFaceMask = imread(frontFullFaceMaskInputPath, IMREAD_GRAYSCALE);

        Mat leftOriginalPPL = new Mat();
        Mat leftOriginalXPL = new Mat();
        Mat leftOriginalUVL = new Mat();
        Mat leftFullFaceMask = new Mat();
        Mat rightOriginalPPL = new Mat();
        Mat rightOriginalXPL = new Mat();
        Mat rightOriginalUVL = new Mat();
        Mat rightFullFaceMask = new Mat();
        if(sideFaceImagesEnabled) {
            if (leftAnonymizedPPLInputImgPath != null)
                leftOriginalPPL = Imgcodecs.imread(leftAnonymizedPPLInputImgPath);
            if (leftAnonymizedXPLInputImgPath != null)
                leftOriginalXPL = Imgcodecs.imread(leftAnonymizedXPLInputImgPath);
            if (leftAnonymizedUVLInputImgPath != null)
                leftOriginalUVL = Imgcodecs.imread(leftAnonymizedUVLInputImgPath);
            if (leftFullFaceMaskInputPath != null)
                leftFullFaceMask = imread(leftFullFaceMaskInputPath, IMREAD_GRAYSCALE);
            if (rightAnonymizedPPLInputImgPath != null)
                rightOriginalPPL = Imgcodecs.imread(rightAnonymizedPPLInputImgPath);
            if (rightAnonymizedXPLInputImgPath != null)
                rightOriginalXPL = Imgcodecs.imread(rightAnonymizedXPLInputImgPath);
            if (rightAnonymizedUVLInputImgPath != null)
                rightOriginalUVL = Imgcodecs.imread(rightAnonymizedUVLInputImgPath);
            if (rightFullFaceMaskInputPath != null)
                rightFullFaceMask = imread(rightFullFaceMaskInputPath, IMREAD_GRAYSCALE);
        }

        final int originalHeight = frontOriginalPPL.rows();
        final int originalWidth = frontOriginalPPL.cols();
        final int channelsNum = frontOriginalPPL.channels();

        Mat frontChinMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontForeheadMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontCheekMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontNoseMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);

        Mat leftChinMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftForeheadMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftCheekMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftNoseMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);

        Mat rightChinMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightForeheadMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightCheekMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightNoseMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);

        Mat oilinessROI = Mat.zeros(originalHeight, originalWidth, CvType.CV_8UC3);
        Mat radianceDullnessROI = Mat.zeros(originalHeight, originalWidth, CvType.CV_8UC3);
        Mat poresROI = Mat.zeros(originalHeight, originalWidth, CvType.CV_8UC3);
        Mat impuritiesROI = Mat.zeros(originalHeight, originalWidth, CvType.CV_8UC3);

        int frontImage = 0;
        int leftImage = 1;
        int rightImage = 2;

        int imgNum = 0;
        if(sideFaceImagesEnabled) imgNum = 3;
        else imgNum = 1;

        // ----- (1) ----- Prepare ROIs using Face Mesh
        Log.d("ImageSegmentation", "-- SHU LI: CFA ROI Begins.");

        for (int position = 0; position < imgNum; position++) {
            // position 0: front
            // position 1: left
            // position 2: right
            if(position == frontImage) {
                // Define lists of FaceMesh key points for ROIs
                // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
                List<Integer> foreheadBottomLine = Collections.unmodifiableList(Arrays.asList(127, 139, 71, 70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300, 368, 356));

                List<Integer> fullFaceContourShape = Collections.unmodifiableList(Arrays.asList(21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162));
                List<Integer> rightCheekBoundShape = Collections.unmodifiableList(Arrays.asList(93, 177, 58, 172, 214, 216, 206, 203, 126, 47, 121, 120, 119, 118, 117, 111, 116));
                List<Integer> leftCheekBoundShape = Collections.unmodifiableList(Arrays.asList(340, 345, 366, 401, 435, 367, 434, 436, 426, 423, 358, 429, 355, 277, 349, 348, 347, 346));
                List<Integer> chinBoundShape = Collections.unmodifiableList(Arrays.asList(43, 202, 210, 169, 150, 149, 176, 148, 152, 377, 400, 378, 379, 394, 430, 422, 273, 335, 406, 313, 18, 83, 182, 106));
                List<Integer> noseBoundShape = Collections.unmodifiableList(Arrays.asList(107, 55, 189, 244, 128, 114, 126, 209, 49, 129, 64, 98, 97, 2, 326, 327, 278, 279, 429, 355, 277, 357, 464, 413, 285, 336, 9));

                int x = 0, y = 0;
                List<Point> rightCheekPoints = new ArrayList<>();
                for(Integer index : rightCheekBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    rightCheekPoints.add(new Point(x, y));
                }
                MatOfPoint rightCheekPointMat = new MatOfPoint();
                rightCheekPointMat.fromList(rightCheekPoints);
                List<MatOfPoint> rightCheekContour = new ArrayList<>();
                rightCheekContour.add(rightCheekPointMat);

                List<Point> leftCheekPoints = new ArrayList<>();
                for(Integer index : leftCheekBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    leftCheekPoints.add(new Point(x, y));
                }
                MatOfPoint leftCheekPointMat = new MatOfPoint();
                leftCheekPointMat.fromList(leftCheekPoints);
                List<MatOfPoint> leftCheekContour = new ArrayList<>();
                leftCheekContour.add(leftCheekPointMat);

                List<Point> nosePoints = new ArrayList<>();
                for(Integer index : noseBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    nosePoints.add(new Point(x, y));
                }
                MatOfPoint nosePointMat = new MatOfPoint();
                nosePointMat.fromList(nosePoints);
                List<MatOfPoint> noseContour = new ArrayList<>();
                noseContour.add(nosePointMat);

                List<Point> chinPoints = new ArrayList<>();
                for(Integer index : chinBoundShape) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    chinPoints.add(new Point(x, y));
                }
                MatOfPoint chinPointMat = new MatOfPoint();
                chinPointMat.fromList(chinPoints);
                List<MatOfPoint> chinContour = new ArrayList<>();
                chinContour.add(chinPointMat);

                // Get cheek mask
                drawContours(frontCheekMask, rightCheekContour, -1, new Scalar(255), FILLED);
                drawContours(frontCheekMask, leftCheekContour, -1, new Scalar(255), FILLED);

                // Get nose mask
                drawContours(frontNoseMask, noseContour, -1, new Scalar(255), FILLED);

                // Get chin mask
                drawContours(frontChinMask, chinContour, -1, new Scalar(255), FILLED);

                // Get forehead mask.
                // --(1)-- face-mesh-forehead and the image region above.
                List<Point> foreheadBoundLinePoints = new ArrayList<>();
                for(Integer index : foreheadBottomLine) {
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    foreheadBoundLinePoints.add(new Point(x, y));
                }
                // --(2)-- Close the bounding shape of forehead region by setting 2 vertices at top edge of the image.
                // Offset by 50 pixels maximum.
                int offset = 50;
                int xLast = (x + offset <= originalWidth - 1) ? (x + offset) : originalWidth - 1;
                int x0 = ((int)foreheadBoundLinePoints.get(0).x - offset) >= 0 ? ((int)foreheadBoundLinePoints.get(0).x - offset) : 0;
                foreheadBoundLinePoints.add(0, new Point(x0, offset));
                foreheadBoundLinePoints.add(new Point(xLast, offset));

                MatOfPoint foreheadPoints = new MatOfPoint();
                foreheadPoints.fromList(foreheadBoundLinePoints);
                List<MatOfPoint> foreheadBoundContour = new ArrayList<>();
                foreheadBoundContour.add(foreheadPoints);

                drawContours(frontForeheadMask, foreheadBoundContour, -1, new Scalar(255), FILLED);
                bitwise_and(frontForeheadMask, frontFullFaceMask, frontForeheadMask);

                // Update full face mask to remove ears
                // -- NOTE -- face mesh full face contour, only includes part of forehead
                List<Point> fullFaceContourLinePoints = new ArrayList<>();
                for(Integer index : fullFaceContourShape){
                    x = frontCoordinates.get(index).get(0).intValue();
                    y = frontCoordinates.get(index).get(1).intValue();
                    fullFaceContourLinePoints.add(new Point(x, y));
                }
                MatOfPoint fullFaceContourLinePointMat = new MatOfPoint();
                fullFaceContourLinePointMat.fromList(fullFaceContourLinePoints);
                List<MatOfPoint> fullFaceContour = new ArrayList<>();
                fullFaceContour.add(fullFaceContourLinePointMat);

                Mat newFullFaceMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
                drawContours(newFullFaceMask, fullFaceContour, -1, new Scalar(255), FILLED);
                bitwise_or(newFullFaceMask, frontForeheadMask, newFullFaceMask);

                // Contain ROI mask images inside newly updated full face mask
                bitwise_and(newFullFaceMask, frontCheekMask, frontCheekMask);
                bitwise_and(newFullFaceMask, frontForeheadMask, frontForeheadMask);
                bitwise_and(newFullFaceMask, frontChinMask, frontChinMask);

                imwrite(frontFullFaceMaskInputPath, newFullFaceMask);
                imwrite(frontForeheadMaskOutputPath, frontForeheadMask);
                imwrite(frontCheekMaskOutputPath, frontCheekMask);
                imwrite(frontNoseMaskOutputPath, frontNoseMask);
                imwrite(frontChinMaskOutputPath, frontChinMask);

                //MyUtil.saveMatToGallery(context, "front cheek.jpg", "cheek roi with facemesh", frontCheekMask);
                //MyUtil.saveMatToGallery(context, "front nose.jpg", "nose roi with facemesh", frontNoseMask);
                //MyUtil.saveMatToGallery(context, "front chin.jpg", "chin roi with facemesh", frontChinMask);
                //MyUtil.saveMatToGallery(context, "front forehead.jpg", "forehead roi with facemesh", frontForeheadMask);
                //MyUtil.saveMatToGallery(context, "front full face.jpg", "updated front full face", newFullFaceMask);
            }

            if(sideFaceImagesEnabled) {
                if(position == leftImage) {
                    // Define lists of FaceMesh key points for ROIs
                    // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
                    List<Integer> foreheadBottomLine = Collections.unmodifiableList(Arrays.asList(108, 151, 337, 299, 333, 298, 301, 368, 389));

                    List<Integer> leftFaceContourLine = Collections.unmodifiableList(Arrays.asList(400, 378, 379, 365, 397, 288, 361, 323, 454, 356));
                    List<Integer> leftCheekBoundShape = Collections.unmodifiableList(Arrays.asList(349, 329, 371, 423, 426, 436, 432, 364, 397, 288, 361, 323, 447, 345, 346, 347, 348));
                    List<Integer> chinBoundShape = Collections.unmodifiableList(Arrays.asList(201, 208, 171, 148, 152, 377, 400, 378, 379, 394, 430, 422, 273, 335, 406, 313, 18, 83));
                    List<Integer> noseBoundShape = Collections.unmodifiableList(Arrays.asList(9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 326, 327, 278, 279, 429, 355, 277, 357, 464, 413, 285, 9));

                    int x = 0, y = 0;

                    List<Point> leftCheekPoints = new ArrayList<>();
                    for(Integer index : leftCheekBoundShape) {
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        leftCheekPoints.add(new Point(x, y));
                    }
                    MatOfPoint leftCheekPointMat = new MatOfPoint();
                    leftCheekPointMat.fromList(leftCheekPoints);
                    List<MatOfPoint> leftCheekContour = new ArrayList<>();
                    leftCheekContour.add(leftCheekPointMat);

                    List<Point> nosePoints = new ArrayList<>();
                    for(Integer index : noseBoundShape) {
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        nosePoints.add(new Point(x, y));
                    }
                    MatOfPoint nosePointMat = new MatOfPoint();
                    nosePointMat.fromList(nosePoints);
                    List<MatOfPoint> noseContour = new ArrayList<>();
                    noseContour.add(nosePointMat);

                    List<Point> chinPoints = new ArrayList<>();
                    for(Integer index : chinBoundShape) {
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        chinPoints.add(new Point(x, y));
                    }
                    MatOfPoint chinPointMat = new MatOfPoint();
                    chinPointMat.fromList(chinPoints);
                    List<MatOfPoint> chinContour = new ArrayList<>();
                    chinContour.add(chinPointMat);

                    // Get cheek mask
                    drawContours(leftCheekMask, leftCheekContour, -1, new Scalar(255), FILLED);

                    // Get nose mask
                    drawContours(leftNoseMask, noseContour, -1, new Scalar(255), FILLED);

                    // Get chin mask
                    drawContours(leftChinMask, chinContour, -1, new Scalar(255), FILLED);

                    // Get forehead mask.
                    // --(1)-- face-mesh-forehead and the image region above.
                    List<Point> foreheadBoundLinePoints = new ArrayList<>();
                    for(Integer index : foreheadBottomLine) {
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        foreheadBoundLinePoints.add(new Point(x, y));
                    }
                    // Close the bounding shape of forehead region by setting 2 vertices at top edge of the image.
                    // Offset by 50 pixels maximum.
                    int offset = 50;
                    int xLast = (x + offset <= originalWidth - 1) ? (x + offset) : originalWidth - 1;
                    int x0 = ((int)foreheadBoundLinePoints.get(0).x - offset) >= 0 ? ((int)foreheadBoundLinePoints.get(0).x - offset) : 0;
                    foreheadBoundLinePoints.add(0, new Point(x0, offset));
                    foreheadBoundLinePoints.add(new Point(xLast, offset));

                    MatOfPoint foreheadPoints = new MatOfPoint();
                    foreheadPoints.fromList(foreheadBoundLinePoints);
                    List<MatOfPoint> foreheadBoundContour = new ArrayList<>();
                    foreheadBoundContour.add(foreheadPoints);

                    drawContours(leftForeheadMask, foreheadBoundContour, -1, new Scalar(255), FILLED);
                    bitwise_and(leftForeheadMask, leftFullFaceMask, leftForeheadMask);

                    // Update full face mask to remove ears
                    // -- NOTE -- Different logic for left side image. Can't use the same full face contour as front face image.
                    List<Point> leftFaceContourLinePoints = new ArrayList<>();
                    for(Integer index : leftFaceContourLine){
                        x = leftCoordinates.get(index).get(0).intValue();
                        y = leftCoordinates.get(index).get(1).intValue();
                        leftFaceContourLinePoints.add(new Point(x, y));
                    }
                    leftFaceContourLinePoints.add(new Point(x, 0));
                    leftFaceContourLinePoints.add(new Point(0, 0));
                    leftFaceContourLinePoints.add(new Point(0, originalHeight - 1));
                    x0 = (int)leftFaceContourLinePoints.get(0).x;
                    leftFaceContourLinePoints.add(0, new Point(x0, originalHeight - 1));

                    MatOfPoint leftFaceBoundContourPointMat = new MatOfPoint();
                    leftFaceBoundContourPointMat.fromList(leftFaceContourLinePoints);
                    List<MatOfPoint> leftFaceBoundContour = new ArrayList<>();
                    leftFaceBoundContour.add(leftFaceBoundContourPointMat);

                    Mat newFullFaceMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
                    drawContours(newFullFaceMask, leftFaceBoundContour, -1, new Scalar(255), FILLED);
                    bitwise_and(newFullFaceMask, leftFullFaceMask, newFullFaceMask);

                    // Contain ROI mask images inside newly updated full face mask
                    bitwise_and(newFullFaceMask, leftCheekMask, leftCheekMask);
                    bitwise_and(newFullFaceMask, leftForeheadMask, leftForeheadMask);
                    bitwise_and(newFullFaceMask, leftChinMask, leftChinMask);

                    imwrite(leftFullFaceMaskInputPath, newFullFaceMask);
                    imwrite(leftForeheadMaskOutputPath, leftForeheadMask);
                    imwrite(leftCheekMaskOutputPath, leftCheekMask);
                    imwrite(leftNoseMaskOutputPath, leftNoseMask);
                    imwrite(leftChinMaskOutputPath, leftChinMask);

                    //MyUtil.saveMatToGallery(context, "left cheek.jpg", "cheek roi with facemesh", leftCheekMask);
                    //MyUtil.saveMatToGallery(context, "left nose.jpg", "nose roi with facemesh", leftNoseMask);
                    //MyUtil.saveMatToGallery(context, "left chin.jpg", "chin roi with facemesh", leftChinMask);
                    //MyUtil.saveMatToGallery(context, "left forehead.jpg", "forehead roi with facemesh", leftForeheadMask);
                    //MyUtil.saveMatToGallery(context, "left full face.jpg", "updated front full face", newFullFaceMask);
                }

                if(position == rightImage) {
                    // Define lists of FaceMesh key points for ROIs
                    // DO NOT USE JAVA SETs as sets are not ordered and the opencv drawing function will generate unusual results.
                    List<Integer> foreheadBottomLine = Collections.unmodifiableList(Arrays.asList(337, 151, 108, 69, 104, 68, 71, 139, 162));

                    List<Integer> rightFaceContourLine = Collections.unmodifiableList(Arrays.asList(149, 150, 136, 172, 58, 132, 93, 234, 127));
                    List<Integer> rightCheekBoundShape = Collections.unmodifiableList(Arrays.asList(93, 132, 58, 172, 136, 150, 210, 212, 216, 206, 203, 142, 101, 118, 117, 111, 116, 227));
                    List<Integer> chinBoundShape = Collections.unmodifiableList(Arrays.asList(43, 202, 210, 169, 150, 149, 176, 148, 152, 377, 400, 378, 379, 394, 430, 422, 273, 335, 406, 313, 18, 83, 182, 106));
                    List<Integer> noseBoundShape = Collections.unmodifiableList(Arrays.asList(9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 97, 98, 64, 129, 49, 209, 126, 114, 128, 244, 189, 55));

                    int x = 0, y = 0;

                    List<Point> rightCheekPoints = new ArrayList<>();
                    for(Integer index : rightCheekBoundShape) {
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        rightCheekPoints.add(new Point(x, y));
                    }
                    MatOfPoint rightCheekPointMat = new MatOfPoint();
                    rightCheekPointMat.fromList(rightCheekPoints);
                    List<MatOfPoint> rightCheekContour = new ArrayList<>();
                    rightCheekContour.add(rightCheekPointMat);

                    List<Point> nosePoints = new ArrayList<>();
                    for(Integer index : noseBoundShape) {
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        nosePoints.add(new Point(x, y));
                    }
                    MatOfPoint nosePointMat = new MatOfPoint();
                    nosePointMat.fromList(nosePoints);
                    List<MatOfPoint> noseContour = new ArrayList<>();
                    noseContour.add(nosePointMat);

                    List<Point> chinPoints = new ArrayList<>();
                    for(Integer index : chinBoundShape) {
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        chinPoints.add(new Point(x, y));
                    }
                    MatOfPoint chinPointMat = new MatOfPoint();
                    chinPointMat.fromList(chinPoints);
                    List<MatOfPoint> chinContour = new ArrayList<>();
                    chinContour.add(chinPointMat);

                    // Get cheek mask
                    drawContours(rightCheekMask, rightCheekContour, -1, new Scalar(255), FILLED);

                    // Get nose mask
                    drawContours(rightNoseMask, noseContour, -1, new Scalar(255), FILLED);

                    // Get chin mask
                    drawContours(rightChinMask, chinContour, -1, new Scalar(255), FILLED);

                    // Get forehead mask.
                    // --(1)-- face-mesh-forehead and the image region above.
                    List<Point> foreheadBoundLinePoints = new ArrayList<>();
                    for(Integer index : foreheadBottomLine) {
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        foreheadBoundLinePoints.add(new Point(x, y));
                    }
                    // Close the bounding shape of forehead region by setting 2 vertices at top edge of the image.
                    // Offset by 50 pixels maximum.
                    int offset = 50;
                    int xLast = (x + offset <= originalWidth - 1) ? (x + offset) : originalWidth - 1;
                    int x0 = ((int)foreheadBoundLinePoints.get(0).x - offset) >= 0 ? ((int)foreheadBoundLinePoints.get(0).x - offset) : 0;
                    foreheadBoundLinePoints.add(0, new Point(x0, offset));
                    foreheadBoundLinePoints.add(new Point(xLast, offset));

                    MatOfPoint foreheadPoints = new MatOfPoint();
                    foreheadPoints.fromList(foreheadBoundLinePoints);
                    List<MatOfPoint> foreheadBoundContour = new ArrayList<>();
                    foreheadBoundContour.add(foreheadPoints);

                    drawContours(rightForeheadMask, foreheadBoundContour, -1, new Scalar(255), FILLED);
                    bitwise_and(rightForeheadMask, rightFullFaceMask, rightForeheadMask);

                    // Update full face mask to remove ears
                    // -- NOTE -- Different logic for left side image. Can't use the same full face contour as front face image.
                    List<Point> rightFaceContourLinePoints = new ArrayList<>();
                    for(Integer index : rightFaceContourLine){
                        x = rightCoordinates.get(index).get(0).intValue();
                        y = rightCoordinates.get(index).get(1).intValue();
                        rightFaceContourLinePoints.add(new Point(x, y));
                    }
                    rightFaceContourLinePoints.add(new Point(x, 0));
                    rightFaceContourLinePoints.add(new Point(originalWidth - 1, 0));
                    rightFaceContourLinePoints.add(new Point(originalWidth - 1, originalHeight - 1));
                    x0 = (int)rightFaceContourLinePoints.get(0).x;
                    rightFaceContourLinePoints.add(0, new Point(x0, originalHeight - 1));

                    MatOfPoint rightFaceBoundContourPointMat = new MatOfPoint();
                    rightFaceBoundContourPointMat.fromList(rightFaceContourLinePoints);
                    List<MatOfPoint> rightFaceBoundContour = new ArrayList<>();
                    rightFaceBoundContour.add(rightFaceBoundContourPointMat);

                    Mat newFullFaceMask = Mat.zeros(new Size(originalWidth, originalHeight), CV_8UC1);
                    drawContours(newFullFaceMask, rightFaceBoundContour, -1, new Scalar(255), FILLED);
                    bitwise_and(newFullFaceMask, rightFullFaceMask, newFullFaceMask);

                    // Contain ROI mask images inside newly updated full face mask
                    bitwise_and(newFullFaceMask, rightCheekMask, rightCheekMask);
                    bitwise_and(newFullFaceMask, rightForeheadMask, rightForeheadMask);
                    bitwise_and(newFullFaceMask, rightChinMask, rightChinMask);

                    imwrite(rightFullFaceMaskInputPath, newFullFaceMask);
                    imwrite(rightForeheadMaskOutputPath, rightForeheadMask);
                    imwrite(rightCheekMaskOutputPath, rightCheekMask);
                    imwrite(rightNoseMaskOutputPath, rightNoseMask);
                    imwrite(rightChinMaskOutputPath, rightChinMask);

                    //MyUtil.saveMatToGallery(context, "right cheek.jpg", "cheek roi with facemesh", rightCheekMask);
                    //MyUtil.saveMatToGallery(context, "right nose.jpg", "nose roi with facemesh", rightNoseMask);
                    //MyUtil.saveMatToGallery(context, "right chin.jpg", "chin roi with facemesh", rightChinMask);
                    //MyUtil.saveMatToGallery(context, "right forehead.jpg", "forehead roi with facemesh", rightForeheadMask);
                    //MyUtil.saveMatToGallery(context, "right full face.jpg", "updated front full face", newFullFaceMask);
                }
            }
        }

        // ------ (1) ------ Get ROIs for other Image Processing measurements.
        //
        Imgproc.cvtColor(frontChinMask, frontChinMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(frontForeheadMask, frontForeheadMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(frontCheekMask, frontCheekMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(frontNoseMask, frontNoseMask, COLOR_GRAY2BGR);

        Imgproc.cvtColor(leftChinMask, leftChinMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftForeheadMask, leftForeheadMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftCheekMask, leftCheekMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftNoseMask, leftNoseMask, COLOR_GRAY2BGR);

        Imgproc.cvtColor(rightChinMask, rightChinMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightForeheadMask, rightForeheadMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightCheekMask, rightCheekMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightNoseMask, rightNoseMask, COLOR_GRAY2BGR);

        // ------ (2) ------ Prepare and save ROI images for redness.
        //
        Mat frontRednessROI = new Mat(), leftRednessROI = new Mat(), rightRednessROI = new Mat();

        Core.bitwise_or(frontChinMask, frontNoseMask, frontRednessROI);
        Core.bitwise_or(frontRednessROI, frontCheekMask, frontRednessROI);
        bitwise_and(frontRednessROI, frontOriginalXPL, frontRednessROI);

        if (frontRednessROIOutputPath != null)
            Imgcodecs.imwrite(frontRednessROIOutputPath, frontRednessROI);

        if(sideFaceImagesEnabled) {
            Core.bitwise_or(leftChinMask, leftNoseMask, leftRednessROI);
            Core.bitwise_or(leftRednessROI, leftCheekMask, leftRednessROI);
            bitwise_and(leftRednessROI, leftOriginalXPL, leftRednessROI);

            Core.bitwise_or(rightChinMask, rightNoseMask, rightRednessROI);
            Core.bitwise_or(rightRednessROI, rightCheekMask, rightRednessROI);
            bitwise_and(rightRednessROI, rightOriginalXPL, rightRednessROI);

            if (leftRednessROIOutputPath != null)
                Imgcodecs.imwrite(leftRednessROIOutputPath, leftRednessROI);
            if (rightRednessROIOutputPath != null)
                Imgcodecs.imwrite(rightRednessROIOutputPath, rightRednessROI);
        }

        //MyUtil.saveMatToGallery(context, "dummy front redness ROI", "front redness face ROI", frontRednessROI);
        if(sideFaceImagesEnabled) {
            //MyUtil.saveMatToGallery(context, "dummy left redness ROI", "left redness face ROI", leftRednessROI);
            //MyUtil.saveMatToGallery(context, "dummy right redness ROI", "right redness face ROI", rightRednessROI);
        }

        // ------ (3) ------ Prepare and save ROI image for oiliness.
        //
        Mat oilinessFrontROI = new Mat();

        Core.bitwise_or(frontChinMask, frontForeheadMask, oilinessFrontROI);
        Core.bitwise_or(oilinessFrontROI, frontCheekMask, oilinessFrontROI);
        Core.bitwise_or(oilinessFrontROI, frontNoseMask, oilinessFrontROI);
        bitwise_and(oilinessFrontROI, frontOriginalPPL, oilinessFrontROI);

        if (oilinessFrontROIOutputPath != null)
            Imgcodecs.imwrite(oilinessFrontROIOutputPath, oilinessFrontROI);

        //MyUtil.saveMatToGallery(context, "dummy oiliness ROI", "oiliness face ROI", oilinessFrontROI);

        // ----- (4) ----- Prepare and save ROI image for Radiance and Dullness.
        //
        Mat radianceFrontROI = new Mat();

        Core.bitwise_or(frontForeheadMask, frontCheekMask, radianceFrontROI);
        bitwise_and(radianceFrontROI, frontOriginalPPL, radianceFrontROI);

        if (radianceFrontROIOutputPath != null)
            Imgcodecs.imwrite(radianceFrontROIOutputPath, radianceFrontROI);

        //MyUtil.saveMatToGallery(context, "dummy radiance ROI", "radiance face ROI", radianceFrontROI);

        // ----- (5) ----- Prepare and save ROI image for impurities.
        //
        if (impuritiesEnabled) {
            Mat impuritiesFrontROI = new Mat();

            Core.bitwise_or(frontNoseMask, frontCheekMask, impuritiesFrontROI);
            Core.bitwise_or(impuritiesFrontROI, frontChinMask, impuritiesFrontROI);
            bitwise_and(impuritiesFrontROI, frontOriginalUVL, impuritiesFrontROI);

            if (impuritiesFrontROIOutputPath != null)
                Imgcodecs.imwrite(impuritiesFrontROIOutputPath, impuritiesFrontROI);

            //MyUtil.saveMatToGallery(context, "dummy impurities ROI", "impurities face ROI", impuritiesFrontROI);
        }
    }
}

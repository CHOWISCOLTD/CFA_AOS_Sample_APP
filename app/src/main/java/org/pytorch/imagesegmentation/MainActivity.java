package org.pytorch.imagesegmentation;

import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import com.chowis.JniComputationPro.JniLocalComputationQA;
import com.chowis.jniimagepro.CFA.JNICFAImageProCW;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import com.google.mediapipe.imagesegmenter.SelfieSegmentation;
import com.google.mediapipe.facelandmarker.FaceMeshLandmarker;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements Runnable {

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("torchvision_ops");
    }

    private Button mButtonSegment;
    //private Module roiAIModule = null;
    private File poresTFModelFile = null;
    private File darkCircleTFModelFile = null;
    private File hyperPigmentationTFModelFile = null;
    private File pigmentationSpotsTFModelFile = null;
    private File wrinklesTFModelFile = null;
    private File impuritiesTFModelFile = null;
    private String localBatchID = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        org.opencv.android.OpenCVLoader.initDebug();

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButtonSegment = findViewById(R.id.segmentButton);
        mButtonSegment.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                mButtonSegment.setText(getString(R.string.run_model));

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        // Create File objects for AI models.
        try {
            poresTFModelFile = new File(MyUtil.assetFilePath(getApplicationContext(), "CFA_pores_segmented_effnet_ep_8_20230329_v1.tflite"));
            darkCircleTFModelFile = new File(MyUtil.assetFilePath(getApplicationContext(), "darkcircles_v1_20231212_metadata.tflite"));
            hyperPigmentationTFModelFile = new File(MyUtil.assetFilePath(getApplicationContext(), "hyperpigmentation_v3_20231212_metadata.tflite"));
            pigmentationSpotsTFModelFile = new File(MyUtil.assetFilePath(getApplicationContext(), "pigmentation_v1_20231212_metadata.tflite"));
            wrinklesTFModelFile = new File(MyUtil.assetFilePath(getApplicationContext(), "CFA_wrinkles_segmented_v3 1.tflite"));
            impuritiesTFModelFile = new File(MyUtil.assetFilePath(getApplicationContext(), "CFA_impurities_segmented_attn_v1_20240326.tflite"));
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }
    }

    @Override
    public void run() {
        long startTimeSeconds = System.currentTimeMillis() / 1000;

        boolean capturedWithFrontCamera = true;
        boolean sideFaceImagesEnabled = true;

        double usedFrontCamera = -1;
        if(capturedWithFrontCamera) usedFrontCamera = 1;
        if(!capturedWithFrontCamera) usedFrontCamera = 0;

        // -------------- Generate local batch ID using current system date and time.
        localBatchID = MyUtil.createLocalBatchID();

        // -------------- Create folder for CFA/CFP APPs.
        String parentFolder = "/Documents/";
        String appFolderName = "mySkinFain";

        // -------------- Create new folder for new BATCH in "Documents".
        parentFolder = "/Documents/" + appFolderName + "/";
        String newBatchFolderName = localBatchID;
        String newBatchFolder = null;
        try {
            newBatchFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, newBatchFolderName);
            System.out.println("Created new batch folder: " + newBatchFolder);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating new batch folder", e);
        }

        parentFolder = parentFolder + newBatchFolderName + "/";

        // ------------------------------ (0) Prepare Input Paths of Captured Original Images ------------------------------
        //
        String frontPPLOriginalAssetPath = null;
        String frontXPLOriginalAssetPath = null;
        String frontUVLOriginalAssetPath = null;

        String leftPPLOriginalAssetPath = null;
        String leftXPLOriginalAssetPath = null;
        String leftUVLOriginalAssetPath = null;

        String rightPPLOriginalAssetPath = null;
        String rightXPLOriginalAssetPath = null;
        String rightUVLOriginalAssetPath = null;

        try {
            // input : original
            frontPPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "ppl front.jpg");
            frontXPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "xpl front.jpg");
            frontUVLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "uvl front.jpg");

            leftPPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "ppl left.jpg");
            leftXPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "xpl left.jpg");
            leftUVLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "uvl left.jpg");

            rightPPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "ppl right.jpg");
            rightXPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "xpl right.jpg");
            rightUVLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "uvl right.jpg");
        } catch(IOException e) {
            e.printStackTrace();
        }

        // Create file paths for original images.
        String originalFrontPPLFolder = "original front ppl";
        String originalFrontXPLFolder = "original front xpl";
        String originalFrontUVLFolder = "original front uvl";

        String originalLeftPPLFolder = "original left ppl";
        String originalLeftXPLFolder = "original left xpl";
        String originalLeftUVLFolder = "original left uvl";

        String originalRightPPLFolder = "original right ppl";
        String originalRightXPLFolder = "original right xpl";
        String originalRightUVLFolder = "original right uvl";

        String originalFrontPPLFilename = "ppl front.jpg";
        String originalFrontXPLFilename = "xpl front.jpg";
        String originalFrontUVLFilename = "uvl front.jpg";

        String originalLeftPPLFilename = "ppl left.jpg";
        String originalLeftXPLFilename = "xpl left.jpg";
        String originalLeftUVLFilename = "uvl left.jpg";

        String originalRightPPLFilename = "ppl right.jpg";
        String originalRightXPLFilename = "xpl right.jpg";
        String originalRightUVLFilename = "uvl right.jpg";

        String originalFrontPPLPath = null;
        String originalFrontXPLPath = null;
        String originalFrontUVLPath = null;

        String originalLeftPPLPath = null;
        String originalLeftXPLPath = null;
        String originalLeftUVLPath = null;

        String originalRightPPLPath = null;
        String originalRightXPLPath = null;
        String originalRightUVLPath = null;

        try{
            originalFrontPPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalFrontPPLFolder, originalFrontPPLFilename);
            originalFrontXPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalFrontXPLFolder, originalFrontXPLFilename);
            originalFrontUVLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalFrontUVLFolder, originalFrontUVLFilename);

            originalLeftPPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalLeftPPLFolder, originalLeftPPLFilename);
            originalLeftXPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalLeftXPLFolder, originalLeftXPLFilename);
            originalLeftUVLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalLeftUVLFolder, originalLeftUVLFilename);

            originalRightPPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalRightPPLFolder, originalRightPPLFilename);
            originalRightXPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalRightXPLFolder, originalRightXPLFilename);
            originalRightUVLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalRightUVLFolder, originalRightUVLFilename);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating anonymized image output folders", e);
        }

        // Read files from assets, then save to their respective file paths.
        Mat frontPPLImg = imread(frontPPLOriginalAssetPath);
        Mat frontXPLImg = imread(frontXPLOriginalAssetPath);
        Mat frontUVLImg = imread(frontUVLOriginalAssetPath);

        Mat leftPPLImg = imread(leftPPLOriginalAssetPath);
        Mat leftXPLImg = imread(leftXPLOriginalAssetPath);
        Mat leftUVLImg = imread(leftUVLOriginalAssetPath);

        Mat rightPPLImg = imread(rightPPLOriginalAssetPath);
        Mat rightXPLImg = imread(rightXPLOriginalAssetPath);
        Mat rightUVLImg = imread(rightUVLOriginalAssetPath);

        imwrite(originalFrontPPLPath, frontPPLImg);
        imwrite(originalFrontXPLPath, frontXPLImg);
        imwrite(originalFrontUVLPath, frontUVLImg);

        imwrite(originalLeftPPLPath, leftPPLImg);
        imwrite(originalLeftXPLPath, leftXPLImg);
        imwrite(originalLeftUVLPath, leftUVLImg);

        imwrite(originalRightPPLPath, rightPPLImg);
        imwrite(originalRightXPLPath, rightXPLImg);
        imwrite(originalRightUVLPath, rightUVLImg);

        File frontPPL = new File(originalFrontPPLPath);
        File frontXPL = new File(originalFrontXPLPath);
        File frontUVL = new File(originalFrontUVLPath);

        File leftPPL = new File(originalLeftPPLPath);
        File leftXPL = new File(originalLeftXPLPath);
        File leftUVL = new File(originalLeftUVLPath);

        File rightPPL = new File(originalRightPPLPath);
        File rightXPL = new File(originalRightXPLPath);
        File rightUVL = new File(originalRightUVLPath);

        // ------------------------------ (1) Face Anonymization to remove background ------------------------------
        //
        // File paths for anonymized images.
        String anonymizedFrontPPLImgFolderName = "anonymized PPL front image";
        String anonymizedFrontXPLImgFolderName = "anonymized XPL front image";
        String anonymizedFrontUVLImgFolderName = "anonymized UVL front image";
        String anonymizedFrontFullFaceMaskFolderName = "anonymized front full face mask";

        String anonymizedLeftPPLImgFolderName = "anonymized PPL left image";
        String anonymizedLeftXPLImgFolderName = "anonymized XPL left image";
        String anonymizedLeftUVLImgFolderName = "anonymized UVL left image";
        String anonymizedLeftFullFaceMaskFolderName = "anonymized left full face mask";

        String anonymizedRightPPLImgFolderName = "anonymized PPL right image";
        String anonymizedRightXPLImgFolderName = "anonymized XPL right image";
        String anonymizedRightUVLImgFolderName = "anonymized UVL right image";
        String anonymizedRightFullFaceMaskFolderName = "anonymized right full face mask";

        String anonymizedFrontPPLImgFileName = "anonymized front PPL.jpg";
        String anonymizedFrontXPLImgFileName = "anonymized front XPL.jpg";
        String anonymizedFrontUVLImgFileName = "anonymized front UVL.jpg";
        String anonymizedFrontFullFaceMaskImgName = "anonymized front full face mask.jpg";

        String anonymizedLeftPPLImgFileName = "anonymized left PPL.jpg";
        String anonymizedLeftXPLImgFileName = "anonymized left XPL.jpg";
        String anonymizedLeftUVLImgFileName = "anonymized left UVL.jpg";
        String anonymizedLeftFullFaceMaskImgName = "anonymized left full face mask.jpg";

        String anonymizedRightPPLImgFileName = "anonymized right PPL.jpg";
        String anonymizedRightXPLImgFileName = "anonymized right XPL.jpg";
        String anonymizedRightUVLImgFileName = "anonymized right UVL.jpg";
        String anonymizedRightFullFaceMaskImgName = "anonymized right full face mask.jpg";

        String anonymizedFrontPPLImgOutputPath = null;
        String anonymizedFrontXPLImgOutputPath = null;
        String anonymizedFrontUVLImgOutputPath = null;
        String anonymizedFrontFullFaceMaskOutputPath = null;

        String anonymizedLeftPPLImgOutputPath = null;
        String anonymizedLeftXPLImgOutputPath = null;
        String anonymizedLeftUVLImgOutputPath = null;
        String anonymizedLeftFullFaceMaskOutputPath = null;

        String anonymizedRightPPLImgOutputPath = null;
        String anonymizedRightXPLImgOutputPath = null;
        String anonymizedRightUVLImgOutputPath = null;
        String anonymizedRightFullFaceMaskOutputPath = null;

        try{
            anonymizedFrontPPLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedFrontPPLImgFolderName, anonymizedFrontPPLImgFileName);
            anonymizedFrontXPLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedFrontXPLImgFolderName, anonymizedFrontXPLImgFileName);
            anonymizedFrontUVLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedFrontUVLImgFolderName, anonymizedFrontUVLImgFileName);
            anonymizedFrontFullFaceMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedFrontFullFaceMaskFolderName, anonymizedFrontFullFaceMaskImgName);

            anonymizedLeftPPLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedLeftPPLImgFolderName, anonymizedLeftPPLImgFileName);
            anonymizedLeftXPLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedLeftXPLImgFolderName, anonymizedLeftXPLImgFileName);
            anonymizedLeftUVLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedLeftUVLImgFolderName, anonymizedLeftUVLImgFileName);
            anonymizedLeftFullFaceMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedLeftFullFaceMaskFolderName, anonymizedLeftFullFaceMaskImgName);

            anonymizedRightPPLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedRightPPLImgFolderName, anonymizedRightPPLImgFileName);
            anonymizedRightXPLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedRightXPLImgFolderName, anonymizedRightXPLImgFileName);
            anonymizedRightUVLImgOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedRightUVLImgFolderName, anonymizedRightUVLImgFileName);
            anonymizedRightFullFaceMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, anonymizedRightFullFaceMaskFolderName, anonymizedRightFullFaceMaskImgName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating anonymized image output folders", e);
        }

        SelfieSegmentation segSelfie = new SelfieSegmentation(MainActivity.this);

        if(sideFaceImagesEnabled == false) {
            leftPPL = null;
            leftXPL = null;
            leftUVL = null;
            rightPPL = null;
            rightXPL = null;
            rightUVL = null;
            originalLeftPPLPath = null;
            anonymizedLeftPPLImgOutputPath = null;
            originalLeftXPLPath = null;
            anonymizedLeftXPLImgOutputPath = null;
            originalLeftUVLPath = null;
            anonymizedLeftUVLImgOutputPath = null;
            originalRightPPLPath = null;
            anonymizedRightPPLImgOutputPath = null;
            originalRightXPLPath = null;
            anonymizedRightXPLImgOutputPath = null;
            originalRightUVLPath = null;
            anonymizedRightUVLImgOutputPath = null;
            anonymizedLeftFullFaceMaskOutputPath = null;
            anonymizedRightFullFaceMaskOutputPath = null;
        }

        segSelfie.runSegmentationOnImage(
                frontPPL,
                frontXPL,
                frontUVL,
                leftPPL,
                leftXPL,
                leftUVL,
                rightPPL,
                rightXPL,
                rightUVL,
                originalFrontPPLPath, anonymizedFrontPPLImgOutputPath,
                originalFrontXPLPath, anonymizedFrontXPLImgOutputPath,
                originalFrontUVLPath, anonymizedFrontUVLImgOutputPath,
                originalLeftPPLPath, anonymizedLeftPPLImgOutputPath,
                originalLeftXPLPath, anonymizedLeftXPLImgOutputPath,
                originalLeftUVLPath, anonymizedLeftUVLImgOutputPath,
                originalRightPPLPath, anonymizedRightPPLImgOutputPath,
                originalRightXPLPath, anonymizedRightXPLImgOutputPath,
                originalRightUVLPath, anonymizedRightUVLImgOutputPath,
                anonymizedFrontFullFaceMaskOutputPath,
                anonymizedLeftFullFaceMaskOutputPath,
                anonymizedRightFullFaceMaskOutputPath,
                sideFaceImagesEnabled);

        String frontAnonymizedPPLInputImgPath = anonymizedFrontPPLImgOutputPath;
        String leftAnonymizedPPLInputImgPath = anonymizedLeftPPLImgOutputPath;
        String rightAnonymizedPPLInputImgPath = anonymizedRightPPLImgOutputPath;

        String frontAnonymizedXPLInputImgPath = anonymizedFrontXPLImgOutputPath;
        String leftAnonymizedXPLInputImgPath = anonymizedLeftXPLImgOutputPath;
        String rightAnonymizedXPLInputImgPath = anonymizedRightXPLImgOutputPath;

        String frontAnonymizedUVLInputImgPath = anonymizedFrontUVLImgOutputPath;
        String leftAnonymizedUVLInputImgPath = anonymizedLeftUVLImgOutputPath;
        String rightAnonymizedUVLInputImgPath = anonymizedRightUVLImgOutputPath;

        // ------------------------------ (2) MediaPipe FaceMesh to Get Coordinates ------------------------------
        //
        long faceMeshStart = System.currentTimeMillis() / 1000;

        FaceMeshLandmarker myFaceMesh = new FaceMeshLandmarker(MainActivity.this);

        Map<Integer, List<Integer>> frontPPLCoordinates = myFaceMesh.runFaceMeshOnImage(frontPPL);
        if(frontPPLCoordinates.isEmpty()) {
            System.out.println("No face detected in front image. Please take proper face image.");
        }
        else {
            MyUtil.drawCoordinates(getApplicationContext(), frontAnonymizedPPLInputImgPath, frontPPLCoordinates);
        }

        Map<Integer, List<Integer>> leftPPLCoordinates = null;
        Map<Integer, List<Integer>> rightPPLCoordinates = null;

        if(sideFaceImagesEnabled) {
            leftPPLCoordinates = myFaceMesh.runFaceMeshOnImage(leftPPL);
            if(leftPPLCoordinates.isEmpty()) {
                System.out.println("No face detected in left image. Please take proper face image.");
            }
            else {
                MyUtil.drawCoordinates(getApplicationContext(), leftAnonymizedPPLInputImgPath, leftPPLCoordinates);
            }

            rightPPLCoordinates = myFaceMesh.runFaceMeshOnImage(rightPPL);
            if(rightPPLCoordinates.isEmpty()) {
                System.out.println("No face detected in right image. Please take proper face image.");
            }
            else {
                MyUtil.drawCoordinates(getApplicationContext(), rightAnonymizedPPLInputImgPath, rightPPLCoordinates);
            }
        }

        long faceMeshEnds = System.currentTimeMillis() / 1000;
        long faceMeshTime = faceMeshEnds - faceMeshStart;
        Log.println(Log.VERBOSE, "FaceMesh Time: ", String.valueOf(faceMeshTime));

        // ------------------------------ (3) Get CFA Face ROIs ------------------------------
        //
        // -------------- Create sub-folders for all output ROI-mask and Face-ROI images under the BATCH folder.
        //
        // ROI mask images.
        String frontFullFaceMaskInputPath = anonymizedFrontFullFaceMaskOutputPath;
        String leftFullFaceMaskInputPath = anonymizedLeftFullFaceMaskOutputPath;
        String rightFullFaceMaskInputPath = anonymizedRightFullFaceMaskOutputPath;

        String frontCheekMaskFolderName = "frontCheekMask";
        String frontChinMaskFolderName = "frontChinMask";
        String frontForeheadMaskFolderName = "frontForeheadMask";
        String frontNoseMaskFolderName = "frontNoseMask";
        String frontEyeMaskFolderName = "frontEyeMask";
        String frontLionMaskFolderName = "frontLionMask";

        String leftCheekMaskFolderName = "leftCheekMask";
        String leftChinMaskFolderName = "leftChinMask";
        String leftForeheadMaskFolderName = "leftForeheadMask";
        String leftNoseMaskFolderName = "leftNoseMask";
        String leftEyeMaskFolderName = "leftEyeMask";
        String leftLionMaskFolderName = "leftLionMask";

        String rightCheekMaskFolderName = "rightCheekMask";
        String rightChinMaskFolderName = "rightChinMask";
        String rightForeheadMaskFolderName = "rightForeheadMask";
        String rightNoseMaskFolderName = "rightNoseMask";
        String rightEyeMaskFolderName = "rightEyeMask";
        String rightLionMaskFolderName = "rightLionMask";

        String frontCheekMaskFileName = "front cheek.jpg";
        String frontChinMaskFileName = "front chin.jpg";
        String frontForeheadMaskFileName = "front forehead.jpg";
        String frontNoseMaskFileName = "front nose.jpg";
        String frontEyeMaskFileName = "front eye.jpg";
        String frontLionMaskFileName = "front lion.jpg";

        String leftCheekMaskFileName = "left cheek.jpg";
        String leftChinMaskFileName = "left chin.jpg";
        String leftForeheadMaskFileName ="left forehead.jpg";
        String leftNoseMaskFileName = "left nose.jpg";
        String leftEyeMaskFileName = "left eye.jpg";
        String leftLionMaskFileName = "left lion.jpg";

        String rightCheekMaskFileName = "right cheek.jpg";
        String rightChinMaskFileName = "right chin.jpg";
        String rightForeheadMaskFileName = "right forehead.jpg";
        String rightNoseMaskFileName ="right nose.jpg";
        String rightEyeMaskFileName = "right eye.jpg";
        String rightLionMaskFileName = "right lion.jpg";

        String frontCheekMaskOutputPath = null;
        String frontChinMaskOutputPath = null;
        String frontForeheadMaskOutputPath = null;
        String frontNoseMaskOutputPath = null;
        String frontEyeMaskOutputPath = null;
        String frontLionMaskOutputPath = null;

        String leftCheekMaskOutputPath = null;
        String leftChinMaskOutputPath = null;
        String leftForeheadMaskOutputPath = null;
        String leftNoseMaskOutputPath = null;
        String leftEyeMaskOutputPath = null;
        String leftLionMaskOutputPath = null;

        String rightCheekMaskOutputPath = null;
        String rightChinMaskOutputPath = null;
        String rightForeheadMaskOutputPath = null;
        String rightNoseMaskOutputPath = null;
        String rightEyeMaskOutputPath = null;
        String rightLionMaskOutputPath = null;

        try {
            frontCheekMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontCheekMaskFolderName, frontCheekMaskFileName);
            frontChinMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontChinMaskFolderName, frontChinMaskFileName);
            frontForeheadMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontForeheadMaskFolderName, frontForeheadMaskFileName);
            frontNoseMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontNoseMaskFolderName, frontNoseMaskFileName);
            frontEyeMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontEyeMaskFolderName, frontEyeMaskFileName);
            frontLionMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontLionMaskFolderName, frontLionMaskFileName);

            leftCheekMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftCheekMaskFolderName, leftCheekMaskFileName);
            leftChinMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftChinMaskFolderName, leftChinMaskFileName);
            leftForeheadMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftForeheadMaskFolderName, leftForeheadMaskFileName);
            leftNoseMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftNoseMaskFolderName, leftNoseMaskFileName);
            leftEyeMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftEyeMaskFolderName, leftEyeMaskFileName);
            leftLionMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftLionMaskFolderName, leftLionMaskFileName);

            rightCheekMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightCheekMaskFolderName, rightCheekMaskFileName);
            rightChinMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightChinMaskFolderName, rightChinMaskFileName);
            rightForeheadMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightForeheadMaskFolderName, rightForeheadMaskFileName);
            rightNoseMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightNoseMaskFolderName, rightNoseMaskFileName);
            rightEyeMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightEyeMaskFolderName, rightEyeMaskFileName);
            rightLionMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightLionMaskFolderName, rightLionMaskFileName);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating file paths for Wrinkles", e);
        }

        // Face-ROI images.
        String frontRednessROIFolderName = "front redness roi";
        String leftRednessROIFolderName = "left redness roi";
        String rightRednessROIFolderName = "right redness roi";
        String frontOilinessROIFolderName = "front oiliness roi";
        String frontRadianceROIFolderName = "front radiance roi";
        String frontImpuritiesROIFolderName = "front impurities roi";

        String frontRednessROIFileName = "front redness roi.jpg";
        String leftRednessROIFileName =  "left redness roi.jpg";
        String rightRednessROIFileName =  "right redness roi.jpg";
        String frontOilinessROIFileName =  "front oiliness roi.jpg";
        String frontRadianceROIFileName =  "front oiliness roi.jpg";
        String frontImpuritiesROIFileName = "front impurities roi.jpg";

        String frontRednessROIOutputPath = null;
        String leftRednessROIOutputPath = null;
        String rightRednessROIOutputPath = null;
        String frontOilinessROIOutputPath = null;
        String frontRadianceROIOutputPath = null;
        String frontImpuritiesROIOutputPath = null;

        try {
            frontRednessROIOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontRednessROIFolderName, frontRednessROIFileName);
            leftRednessROIOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftRednessROIFolderName, leftRednessROIFileName);
            rightRednessROIOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightRednessROIFolderName, rightRednessROIFileName);
            frontOilinessROIOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontOilinessROIFolderName, frontOilinessROIFileName);
            frontRadianceROIOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontRadianceROIFolderName, frontRadianceROIFileName);
            frontImpuritiesROIOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontImpuritiesROIFolderName, frontImpuritiesROIFileName);
        } catch(IOException e){
            Log.e("ImageSegmentation", "Error creating Face ROI output file paths", e);
        }

        boolean impuritiesEnabled = true;

        if (sideFaceImagesEnabled == false) {
            leftAnonymizedPPLInputImgPath = null;
            rightAnonymizedPPLInputImgPath = null;

            leftAnonymizedXPLInputImgPath = null;
            rightAnonymizedXPLInputImgPath = null;

            leftAnonymizedUVLInputImgPath = null;
            rightAnonymizedUVLInputImgPath = null;

            leftFullFaceMaskInputPath = null;
            rightFullFaceMaskInputPath = null;

            leftRednessROIOutputPath = null;
            rightRednessROIOutputPath = null;

            leftForeheadMaskOutputPath = null;
            leftNoseMaskOutputPath = null;
            leftCheekMaskOutputPath = null;
            leftChinMaskOutputPath = null;
            rightForeheadMaskOutputPath = null;
            rightNoseMaskOutputPath = null;
            rightCheekMaskOutputPath = null;
            rightChinMaskOutputPath = null;
        }

        CFAGetROIs101.doAnalysis(MainActivity.this,
                frontAnonymizedPPLInputImgPath,
                leftAnonymizedPPLInputImgPath,
                rightAnonymizedPPLInputImgPath,
                frontAnonymizedXPLInputImgPath,
                leftAnonymizedXPLInputImgPath,
                rightAnonymizedXPLInputImgPath,
                frontAnonymizedUVLInputImgPath,
                leftAnonymizedUVLInputImgPath,
                rightAnonymizedUVLInputImgPath,
                frontFullFaceMaskInputPath,
                leftFullFaceMaskInputPath,
                rightFullFaceMaskInputPath,
                frontRednessROIOutputPath,
                leftRednessROIOutputPath,
                rightRednessROIOutputPath,
                frontOilinessROIOutputPath,
                frontRadianceROIOutputPath,
                frontImpuritiesROIOutputPath,
                frontForeheadMaskOutputPath,
                frontNoseMaskOutputPath,
                frontCheekMaskOutputPath,
                frontChinMaskOutputPath,
                leftForeheadMaskOutputPath,
                leftNoseMaskOutputPath,
                leftCheekMaskOutputPath,
                leftChinMaskOutputPath,
                rightForeheadMaskOutputPath,
                rightNoseMaskOutputPath,
                rightCheekMaskOutputPath,
                rightChinMaskOutputPath,
                frontPPLCoordinates,
                leftPPLCoordinates,
                rightPPLCoordinates,
                impuritiesEnabled,
                sideFaceImagesEnabled);

        // ------------------------------ (4) Prepare Input Paths for Skin Analyses ------------------------------
        //
        String frontForeheadMaskInputPath = frontForeheadMaskOutputPath;
        String frontNoseMaskInputPath = frontNoseMaskOutputPath;
        String frontCheekMaskInputPath = frontCheekMaskOutputPath;
        String frontChinMaskInputPath = frontChinMaskOutputPath;

        String leftForeheadMaskInputPath = leftForeheadMaskOutputPath;
        String leftNoseMaskInputPath = leftNoseMaskOutputPath;
        String leftCheekMaskInputPath = leftCheekMaskOutputPath;
        String leftChinMaskInputPath = leftChinMaskOutputPath;

        String rightForeheadMaskInputPath = rightForeheadMaskOutputPath;
        String rightNoseMaskInputPath = rightNoseMaskOutputPath;
        String rightCheekMaskInputPath = rightCheekMaskOutputPath;
        String rightChinMaskInputPath = rightChinMaskOutputPath;

        String frontEyeMaskInputPath = frontEyeMaskOutputPath;
        String frontLionMaskInputPath = frontLionMaskOutputPath;

        String leftEyeMaskInputPath = leftEyeMaskOutputPath;
        String leftLionMaskInputPath = leftLionMaskOutputPath;

        String rightEyeMaskInputPath = rightEyeMaskOutputPath;
        String rightLionMaskInputPath = rightLionMaskOutputPath;

        // ------------------------------ (5) CFA Wrinkles AI Analysis ------------------------------
        //
        // -------------- Create sub-folders for output mask images and result images for Wrinkles, under the BATCH folder.
        String frontWrinkleMaskFolderName = "frontWrinkleMask";
        String frontWrinkleResultFolderName = "frontWrinkleResult";

        String leftWrinkleMaskFolderName = "leftWrinkleMask";
        String leftWrinkleResultFolderName = "leftWrinkleResult";

        String rightWrinkleMaskFolderName = "rightWrinkleMask";
        String rightWrinkleResultFolderName = "rightWrinkleResult";

        String frontWrinkleMaskFileName =  "front_wrinkle_mask.jpg";
        String leftWrinkleMaskFileName =  "left_wrinkle_mask.jpg";
        String rightWrinkleMaskFileName =  "right_wrinkle_mask.jpg";

        String frontWrinkleResultFileName = "front_wrinkle_result.jpg";
        String leftWrinkleResultFileName = "left_wrinkle_result.jpg";
        String rightWrinkleResultFileName = "right_wrinkle_result.jpg";

        String frontWrinkleMaskOutputPath = null;
        String leftWrinkleMaskOutputPath = null;
        String rightWrinkleMaskOutputPath = null;

        String frontWrinkleResultOutputPath = null;
        String leftWrinkleResultOutputPath = null;
        String rightWrinkleResultOutputPath = null;

        try {
            frontWrinkleMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontWrinkleMaskFolderName, frontWrinkleMaskFileName);
            leftWrinkleMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftWrinkleMaskFolderName, leftWrinkleMaskFileName);
            rightWrinkleMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightWrinkleMaskFolderName, rightWrinkleMaskFileName);

            frontWrinkleResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontWrinkleResultFolderName, frontWrinkleResultFileName);
            leftWrinkleResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftWrinkleResultFolderName, leftWrinkleResultFileName);
            rightWrinkleResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightWrinkleResultFolderName, rightWrinkleResultFileName);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating wrinkle output file paths", e);
        }

        if (sideFaceImagesEnabled == false) {
            leftAnonymizedPPLInputImgPath = null;
            rightAnonymizedPPLInputImgPath = null;

            originalLeftPPLPath = null;
            originalRightPPLPath = null;

            leftFullFaceMaskInputPath = null;
            rightFullFaceMaskInputPath = null;

            leftForeheadMaskInputPath = null;
            rightForeheadMaskInputPath = null;

            leftWrinkleResultOutputPath = null;
            rightWrinkleResultOutputPath = null;

            leftWrinkleMaskOutputPath = null;
            rightWrinkleMaskOutputPath = null;
        }

       String wrinkleResStr = CFALocalWrinkle101.doAnalysis(MainActivity.this,
                wrinklesTFModelFile,
                frontAnonymizedPPLInputImgPath,
                leftAnonymizedPPLInputImgPath,
                rightAnonymizedPPLInputImgPath,

                originalFrontPPLPath,
                originalLeftPPLPath,
                originalRightPPLPath,

                frontFullFaceMaskInputPath,
                leftFullFaceMaskInputPath,
                rightFullFaceMaskInputPath,

                frontForeheadMaskInputPath,
                leftForeheadMaskInputPath,
                rightForeheadMaskInputPath,

                frontWrinkleResultOutputPath,
                leftWrinkleResultOutputPath,
                rightWrinkleResultOutputPath,

                frontWrinkleMaskOutputPath,
                leftWrinkleMaskOutputPath,
                rightWrinkleMaskOutputPath,

                frontPPLCoordinates,
                leftPPLCoordinates,
                rightPPLCoordinates,

                capturedWithFrontCamera,
                sideFaceImagesEnabled);

        // ------------------------------ (6) CFA Dark Circles AI Analysis ------------------------------
        //
        String darkCircleMaskFolderName = "darkCircleMask";
        String darkCircleResultFolderName = "darkCircleResult";

        String darkCircleMaskFileName = "dark_circle_mask.jpg";
        String darkCircleResultFileName = "dark_circle_result.jpg";

        String darkCircleMaskOutputPath = null;
        String darkCircleResultOutputPath = null;

        try {
            darkCircleMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, darkCircleMaskFolderName, darkCircleMaskFileName);
            darkCircleResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, darkCircleResultFolderName, darkCircleResultFileName);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating dark circle output file paths", e);
        }

        String darkCircleResStr = CFADarkCircle.doAnalysis(MainActivity.this,
                                                                    darkCircleTFModelFile,
                                                                    frontAnonymizedPPLInputImgPath,
                                                                    originalFrontPPLPath,
                                                                    frontFullFaceMaskInputPath,
                                                                    darkCircleMaskOutputPath,
                                                                    darkCircleResultOutputPath,
                                                                    capturedWithFrontCamera);

        // ------------------------------ (7) CFA Hyperpigmentation AI Analysis ------------------------------
        //
        String frontHyperPigMaskOutputFolderName = "front hyperpigmentation mask";
        String leftHyperPigMaskOutputFolderName = "left hyperpigmentation mask";
        String rightHyperPigMaskOutputFolderName = "right hyperpigmentation mask";
        String frontHyperPigResultOutputFolderName = "front hyperpigmentation result";
        String leftHyperPigResultOutputFolderName = "left hyperpigmentation result";
        String rightHyperPigResultOutputFolderName = "right hyperpigmentation result";

        String frontHyperPigMaskFileName =  "front hyperPig mask.jpg";
        String leftHyperPigMaskFileName =  "left hyperPig mask.jpg";
        String rightHyperPigMaskFileName = "right hyperPig mask.jpg";
        String frontHyperPigResultFileName = "front hyperPig result.jpg";
        String leftHyperPigResultFileName =  "left hyperPig result.jpg";
        String rightHyperPigResultFileName =  "right hyperPig result.jpg";

        String frontHyperPigmentationMaskOutputPath = null;
        String leftHyperPigmentationMaskOutputPath = null;
        String rightHyperPigmentationMaskOutputPath = null;
        String frontHyperPigmentationResultOutputPath = null;
        String leftHyperPigmentationResultOutputPath = null;
        String rightHyperPigmentationResultOutputPath = null;

        try {
            frontHyperPigmentationMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontHyperPigMaskOutputFolderName, frontHyperPigMaskFileName);
            leftHyperPigmentationMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftHyperPigMaskOutputFolderName, leftHyperPigMaskFileName);
            rightHyperPigmentationMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightHyperPigMaskOutputFolderName, rightHyperPigMaskFileName);
            frontHyperPigmentationResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontHyperPigResultOutputFolderName, frontHyperPigResultFileName);
            leftHyperPigmentationResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftHyperPigResultOutputFolderName, leftHyperPigResultFileName);
            rightHyperPigmentationResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightHyperPigResultOutputFolderName, rightHyperPigResultFileName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating hyperPigmentation output file paths", e);
        }

        if(sideFaceImagesEnabled == false) {
            leftAnonymizedUVLInputImgPath = null;
            rightAnonymizedUVLInputImgPath = null;

            originalLeftUVLPath = null;
            originalRightUVLPath = null;

            leftForeheadMaskInputPath = null;
            leftNoseMaskInputPath = null;
            leftCheekMaskInputPath = null;
            leftFullFaceMaskInputPath = null;
            rightForeheadMaskInputPath = null;
            rightNoseMaskInputPath = null;
            rightCheekMaskInputPath = null;
            rightFullFaceMaskInputPath = null;

            leftHyperPigmentationResultOutputPath = null;
            rightHyperPigmentationResultOutputPath = null;

            leftHyperPigmentationMaskOutputPath = null;
            rightHyperPigmentationMaskOutputPath = null;

            leftHyperPigmentationResultOutputPath = null;
            rightHyperPigmentationResultOutputPath = null;

            leftHyperPigmentationMaskOutputPath = null;
            rightHyperPigmentationMaskOutputPath = null;
        }

        String hyperPigResStr = CFALocalHyperPigmentation101.doAnalysis( MainActivity.this,
                hyperPigmentationTFModelFile,
                frontAnonymizedUVLInputImgPath,
                leftAnonymizedUVLInputImgPath,
                rightAnonymizedUVLInputImgPath,
                originalFrontUVLPath,
                originalLeftUVLPath,
                originalRightUVLPath,
                frontForeheadMaskInputPath,
                frontNoseMaskInputPath,
                frontCheekMaskInputPath,
                frontFullFaceMaskInputPath,
                leftForeheadMaskInputPath,
                leftNoseMaskInputPath,
                leftCheekMaskInputPath,
                leftFullFaceMaskInputPath,
                rightForeheadMaskInputPath,
                rightNoseMaskInputPath,
                rightCheekMaskInputPath,
                rightFullFaceMaskInputPath,
                frontHyperPigmentationResultOutputPath,
                leftHyperPigmentationResultOutputPath,
                rightHyperPigmentationResultOutputPath,
                frontHyperPigmentationMaskOutputPath,
                leftHyperPigmentationMaskOutputPath,
                rightHyperPigmentationMaskOutputPath,
                sideFaceImagesEnabled,
                capturedWithFrontCamera);

        // ------------------------------ (8) CFA pigmentation/spots AI Analysis ------------------------------
        String frontSpotsResultOutputFolderName = "front spots result";
        String leftSpotsResultOutputFolderName = "left spots result";
        String rightSpotsResultOutputFolderName = "right spots result";
        String frontSpotsMaskOutputFolderName = "front spots mask";
        String leftSpotsMaskOutputFolderName = "left spots mask";
        String rightSpotsMaskOutputFolderName = "right spots mask";

        String frontSpotsResultFileName =  "front spots result.jpg";
        String leftSpotsResultFileName =  "left spots result.jpg";
        String rightSpotsResultFileName = "right spots result.jpg";
        String frontSpotsMaskFileName = "front spots mask.jpg";
        String leftSpotsMaskFileName = "left spots mask.jpg";
        String rightSpotsMaskFileName = "right spots mask.jpg";

        String frontSpotsResultOutputPath = null;
        String leftSpotsResultOutputPath = null;
        String rightSpotsResultOutputPath = null;
        String frontSpotsMaskOutputPath = null;
        String leftSpotsMaskOutputPath = null;
        String rightSpotsMaskOutputPath = null;

        try {
            frontSpotsResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontSpotsResultOutputFolderName, frontSpotsResultFileName);
            leftSpotsResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftSpotsResultOutputFolderName, leftSpotsResultFileName);
            rightSpotsResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightSpotsResultOutputFolderName, rightSpotsResultFileName);

            frontSpotsMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontSpotsMaskOutputFolderName, frontSpotsMaskFileName);
            leftSpotsMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftSpotsMaskOutputFolderName, leftSpotsMaskFileName);
            rightSpotsMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightSpotsMaskOutputFolderName, rightSpotsMaskFileName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Pigmentation/Spots output file paths", e);
        }

        if(sideFaceImagesEnabled == false) {
            leftAnonymizedXPLInputImgPath = null;
            rightAnonymizedXPLInputImgPath = null;

            originalLeftXPLPath = null;
            originalRightXPLPath = null;

            leftSpotsResultOutputPath = null;
            rightSpotsResultOutputPath = null;

            leftSpotsMaskOutputPath = null;
            rightSpotsMaskOutputPath = null;

            leftForeheadMaskInputPath = null;
            leftNoseMaskInputPath = null;
            leftCheekMaskInputPath = null;
            leftChinMaskInputPath = null;
            leftFullFaceMaskInputPath = null;
            rightForeheadMaskInputPath = null;
            rightNoseMaskInputPath = null;
            rightCheekMaskInputPath = null;
            rightChinMaskInputPath = null;
            rightFullFaceMaskInputPath = null;
        }

        String pigSpotsResStr = CFALocalPigmentationSpots102.doAnalysis(MainActivity.this,
                pigmentationSpotsTFModelFile,
                frontAnonymizedXPLInputImgPath,
                leftAnonymizedXPLInputImgPath,
                rightAnonymizedXPLInputImgPath,
                originalFrontXPLPath,
                originalLeftXPLPath,
                originalRightXPLPath,
                frontSpotsResultOutputPath,
                leftSpotsResultOutputPath,
                rightSpotsResultOutputPath,
                frontSpotsMaskOutputPath,
                leftSpotsMaskOutputPath,
                rightSpotsMaskOutputPath,
                frontForeheadMaskInputPath,
                frontNoseMaskInputPath,
                frontCheekMaskInputPath,
                frontChinMaskInputPath,
                frontFullFaceMaskInputPath,
                leftForeheadMaskInputPath,
                leftNoseMaskInputPath,
                leftCheekMaskInputPath,
                leftChinMaskInputPath,
                leftFullFaceMaskInputPath,
                rightForeheadMaskInputPath,
                rightNoseMaskInputPath,
                rightCheekMaskInputPath,
                rightChinMaskInputPath,
                rightFullFaceMaskInputPath,
                sideFaceImagesEnabled,
                capturedWithFrontCamera);

        // ------------------------------ (9) CFA Pores AI Analysis ------------------------------
        String poresResultOutputFolderName = "pores result";
        String poresResultFileName = "pores result.jpg";
        String poresResultOutputPath = null;

        String poresMaskOutputFolderName = "pores mask";
        String poresMaskFileName = "pores mask.jpg";
        String poresMaskOutputPath = null;

        try {
            poresResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, poresResultOutputFolderName, poresResultFileName);
            poresMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, poresMaskOutputFolderName, poresMaskFileName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Pores ROI output file path", e);
        }

        CFALocalPores101.doAnalysis(MainActivity.this,
                poresTFModelFile,
                frontAnonymizedPPLInputImgPath,
                originalFrontPPLPath,
                frontFullFaceMaskInputPath,
                poresResultOutputPath,
                poresMaskOutputPath,
                frontPPLCoordinates,
                capturedWithFrontCamera);

        // ------------------------------ (10) CFA Impurities AI Analysis ------------------------------
        //
        if (impuritiesEnabled) {
            String impuritiesMaskOutputFolderName = "impurities mask";
            String impuritiesResultOutputFolderName = "impurities result";

            String impuritiesMaskFileName= "impurities mask.jpg";
            String impuritiesResultFileName = "impurities result.jpg";

            String impuritiesMaskOutputPath = null;
            String impuritiesResultOutputPath = null;

            try {
                impuritiesMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, impuritiesMaskOutputFolderName, impuritiesMaskFileName);
                impuritiesResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, impuritiesResultOutputFolderName, impuritiesResultFileName);
            } catch (IOException e) {
                Log.e("ImageSegmentation", "Error creating Impurities output folders", e);
            }

            String impuritiesResStr = CFALocalImpurities100.doAnalysis(MainActivity.this,
                    impuritiesTFModelFile,
                    frontAnonymizedUVLInputImgPath,
                    originalFrontUVLPath,
                    frontFullFaceMaskInputPath,
                    impuritiesResultOutputPath,
                    impuritiesMaskOutputPath,
                    frontPPLCoordinates,
                    capturedWithFrontCamera);

            System.out.println("Returned Impurities String is: " + impuritiesResStr);
        }
    /*
        // ------------------------------ (11) Test Image Processing Algorithms ------------------------------
        //
        com.chowis.jniimagepro.CFA.JNICFAImageProCW myCFAImgProc = new JNICFAImageProCW();

        // ----- Redness algorithm -----
        //
        String rednessFrontRoiImgPath = frontRednessROIOutputPath;
        String rednessLeftRoiImgPath = leftRednessROIOutputPath;
        String rednessRightRoiImgPath = rightRednessROIOutputPath;

        String frontRednessResultOutputFolderName = "front redness result";
        String leftRednessResultOutputFolderName = "left redness result";
        String rightRednessResultOutputFolderName = "right redness result";
        String frontRednessMaskOutputFolderName = "front redness mask";
        String leftRednessMaskOutputFolderName = "left redness mask";
        String rightRednessMaskOutputFolderName = "right redness mask";

        String frontRednessResultFileName = "front redness result.jpg";
        String leftRednessResultFileName ="left redness result.jpg";
        String rightRednessResultFileName ="right redness result.jpg";
        String frontRednessMaskFileName = "front redness mask.jpg";
        String leftRednessMaskFileName = "left redness mask.jpg";
        String rightRednessMaskFileName = "right redness mask.jpg";

        String frontRednessResultOutputPath = null;
        String leftRednessResultOutputPath = null;
        String rightRednessResultOutputPath = null;
        String frontRednessMaskOutputPath = null;
        String leftRednessMaskOutputPath = null;
        String rightRednessMaskOutputPath = null;

        try {
            frontRednessResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontRednessResultOutputFolderName, frontRednessResultFileName);
            leftRednessResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftRednessResultOutputFolderName, leftRednessResultFileName);
            rightRednessResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightRednessResultOutputFolderName, rightRednessResultFileName);

            frontRednessMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, frontRednessMaskOutputFolderName, frontRednessMaskFileName);
            leftRednessMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, leftRednessMaskOutputFolderName, leftRednessMaskFileName);
            rightRednessMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, rightRednessMaskOutputFolderName, rightRednessMaskFileName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Redness output folders", e);
        }

        double rednessSideImageEnabled = -1;
        if(sideFaceImagesEnabled) rednessSideImageEnabled = 1;
        else rednessSideImageEnabled = 0;

        if(sideFaceImagesEnabled == false) {
            originalLeftXPLPath = "";
            originalRightXPLPath = "";

            rednessLeftRoiImgPath = "";
            rednessRightRoiImgPath = "";

            leftRednessResultOutputPath = "";
            rightRednessResultOutputPath = "";

            leftRednessMaskOutputPath = "";
            rightRednessMaskOutputPath = "";
        }

        String rednessResStr = myCFAImgProc.CFALocalRedness104Jni(originalFrontXPLPath,
                originalLeftXPLPath,
                originalRightXPLPath,
                rednessFrontRoiImgPath,
                rednessLeftRoiImgPath,
                rednessRightRoiImgPath,
                frontRednessResultOutputPath,
                leftRednessResultOutputPath,
                rightRednessResultOutputPath,
                frontRednessMaskOutputPath,
                leftRednessMaskOutputPath,
                rightRednessMaskOutputPath,
                usedFrontCamera,
                rednessSideImageEnabled);

        System.out.println("Returned redness string is: " + rednessResStr);

        // ----- Oiliness algorithm -----
        //
        String oilinessRoiImgPath = frontOilinessROIOutputPath;

        String oilinessResultOutputFolderName = "oiliness result";
        String oilinessGreenMaskOutputFolderName = "oiliness green mask";
        String oilinessWhiteMaskOutputFolderName = "oiliness white mask";

        String oilinessResultFileName ="oiliness result.jpg";
        String oilinessGreenMaskFileName = "oiliness green mask.jpg";
        String oilinessWhiteMaskFileName = "oiliness white mask.jpg";

        String oilinessResultOutputPath = null;
        String oilinessGreenMaskOutputPath = null;
        String oilinessWhiteMaskOutputPath = null;

        try {
            oilinessResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, oilinessResultOutputFolderName, oilinessResultFileName);
            oilinessGreenMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, oilinessGreenMaskOutputFolderName, oilinessGreenMaskFileName);
            oilinessWhiteMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, oilinessWhiteMaskOutputFolderName, oilinessWhiteMaskFileName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Oiliness output folders", e);
        }

        String oilinessResStr = myCFAImgProc.CFALocalOiliness100Jni(
                originalFrontPPLPath,
                oilinessRoiImgPath,
                oilinessResultOutputPath,
                oilinessGreenMaskOutputPath,
                oilinessWhiteMaskOutputPath,
                usedFrontCamera);

        System.out.println("Returned oiliness string is: " + oilinessResStr);

        // ----- Radiance & Dullness algorithm -----
        //
        String radianceRoiImgPath = frontRadianceROIOutputPath;

        String radianceResultOutputFolderName = "radiance result";
        String radianceGrayMaskOutputFolderName = "radiance gray mask";
        String radianceWhiteMaskOutputFolderName = "radiance white mask";

        String radianceResultFileName = "radiance result.jpg";
        String radianceGrayMaskFileName  = "radiance gray mask.jpg";
        String radianceWhiteMaskFileName  =  "radiance white mask.jpg";

        String radianceResultOutputPath = null;
        String radianceGrayMaskOutputPath = null;
        String radianceWhiteMaskOutputPath = null;

        try {
            radianceResultOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, radianceResultOutputFolderName, radianceResultFileName);
            radianceGrayMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, radianceGrayMaskOutputFolderName, radianceGrayMaskFileName);
            radianceWhiteMaskOutputPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, radianceWhiteMaskOutputFolderName, radianceWhiteMaskFileName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Radiance output folders", e);
        }

        String radianceDullnessResStr = myCFAImgProc.CFALocalRadianceDullness100Jni(
                originalFrontPPLPath,
                radianceRoiImgPath,
                radianceResultOutputPath,
                radianceGrayMaskOutputPath,
                radianceWhiteMaskOutputPath,
                usedFrontCamera);

        System.out.println("Returned Radiance & Dullness String is: " + radianceDullnessResStr);

        // ------------------------------ (12) Test Computation Algorithms ------------------------------
        //
        // Test Skin Age from Unified Computation
        JniLocalComputationQA myComputation = new JniLocalComputationQA();
        double mySkinAge = myComputation.computeSkinAge101Jni(20, 40, 28);
        Log.println(Log.VERBOSE, "Got Skin Age as: ", String.valueOf(mySkinAge));
        */
        // ------------------------------ Finishing ------------------------------
        //
        // Processing results, and local files etc.

        long endTimeSeconds = System.currentTimeMillis() / 1000;
        long processTime = endTimeSeconds - startTimeSeconds;
        Log.println(Log.VERBOSE, "Processing Time for CFAAI: ", String.valueOf(processTime));

        runOnUiThread(new Runnable(){
            @Override
            public void run() {
                mButtonSegment.setText(getString(R.string.segment));
            }
        });
    }
}
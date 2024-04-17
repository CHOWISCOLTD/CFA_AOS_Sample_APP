package com.google.mediapipe.imagesegmenter

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.util.Log
import android.widget.Toast
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Core.bitwise_and
import org.opencv.core.CvType.CV_8UC1
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR
import org.opencv.imgproc.Imgproc.INTER_AREA
import org.opencv.imgproc.Imgproc.INTER_LINEAR
import org.opencv.imgproc.Imgproc.cvtColor
import org.opencv.imgproc.Imgproc.resize
import java.io.File
import java.nio.ByteBuffer
import java.util.Timer

class SelfieSegmentation(
    private val context: Context
) : ImageSegmenterHelper.SegmenterListener {

    private lateinit var imageSegmenterHelper: ImageSegmenterHelper
    private var backgroundScope: CoroutineScope? = null
    private var fixedRateTimer: Timer? = null

    // Load and segment the image and get anonymized hair + face mask.
    fun runSegmentationOnImage(frontPPL : File,
                               frontXPL : File,
                               frontUVL : File,
                               leftPPL : File?,
                               leftXPL : File?,
                               leftUVL : File?,
                               rightPPL : File?,
                               rightXPL : File?,
                               rightUVL : File?,
                               inputFrontPPLPath: String, outputFrontPPLPath: String,
                               inputFrontXPLPath : String, outputFrontXPLPath : String,
                               inputFrontUVLPath : String, outputFrontUVLPath : String,
                               inputLeftPPLPath: String?, outputLeftPPLPath: String?,
                               inputLeftXPLPath : String?, outputLeftXPLPath : String?,
                               inputLeftUVLPath : String?, outputLeftUVLPath : String?,
                               inputRightPPLPath: String?, outputRightPPLPath: String?,
                               inputRightXPLPath : String?, outputRightXPLPath : String?,
                               inputRightUVLPath : String?, outputRightUVLPath : String?,
                                outputFrontFullFaceMaskPath : String,
                                outputLeftFullFaceMaskPath : String?,
                                outputRightFullFaceMaskPath : String?,
                                sideFaceImagesEnabled: Boolean) {

        // Configure coroutine and AI model.
        backgroundScope = CoroutineScope(Dispatchers.IO)

        imageSegmenterHelper = ImageSegmenterHelper(
            //context = requireContext(),
            context = context,
            runningMode = RunningMode.IMAGE,
            currentModel = ImageSegmenterHelper.MODEL_SELFIE_MULTICLASS,
            currentDelegate = ImageSegmenterHelper.DELEGATE_CPU,
            imageSegmenterListener = this
        )

        var inputImageFrontPPL = toBitmap(frontPPL)
        inputImageFrontPPL = inputImageFrontPPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        var inputImageFrontXPL = toBitmap(frontXPL)
        inputImageFrontXPL = inputImageFrontXPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        var inputImageFrontUVL = toBitmap(frontUVL)
        inputImageFrontUVL = inputImageFrontUVL.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        // Run image segmentation on the input image.
        val mpImageFrontPPL = BitmapImageBuilder(inputImageFrontPPL).build()
        val resultFrontPPL = imageSegmenterHelper?.segmentImageFile(mpImageFrontPPL)

        val mpImageFrontXPL = BitmapImageBuilder(inputImageFrontXPL).build()
        val resultFrontXPL = imageSegmenterHelper?.segmentImageFile(mpImageFrontXPL)

        val mpImageFrontUVL = BitmapImageBuilder(inputImageFrontUVL).build()
        val resultFrontUVL = imageSegmenterHelper?.segmentImageFile(mpImageFrontUVL)

        // process AI output results.
        getAnonymizedFaceImg(resultFrontPPL!!, inputFrontPPLPath, outputFrontPPLPath)
        getAnonymizedFaceImg(resultFrontXPL!!, inputFrontXPLPath, outputFrontXPLPath)
        getAnonymizedFaceImg(resultFrontUVL!!, inputFrontUVLPath, outputFrontUVLPath)
        getAnonymizedFullFaceMask(resultFrontXPL!!, inputFrontXPLPath, outputFrontFullFaceMaskPath)

        if(sideFaceImagesEnabled) {
            var inputImageLeftPPL = leftPPL?.let { toBitmap(it) }
            if (inputImageLeftPPL != null) {
                inputImageLeftPPL = inputImageLeftPPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)
            }

            var inputImageLeftXPL = leftXPL?.let { toBitmap(it) }
            if (inputImageLeftXPL != null) {
                inputImageLeftXPL = inputImageLeftXPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)
            }

            var inputImageLeftUVL = leftUVL?.let { toBitmap(it) }
            if (inputImageLeftUVL != null) {
                inputImageLeftUVL = inputImageLeftUVL.scaleDown(INPUT_IMAGE_MAX_WIDTH)
            }

            var inputImageRightPPL = rightPPL?.let { toBitmap(it) }
            if (inputImageRightPPL != null) {
                inputImageRightPPL = inputImageRightPPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)
            }

            var inputImageRightXPL = rightXPL?.let { toBitmap(it) }
            if (inputImageRightXPL != null) {
                inputImageRightXPL = inputImageRightXPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)
            }

            var inputImageRightUVL = rightUVL?.let { toBitmap(it) }
            if (inputImageRightUVL != null) {
                inputImageRightUVL = inputImageRightUVL.scaleDown(INPUT_IMAGE_MAX_WIDTH)
            }

            // Run image segmentation on the input image.
            // Disable backgroundScope so file written IO is fully done before trying to read anonymized images.
            //backgroundScope?.launch {
            val mpImageLeftPPL = BitmapImageBuilder(inputImageLeftPPL).build()
            val resultLeftPPL = imageSegmenterHelper?.segmentImageFile(mpImageLeftPPL)

            val mpImageLeftXPL = BitmapImageBuilder(inputImageLeftXPL).build()
            val resultLeftXPL = imageSegmenterHelper?.segmentImageFile(mpImageLeftXPL)

            val mpImageLeftUVL = BitmapImageBuilder(inputImageLeftUVL).build()
            val resultLeftUVL = imageSegmenterHelper?.segmentImageFile(mpImageLeftUVL)

            val mpImageRightPPL = BitmapImageBuilder(inputImageRightPPL).build()
            val resultRightPPL = imageSegmenterHelper?.segmentImageFile(mpImageRightPPL)

            val mpImageRightXPL = BitmapImageBuilder(inputImageRightXPL).build()
            val resultRightXPL = imageSegmenterHelper?.segmentImageFile(mpImageRightXPL)

            val mpImageRightUVL = BitmapImageBuilder(inputImageRightUVL).build()
            val resultRightUVL = imageSegmenterHelper?.segmentImageFile(mpImageRightUVL)

            Log.println(Log.VERBOSE, "selfie multi class", "---------- segmented ----------")

            // process AI output results.
            if (inputLeftPPLPath != null && outputLeftPPLPath != null) {
                    getAnonymizedFaceImg(resultLeftPPL!!, inputLeftPPLPath, outputLeftPPLPath)
            }
            if (inputLeftXPLPath != null && outputLeftXPLPath != null) {
                    getAnonymizedFaceImg(resultLeftXPL!!, inputLeftXPLPath, outputLeftXPLPath)
            }
            if (inputLeftUVLPath != null && outputLeftUVLPath != null) {
                    getAnonymizedFaceImg(resultLeftUVL!!, inputLeftUVLPath, outputLeftUVLPath)
            }
            if (inputLeftXPLPath != null && outputLeftFullFaceMaskPath != null) {
                    getAnonymizedFullFaceMask(resultLeftXPL!!, inputLeftXPLPath, outputLeftFullFaceMaskPath)
            }

            if (inputRightPPLPath != null && outputRightPPLPath != null) {
                    getAnonymizedFaceImg(resultRightPPL!!, inputRightPPLPath, outputRightPPLPath)
            }
            if (inputRightXPLPath != null && outputRightXPLPath != null) {
                    getAnonymizedFaceImg(resultRightXPL!!, inputRightXPLPath, outputRightXPLPath)
            }
            if (inputRightUVLPath != null && outputRightUVLPath != null) {
                    getAnonymizedFaceImg(resultRightUVL!!, inputRightUVLPath, outputRightUVLPath)
            }
            if (inputRightXPLPath != null && outputRightFullFaceMaskPath != null) {
                    getAnonymizedFullFaceMask(
                        resultRightXPL!!,
                        inputRightXPLPath,
                        outputRightFullFaceMaskPath
                    )
            }
            //}
        }
    }

    // convert Uri/assets to bitmap image.
    private fun toBitmap(file : File): Bitmap {
        val source = ImageDecoder.createSource(file)
        return (ImageDecoder.decodeBitmap(source)).copy(Bitmap.Config.ARGB_8888, true)
    }

    /**
     * Scales down the given bitmap to the specified target width while maintaining aspect ratio.
     * If the original image is already smaller than the target width, the original image is returned.
     */
    private fun Bitmap.scaleDown(targetWidth: Float): Bitmap {
        // if this image smaller than widthSize, return original image
        Log.println(Log.VERBOSE, "original image width:", width.toString())

        if (targetWidth >= width) return this
        val scaleFactor = targetWidth / width
        return Bitmap.createScaledBitmap(
            this,
            (width * scaleFactor).toInt(),
            (height * scaleFactor).toInt(),
            false
        )
    }

    private fun getAnonymizedFaceImg(result: ImageSegmenterResult, inputPath : String, outputPath: String) {
        val newImage = result.categoryMask().get()

        val scaledWidth = newImage.width
        val scaledHeight = newImage.height
        val byteBuffer : ByteBuffer = ByteBufferExtractor.extract(newImage)

        val originalImg : Mat = imread(inputPath)
        val width = originalImg.width()
        val height = originalImg.height()

        // Create the mask for hair (category 1) and face skin (category 3).
        var mpMask : Mat = Mat.zeros(scaledHeight, scaledWidth, CV_8UC1)

        for (i in 0 until scaledHeight) {
            for (j in 0 until scaledWidth) {
                // Using unsigned int here because selfie segmentation returns 0 or 255U (-1 signed)
                // with 0 being the found person, 255U for no label.
                // Deeplab uses 0 for background and other labels are 1-19,
                // so only providing 20 colors from ImageSegmenterHelper -> labelColors
                val category = (byteBuffer.get(i * scaledWidth + j).toUInt() % 20U).toInt()

                if (category == 1 || category == 3) {
                    mpMask.put(i, j, 255.toDouble())
                } else {
                    mpMask.put(i, j, 0.toDouble())
                }
            }
        }

        // resize mask image to original size.
        if (width * height > scaledWidth * scaledHeight) {
            resize(mpMask, mpMask, Size(width.toDouble(), height.toDouble()), INTER_LINEAR.toDouble())
        }
        if (width * height < scaledWidth * scaledHeight) {
            resize(mpMask, mpMask, Size(width.toDouble(), height.toDouble()), INTER_AREA.toDouble())
        }

        var anonymizedImg = Mat()
        cvtColor(mpMask, mpMask, COLOR_GRAY2BGR)
        bitwise_and(mpMask, originalImg, anonymizedImg)

        // save image.
        //MyUtil.saveMatToGallery(context, "anonymized image", "anonymized front ppl image", anonymizedImg)
        imwrite(outputPath, anonymizedImg)
    }

    private fun getAnonymizedFullFaceMask(result: ImageSegmenterResult, inputPath : String, outputPath: String) {
        val newImage = result.categoryMask().get()

        val scaledWidth = newImage.width
        val scaledHeight = newImage.height
        val byteBuffer : ByteBuffer = ByteBufferExtractor.extract(newImage)

        val originalImg : Mat = imread(inputPath)
        val width = originalImg.width()
        val height = originalImg.height()

        // Create the mask for face skin (category 3).
        var fullFaceMask : Mat = Mat.zeros(scaledHeight, scaledWidth, CV_8UC1)

        for (i in 0 until scaledHeight) {
            for (j in 0 until scaledWidth) {
                // Using unsigned int here because selfie segmentation returns 0 or 255U (-1 signed)
                // with 0 being the found person, 255U for no label.
                // Model uses 0 for background and other labels are 1-19,
                // so only providing 20 colors from ImageSegmenterHelper -> labelColors
                val category = (byteBuffer.get(i * scaledWidth + j).toUInt() % 20U).toInt()

                if (category == 3) {
                    fullFaceMask.put(i, j, 255.toDouble())
                } else {
                    fullFaceMask.put(i, j, 0.toDouble())
                }
            }
        }

        // resize mask image to original size.
        if (width * height > scaledWidth * scaledHeight) {
            resize(fullFaceMask, fullFaceMask, Size(width.toDouble(), height.toDouble()), INTER_LINEAR.toDouble())
        }
        if (width * height < scaledWidth * scaledHeight) {
            resize(fullFaceMask, fullFaceMask, Size(width.toDouble(), height.toDouble()), INTER_AREA.toDouble())
        }

        // save image.
        //MyUtil.saveMatToGallery(context, "anonymized image", "anonymized front ppl image", anonymizedImg)
        imwrite(outputPath, fullFaceMask)
    }


    private fun stopAllTasks() {
        // cancel all jobs
        fixedRateTimer?.cancel()
        fixedRateTimer = null
        backgroundScope?.cancel()
        backgroundScope = null

        // clear Image Segmenter
        imageSegmenterHelper?.clearListener()
        imageSegmenterHelper?.clearImageSegmenter()
        //imageSegmenterHelper = null
    }

    private fun segmentationError() {
        stopAllTasks()
    }

    override fun onError(error: String, errorCode: Int) {
        backgroundScope?.launch {
            withContext(Dispatchers.Main) {
                segmentationError()
                Toast.makeText(context, error, Toast.LENGTH_SHORT)
                    .show()
                /*if (errorCode == ImageSegmenterHelper.GPU_ERROR) {
                    fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                        ImageSegmenterHelper.DELEGATE_CPU, false
                    )
                }*/
            }
        }
    }

    override fun onResults(resultBundle: ImageSegmenterHelper.ResultBundle) {
        TODO("Not yet implemented")
    }

    companion object {
        private const val INPUT_IMAGE_MAX_WIDTH = 512F
    }
}
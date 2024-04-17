/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.facelandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.widget.Toast
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService

class FaceMeshLandmarker(private val context: Context) : FaceLandmarkerHelper.LandmarkerListener {

    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ScheduledExecutorService

    // Load and display the image.
    fun runFaceMeshOnImage(inputFile: File) : MutableMap<Int, MutableList<Int>>{

        backgroundExecutor = Executors.newSingleThreadScheduledExecutor()
        var inputBitmap = toBitmap(inputFile)

        // Run face landmarker on the input image
        val coordinatesMap = mutableMapOf<Int, MutableList<Int>>()

        //backgroundExecutor.execute {

        faceLandmarkerHelper = FaceLandmarkerHelper(
                context = context,
                runningMode = RunningMode.IMAGE,
                minFaceDetectionConfidence = FaceLandmarkerHelper.DEFAULT_FACE_DETECTION_CONFIDENCE,
                minFaceTrackingConfidence = FaceLandmarkerHelper.DEFAULT_FACE_TRACKING_CONFIDENCE,
                minFacePresenceConfidence = FaceLandmarkerHelper.DEFAULT_FACE_PRESENCE_CONFIDENCE,
                maxNumFaces = FaceLandmarkerHelper.DEFAULT_NUM_FACES,
                currentDelegate = FaceLandmarkerHelper.DELEGATE_CPU
        )

        val imageHeight = inputBitmap.height
        val imageWidth = inputBitmap.width
        val scaleFactor = 1

        (faceLandmarkerHelper.detectImage(inputBitmap))?.result?.let{faceLandmarkerResult ->
            var meshID = 0
            for(landmark in faceLandmarkerResult.faceLandmarks()) {
                for(normalizedLandmark in landmark) {
                    var x = normalizedLandmark.x() * imageWidth * scaleFactor
                    var y = normalizedLandmark.y() * imageHeight * scaleFactor

                    var coordinate = mutableListOf<Int>(x.toInt(), y.toInt())
                    coordinatesMap[meshID] = coordinate
                    meshID += 1
                }
            }
        }

        faceLandmarkerHelper.clearFaceLandmarker()
        //}
        return coordinatesMap
    }

    // convert Uri/assets to bitmap image.
    private fun toBitmap(file : File): Bitmap {
        val source = ImageDecoder.createSource(file)
        return (ImageDecoder.decodeBitmap(source)).copy(Bitmap.Config.ARGB_8888, true)
    }

    // 2024-03-28, added by Shu Li following selfie segmentation AI.
    private fun stopAllTasks() {
        backgroundExecutor?.shutdownNow()

        faceLandmarkerHelper.clearFaceLandmarker()
    }

    // 2024-03-28, updated by Shu Li following selfie segmentation AI.
    private fun classifyingError() {
        stopAllTasks()
    }

    override fun onError(error: String, errorCode: Int) {
        classifyingError()
        Toast.makeText(context, error, Toast.LENGTH_SHORT).show()
    }

    override fun onResults(resultBundle: FaceLandmarkerHelper.ResultBundle) {
        // no-op
    }

    companion object {
        private const val TAG = "GalleryFragment"

        // Value used to get frames at specific intervals for inference (e.g. every 300ms)
        private const val VIDEO_INTERVAL_MS = 300L
    }
}
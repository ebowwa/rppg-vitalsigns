import Foundation
import Vision
import CoreImage
import AVFoundation

public struct FaceRegion {
    public let boundingBox: CGRect
    public let landmarks: [CGPoint]
    public let confidence: Float
    
    public init(boundingBox: CGRect, landmarks: [CGPoint], confidence: Float) {
        self.boundingBox = boundingBox
        self.landmarks = landmarks
        self.confidence = confidence
    }
}

public class FaceDetector {
    private let faceDetectionRequest: VNDetectFaceRectanglesRequest
    private let landmarkRequest: VNDetectFaceLandmarksRequest
    private var lastFaceRegion: FaceRegion?
    
    public init() {
        self.faceDetectionRequest = VNDetectFaceRectanglesRequest()
        self.landmarkRequest = VNDetectFaceLandmarksRequest()
        
        if #available(iOS 15.0, macOS 12.0, *) {
            self.faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision3
            self.landmarkRequest.revision = VNDetectFaceLandmarksRequestRevision3
        } else {
            self.faceDetectionRequest.revision = VNDetectFaceRectanglesRequestRevision2
            self.landmarkRequest.revision = VNDetectFaceLandmarksRequestRevision2
        }
    }
    
    public func detectFace(in image: CVPixelBuffer) -> FaceRegion? {
        let handler = VNImageRequestHandler(cvPixelBuffer: image, options: [:])
        
        do {
            try handler.perform([faceDetectionRequest])
            
            guard let faceObservation = faceDetectionRequest.results?.first as? VNFaceObservation else {
                return nil
            }
            
            let boundingBox = faceObservation.boundingBox
            let confidence = faceObservation.confidence
            
            try handler.perform([landmarkRequest])
            
            var landmarks: [CGPoint] = []
            if let landmarkObservation = landmarkRequest.results?.first as? VNFaceObservation,
               let allPoints = landmarkObservation.landmarks?.allPoints {
                landmarks = allPoints.normalizedPoints.map { point in
                    CGPoint(x: boundingBox.origin.x + point.x * boundingBox.width,
                           y: boundingBox.origin.y + point.y * boundingBox.height)
                }
            }
            
            let faceRegion = FaceRegion(boundingBox: boundingBox, landmarks: landmarks, confidence: confidence)
            lastFaceRegion = faceRegion
            return faceRegion
            
        } catch {
            print("Face detection error: \(error)")
            return nil
        }
    }
    
    public func trackFace(in image: CVPixelBuffer, previousRegion: FaceRegion) -> FaceRegion? {
        let trackingRequest = VNTrackRectangleRequest(rectangleObservation: VNRectangleObservation(boundingBox: previousRegion.boundingBox))
        
        let handler = VNImageRequestHandler(cvPixelBuffer: image, options: [:])
        
        do {
            try handler.perform([trackingRequest])
            
            guard let trackingObservation = trackingRequest.results?.first as? VNRectangleObservation else {
                return detectFace(in: image)
            }
            
            let updatedBoundingBox = trackingObservation.boundingBox
            let updatedRegion = FaceRegion(
                boundingBox: updatedBoundingBox,
                landmarks: previousRegion.landmarks,
                confidence: trackingObservation.confidence
            )
            
            return updatedRegion
            
        } catch {
            print("Face tracking error: \(error)")
            return detectFace(in: image)
        }
    }
    
    public func extractROI(from boundingBox: CGRect) -> CGRect {
        let roiX = boundingBox.origin.x + 0.3 * boundingBox.width
        let roiY = boundingBox.origin.y + 0.1 * boundingBox.height
        let roiWidth = 0.4 * boundingBox.width
        let roiHeight = 0.15 * boundingBox.height
        
        return CGRect(x: roiX, y: roiY, width: roiWidth, height: roiHeight)
    }
}

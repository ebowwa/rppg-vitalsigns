import XCTest
@testable import RPPGKit

final class RPPGKitTests: XCTestCase {
    
    func testRPPGAlgorithmEnum() {
        XCTAssertEqual(RPPGAlgorithm.g.rawValue, "g")
        XCTAssertEqual(RPPGAlgorithm.pca.rawValue, "pca")
        XCTAssertEqual(RPPGAlgorithm.xminay.rawValue, "xminay")
    }
    
    func testFaceDetectionAlgorithmEnum() {
        XCTAssertEqual(FaceDetectionAlgorithm.haar.rawValue, "haar")
        XCTAssertEqual(FaceDetectionAlgorithm.deep.rawValue, "deep")
    }
    
    func testRPPGConstants() {
        XCTAssertEqual(RPPGConstants.lowBPM, 42)
        XCTAssertEqual(RPPGConstants.highBPM, 240)
        XCTAssertEqual(RPPGConstants.secPerMin, 60)
    }
    
    func testSignalProcessingFPS() {
        let timeArray = [0.0, 1.0, 2.0, 3.0, 4.0]
        let fps = SignalProcessing.getFps(timeArray: timeArray, timeBase: 1.0)
        XCTAssertEqual(fps, 1.25, accuracy: 0.01)
    }
    
    func testSignalProcessingNormalization() {
        let input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        let normalized = SignalProcessing.normalization(input: input)
        
        XCTAssertEqual(normalized.count, 3)
        XCTAssertEqual(normalized[0].count, 3)
        
        for col in 0..<3 {
            let column = normalized.map { $0[col] }
            let mean = column.reduce(0, +) / Double(column.count)
            XCTAssertEqual(mean, 0.0, accuracy: 0.001)
        }
    }
    
    func testSignalProcessingDenoise() {
        let input = [1.0, 2.0, 5.0, 4.0, 5.0]
        let jumps = [false, false, true, false, false]
        let denoised = SignalProcessing.denoise(input: input, jumps: jumps)
        
        XCTAssertEqual(denoised.count, input.count)
        XCTAssertNotEqual(denoised[2], input[2])
    }
    
    func testSignalProcessingMovingAverage() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0]
        let smoothed = SignalProcessing.movingAverage(input: input, iterations: 1, windowSize: 3)
        
        XCTAssertEqual(smoothed.count, input.count)
        XCTAssertTrue(smoothed[2] > input[1] && smoothed[2] < input[3])
    }
    
    func testRPPGConfiguration() {
        let config = RPPG.Configuration(
            algorithm: .pca,
            faceDetectionAlgorithm: .deep,
            samplingFrequency: 2.0,
            rescanFrequency: 0.5,
            minSignalSize: 10,
            maxSignalSize: 20
        )
        
        XCTAssertEqual(config.algorithm, .pca)
        XCTAssertEqual(config.faceDetectionAlgorithm, .deep)
        XCTAssertEqual(config.samplingFrequency, 2.0)
        XCTAssertEqual(config.rescanFrequency, 0.5)
        XCTAssertEqual(config.minSignalSize, 10)
        XCTAssertEqual(config.maxSignalSize, 20)
    }
    
    func testRPPGResults() {
        let results = RPPG.Results(
            heartRate: 72.0,
            meanBPM: 70.0,
            minBPM: 65.0,
            maxBPM: 75.0,
            faceValid: true,
            fps: 30.0
        )
        
        XCTAssertEqual(results.heartRate, 72.0)
        XCTAssertEqual(results.meanBPM, 70.0)
        XCTAssertEqual(results.minBPM, 65.0)
        XCTAssertEqual(results.maxBPM, 75.0)
        XCTAssertTrue(results.faceValid)
        XCTAssertEqual(results.fps, 30.0)
    }
    
    func testFaceRegion() {
        let boundingBox = CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        let landmarks = [CGPoint(x: 0.15, y: 0.25), CGPoint(x: 0.35, y: 0.25)]
        let confidence: Float = 0.95
        
        let faceRegion = FaceRegion(boundingBox: boundingBox, landmarks: landmarks, confidence: confidence)
        
        XCTAssertEqual(faceRegion.boundingBox, boundingBox)
        XCTAssertEqual(faceRegion.landmarks.count, 2)
        XCTAssertEqual(faceRegion.confidence, confidence)
    }
    
    func testFaceDetectorROIExtraction() {
        let detector = FaceDetector()
        let boundingBox = CGRect(x: 0.0, y: 0.0, width: 1.0, height: 1.0)
        let roi = detector.extractROI(from: boundingBox)
        
        XCTAssertEqual(roi.origin.x, 0.3)
        XCTAssertEqual(roi.origin.y, 0.1)
        XCTAssertEqual(roi.width, 0.4)
        XCTAssertEqual(roi.height, 0.15)
    }
}

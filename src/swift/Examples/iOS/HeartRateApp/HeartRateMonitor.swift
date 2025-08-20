import Foundation
import SwiftUI
import RPPGKit
import AVFoundation

struct HeartRateMeasurement {
    let heartRate: Double
    let timestamp: Date
    let algorithm: RPPGAlgorithm
}

class HeartRateMonitor: NSObject, ObservableObject {
    @Published var currentHeartRate: Double = 0
    @Published var fps: Double = 0
    @Published var faceDetected: Bool = false
    @Published var cameraPermissionGranted: Bool = false
    @Published var measurements: [HeartRateMeasurement] = []
    
    private var videoProcessor: VideoProcessor?
    let previewView = UIView()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    override init() {
        super.init()
        checkCameraPermission()
    }
    
    func requestCameraPermission() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                self?.cameraPermissionGranted = granted
                if granted {
                    self?.setupPreview()
                }
            }
        }
    }
    
    private func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraPermissionGranted = true
            setupPreview()
        case .notDetermined:
            requestCameraPermission()
        case .denied, .restricted:
            cameraPermissionGranted = false
        @unknown default:
            cameraPermissionGranted = false
        }
    }
    
    private func setupPreview() {
        guard cameraPermissionGranted else { return }
        
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        
        guard let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: frontCamera) else {
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = previewView.bounds
        
        if let previewLayer = previewLayer {
            previewView.layer.addSublayer(previewLayer)
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            captureSession.startRunning()
        }
    }
    
    func startProcessing(algorithm: RPPGAlgorithm) {
        let config = RPPG.Configuration(
            algorithm: algorithm,
            faceDetectionAlgorithm: .deep,
            samplingFrequency: 1.0,
            rescanFrequency: 1.0,
            minSignalSize: 5,
            maxSignalSize: 10,
            timeBase: 1.0 / 30.0,
            enableLogging: false,
            enableGUI: true
        )
        
        videoProcessor = VideoProcessor(configuration: config)
        videoProcessor?.delegate = self
        videoProcessor?.startProcessing()
    }
    
    func stopProcessing() {
        videoProcessor?.stopProcessing()
        videoProcessor = nil
        
        currentHeartRate = 0
        fps = 0
        faceDetected = false
    }
    
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        if keyPath == "bounds" {
            DispatchQueue.main.async { [weak self] in
                self?.previewLayer?.frame = self?.previewView.bounds ?? .zero
            }
        }
    }
}

extension HeartRateMonitor: VideoProcessorDelegate {
    func videoProcessor(_ processor: VideoProcessor, didProcess results: RPPG.Results) {
        DispatchQueue.main.async { [weak self] in
            self?.currentHeartRate = results.heartRate
            self?.fps = results.fps
            self?.faceDetected = results.faceValid
            
            if results.faceValid && results.heartRate > 0 {
                let measurement = HeartRateMeasurement(
                    heartRate: results.heartRate,
                    timestamp: Date(),
                    algorithm: .g
                )
                self?.measurements.append(measurement)
                
                if let measurements = self?.measurements, measurements.count > 100 {
                    self?.measurements.removeFirst()
                }
            }
        }
    }
    
    func videoProcessor(_ processor: VideoProcessor, didEncounterError error: Error) {
        print("Video processing error: \(error)")
    }
}

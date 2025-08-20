import Foundation
import AVFoundation
import CoreVideo

public protocol VideoProcessorDelegate: AnyObject {
    func videoProcessor(_ processor: VideoProcessor, didProcess results: RPPG.Results)
    func videoProcessor(_ processor: VideoProcessor, didEncounterError error: Error)
}

public class VideoProcessor: NSObject {
    
    public weak var delegate: VideoProcessorDelegate?
    
    private let rppg: RPPG
    private let captureSession: AVCaptureSession
    private let videoOutput: AVCaptureVideoDataOutput
    private let processingQueue: DispatchQueue
    
    private var isProcessing: Bool = false
    private var frameCount: Int64 = 0
    
    public init(configuration: RPPG.Configuration) {
        self.rppg = RPPG(configuration: configuration)
        self.captureSession = AVCaptureSession()
        self.videoOutput = AVCaptureVideoDataOutput()
        self.processingQueue = DispatchQueue(label: "com.rppg.processing", qos: .userInitiated)
        
        super.init()
        
        setupCaptureSession()
    }
    
    private func setupCaptureSession() {
        captureSession.sessionPreset = .medium
        
        guard let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            delegate?.videoProcessor(self, didEncounterError: VideoProcessorError.cameraNotAvailable)
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: frontCamera)
            
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
            }
            
        } catch {
            delegate?.videoProcessor(self, didEncounterError: error)
        }
    }
    
    public func startProcessing() {
        guard !isProcessing else { return }
        
        isProcessing = true
        frameCount = 0
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }
    
    public func stopProcessing() {
        guard isProcessing else { return }
        
        isProcessing = false
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }
    
    public func processVideoFile(at url: URL, completion: @escaping ([RPPG.Results]) -> Void) {
        var results: [RPPG.Results] = []
        
        let asset = AVAsset(url: url)
        let reader: AVAssetReader
        
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            delegate?.videoProcessor(self, didEncounterError: error)
            completion([])
            return
        }
        
        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            delegate?.videoProcessor(self, didEncounterError: VideoProcessorError.noVideoTrack)
            completion([])
            return
        }
        
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        let readerOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        reader.add(readerOutput)
        
        reader.startReading()
        
        var frameIndex: Int64 = 0
        
        while reader.status == .reading {
            if let sampleBuffer = readerOutput.copyNextSampleBuffer(),
               let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                
                let result = rppg.processFrame(pixelBuffer, timestamp: frameIndex)
                results.append(result)
                frameIndex += 1
            }
        }
        
        completion(results)
    }
}

extension VideoProcessor: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isProcessing,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let result = rppg.processFrame(pixelBuffer, timestamp: frameCount)
        frameCount += 1
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.videoProcessor(self, didProcess: result)
        }
    }
    
    public func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        print("Dropped frame")
    }
}

public enum VideoProcessorError: Error {
    case cameraNotAvailable
    case noVideoTrack
    case processingFailed
    
    public var localizedDescription: String {
        switch self {
        case .cameraNotAvailable:
            return "Camera not available"
        case .noVideoTrack:
            return "No video track found"
        case .processingFailed:
            return "Video processing failed"
        }
    }
}

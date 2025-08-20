import Foundation
import AVFoundation
import CoreImage
import Vision

public class RPPG {
    
    public struct Configuration {
        public let algorithm: RPPGAlgorithm
        public let faceDetectionAlgorithm: FaceDetectionAlgorithm
        public let samplingFrequency: Double
        public let rescanFrequency: Double
        public let minSignalSize: Int
        public let maxSignalSize: Int
        public let timeBase: Double
        public let enableLogging: Bool
        public let enableGUI: Bool
        
        public init(
            algorithm: RPPGAlgorithm = .g,
            faceDetectionAlgorithm: FaceDetectionAlgorithm = .deep,
            samplingFrequency: Double = 1.0,
            rescanFrequency: Double = 1.0,
            minSignalSize: Int = 5,
            maxSignalSize: Int = 5,
            timeBase: Double = 1.0,
            enableLogging: Bool = false,
            enableGUI: Bool = true
        ) {
            self.algorithm = algorithm
            self.faceDetectionAlgorithm = faceDetectionAlgorithm
            self.samplingFrequency = samplingFrequency
            self.rescanFrequency = rescanFrequency
            self.minSignalSize = minSignalSize
            self.maxSignalSize = maxSignalSize
            self.timeBase = timeBase
            self.enableLogging = enableLogging
            self.enableGUI = enableGUI
        }
    }
    
    public struct Results {
        public let heartRate: Double
        public let meanBPM: Double
        public let minBPM: Double
        public let maxBPM: Double
        public let faceValid: Bool
        public let fps: Double
        
        public init(heartRate: Double, meanBPM: Double, minBPM: Double, maxBPM: Double, faceValid: Bool, fps: Double) {
            self.heartRate = heartRate
            self.meanBPM = meanBPM
            self.minBPM = minBPM
            self.maxBPM = maxBPM
            self.faceValid = faceValid
            self.fps = fps
        }
    }
    
    private let configuration: Configuration
    private let faceDetector: FaceDetector
    
    private var faceValid: Bool = false
    private var lastScanTime: Int64 = 0
    private var lastSamplingTime: Int64 = 0
    private var currentTime: Int64 = 0
    private var fps: Double = 0
    private var rescanFlag: Bool = false
    
    private var currentFaceRegion: FaceRegion?
    private var roi: CGRect = .zero
    
    private var signalBuffer: [[Double]] = []
    private var timeBuffer: [Double] = []
    private var rescanBuffer: [Bool] = []
    
    private var filteredSignal: [Double] = []
    private var powerSpectrum: [Double] = []
    private var bpmHistory: [Double] = []
    
    private var currentBPM: Double = 0
    private var meanBPM: Double = 0
    private var minBPM: Double = 0
    private var maxBPM: Double = 0
    
    private var lowBand: Int = 0
    private var highBand: Int = 0
    
    public init(configuration: Configuration) {
        self.configuration = configuration
        self.faceDetector = FaceDetector()
    }
    
    public func processFrame(_ pixelBuffer: CVPixelBuffer, timestamp: Int64) -> Results {
        currentTime = timestamp
        
        if !faceValid {
            print("Not valid, finding a new face")
            lastScanTime = timestamp
            detectFace(in: pixelBuffer)
        } else if Double(timestamp - lastScanTime) * configuration.timeBase >= 1.0 / configuration.rescanFrequency {
            print("Valid, but rescanning face")
            lastScanTime = timestamp
            detectFace(in: pixelBuffer)
            rescanFlag = true
        } else {
            print("Tracking face")
            trackFace(in: pixelBuffer)
        }
        
        if faceValid {
            updateFPS()
            
            while signalBuffer.count > Int(fps * Double(configuration.maxSignalSize)) {
                signalBuffer.removeFirst()
                timeBuffer.removeFirst()
                rescanBuffer.removeFirst()
            }
            
            let rgbMeans = extractRGBMeans(from: pixelBuffer, roi: roi)
            signalBuffer.append(rgbMeans)
            timeBuffer.append(Double(timestamp))
            rescanBuffer.append(rescanFlag)
            
            updateFPS()
            updateBandLimits()
            
            if signalBuffer.count >= Int(fps * Double(configuration.minSignalSize)) {
                switch configuration.algorithm {
                case .g:
                    extractSignalG()
                case .pca:
                    extractSignalPCA()
                case .xminay:
                    extractSignalXMinAY()
                }
                
                estimateHeartRate()
            }
        }
        
        rescanFlag = false
        
        return Results(
            heartRate: currentBPM,
            meanBPM: meanBPM,
            minBPM: minBPM,
            maxBPM: maxBPM,
            faceValid: faceValid,
            fps: fps
        )
    }
    
    private func detectFace(in pixelBuffer: CVPixelBuffer) {
        print("Scanning for faces...")
        
        if let faceRegion = faceDetector.detectFace(in: pixelBuffer) {
            print("Found a face")
            currentFaceRegion = faceRegion
            roi = faceDetector.extractROI(from: faceRegion.boundingBox)
            faceValid = true
        } else {
            print("Found no face")
            invalidateFace()
        }
    }
    
    private func trackFace(in pixelBuffer: CVPixelBuffer) {
        guard let previousRegion = currentFaceRegion else {
            detectFace(in: pixelBuffer)
            return
        }
        
        if let updatedRegion = faceDetector.trackFace(in: pixelBuffer, previousRegion: previousRegion) {
            currentFaceRegion = updatedRegion
            roi = faceDetector.extractROI(from: updatedRegion.boundingBox)
        } else {
            print("Tracking failed!")
            invalidateFace()
        }
    }
    
    private func invalidateFace() {
        signalBuffer.removeAll()
        filteredSignal.removeAll()
        timeBuffer.removeAll()
        rescanBuffer.removeAll()
        powerSpectrum.removeAll()
        faceValid = false
        currentFaceRegion = nil
    }
    
    private func updateFPS() {
        fps = SignalProcessing.getFps(timeArray: timeBuffer, timeBase: configuration.timeBase)
    }
    
    private func updateBandLimits() {
        lowBand = Int(Double(signalBuffer.count) * RPPGConstants.lowBPM / RPPGConstants.secPerMin / fps)
        highBand = Int(Double(signalBuffer.count) * RPPGConstants.highBPM / RPPGConstants.secPerMin / fps) + 1
    }
    
    private func extractRGBMeans(from pixelBuffer: CVPixelBuffer, roi: CGRect) -> [Double] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return [0, 0, 0]
        }
        
        let roiX = Int(roi.origin.x * CGFloat(width))
        let roiY = Int(roi.origin.y * CGFloat(height))
        let roiWidth = Int(roi.width * CGFloat(width))
        let roiHeight = Int(roi.height * CGFloat(height))
        
        var rSum: Double = 0
        var gSum: Double = 0
        var bSum: Double = 0
        var pixelCount: Double = 0
        
        for y in roiY..<min(roiY + roiHeight, height) {
            for x in roiX..<min(roiX + roiWidth, width) {
                let pixelOffset = y * bytesPerRow + x * 4
                let pixel = baseAddress.advanced(by: pixelOffset).assumingMemoryBound(to: UInt8.self)
                
                bSum += Double(pixel[0])
                gSum += Double(pixel[1])
                rSum += Double(pixel[2])
                pixelCount += 1
            }
        }
        
        guard pixelCount > 0 else { return [0, 0, 0] }
        
        return [rSum / pixelCount, gSum / pixelCount, bSum / pixelCount]
    }
    
    private func extractSignalG() {
        let greenChannel = signalBuffer.map { $0[1] }
        
        let denoised = SignalProcessing.denoise(input: greenChannel, jumps: rescanBuffer)
        let normalized = SignalProcessing.normalization(input: [denoised])[0]
        let detrended = SignalProcessing.detrend(input: normalized, lambda: fps)
        let smoothed = SignalProcessing.movingAverage(
            input: detrended,
            iterations: 3,
            windowSize: max(Int(floor(fps / 6)), 2)
        )
        
        filteredSignal = smoothed
    }
    
    private func extractSignalPCA() {
        let denoised = SignalProcessing.normalization(input: signalBuffer.map { row in
            SignalProcessing.denoise(input: row, jumps: rescanBuffer)
        })
        
        let normalized = SignalProcessing.normalization(input: denoised)
        
        let detrended = normalized.map { row in
            SignalProcessing.detrend(input: row, lambda: fps)
        }
        
        let transposed = Array(0..<detrended[0].count).map { col in
            detrended.map { $0[col] }
        }
        
        let (pcaSignal, _) = SignalProcessing.pcaComponent(
            input: transposed,
            lowBand: lowBand,
            highBand: highBand
        )
        
        let smoothed = SignalProcessing.movingAverage(
            input: pcaSignal,
            iterations: 3,
            windowSize: max(Int(floor(fps / 6)), 2)
        )
        
        filteredSignal = smoothed
    }
    
    private func extractSignalXMinAY() {
        let denoised = SignalProcessing.normalization(input: signalBuffer.map { row in
            SignalProcessing.denoise(input: row, jumps: rescanBuffer)
        })
        
        let normalized = SignalProcessing.normalization(input: denoised)
        
        let rChannel = normalized.map { $0[0] }
        let gChannel = normalized.map { $0[1] }
        let bChannel = normalized.map { $0[2] }
        
        var xSignal = [Double](repeating: 0, count: rChannel.count)
        var ySignal = [Double](repeating: 0, count: rChannel.count)
        
        for i in 0..<rChannel.count {
            xSignal[i] = 3 * rChannel[i] - 2 * gChannel[i]
            ySignal[i] = 1.5 * rChannel[i] + gChannel[i] - 1.5 * bChannel[i]
        }
        
        let xFiltered = SignalProcessing.bandpass(input: xSignal, lowFreq: Double(lowBand), highFreq: Double(highBand))
        let yFiltered = SignalProcessing.bandpass(input: ySignal, lowFreq: Double(lowBand), highFreq: Double(highBand))
        
        let xStdDev = sqrt(xFiltered.map { $0 * $0 }.reduce(0, +) / Double(xFiltered.count))
        let yStdDev = sqrt(yFiltered.map { $0 * $0 }.reduce(0, +) / Double(yFiltered.count))
        
        let alpha = xStdDev / yStdDev
        
        var xMinAY = [Double](repeating: 0, count: xFiltered.count)
        for i in 0..<xFiltered.count {
            xMinAY[i] = xFiltered[i] - alpha * yFiltered[i]
        }
        
        let smoothed = SignalProcessing.movingAverage(
            input: xMinAY,
            iterations: 3,
            windowSize: max(Int(floor(fps / 6)), 2)
        )
        
        filteredSignal = smoothed
    }
    
    private func estimateHeartRate() {
        powerSpectrum = SignalProcessing.timeToFrequency(input: filteredSignal, magnitude: true)
        
        guard !powerSpectrum.isEmpty else { return }
        
        var maxPower = 0.0
        var maxIndex = 0
        
        for i in lowBand..<min(highBand, powerSpectrum.count) {
            if powerSpectrum[i] > maxPower {
                maxPower = powerSpectrum[i]
                maxIndex = i
            }
        }
        
        currentBPM = Double(maxIndex) * fps / Double(filteredSignal.count) * RPPGConstants.secPerMin
        bpmHistory.append(currentBPM)
        
        print("FPS=\(fps) Vals=\(powerSpectrum.count) Peak=\(maxIndex) BPM=\(currentBPM)")
        
        if Double(currentTime - lastSamplingTime) * configuration.timeBase >= 1.0 / configuration.samplingFrequency {
            lastSamplingTime = currentTime
            
            let sortedBPMs = bpmHistory.sorted()
            meanBPM = bpmHistory.reduce(0, +) / Double(bpmHistory.count)
            minBPM = sortedBPMs.first ?? 0
            maxBPM = sortedBPMs.last ?? 0
            
            print("meanBPM=\(meanBPM) minBpm=\(minBPM) maxBpm=\(maxBPM)")
            
            bpmHistory.removeAll()
        }
    }
}

# RPPGKit - Swift Implementation

A Swift port of the rPPG (remote photoplethysmography) algorithms from the [heartbeat](https://github.com/prouast/heartbeat) repository, designed for iOS and macOS applications.

## Overview

RPPGKit provides real-time heart rate estimation from facial videos using remote photoplethysmography techniques. This Swift implementation replaces OpenCV dependencies with native iOS frameworks for optimal performance on Apple platforms.

## Features

- **Multiple rPPG Algorithms**: G (green channel), PCA, and XMINAY implementations
- **Native iOS Integration**: Uses Vision framework for face detection and Accelerate for signal processing
- **Real-time Processing**: Optimized for live camera input with AVFoundation
- **Video File Support**: Process pre-recorded videos for analysis
- **Comprehensive Testing**: Unit tests for all core algorithms and components

## Architecture

### Core Components

- **RPPG**: Main class implementing the three rPPG algorithms
- **SignalProcessing**: High-performance signal processing utilities using Accelerate framework
- **FaceDetection**: Face detection and tracking using Vision framework
- **VideoProcessor**: Real-time video processing with AVFoundation

### Swift Framework Replacements

| Original (C++/OpenCV) | Swift Equivalent |
|----------------------|------------------|
| OpenCV face detection | Vision framework |
| OpenCV signal processing | Accelerate framework |
| OpenCV video capture | AVFoundation |
| OpenCV matrix operations | Native Swift arrays + vDSP |

## Installation

### Swift Package Manager

Add RPPGKit to your project using Swift Package Manager:

```swift
dependencies: [
    .package(path: "path/to/rppg-vitalsigns/src/swift")
]
```

### Requirements

- iOS 14.0+ / macOS 11.0+
- Xcode 12.0+
- Swift 5.3+

## Usage

### Basic Real-time Processing

```swift
import RPPGKit

class HeartRateViewController: UIViewController {
    private var videoProcessor: VideoProcessor?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let config = RPPG.Configuration(
            algorithm: .g,
            faceDetectionAlgorithm: .deep,
            samplingFrequency: 1.0,
            rescanFrequency: 1.0
        )
        
        videoProcessor = VideoProcessor(configuration: config)
        videoProcessor?.delegate = self
    }
    
    func startHeartRateDetection() {
        videoProcessor?.startProcessing()
    }
    
    func stopHeartRateDetection() {
        videoProcessor?.stopProcessing()
    }
}

extension HeartRateViewController: VideoProcessorDelegate {
    func videoProcessor(_ processor: VideoProcessor, didProcess results: RPPG.Results) {
        DispatchQueue.main.async {
            if results.faceValid {
                print("Heart Rate: \(results.heartRate) BPM")
                print("Mean BPM: \(results.meanBPM)")
            }
        }
    }
    
    func videoProcessor(_ processor: VideoProcessor, didEncounterError error: Error) {
        print("Processing error: \(error)")
    }
}
```

### Processing Video Files

```swift
let config = RPPG.Configuration(algorithm: .pca)
let processor = VideoProcessor(configuration: config)

let videoURL = URL(fileURLWithPath: "path/to/video.mp4")
processor.processVideoFile(at: videoURL) { results in
    for result in results {
        if result.faceValid {
            print("Frame heart rate: \(result.heartRate) BPM")
        }
    }
}
```

### Algorithm Comparison

```swift
let algorithms: [RPPGAlgorithm] = [.g, .pca, .xminay]

for algorithm in algorithms {
    let config = RPPG.Configuration(algorithm: algorithm)
    let processor = VideoProcessor(configuration: config)
    
    // Process same video with different algorithms
    processor.processVideoFile(at: videoURL) { results in
        let validResults = results.filter { $0.faceValid }
        let averageHR = validResults.map { $0.heartRate }.reduce(0, +) / Double(validResults.count)
        print("\(algorithm.rawValue) algorithm average HR: \(averageHR) BPM")
    }
}
```

## Algorithm Details

### G Algorithm (Green Channel)
- Extracts heart rate signal from green color channel
- Fastest processing, suitable for real-time applications
- Best for controlled lighting conditions

### PCA Algorithm
- Uses Principal Component Analysis on RGB channels
- Better noise reduction than G algorithm
- Good balance between accuracy and performance

### XMINAY Algorithm
- Advanced chrominance-based approach
- Highest accuracy, especially in challenging conditions
- More computationally intensive

## Performance

The Swift implementation achieves comparable performance to the original C++ version:

- **Real-time processing**: 30+ FPS on modern iOS devices
- **Memory efficient**: Optimized for mobile constraints
- **Low latency**: Minimal processing delay for live applications

## Testing

Run the test suite:

```bash
cd src/swift
swift test
```

The test suite includes:
- Unit tests for all signal processing functions
- Algorithm validation tests
- Face detection component tests
- Integration tests for the complete pipeline

## Integration with Python Evaluation Framework

The Swift implementation can be integrated with the existing Python evaluation tools in the rppg-vitalsigns repository:

1. Export results to CSV format compatible with existing evaluation scripts
2. Use the same performance metrics (MAE, SNR, correlation)
3. Compare Swift implementation against reference algorithms

## Contributing

1. Follow Swift coding conventions and best practices
2. Add unit tests for new functionality
3. Update documentation for API changes
4. Ensure compatibility with existing evaluation framework

## License

This Swift implementation follows the same license as the parent rppg-vitalsigns repository.

## Acknowledgments

- Original C++ implementation: [heartbeat](https://github.com/prouast/heartbeat) by Philipp Rouast
- rPPG research community for algorithm development
- Apple's Vision and Accelerate frameworks for enabling native iOS implementation

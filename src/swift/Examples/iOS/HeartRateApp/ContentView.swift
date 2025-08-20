import SwiftUI
import RPPGKit
import AVFoundation

struct ContentView: View {
    @StateObject private var heartRateMonitor = HeartRateMonitor()
    @State private var selectedAlgorithm: RPPGAlgorithm = .g
    @State private var isProcessing = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("rPPG Heart Rate Monitor")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            if heartRateMonitor.cameraPermissionGranted {
                CameraPreviewView(heartRateMonitor: heartRateMonitor)
                    .frame(height: 300)
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(heartRateMonitor.faceDetected ? Color.green : Color.red, lineWidth: 3)
                    )
            } else {
                VStack {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.gray)
                    Text("Camera permission required")
                        .font(.headline)
                    Button("Grant Permission") {
                        heartRateMonitor.requestCameraPermission()
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .frame(height: 300)
            }
            
            VStack(spacing: 10) {
                HStack {
                    Text("Algorithm:")
                        .font(.headline)
                    Picker("Algorithm", selection: $selectedAlgorithm) {
                        ForEach(RPPGAlgorithm.allCases, id: \.self) { algorithm in
                            Text(algorithm.rawValue.uppercased()).tag(algorithm)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                
                HStack {
                    VStack {
                        Text("Heart Rate")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(Int(heartRateMonitor.currentHeartRate))")
                            .font(.title)
                            .fontWeight(.bold)
                            .foregroundColor(heartRateMonitor.faceDetected ? .primary : .gray)
                        Text("BPM")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    VStack {
                        Text("FPS")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(Int(heartRateMonitor.fps))")
                            .font(.title2)
                            .fontWeight(.semibold)
                    }
                    
                    Spacer()
                    
                    VStack {
                        Text("Face")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Image(systemName: heartRateMonitor.faceDetected ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(heartRateMonitor.faceDetected ? .green : .red)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
            }
            
            Button(action: {
                if isProcessing {
                    heartRateMonitor.stopProcessing()
                } else {
                    heartRateMonitor.startProcessing(algorithm: selectedAlgorithm)
                }
                isProcessing.toggle()
            }) {
                HStack {
                    Image(systemName: isProcessing ? "stop.fill" : "play.fill")
                    Text(isProcessing ? "Stop" : "Start")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding()
                .frame(maxWidth: .infinity)
                .background(isProcessing ? Color.red : Color.blue)
                .cornerRadius(10)
            }
            .disabled(!heartRateMonitor.cameraPermissionGranted)
            
            if !heartRateMonitor.measurements.isEmpty {
                VStack(alignment: .leading) {
                    Text("Recent Measurements")
                        .font(.headline)
                    
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 5) {
                            ForEach(heartRateMonitor.measurements.suffix(10).reversed(), id: \.timestamp) { measurement in
                                HStack {
                                    Text("\(Int(measurement.heartRate)) BPM")
                                        .fontWeight(.medium)
                                    Spacer()
                                    Text(measurement.timestamp, style: .time)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    .frame(maxHeight: 150)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
            }
            
            Spacer()
        }
        .padding()
        .onAppear {
            heartRateMonitor.requestCameraPermission()
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let heartRateMonitor: HeartRateMonitor
    
    func makeUIView(context: Context) -> UIView {
        return heartRateMonitor.previewView
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

import Foundation

public enum RPPGAlgorithm: String, CaseIterable {
    case g = "g"
    case pca = "pca"
    case xminay = "xminay"
}

public enum FaceDetectionAlgorithm: String, CaseIterable {
    case haar = "haar"
    case deep = "deep"
}

public struct RPPGConstants {
    public static let lowBPM: Double = 42
    public static let highBPM: Double = 240
    public static let relMinFaceSize: Double = 0.4
    public static let secPerMin: Double = 60
    public static let maxCorners: Int = 10
    public static let minCorners: Int = 5
    public static let qualityLevel: Double = 0.01
    public static let minDistance: Double = 25
}

// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "RPPGKit",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "RPPGKit",
            targets: ["RPPGKit"]),
        .executable(
            name: "HeartRateApp",
            targets: ["HeartRateApp"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "RPPGKit",
            dependencies: [],
            path: "Sources/RPPGKit"),
        .executableTarget(
            name: "HeartRateApp",
            dependencies: ["RPPGKit"],
            path: "Examples/iOS/HeartRateApp"),
        .testTarget(
            name: "RPPGKitTests",
            dependencies: ["RPPGKit"],
            path: "Tests/RPPGKitTests"),
    ]
)

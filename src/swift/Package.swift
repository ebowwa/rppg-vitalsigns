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
    ],
    dependencies: [],
    targets: [
        .target(
            name: "RPPGKit",
            dependencies: [],
            path: "Sources/RPPGKit"),
        .testTarget(
            name: "RPPGKitTests",
            dependencies: ["RPPGKit"],
            path: "Tests/RPPGKitTests"),
    ]
)

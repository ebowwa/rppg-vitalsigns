import Foundation
import Accelerate

public struct SignalProcessing {
    
    public static func getFps(timeArray: [Double], timeBase: Double) -> Double {
        guard !timeArray.isEmpty else { return 1.0 }
        guard timeArray.count > 1 else { return Double.greatestFiniteMagnitude }
        
        let diff = (timeArray.last! - timeArray.first!) * timeBase
        return diff == 0 ? Double.greatestFiniteMagnitude : Double(timeArray.count) / diff
    }
    
    public static func normalization(input: [[Double]]) -> [[Double]] {
        var result = input
        
        for col in 0..<input[0].count {
            var column = input.map { $0[col] }
            
            let mean = vDSP.mean(column)
            let variance = vDSP.meanSquare(column) - mean * mean
            let stdDev = sqrt(variance)
            
            if stdDev > 0 {
                var meanArray = [Double](repeating: mean, count: column.count)
                vDSP_vsubD(meanArray, 1, column, 1, &column, 1, vDSP_Length(column.count))
                var stdDevArray = [Double](repeating: stdDev, count: column.count)
                vDSP_vdivD(stdDevArray, 1, column, 1, &column, 1, vDSP_Length(column.count))
            }
            
            for row in 0..<result.count {
                result[row][col] = column[row]
            }
        }
        
        return result
    }
    
    public static func denoise(input: [Double], jumps: [Bool]) -> [Double] {
        var result = input
        guard input.count == jumps.count else { return result }
        
        var diff = [Double](repeating: 0, count: input.count - 1)
        let dropFirst = Array(input.dropFirst())
        let dropLast = Array(input.dropLast())
        vDSP_vsubD(dropLast, 1, dropFirst, 1, &diff, 1, vDSP_Length(diff.count))
        
        for i in 1..<jumps.count {
            if jumps[i] {
                for j in i..<result.count {
                    result[j] -= diff[i-1]
                }
            }
        }
        
        return result
    }
    
    public static func detrend(input: [Double], lambda: Double) -> [Double] {
        let n = input.count
        guard n >= 3 else { return input }
        
        var identity = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            identity[i * n + i] = 1.0
        }
        
        var d2 = [Double](repeating: 0, count: (n-2) * n)
        for i in 0..<(n-2) {
            d2[i * n + i] = 1.0
            d2[i * n + i + 1] = -2.0
            d2[i * n + i + 2] = 1.0
        }
        
        var d2t = [Double](repeating: 0, count: n * (n-2))
        vDSP_mtransD(d2, 1, &d2t, 1, vDSP_Length(n), vDSP_Length(n-2))
        
        var d2td2 = [Double](repeating: 0, count: n * n)
        vDSP_mmulD(d2t, 1, d2, 1, &d2td2, 1, vDSP_Length(n), vDSP_Length(n), vDSP_Length(n-2))
        
        var lambdaD2td2 = [Double](repeating: 0, count: n * n)
        var lambdaSquared = lambda * lambda
        vDSP_vsmulD(d2td2, 1, &lambdaSquared, &lambdaD2td2, 1, vDSP_Length(n * n))
        
        var matrix = [Double](repeating: 0, count: n * n)
        vDSP_vaddD(identity, 1, lambdaD2td2, 1, &matrix, 1, vDSP_Length(n * n))
        
        var ipiv = [Int32](repeating: 0, count: n)
        var info: Int32 = 0
        var nInt = Int32(n)
        var nInt2 = Int32(n)
        var nInt3 = Int32(n)
        
        dgetrf_(&nInt, &nInt2, &matrix, &nInt3, &ipiv, &info)
        
        var result = input
        var nrhs: Int32 = 1
        var nInt4 = Int32(n)
        var nInt5 = Int32(n)
        var nInt6 = Int32(n)
        dgetrs_(UnsafeMutablePointer(mutating: "N".cString(using: .ascii)), &nInt4, &nrhs, &matrix, &nInt5, &ipiv, &result, &nInt6, &info)
        
        vDSP_vsubD(input, 1, result, 1, &result, 1, vDSP_Length(n))
        
        return result
    }
    
    public static func movingAverage(input: [Double], iterations: Int, windowSize: Int) -> [Double] {
        var result = input
        let kernel = [Double](repeating: 1.0 / Double(windowSize), count: windowSize)
        
        for _ in 0..<iterations {
            var convolved = [Double](repeating: 0, count: result.count)
            vDSP_convD(result, 1, kernel, 1, &convolved, 1, vDSP_Length(convolved.count), vDSP_Length(kernel.count))
            result = convolved
        }
        
        return result
    }
    
    public static func timeToFrequency(input: [Double], magnitude: Bool = false) -> [Double] {
        let n = input.count
        let log2n = vDSP_Length(log2(Double(n)))
        
        guard let fftSetup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else {
            return input
        }
        defer { vDSP_destroy_fftsetupD(fftSetup) }
        
        var realp = [Double](input)
        var imagp = [Double](repeating: 0, count: n)
        
        var splitComplex = DSPDoubleSplitComplex(realp: &realp, imagp: &imagp)
        
        vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
        
        if magnitude {
            var magnitudes = [Double](repeating: 0, count: n)
            vDSP_vdistD(realp, 1, imagp, 1, &magnitudes, 1, vDSP_Length(n))
            return magnitudes
        } else {
            return realp + imagp
        }
    }
    
    public static func bandpass(input: [Double], lowFreq: Double, highFreq: Double) -> [Double] {
        guard input.count >= 3 else { return input }
        
        let frequencySpectrum = timeToFrequency(input: input)
        let n = input.count
        
        var filter = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let freq = Double(i)
            if freq >= lowFreq && freq <= highFreq {
                filter[i] = 1.0
            }
        }
        
        var filtered = [Double](repeating: 0, count: n)
        vDSP_vmulD(frequencySpectrum, 1, filter, 1, &filtered, 1, vDSP_Length(n))
        
        return frequencyToTime(input: filtered)
    }
    
    public static func frequencyToTime(input: [Double]) -> [Double] {
        let n = input.count / 2
        let log2n = vDSP_Length(log2(Double(n)))
        
        guard let fftSetup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else {
            return Array(input.prefix(n))
        }
        defer { vDSP_destroy_fftsetupD(fftSetup) }
        
        var realp = Array(input.prefix(n))
        var imagp = Array(input.suffix(n))
        
        var splitComplex = DSPDoubleSplitComplex(realp: &realp, imagp: &imagp)
        
        vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))
        
        var scale = 1.0 / Double(n)
        vDSP_vsmulD(realp, 1, &scale, &realp, 1, vDSP_Length(n))
        
        return realp
    }
    
    public static func pcaComponent(input: [[Double]], lowBand: Int, highBand: Int) -> ([Double], [[Double]]) {
        let rows = input.count
        let cols = input[0].count
        
        var flatInput = input.flatMap { $0 }
        var mean = [Double](repeating: 0, count: cols)
        
        for col in 0..<cols {
            var column = [Double](repeating: 0, count: rows)
            for row in 0..<rows {
                column[row] = input[row][col]
            }
            mean[col] = vDSP.mean(column)
        }
        
        for row in 0..<rows {
            for col in 0..<cols {
                flatInput[row * cols + col] -= mean[col]
            }
        }
        
        var covariance = [Double](repeating: 0, count: cols * cols)
        vDSP_mmulD(flatInput, 1, flatInput, 1, &covariance, 1, vDSP_Length(cols), vDSP_Length(cols), vDSP_Length(rows))
        
        var eigenvalues = [Double](repeating: 0, count: cols)
        var eigenvectors = [Double](repeating: 0, count: cols * cols)
        var work = [Double](repeating: 0, count: 3 * cols)
        var info: Int32 = 0
        var colsInt = Int32(cols)
        var colsInt2 = Int32(cols)
        
        var workSize = Int32(3 * cols)
        dsyev_(UnsafeMutablePointer(mutating: "V".cString(using: .ascii)), 
               UnsafeMutablePointer(mutating: "U".cString(using: .ascii)), 
               &colsInt, &covariance, &colsInt2, &eigenvalues, &work, &workSize, &info)
        
        var pcComponents = [[Double]](repeating: [Double](repeating: 0, count: rows), count: cols)
        for i in 0..<cols {
            var component = [Double](repeating: 0, count: rows)
            for j in 0..<rows {
                for k in 0..<cols {
                    component[j] += input[j][k] * covariance[i * cols + k]
                }
            }
            pcComponents[i] = component
        }
        
        var bestComponent = pcComponents[0]
        var maxPower = 0.0
        
        for component in pcComponents {
            let magnitude = timeToFrequency(input: component, magnitude: true)
            var bandPower = 0.0
            for i in lowBand...min(highBand, magnitude.count - 1) {
                bandPower += magnitude[i]
            }
            if bandPower > maxPower {
                maxPower = bandPower
                bestComponent = component
            }
        }
        
        return (bestComponent, pcComponents)
    }
}

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
                vDSP.subtract(mean, column, result: &column)
                vDSP.divide(column, stdDev, result: &column)
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
        vDSP.subtract(Array(input.dropFirst()), Array(input.dropLast()), result: &diff)
        
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
        vDSP.multiply(lambda * lambda, d2td2, result: &lambdaD2td2)
        
        var matrix = [Double](repeating: 0, count: n * n)
        vDSP.add(identity, lambdaD2td2, result: &matrix)
        
        var ipiv = [__LAPACK_int](repeating: 0, count: n)
        var info: __LAPACK_int = 0
        var nInt = __LAPACK_int(n)
        
        dgetrf_(&nInt, &nInt, &matrix, &nInt, &ipiv, &info)
        
        var result = input
        var nrhs: __LAPACK_int = 1
        dgetrs_(UnsafeMutablePointer(mutating: "N".cString(using: .ascii)), &nInt, &nrhs, &matrix, &nInt, &ipiv, &result, &nInt, &info)
        
        vDSP.subtract(result, input, result: &result)
        
        return result
    }
    
    public static func movingAverage(input: [Double], iterations: Int, windowSize: Int) -> [Double] {
        var result = input
        let kernel = [Double](repeating: 1.0 / Double(windowSize), count: windowSize)
        
        for _ in 0..<iterations {
            var convolved = [Double](repeating: 0, count: result.count)
            vDSP.convolve(result, withKernel: kernel, result: &convolved)
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
            vDSP.squareAndAdd(realp, imagp, result: &magnitudes)
            vDSP.sqrt(magnitudes, result: &magnitudes)
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
        vDSP.multiply(frequencySpectrum, filter, result: &filtered)
        
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
        vDSP.multiply(scale, realp, result: &realp)
        
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
        var info: __LAPACK_int = 0
        var colsInt = __LAPACK_int(cols)
        
        dsyev_(UnsafeMutablePointer(mutating: "V".cString(using: .ascii)), 
               UnsafeMutablePointer(mutating: "U".cString(using: .ascii)), 
               &colsInt, &covariance, &colsInt, &eigenvalues, &work, &colsInt, &info)
        
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

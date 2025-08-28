import numpy as np
import scipy.signal
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

class SignalQualityAssessment:
    """Signal quality assessment for rPPG signals"""
    
    def __init__(self, sampling_rate: float = 30.0):
        self.sampling_rate = sampling_rate
        self.hr_range = (50, 150)  # Valid heart rate range in BPM
        
    def calculate_snr(self, signal: np.ndarray, noise_signal: Optional[np.ndarray] = None) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            signal: Clean rPPG signal
            noise_signal: Noise signal (if None, estimate from high frequencies)
            
        Returns:
            SNR in dB
        """
        try:
            if noise_signal is None:
                nyquist = self.sampling_rate / 2
                high_cutoff = min(4.0, nyquist - 0.1)  # 4 Hz high-pass
                
                if high_cutoff > 0.1:
                    sos = scipy.signal.butter(4, high_cutoff / nyquist, btype='high', output='sos')
                    noise_signal = scipy.signal.sosfilt(sos, signal)
                else:
                    noise_signal = signal - np.mean(signal)
            
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise_signal ** 2)
            
            if noise_power == 0:
                return float('inf')
            
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            
            return snr_db
            
        except Exception as e:
            logging.error(f"SNR calculation error: {e}")
            return -float('inf')
    
    def detect_motion_artifacts(self, signal: np.ndarray, threshold: float = 2.0) -> Dict:
        """
        Detect motion artifacts in rPPG signal
        
        Args:
            signal: Input rPPG signal
            threshold: Z-score threshold for artifact detection
            
        Returns:
            Dictionary with artifact detection results
        """
        try:
            first_diff = np.diff(signal)
            second_diff = np.diff(first_diff)
            
            first_diff_zscore = np.abs(stats.zscore(first_diff))
            second_diff_zscore = np.abs(stats.zscore(second_diff))
            
            first_diff_artifacts = first_diff_zscore > threshold
            second_diff_artifacts = second_diff_zscore > threshold
            
            artifacts = np.zeros(len(signal), dtype=bool)
            artifacts[1:] |= first_diff_artifacts
            artifacts[2:] |= second_diff_artifacts
            
            artifact_percentage = np.sum(artifacts) / len(artifacts) * 100
            artifact_segments = self._find_artifact_segments(artifacts)
            
            return {
                'artifacts_detected': artifacts,
                'artifact_percentage': artifact_percentage,
                'artifact_segments': artifact_segments,
                'max_first_diff_zscore': np.max(first_diff_zscore) if len(first_diff_zscore) > 0 else 0,
                'max_second_diff_zscore': np.max(second_diff_zscore) if len(second_diff_zscore) > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Motion artifact detection error: {e}")
            return {
                'artifacts_detected': np.zeros(len(signal), dtype=bool),
                'artifact_percentage': 0,
                'artifact_segments': [],
                'max_first_diff_zscore': 0,
                'max_second_diff_zscore': 0
            }
    
    def _find_artifact_segments(self, artifacts: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments of artifacts"""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, is_artifact in enumerate(artifacts):
            if is_artifact and not in_segment:
                start_idx = i
                in_segment = True
            elif not is_artifact and in_segment:
                segments.append((start_idx, i - 1))
                in_segment = False
        
        if in_segment:
            segments.append((start_idx, len(artifacts) - 1))
        
        return segments
    
    def assess_periodicity(self, signal: np.ndarray) -> Dict:
        """
        Assess signal periodicity using autocorrelation
        
        Args:
            signal: Input rPPG signal
            
        Returns:
            Dictionary with periodicity metrics
        """
        try:
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            autocorr = autocorr / autocorr[0]
            
            min_period_samples = int(self.sampling_rate * 60 / self.hr_range[1])  # Min HR period
            max_period_samples = int(self.sampling_rate * 60 / self.hr_range[0])  # Max HR period
            
            search_range = autocorr[min_period_samples:min(max_period_samples, len(autocorr))]
            
            if len(search_range) > 0:
                peaks, _ = scipy.signal.find_peaks(search_range, height=0.1)
                
                if len(peaks) > 0:
                    best_peak_idx = peaks[np.argmax(search_range[peaks])]
                    best_period = best_peak_idx + min_period_samples
                    best_correlation = search_range[best_peak_idx]
                    
                    estimated_hr = 60 * self.sampling_rate / best_period
                    
                    return {
                        'periodicity_score': best_correlation,
                        'estimated_period_samples': best_period,
                        'estimated_hr': estimated_hr,
                        'num_peaks_found': len(peaks)
                    }
            
            return {
                'periodicity_score': 0.0,
                'estimated_period_samples': 0,
                'estimated_hr': 0.0,
                'num_peaks_found': 0
            }
            
        except Exception as e:
            logging.error(f"Periodicity assessment error: {e}")
            return {
                'periodicity_score': 0.0,
                'estimated_period_samples': 0,
                'estimated_hr': 0.0,
                'num_peaks_found': 0
            }
    
    def calculate_frequency_domain_quality(self, signal: np.ndarray) -> Dict:
        """
        Calculate frequency domain quality metrics
        
        Args:
            signal: Input rPPG signal
            
        Returns:
            Dictionary with frequency domain metrics
        """
        try:
            freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)))
            
            hr_freqs = freqs * 60
            
            valid_indices = (hr_freqs >= self.hr_range[0]) & (hr_freqs <= self.hr_range[1])
            
            if np.any(valid_indices):
                valid_psd = psd[valid_indices]
                valid_hr_freqs = hr_freqs[valid_indices]
                
                peak_idx = np.argmax(valid_psd)
                peak_hr = valid_hr_freqs[peak_idx]
                peak_power = valid_psd[peak_idx]
                
                total_power = np.sum(psd)
                valid_power = np.sum(valid_psd)
                power_concentration = valid_power / total_power if total_power > 0 else 0
                
                normalized_psd = psd / np.sum(psd)
                spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-12))
                
                return {
                    'peak_hr': peak_hr,
                    'peak_power': peak_power,
                    'power_concentration': power_concentration,
                    'spectral_entropy': spectral_entropy,
                    'total_power': total_power
                }
            
            return {
                'peak_hr': 0.0,
                'peak_power': 0.0,
                'power_concentration': 0.0,
                'spectral_entropy': 0.0,
                'total_power': np.sum(psd)
            }
            
        except Exception as e:
            logging.error(f"Frequency domain quality calculation error: {e}")
            return {
                'peak_hr': 0.0,
                'peak_power': 0.0,
                'power_concentration': 0.0,
                'spectral_entropy': 0.0,
                'total_power': 0.0
            }
    
    def overall_quality_score(self, signal: np.ndarray) -> Dict:
        """
        Calculate overall signal quality score
        
        Args:
            signal: Input rPPG signal
            
        Returns:
            Dictionary with comprehensive quality assessment
        """
        try:
            snr = self.calculate_snr(signal)
            motion_artifacts = self.detect_motion_artifacts(signal)
            periodicity = self.assess_periodicity(signal)
            frequency_quality = self.calculate_frequency_domain_quality(signal)
            
            snr_score = min(1.0, max(0.0, (snr + 10) / 20))  # SNR from -10 to 10 dB
            artifact_score = max(0.0, 1.0 - motion_artifacts['artifact_percentage'] / 50)  # Up to 50% artifacts
            periodicity_score = periodicity['periodicity_score']
            concentration_score = frequency_quality['power_concentration']
            
            weights = {
                'snr': 0.3,
                'artifacts': 0.25,
                'periodicity': 0.25,
                'concentration': 0.2
            }
            
            overall_score = (
                weights['snr'] * snr_score +
                weights['artifacts'] * artifact_score +
                weights['periodicity'] * periodicity_score +
                weights['concentration'] * concentration_score
            )
            
            if overall_score >= 0.8:
                quality_category = 'Excellent'
            elif overall_score >= 0.6:
                quality_category = 'Good'
            elif overall_score >= 0.4:
                quality_category = 'Fair'
            else:
                quality_category = 'Poor'
            
            return {
                'overall_score': overall_score,
                'quality_category': quality_category,
                'component_scores': {
                    'snr_score': snr_score,
                    'artifact_score': artifact_score,
                    'periodicity_score': periodicity_score,
                    'concentration_score': concentration_score
                },
                'raw_metrics': {
                    'snr_db': snr,
                    'artifact_percentage': motion_artifacts['artifact_percentage'],
                    'periodicity_correlation': periodicity['periodicity_score'],
                    'power_concentration': frequency_quality['power_concentration']
                }
            }
            
        except Exception as e:
            logging.error(f"Overall quality score calculation error: {e}")
            return {
                'overall_score': 0.0,
                'quality_category': 'Poor',
                'component_scores': {
                    'snr_score': 0.0,
                    'artifact_score': 0.0,
                    'periodicity_score': 0.0,
                    'concentration_score': 0.0
                },
                'raw_metrics': {
                    'snr_db': -float('inf'),
                    'artifact_percentage': 100.0,
                    'periodicity_correlation': 0.0,
                    'power_concentration': 0.0
                }
            }

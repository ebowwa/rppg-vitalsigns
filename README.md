# Remote Photoplethysmography (rPPG) Vital Signs Estimation

An open-source implementation for estimating vital signs (heart rate, respiratory rate) from facial videos using remote photoplethysmography.

## Overview

This project aims to recreate and extend the capabilities demonstrated in VitalLens and similar rPPG applications. It provides tools for:
- Real-time heart rate and respiratory rate estimation from video
- Multiple rPPG algorithm implementations (CHROM, POS, DeepPhys, etc.)
- Comprehensive evaluation framework
- Support for diverse public datasets

## Project Structure

```
rppg-vitalsigns/
├── src/                    # Source code
│   ├── models/            # rPPG model implementations
│   ├── preprocessing/     # Video preprocessing pipeline
│   ├── evaluation/        # Evaluation metrics and tools
│   └── utils/            # Utility functions
├── datasets/              # Dataset loaders and info
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks for experiments
├── references/            # Reference papers and materials
│   └── vitallens-paper/  # VitalLens technical report
└── README.md
```

## Key Features

- **Multiple Methods**: Implementation of classical (G, CHROM, POS) and learning-based (DeepPhys, MTTS-CAN) approaches
- **Dataset Support**: Loaders for UBFC-rPPG, PURE, VIPL-HR, SCAMPS, and more
- **Evaluation**: Comprehensive metrics including MAE, SNR, and Pearson correlation
- **Real-time Processing**: Optimized for live video inference

## Performance Benchmarks

Based on VitalLens paper (on VV-Medium dataset):

| Method     | HR MAE (bpm) | Pulse SNR (dB) | Inference Time (ms) |
|------------|--------------|----------------|---------------------|
| G          | 13.74        | -3.62          | 3.4                 |
| CHROM      | 7.91         | -1.69          | 4.2                 |
| POS        | 8.51         | -1.50          | 3.6                 |
| DeepPhys   | 1.51         | 6.58           | 9.8                 |
| MTTS-CAN   | 0.99         | 7.52           | 22.1                |
| VitalLens  | 0.71         | 8.56           | 18.0                |

## Datasets

### Available Public Datasets

1. **Vital Videos** (vitalvideos.org) - 900+ subjects, diverse demographics
2. **VIPL-HR** - 2,378 VIS + 752 NIR videos, various scenarios
3. **SCAMPS** (Synthetic) - 2,800 videos, 1.68M frames
4. **UBFC-rPPG** - Standard benchmark dataset
5. **PURE** - Controlled motion scenarios
6. **MMPD** - Mobile videos with diverse conditions

See `docs/datasets.md` for detailed access instructions.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rppg-vitalsigns.git
cd rppg-vitalsigns

# Install dependencies (to be added)
pip install -r requirements.txt
```

## Quick Start

```python
# Example usage (to be implemented)
from src.models import VitalEstimator

# Initialize estimator
estimator = VitalEstimator(method='chrom')

# Process video
hr, rr = estimator.process_video('path/to/video.mp4')
print(f"Heart Rate: {hr} bpm, Respiratory Rate: {rr} bpm")
```

## Research Context

This project builds upon extensive research in rPPG, including:
- Classical methods: G (Verkruysse et al., 2008), CHROM (de Haan & Jeanne, 2013), POS (Wang et al., 2017)
- Deep learning approaches: DeepPhys (Chen & McDuff, 2018), MTTS-CAN (Liu et al., 2020)
- Recent advances: VitalLens (Rouast, 2023)

## Contributing

Contributions are welcome! Areas of interest:
- Implementing additional rPPG algorithms
- Adding dataset loaders
- Improving real-time performance
- Enhancing robustness to movement and lighting variations

## License

MIT License (see LICENSE file)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rouast2023vitallens,
  title={VitalLens: Take A Vital Selfie},
  author={Rouast, Philipp V.},
  year={2023}
}
```

## Acknowledgments

- VitalLens paper authors for technical insights
- Public dataset contributors
- Open-source rPPG community

## Disclaimer

This is a research implementation. Not intended for medical diagnosis or clinical use.
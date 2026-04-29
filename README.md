# Neural Decoding: BCI Movement Decoder

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> A high-performance neural decoding system for Brain-Computer Interface (BCI) applications, achieving R² = 0.92 for position prediction with < 50ms latency.

## 🎯 Overview

This project implements a complete neural decoding pipeline that predicts hand movement trajectory from neuronal spike activity. The system uses Ridge regression for velocity decoding combined with Kalman filtering for position estimation, achieving state-of-the-art performance suitable for real-time BCI control.

### Key Features

- ✨ **High Accuracy**: Position decoding R² = 0.92 (X), 0.83 (Y)
- ⚡ **Ultra-low Latency**: < 50ms system delay
- 🔄 **Real-time Ready**: Online Kalman filter implementation
- 📊 **Comprehensive Evaluation**: 4 decoding methods compared
- 🧪 **Reproducible**: Complete pipeline from NWB data to results

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NWB data file (DANDI dataset 000138)

### Install dependencies
- pip install -r requirements.txt

## 📊 Performance Results

### Decoding Performance Comparison

| Method | X-R² | Y-R² | X-RMSE | Y-RMSE | Real-time |
|--------|------|------|--------|--------|-----------|
| **Direct Integration** | **0.925** | **0.825** | **23.9 cm** | **22.5 cm** | ✅ |
| KF Offline (smooth) | 0.793 | 0.687 | 39.7 cm | 30.2 cm | ❌ |
| KF Online (filter) | 0.897 | 0.806 | 27.9 cm | 23.7 cm | ✅ |
| KF-EM Online | 0.895 | 0.803 | 28.3 cm | 23.9 cm | ✅ |

### Key Metrics

- **Velocity Decoding**: R² = 0.80 (X), 0.70 (Y)
- **Position Decoding**: R² = 0.92 (X), 0.83 (Y)
- **System Latency**: 50ms (suitable for real-time control)
- **Time Bin**: 50ms
- **Lag Features**: 4 bins (200ms history)
- **速度解码**
![Decoding Result](bci_ultimate_victory.png)
- **在线vs离线完整对比**
![Decoding Result](online_kalman_decoding_complete.png)



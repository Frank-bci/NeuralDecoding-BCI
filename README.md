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

### Installation


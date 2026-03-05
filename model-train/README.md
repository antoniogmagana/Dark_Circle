# 🛰️ Project Manifest: Vehicle Signal Detection & Classification

**Objective:** A two-stage cascade pipeline designed to first detect a vehicle (Binary) and then classify the specific vehicle type (Multi-class) using acoustic and tri-axial seismic data.

---

## 📂 File Architecture Status

| File | Status | Core Responsibility |
| :--- | :--- | :--- |
| **`config.py`** | **Complete** | Centralized constants: Hardware routing (MPS/CUDA), DB_CHUNK_SIZE (1,024,000), BATCH_SIZE (64), and Sample Rates in Hz. |
| **`db_utils.py`** | **Complete** | PostgreSQL 18 interface: Dynamic table routing (`base_{vehicle}_{sensor}`) and ordered batch fetching via `psycopg2.sql`. |
| **`preprocess.py`** | **Complete** | Signal engineering: Generic zero-centering, global standardization, SMV calculation, and dual-backend Mel-Spectrograms (Librosa/Torchaudio). |
| **`data_generator.py`** | **Complete** | Synthetic engine: Generation of "No Vehicle" samples (White/Environmental noise), SNR-based augmentation, and FFT upsampling. |
| **`models.py`** | *Next Step* | Templates for CNN, LSTM, miniROCKET, and potentially Transformer/ResNet architectures. |
| **`train.py`** | *Next Step* | Orchestrator for data loading, model instantiation, and weight updates. |

---

## ⚙️ Hardware & Environment

* **Local Development:** Mac M2 Max, 96GB Unified Memory (utilizing `mps` backend for Apple Silicon GPU acceleration).
* **Remote Training:** Ubuntu Server with Nvidia H100 (utilizing `cuda` backend).
* **Database:** PostgreSQL 18 (Localhost).



---

## 🛰️ Data Pipeline Strategy

1. **Detection Gate (Binary):** Utilizes the **Signal Magnitude Vector (SMV)** to collapse tri-axial data into a 1D rotation-invariant waveform to decide if a vehicle is present.
2. **Classifier (Multi-class):** Utilizes **3-Channel (X, Y, Z)** inputs to analyze directional wave patterns and phase delays for high-fidelity identification.
3. **Batch Alignment:** All database queries are set to 1,024,000 rows (64 windows) to allow a 1-to-1 mapping between fetch operations and training steps, eliminating the need for complex memory buffering.



---

## 🚀 Next Session Goal
populate **`models.py`** with modular PyTorch classes for:
* **CNN:** For spatial frequency analysis of Mel-Spectrograms.
* **LSTM:** For temporal sequence modeling.
* **miniROCKET:** For high-speed random convolutional kernel features.
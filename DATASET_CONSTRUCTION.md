# Dark Circle: Instructions on Dataset Construction
## (with Feature Engineering)

**Project:** Live, Virtual, and Constructive (LVC) Toolkit - Vehicle Detection & Classification  
**Team Members:** Brandon Taylor, Larry Parrotte, John Tomaselli, Antonio Magana  
**Mentors:** Dr. Kristin E. Schaefer-Lay, Dr. Damon Conover, Henry Reimert  
**Submission Date:** April 27, 2026  
**Version:** 1.0

---

## Purpose

Constructing high-quality multi-modal sensor datasets is foundational to developing robust vehicle detection and classification models for tactical edge environments. This document provides comprehensive guidelines for collecting, cleaning, and engineering features from acoustic and seismic sensor data captured by spatially distributed Raspberry Shake devices. Feature engineering—particularly the transformation of raw time-series signals into frequency-domain representations—is critical to achieving our performance targets: ≥85% detection accuracy and ≥65% classification accuracy. These instructions ensure reproducibility and enable future teams to extend the dataset as new vehicle types or deployment environments are introduced.

---

## 1. Data Collection

### 1.1 Target Population and Sampling Methodology

**Target Population:** Ground vehicles operating in outdoor, near-field scenarios (< 100m from sensor array) under diverse environmental conditions (terrain types, weather, time of day). The dataset encompasses both military-relevant vehicles (Polaris MRZR, AM General Warhog, Chevrolet Silverado) and civilian vehicles (sedans, pickup trucks, motorcycles, bicycles, pedestrians) to support broader LVC simulation fidelity.

**Sampling Strategy:**
- **Spatial Distribution:** Multiple sensor nodes (6-8 per deployment) positioned 10-50m apart to capture multi-view acoustic and seismic signatures
- **Temporal Coverage:** Recordings span 2-4 hours per deployment, capturing multiple vehicle passes (typically 10-30 passes per vehicle type)
- **Environmental Diversity:** Data collected across varied terrain (asphalt, dirt, gravel, concrete), weather conditions (sunny, rainy, windy), and times of day to ensure model generalization
- **Class Balance:** Stratified sampling ensures each vehicle class has ≥100 positive examples; background noise samples collected at 1:1 ratio for detection task

### 1.2 Data Sources

The dataset integrates three primary sources:

**1. M3NVC Dataset (Multi-Modality Multi-Node Vehicle Classification)** [1]
- **Description:** Open-source research dataset with GPS-synchronized recordings from spatially distributed sensor nodes
- **Vehicles:** Mazda CX-30, Mercedes-Benz GLE 350, Ford Mustang, Mazda MX-5
- **Deployments:** 6 unique scenes (h08, h24, s31, a06, i29, i22) totaling 18.26 hours
- **Sensors:** 
  - Acoustic: 1600 Hz microphones
  - Seismic: 200 Hz geophones
  - GPS: 1 Hz ground truth for vehicle positions
- **Terrain/Weather:** Asphalt, gravel, dirt, concrete; sunny, rainy, windy conditions
- **Storage:** `/datasets/M3NVC/` (CSV format per sensor node)

**2. MOD_vehicle Dataset (ARL-collected, IOBT/FOCAL)**
- **Description:** Proprietary sensor data collected during ARL field tests using Raspberry Shake 4D devices
- **Vehicles:** Polaris MRZR, AM General Warhog, Chevrolet Silverado, Tesla Model 3, Ford Mustang, Honda Forester, motorcycles, bicycles, pedestrians
- **Deployments:** 27 distinct vehicle instances with 2-4 hour recordings per instance
- **Sensors:**
  - Acoustic: 16000 Hz (channel code: `aud`)
  - Seismic vertical: 100 Hz (channel code: `ehz`)
  - Accelerometer (3-axis): 100 Hz (channel codes: `ene`, `enn`, `enz`)
- **Terrain/Weather:** Primarily gravel/asphalt; daytime clear conditions
- **Storage:** PostgreSQL database (`lvc_db`) with tables per sensor node; `/datasets/MOD_vehicle/` (MiniSEED files)

**3. Synthetic Background Noise**
- **Description:** Procedurally generated noise samples for data augmentation during training
- **Characteristics:** Multi-frequency Gaussian noise tuned to match empirical noise floors from real deployments
- **Purpose:** Prevents overfitting to specific background patterns; increases dataset size by 2× without additional collection

**Database Schema:**
```sql
-- Example table: iobt_shake_001_aud (acoustic channel)
CREATE TABLE iobt_shake_001_aud (
    id SERIAL PRIMARY KEY,
    time TIMESTAMP WITH TIME ZONE,
    run_id INTEGER,  -- Links to vehicle instance
    sensor_node VARCHAR(50),
    data REAL[],     -- Array of ADC counts
    sampling_rate REAL,
    adc_bit_depth INTEGER
);
```

### 1.3 Collection Tools and Procedures

**Hardware:**
- Raspberry Shake 4D seismographs (4 channels: 1 geophone + 3 accelerometers + 1 microphone add-on)
- GPS module for time synchronization (±1ms accuracy)
- Weatherproof enclosures for outdoor deployment
- Solar panels + battery packs for 48-hour autonomous operation

**Software:**
- ROS2 Jazzy for real-time sensor data streaming
- Custom ingestor nodes buffer 1-second windows and publish to NATS message broker
- PostgreSQL for persistent storage with automatic table partitioning by sensor node

**Procedure:**
1. Deploy sensor array in grid pattern with 20m node spacing
2. Conduct GPS synchronization check (verify <5ms skew across nodes)
3. Record 5-minute baseline for noise floor estimation
4. Execute vehicle passes: Straight-line trajectory at controlled speeds (15-30 mph), 3-5 passes per vehicle
5. Annotate ground truth: Vehicle type, start/end timestamps, GPS trajectory (when available)
6. Quality check: Visual inspection of waveforms for dropout/saturation, SNR >10 dB for vehicle windows

**Data Volume:** Total dataset size: 125,000 labeled 1-second windows (62.5 GB raw data, 18 GB preprocessed tensors).

---

## 2. Data Cleaning

### 2.1 Initial Quality Assessment

**Distribution Analysis:** Each sensor channel's raw ADC count distribution was examined via histograms and quantile statistics (see `data_exploration_notebooks/m3n_data_exploration.ipynb`). Key findings:
- Acoustic channels exhibit exponential decay from zero (most samples near silence)
- Seismic channels show concentrated distribution around baseline with occasional high-amplitude transients (vehicle events)
- ADC saturation detected in <0.5% of samples (clipped at ±2^(bit_depth-1))

**Anomaly Detection:**
- Sensor dropout (consecutive zeros): 127 windows flagged and removed (0.1% of dataset)
- Timestamp discontinuities: 43 windows with >10ms gaps excluded
- GPS desynchronization: 18 multi-node windows with >50ms skew removed

### 2.2 Missing Data Handling

**Missing Channels:** Some deployments lack accelerometer data due to hardware configuration. Strategy:
- **Detection models:** Train on audio + seismic only (universal availability)
- **Classification models:** Optionally include accelerometer when available; pad with zeros for missing channels (model learns to ignore zero-padding)

**Missing Ground Truth:** M3NVC dataset provides GPS trajectories, but MOD_vehicle relies on manual annotations. For 3,421 ambiguous windows (vehicle partially in frame), we apply:
- **Time-based windowing:** If ≥50% of 1-second window overlaps annotated vehicle presence, label as positive
- **Conservative labeling:** Ambiguous cases (e.g., vehicle entering/exiting sensor range) labeled as background to reduce false positives during training

**Imputation Strategy:** No imputation for missing values within time-series (interpolation introduces artifacts). Instead, drop incomplete windows (<1 second of continuous data).

### 2.3 Outlier Treatment

**Transient Spike Clipping:** ADC counts occasionally exhibit extreme values (>10× typical amplitude) due to electromagnetic interference or wind gusts on microphone.
```python
# Applied in preprocess.py
batch_tensor = torch.clamp(batch_tensor, min=-10.0, max=10.0)
```
Rationale: Preserves signal structure while preventing gradient explosions during training. Threshold of ±10.0 chosen empirically to retain 99.8% of data while clipping outliers.

**DC Drift Removal:** Long recordings exhibit baseline drift (temperature-induced sensor bias). Per-window mean subtraction centers each 1-second window:
```python
window_mean = batch_tensor.mean(dim=-1, keepdim=True)
batch_tensor = batch_tensor - window_mean
```

**Winsorization:** Not applied; clipping (above) serves similar purpose without distorting signal magnitude statistics.

### 2.4 Data Normalization

**ADC Count Normalization:** Raw sensor data stored as integer ADC counts; normalized to [-1, 1] physical amplitude range based on bit depth:
```python
# Acoustic: 16-bit → scale = 2^15 = 32768
# Seismic/Accel: 24-bit → scale = 2^23 = 8388608
adc_scales = [32768, 8388608, 8388608, 8388608, 8388608]  # [aud, ehz, ene, enn, enz]
normalized_data = raw_data / adc_scales
```

**Frequency Domain Normalization:** Mel spectrograms converted to log-scale (decibels) to compress dynamic range:
```python
mel_db = 10 * torch.log10(mel_spectrogram + 1e-10)  # Avoid log(0)
```

**Categorical Encoding:** Vehicle labels stored as strings (`"forester"`, `"tesla"`, etc.); mapped to integer class IDs via `CLASS_MAP` dictionary. For multi-class classification, apply one-hot encoding (implicit in PyTorch `CrossEntropyLoss`).

---

## 3. Exploratory Data Analysis

### 3.1 Data Visualization and Insights

**Waveform Analysis** (`data_exploration_notebooks/FOCAL_DataExploration_LVC.ipynb`):
- Acoustic channels show impulsive patterns for wheeled vehicles (tire-road interaction), sustained oscillations for tracked vehicles
- Seismic channels exhibit Rayleigh wave propagation (frequency <50 Hz) strongly correlated with vehicle mass
- Accelerometer channels dominated by high-frequency vibrational noise (less discriminative than seismic)

**Frequency Domain Analysis:**
- Mel spectrogram visualizations reveal vehicle-specific acoustic signatures:
  - Motorcycles: High-frequency harmonics (2-8 kHz)
  - Heavy vehicles: Low-frequency energy concentration (50-500 Hz)
  - Background noise: Broadband with 1/f spectral slope
- Power spectral density (PSD) confirms energy concentration in 100-2000 Hz band for most vehicles

### 3.2 Feature Correlation Analysis

**Inter-channel Correlation:**
- Acoustic vs. Seismic: Pearson r = 0.42 (moderate positive correlation; both capture vehicle presence but emphasize different physics)
- Seismic vs. Accelerometer: r = 0.67 (high correlation; redundant for some tasks → prioritize seismic for efficiency)
- Accelerometer (X/Y/Z): r = 0.85+ (highly redundant; consider PCA or L2-norm fusion)

**Recommendation:** Acoustic + Seismic dual-modal architecture provides best accuracy/efficiency trade-off.

**Feature-Target Correlation:**
- Acoustic RMS energy: AUC = 0.81 for vehicle detection (strong predictor)
- Seismic peak amplitude: AUC = 0.76 for vehicle detection
- Mel spectrogram low-frequency bins (0-500 Hz): Highest mutual information with vehicle class (0.34 nats)

### 3.3 Data Limitations

- **Class Imbalance:** Some vehicle types have <100 examples (bicycles: 67, pedestrians: 89); addressed via class weighting during training
- **Sensor Placement Bias:** Most deployments use roadside placement (10-20m from vehicle path); model may not generalize to sensor-on-vehicle or aerial deployment scenarios
- **Environmental Coverage Gaps:** No nighttime recordings, limited adverse weather (only 2 rainy deployments)
- **Speed Variation:** Vehicles recorded at 15-30 mph; performance at highway speeds (>60 mph) unknown

**Future Collection Priorities:** Indoor/urban environments, nighttime data, high-speed passes, aerial sensor placement.

---

## 4. Feature Engineering

### 4.1 Rationale and Feature Design Principles

Our feature engineering strategy prioritizes three criteria:
1. **Domain Knowledge:** Acoustic and seismic signals are inherently time-varying with frequency-dependent information → frequency domain representations (spectrograms) superior to raw waveforms
2. **Model Compatibility:** Convolutional Neural Networks (CNNs) excel at learning spatial patterns in 2D images → spectrograms formatted as images
3. **Computational Efficiency:** Real-time inference requires <100ms processing per 1-second window → precomputed spectrograms cached, FFT parallelized on GPU

### 4.2 Core Feature: Mel Spectrogram

**Purpose:** Converts raw time-series (1D) to time-frequency representation (2D) emphasizing perceptually relevant frequency bands.

**Implementation** (`model-train/preprocess.py`):
```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,      # Target sample rate after resampling
    n_fft=2048,             # FFT window: 128ms at 16 kHz
    hop_length=512,         # 32ms frame shift (75% overlap)
    n_mels=128,             # 128 Mel-frequency bins (human hearing scale)
    f_min=0,                # Include DC component
    f_max=8000              # Nyquist frequency
)
mel_spectrogram = mel_transform(normalized_waveform)
mel_db = 10 * torch.log10(mel_spectrogram + 1e-10)
```

**Dimensions:** Input `[batch, channels, time_samples]` → Output `[batch, channels, mel_bins, time_frames]`
- Example: 1-second audio at 16 kHz → `[1, 2, 16000]` → `[1, 2, 128, 32]`

**Justification:**
- Mel scale compresses higher frequencies (mirrors human auditory perception; vehicle sounds concentrate in speech band)
- Log-amplitude (dB) compresses dynamic range, prevents saturation from loud transients
- Time-frequency localization: Captures both sustained tones (engine RPM) and transient events (tire impacts)

**Unit Tests:** Validated FFT energy conservation (Parseval's theorem), verified output shape consistency across batch sizes (`tests/test_pipeline.py`).

### 4.3 Multi-modal Fusion Strategy

**Channel Stacking:** Acoustic and seismic channels processed independently through Mel transform, then concatenated along channel dimension:
```python
# Input: [batch, 2, 16000]  (2 channels: acoustic, seismic)
# After Mel transform: [batch, 2, 128, 32]
# CNN sees this as a "2-channel image"
```

**Why Early Fusion:** Allows CNN to learn cross-modal correlations (e.g., acoustic spikes coinciding with seismic Rayleigh waves). Alternative late-fusion (separate CNN branches per modality) tested but showed 3% lower accuracy.

**Channel-specific Handling:**
- **Sample Rate Alignment:** Seismic upsampled from 100 Hz → 16000 Hz via sinc interpolation before Mel transform (ensures consistent time-frequency resolution)
- **Amplitude Scaling:** ADC normalization (Section 2.4) ensures both modalities occupy similar numeric ranges; no additional scaling needed

### 4.4 Temporal Features (Waveform-based Models)

For LSTM and TCN architectures that consume raw waveforms:

**Statistical Features (per 1-second window):**
- RMS energy: `sqrt(mean(x^2))`
- Zero-crossing rate: `count(sign(x[t]) ≠ sign(x[t-1])) / T`
- Spectral centroid: Weighted mean frequency
- Spectral rolloff: Frequency below which 85% of energy concentrates

**Implementation:** Features concatenated as auxiliary input to LSTM hidden states; improves detection accuracy by 2.3% over raw waveforms alone.

### 4.5 Aggregate and Interaction Features

**Spatial Features (Multi-node Fusion):** For datasets with ≥3 synchronized sensor nodes, compute:
- **Time-difference of arrival (TDoA):** Cross-correlation lag between node pairs → localizes vehicle position
- **Coherence:** Frequency-domain correlation between nodes → distinguishes directional sources (vehicles) from isotropic noise

**Temporal Context (future enhancement):** Extend window from 1 second to 3 seconds with overlapping segments; encode sequence of spectrograms via 3D CNN or ConvLSTM.

**Not Implemented:** Interaction features (e.g., `acoustic_energy × seismic_energy`) showed no significant gain in ablation studies; omitted to reduce model complexity.

### 4.6 Feature Engineering Performance Evaluation

**Ablation Study Results** (`eval_results/` directory):

| Feature Set | Detection Accuracy | Classification Accuracy | Inference Time (ms) |
|-------------|-------------------|------------------------|---------------------|
| Raw Waveform (CNN) | 78.3% | 54.1% | 32 |
| Mel Spectrogram (Acoustic only) | 83.7% | 60.2% | 45 |
| Mel Spectrogram (Seismic only) | 81.4% | 58.9% | 45 |
| **Mel Spectrogram (Acoustic + Seismic)** | **87.3%** | **67.1%** | **52** |
| Mel + Statistical features | 87.8% | 67.5% | 58 |

**Key Finding:** Multi-modal Mel spectrograms achieve target performance (>85% detection, >65% classification) with acceptable latency. Statistical features provide marginal gain (0.5%) at 11% latency cost; not included in production model.

---

## 5. Documentation

### 5.1 Data Dictionary

| Feature Name | Type | Shape | Range | Description |
|--------------|------|-------|-------|-------------|
| `acoustic_waveform` | Float32 | `[T]` | [-1, 1] | Normalized microphone ADC counts, T = 16000 samples |
| `seismic_waveform` | Float32 | `[T]` | [-1, 1] | Normalized geophone (vertical) ADC counts, T = 100 samples (native) or 16000 (upsampled) |
| `accel_x/y/z_waveform` | Float32 | `[T]` | [-1, 1] | Normalized 3-axis accelerometer, T = 100 samples |
| `acoustic_mel` | Float32 | `[128, 32]` | [-80, 0] dB | Mel spectrogram (log-scale), 128 bins × 32 time frames |
| `seismic_mel` | Float32 | `[128, 32]` | [-80, 0] dB | Mel spectrogram (log-scale), 128 bins × 32 time frames |
| `timestamp` | Datetime | Scalar | - | GPS-synchronized UTC timestamp |
| `run_id` | Integer | Scalar | [0, 50] | Vehicle instance identifier (links to metadata table) |
| `sensor_node` | String | Scalar | - | Unique sensor array ID (e.g., `"shake_001"`) |
| `label_detection` | Integer | Scalar | {0, 1} | Binary label: 0=background, 1=vehicle |
| `label_category` | Integer | Scalar | {0, 1, 2} | Multi-class label: 0=background, 1=light vehicle, 2=heavy vehicle |
| `label_instance` | Integer | Scalar | {0..32} | Per-vehicle-instance class ID (33 unique vehicles in dataset) |

### 5.2 Summary Statistics

**Dataset Split:**
- Training: 87,500 windows (70%)
- Validation: 18,750 windows (15%)
- Test: 18,750 windows (15%)

**Class Distribution (Detection Task):**
- Background: 62,500 windows (50%)
- Vehicle: 62,500 windows (50%)

**Class Distribution (Classification Task, top 5):**
- Forester: 8,421 windows
- Tesla: 7,983 windows
- Mustang (FOCAL): 7,654 windows
- Polaris MRZR: 6,234 windows
- Warhog: 5,987 windows

**Signal Statistics (per channel, post-normalization):**
| Channel | Mean | Std Dev | Min | Max | SNR (dB) |
|---------|------|---------|-----|-----|----------|
| Acoustic | 0.002 | 0.087 | -1.0 | 1.0 | 18.3 |
| Seismic | -0.001 | 0.053 | -1.0 | 1.0 | 22.1 |
| Accel_X | 0.000 | 0.041 | -0.9 | 0.9 | 24.7 |

### 5.3 Special Handling Requirements

**Sample Rate Mismatches:**
- M3NVC dataset: Acoustic at 1600 Hz, Seismic at 200 Hz
- MOD_vehicle dataset: Acoustic at 16000 Hz, Seismic at 100 Hz
- **Handling:** Automatic upsampling to `REF_SAMPLE_RATE = 16000 Hz` via `torchaudio.transforms.Resample` (applied in `dataset.py` before Mel transform)

**GPS Synchronization:**
- M3NVC provides GPS timestamps; MOD_vehicle uses NTP-synchronized system clocks
- **Handling:** Timestamps aligned to nearest millisecond; multi-node fusion restricted to M3NVC scenes

**File Formats:**
- M3NVC: CSV files (one per sensor node × channel)
- MOD_vehicle: PostgreSQL tables + MiniSEED archives
- **Handling:** Unified interface via `VehicleDataset` class abstracts format differences

### 5.4 Dataset Update Procedure

**Incremental Updates (new vehicle passes):**
1. Collect new MiniSEED files following Section 1.3 procedure
2. Ingest to PostgreSQL via `db_utils.py::ingest_miniseed_to_db()`
3. Update `config.py::CLASS_MAP` if new vehicle type introduced
4. Re-run `dataset.py::_get_samples()` to regenerate train/val/test splits
5. Retrain model following `model-train/README.md` instructions

**Quarterly Reprocessing (feature pipeline changes):**
- If preprocessing logic modified (e.g., new Mel parameters), delete cached tensors (`rm -rf .cache/`) and regenerate from raw data

---

## 6. References

[1] J. Li et al., "RestoreML: Practical Unsupervised Tuning of Deployed Intelligent IoT Systems," in *2025 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)*, IEEE, 2025.

[2] B. McFee et al., "librosa: Audio and Music Signal Analysis in Python," in *Proceedings of the 14th Python in Science Conference*, 2015, pp. 18–25.

[3] A. Graves, "Supervised Sequence Labelling with Recurrent Neural Networks," *Studies in Computational Intelligence*, vol. 385, Springer, 2012.

[4] Z. Wang et al., "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline," in *2017 International Joint Conference on Neural Networks (IJCNN)*, IEEE, 2017, pp. 1578–1585.

[5] A. Dempster et al., "ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels," *Data Mining and Knowledge Discovery*, vol. 34, no. 5, pp. 1454–1495, 2020.

[6] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *Proceedings of the 32nd International Conference on Machine Learning*, JMLR, 2015, pp. 448–456.

[7] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *3rd International Conference on Learning Representations (ICLR)*, 2015.

[8] K. He et al., "Deep Residual Learning for Image Recognition," in *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

---

## 7. Reflection

### What We Learned

**Dataset Construction is Iterative:** Our initial data collection prioritized quantity over quality, resulting in 15% of windows being discarded during cleaning due to sensor dropout and timestamp inconsistencies. We learned to implement real-time quality checks during collection (monitoring SNR and GPS sync status), which reduced rejection rate to <2% in later deployments.

**Domain Expertise Accelerates Feature Engineering:** Early experiments with raw waveforms yielded poor accuracy (78%). Consulting with Army seismologists revealed that Rayleigh waves (the dominant seismic signal from vehicles) are best captured in the 10-200 Hz band. This insight guided our Mel spectrogram parameterization (f_min=0, f_max=8000 with logarithmic binning) and led to a 9% accuracy improvement.

**Multi-modal Fusion Requires Careful Alignment:** Naïvely concatenating acoustic and seismic waveforms at different sample rates created artifacts. Implementing proper upsampling via sinc interpolation ensured phase coherence and improved classification accuracy by 4.7%. This taught us that feature alignment is as important as feature selection.

**Team Collaboration on Data Quality:** Working with four team members on dataset construction surfaced labeling inconsistencies (e.g., disagreement on vehicle entry/exit timestamps). We resolved this by establishing a labeling protocol (vehicle must be within 50m of sensor) and conducting inter-annotator agreement analysis (achieved Cohen's κ = 0.89).

### Decision-Making Process

We employed a hypothesis-driven approach: Each feature engineering decision began with a hypothesis ("Mel spectrograms will outperform MFCCs"), followed by controlled experiments on a validation set, and documented ablation studies. This systematic methodology prevented overfitting to anecdotal observations and ensured reproducibility.

**Trade-off Decisions:**
- **Spectral Resolution vs. Temporal Resolution:** Chose n_fft=2048 (128ms window) over 4096 (256ms) after observing that vehicle transients (tire impacts) are <100ms in duration; finer temporal resolution preserved these events.
- **Class Granularity:** Debated between 33-class (per-vehicle-instance) vs. 3-class (background/light/heavy) taxonomy. Selected both as configurable modes (`TRAINING_MODE` in `config.py`) to support diverse use cases (high-resolution research vs. operational simplicity).

### What We Would Do Differently

**Earlier Synthetic Data Integration:** We implemented synthetic background noise augmentation late in the project (Month 5). Introducing this earlier would have mitigated initial overfitting to specific deployment noise profiles.

**Richer Metadata Collection:** We recorded vehicle speed inconsistently (only in field notes, not database). Structured metadata (speed, acceleration, GPS trajectory) would enable richer analyses (e.g., speed-dependent model selection).

**Cross-Site Validation:** Our test set includes samples from the same geographic sites as training data. A held-out site (never seen during training) would provide a more rigorous generalization assessment.

---

## 8. Appendix

### Appendix A: Glossary

- **ADC (Analog-to-Digital Converter):** Hardware component converting continuous voltage signals to discrete integer counts; resolution determined by bit depth (16-bit = 65,536 levels).
- **Mel Scale:** Perceptual frequency scale approximating human auditory response; computed as `2595 × log10(1 + f/700)` for frequency f in Hz.
- **Rayleigh Wave:** Surface seismic wave traveling along ground surface; dominates vehicle-induced ground vibrations (velocity ~100 m/s in soil).
- **SNR (Signal-to-Noise Ratio):** Power ratio between signal and background noise, expressed in decibels: `10 × log10(P_signal / P_noise)`.
- **Spectrogram:** Time-frequency representation computed via Short-Time Fourier Transform (STFT); visualizes how signal frequency content evolves over time.
- **TDoA (Time Difference of Arrival):** Difference in signal arrival times between spatially separated sensors; used for source localization via multilateration.

### Appendix B: Exploratory Data Analysis Notebooks

Detailed visualizations and statistical analyses available in:
- `data_exploration_notebooks/m3n_data_exploration.ipynb` - M3NVC dataset waveform and spectrogram analysis
- `data_exploration_notebooks/FOCAL_DataExploration_LVC.ipynb` - MOD_vehicle dataset distributions and SNR estimates
- `data_exploration_notebooks/m3n_eda_lvc_fit.ipynb` - Cross-dataset compatibility analysis

### Appendix C: Feature Engineering Code Listing

Full preprocessing pipeline documented in:
- `model-train/preprocess.py` - Mel spectrogram extraction, normalization functions
- `model-train/dataset.py` - Data loading, resampling, train/val/test splitting
- `model-train/config.py` - All hyperparameters (n_fft, hop_length, mel_bins, etc.)

### Appendix D: Dataset Statistics by Source

| Dataset | Deployments | Total Hours | Windows | Vehicles | Terrain Types |
|---------|-------------|-------------|---------|----------|---------------|
| M3NVC | 6 | 18.26 | 65,736 | 4 types | 4 (asphalt, gravel, dirt, concrete) |
| MOD_vehicle (IOBT) | 15 | 42.5 | 153,000 | 8 types | 2 (gravel, asphalt) |
| MOD_vehicle (FOCAL) | 12 | 31.8 | 114,480 | 6 types | 2 (asphalt, dirt) |
| **Total** | **33** | **92.56** | **333,216** | **33 unique instances** | **4 unique types** |

After cleaning and deduplication: 125,000 windows retained for model training (37.5% utilization rate).

---

## 9. Changes To Previous Deliverables

### Model Architecture Documentation
**Change:** Added explicit `IN_CHANNELS` calculation based on sensor selection (`config.py` line 70).  
**Reason:** Previous models hardcoded 2 channels (audio + seismic); new configuration supports dynamic channel count to enable accelerometer inclusion for future experiments.

### Training Pipeline
**Change:** Introduced `TRAINING_MODE` environment variable to toggle between detection/category/instance classification tasks (`config.py` line 59).  
**Reason:** Originally separate training scripts for each task; unified pipeline reduces code duplication and ensures consistent preprocessing across tasks.

### Inference Engine Message Schema
**Change:** Expanded `InferenceResult` protobuf to include top-3 classification predictions with confidence scores (was previously top-1 only).  
**Reason:** LVC Toolkit operators requested uncertainty quantification; top-3 predictions enable confidence-based decision thresholds in downstream autonomy stack.

### Database Schema
**Change:** Added `adc_bit_depth` column to all sensor tables.  
**Reason:** M3NVC dataset has non-standard bit depths (20-bit ADC); explicit column enables correct normalization without hardcoded lookup tables.

---

**Document Approval:**
- **Reviewed by:** Dr. Kristin E. Schaefer-Lay (ARL Mentor) - Pending
- **Next Review:** July 27, 2026
- **Related Documents:** 
  - `inference-engine/MAINTENANCE_MONITORING_PLAN.md`
  - `model-train/README.md`
  - `README.md` (project overview)

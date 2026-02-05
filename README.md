# Live, Virtual, and Constructive (LVC) Toolkit

**Project Codename:** Dark Circle  
**Team:** AI Technician Capstone Group 5, Carnegie Mellon University, ARL  
**Team Members:** Brandon Taylor, Larry Parrotte, John Tomaselli, Antonio Magana  
**Mentors:** Dr. Kristin E. Schaefer-Lay, Dr. Damon Conover, Henry Reimert

---

## 1. Project Overview

This project is a Capstone effort for the Carnegie Mellon University AI Technician Program, developed in collaboration with the Army Research Laboratory (ARL). The primary goal is to explore and develop Artificial Intelligence/Machine Learning (AI/ML) techniques to enhance ARL's Live, Virtual, and Constructive (LVC) Toolkit.

The project focuses on developing a **multi-modal classification model** that utilizes acoustic and seismic sensor data to automatically detect the presence of vehicles and classify their specific types. This capability aims to provide high-fidelity, actionable intelligence to improve situational awareness for researchers, soldiers, and test & evaluation teams operating within the LVC simulation environment.

## 2. Problem Statement

In modern military operations, the ability to detect and identify potential threats early and accurately is a decisive advantage. However, visual identification is often hindered by terrain, foliage, and adversary camouflage. This creates a critical need for non-visual detection methods.

The Army Research Laboratory faces a capability gap in the automated interpretation of data from tactical edge sensors. While the LVC Toolkit can aggregate raw sensor data, it lacks the robust AI mechanisms to transform this data into actionable intelligence. Any technical solution must be:
* **Lightweight and Power-Efficient:** To run on soldier-carried or embedded devices.
* **Resilient:** To function in austere field conditions with limited connectivity.
* **Integratable:** To work with existing Army tactical systems.

This project addresses the challenge of fusing "big data" from multiple battlefield sources to reduce noise, provide intelligent information, and enable more optimal outcomes for Army operations.

## 3. Proposed Solution

We propose a two-part solution: a multi-modal AI model and a containerized inference engine.

1.  **AI Model:** A classification model that processes synchronized acoustic and seismic time-series data to perform:
    * **Vehicle Detection:** A binary classification to determine if a vehicle is present versus background noise.
    * **Vehicle Classification:** A multi-class classification to identify the specific type of vehicle detected (e.g., "Pickup Truck," "Tank").

2.  **Inference Engine:** A containerized service (using Docker) designed for seamless integration into the ARL LVC Toolkit. The engine will:
    * Ingest live or historical sensor data from LVC data sources (e.g., PostgreSQL database).
    * Execute the AI model to produce near real-time classifications.
    * Export enriched data (e.g., detection status, vehicle class, confidence score) back into the LVC environment for use by autonomy stacks or for operator visualization.

![Solution Concept Diagram](https://github.com/antoniogmagana/Dark_Circle/blob/main/images/solution-concept.png)

## 4. Scientific Hypothesis

The core of our research is based on the following empirical hypothesis:

> Using ARL-approved seismic and acoustic data, a **multi-modal classification model** will perform better than the most promising single-mode classification models for identifying selected vehicles from each other and from random background noise.

#### Target Performance Metrics:

| Task | Metric | Target |
| :--- | :--- | :--- |
| **Vehicle vs. Background Noise** | Accuracy | ≥ 85% |
| | Recall (Pd) | ≥ 85% |
| | False Alarm Rate | ≤ 15% |
| **Vehicle vs. Other Vehicles** | Accuracy | ≥ 65% |
| | Recall (Pd) | ≥ 65% |
| | False Alarm Rate | ≤ 25% |

## 5. System Architecture & Design

The solution operates as an inference engine within the broader LVC ecosystem. It can process data from live sensor feeds or a historical database. The processed output is then fed back into the LVC Autonomy Stack and Simulation Environment.

The development environment is an on-premise server infrastructure at the Army Artificial Intelligence Integration Center (AI2C), leveraging Proxmox, Kubernetes (K8s), Docker, and a PostgreSQL database for a robust and scalable workflow.

![Architecture Diagram](https://github.com/antoniogmagana/Dark_Circle/blob/main/images/architecture.png)

## 6. Methodology

Our development process follows a structured experimental design for signal processing and machine learning:

1.  **Data Preparation:** Ingesting and synchronizing raw acoustic/seismic data, followed by applying digital signal processing (DSP) techniques for noise reduction and feature extraction.
2.  **Model Exploration:** Evaluating several candidate algorithms suitable for time-series classification, including **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory (LSTM)** networks, and **ROCKET**.
3.  **Model Training & Tuning:** Training the models on labeled datasets and iteratively optimizing hyperparameters to maximize performance.
4.  **Evaluation & Analysis:** Assessing model performance with metrics like F1-score, precision-recall curves, and confusion matrices, and performing error analysis to identify failure points.

## 7. Project Status & Roadmap

### Anticipated Tasks
- [x] Requirements Gathering and Design
- [ ] Data Engineering and Modeling
- [ ] Deployment and Integration
- [ ] Evaluation and Redeployment

---

## 8. Data Management

### Large Files (Not Included in Repository)
Due to GitHub file size limits, the following raw data files are excluded from the main repository:

* `raw_data/**/rs1/aud16000.csv` - Audio data CSV files (16kHz audio samples)
* `raw_data/**/rs1/aud16k.wav` - Audio WAV files

> [!WARNING]
> **These files are required to run the data exploration notebook.**

### Download Data Files
* **Option 1: Cloud Storage (Recommended):** [Add your shared link here once you upload the data]
* **Option 2: Git LFS:** If your team has Git LFS set up, use:
    ```bash
    git lfs install
    git lfs pull
    ```
* **Option 3: Team Access:** Contact the CMU Capstone Team Group 5 for direct access.

### Expected Data Structure
After downloading and extracting data, ensure your directory is organized as follows:

```text
Dark_Circle/
├── Data exploration.ipynb
├── raw_data/
│   ├── Polaris0150pm/
│   │   └── rs1/
│   │       ├── aud16000.csv     (Separately Downloaded)
│   │       ├── aud16k.wav       (Separately Downloaded)
│   │       ├── ehz.csv
│   │       └── gps.csv
│   └── ... (Other experiment folders)
```

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/antoniogmagana/Dark_Circle.git
cd Dark_Circle
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data Files
Download the large data files from [shared drive link] and place them in the appropriate `raw_data/*/rs1/` folders.

### 4. Run the Notebook
```bash
jupyter notebook "Data exploration.ipynb"
```

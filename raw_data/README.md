# Raw Data Directory

This directory contains the seismic (EHZ) and GPS data from vehicle movement experiments.

## Structure
Each experiment folder contains:
- `rs1/ehz.csv` - Seismic sensor data (included in repository)
- `rs1/gps.csv` - GPS position data (included in repository)
- `rs1/aud16000.csv` - Audio data (NOT included - too large for GitHub)
- `rs1/aud16k.wav` - Audio WAV file (NOT included - too large for GitHub)

## Available Experiments
- Polaris0150pm
- Polaris0215pm
- Polaris0235pm-NoLineOfSight
- Silverado0255pm
- Silverado0315pm
- Warhog-NoLineOfSight
- Warhog1135am
- Warhog1149am

## Audio Files (Large Files)
The audio files (aud16000.csv and aud16k.wav) are excluded from the repository due to GitHub's file size limits.

**To use the full dataset:**

### Option 1: Download from Shared Drive (Recommended)
- **Google Drive / OneDrive / Dropbox:** [Add link here once uploaded]
- Download the `audio_files.zip` or individual files
- Extract to the appropriate `raw_data/[experiment]/rs1/` folders

### Option 2: Use Git LFS (If configured)
```bash
git lfs install
git lfs pull
```

### Option 3: Contact Team Members
Contact the capstone team for access to the complete dataset:
- Antonio Magana
- Larry Parrotte
- Brandon Taylor
- John Tomaselli

### Option 4: Original Data Source
Download from the original IoBT Moving Object Detection (MOD) Data Subset repository or source.

**After downloading, place files in this structure:**
```
raw_data/
├── Polaris0150pm/
│   └── rs1/
│       ├── aud16000.csv  ← Add this
│       ├── aud16k.wav    ← Add this
│       ├── ehz.csv       ✓ Already included
│       └── gps.csv       ✓ Already included
└── ... (repeat for each experiment folder)
```

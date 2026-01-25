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
1. Download the complete data from the original source
2. Place the audio files in the respective `rs1/` folders
3. The notebook will automatically detect and load them

**Original Data Source:**
IoBT Moving Object Detection (MOD) Data Subset

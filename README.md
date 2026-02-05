# Dark Circle

## Description
**Capstone LVC Toolkit**

*Detecting potential adversaries earlier and more accurately often provides a decisive advantage in combat.  This reason underlies many of the frameworks through which intellectuals often articulate desired combatant and system behavior, most prominently the Observe, Orient, Decide, and Act*

## Project Breakdown

### Problem
How can LVC Toolkit users enhance situational awareness and operational decision-making during simulated small-unit tactical robotic engagements?

### Motivation
What motivated the solution of the problem, and why is this problem important?

### Related Work
List similar papers and repositories that do similar things to what you are solving, which you consulted when trying to solve your problem of interest.

### Anticipated Tasks
List the tasks you are planning to do, and which have been solved. 
- [ ] Requirements Gathering and Design
- [ ] Data Engineering and Modeling
- [ ] Deployment and Intergration
- [ ] Evaluation and Redeployment

### Capacity Gaps
What are the capacity gaps that can be critical?

### Capability Gaps
What are the capability gaps that can be critical?

### AI2C Fit
How does this project fit to AI2C goals?

### RFI’s for Customer
How do you request information about the customer your capstone targets?

## Stakeholders
* **Mentor Info:** Dr. Kristin E. Schaefer-Lay, Dr. Damon Conover, Henry Reimert
* **Customer Info:** Dr. Carl Busart

* **Capstone Team:** Brandon Taylor, John Tomaselli, Larry Parrotte, Antonio Magana

## Data Files

### Large Files (Not Included in Repository)
The following large data files are excluded from the repository due to GitHub file size limits:

- `raw_data/**/rs1/aud16000.csv` - Audio data CSV files (16kHz audio samples)
- `raw_data/**/rs1/aud16k.wav` - Audio WAV files

**⚠️ These files are required to run the data exploration notebook.**

### Download Data Files

**Option 1: Google Drive / OneDrive / Dropbox (Recommended)**
[Add your shared link here once you upload the data]

**Option 2: Git LFS (For files < 2GB)**
If your team has Git LFS set up:
```bash
git lfs install
git lfs pull
```

**Option 3: Contact the Team**
Contact the capstone team for access to the data files.

### Expected Data Structure
After downloading, your directory should look like:
```
Dark_Circle/
├── Data exploration.ipynb
├── raw_data/
│   ├── Polaris0150pm/
│   │   └── rs1/
│   │       ├── aud16000.csv     (LARGE - download separately)
│   │       ├── aud16k.wav       (LARGE - download separately)
│   │       ├── ehz.csv
│   │       └── gps.csv
│   ├── Polaris0215pm/
│   │   └── rs1/...
│   └── ... (other experiment folders)
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

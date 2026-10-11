# Live audio spectrogram

This script captures audio from the default system microphone and displays a continuously updated, rolling 5-second spectrogram in one window:

- A 2D frequency-versus-time heat map.
- A 3D frequency-versus-time surface, with amplitude in decibels on the z-axis.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python src/spectrogram-generator.py
```

Allow microphone access if your operating system asks. Close the plot window to stop audio capture.

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

When launched without options, a microphone selector opens first. Choose an input device, speak or make a sound to check its live level meter, then select **Start** to open the spectrogram. The preview stream is closed before the spectrogram starts. Select **Cancel** or close the selector to exit.

```powershell
python src/spectrogram-generator.py
```

To bypass the selector and use a specific microphone directly, first list available input devices:

```powershell
python src/spectrogram-generator.py --list-devices
```

Then pass the desired device ID (for example, device `2`):

```powershell
python src/spectrogram-generator.py --device 2
```

Allow microphone access if your operating system asks. The selected device ID and name are logged when capture starts. Audio is captured at 48 kHz and analyzed with a 4,096-sample FFT for finer frequency detail. The spectrogram window also shows a live input-level meter in dBFS. Its frequency axis follows detected audio within a speech-focused 70 Hz–3.5 kHz range, with smoothing and padding. The 3D plot averages nearby spectral power into each surface cell and uses the same live dB range for height and color, with a compressed vertical axis. Black areas indicate frequency/time bins with little or no signal above the display floor. Press **q** in the spectrogram window to quit.

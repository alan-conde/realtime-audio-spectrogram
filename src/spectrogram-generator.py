"""Plot a live microphone spectrogram in 2D and 3D."""

from __future__ import annotations

import logging
import queue

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation


SAMPLE_RATE = 44_100
WINDOW_SECONDS = 5
FFT_SIZE = 2_048
HOP_SIZE = 512
UPDATE_INTERVAL_MS = 250
DB_FLOOR = -100.0
SURFACE_FREQUENCY_BINS = 128
SURFACE_TIME_BINS = 120

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _compute_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies and decibel magnitudes, shaped (frequency, time)."""
    if samples.size < fft_size:
        samples = np.pad(samples, (fft_size - samples.size, 0))

    frames = np.lib.stride_tricks.sliding_window_view(samples, fft_size)[::hop_size]
    window = np.hanning(fft_size).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(frames * window, axis=1))
    magnitudes *= 2.0 / np.sum(window)
    decibels = 20.0 * np.log10(np.maximum(magnitudes, 10 ** (DB_FLOOR / 20)))
    decibels = np.maximum(decibels, DB_FLOOR).T
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    return frequencies, decibels


class LiveSpectrogram:
    def __init__(self) -> None:
        self.sample_count = SAMPLE_RATE * WINDOW_SECONDS
        self.samples = np.zeros(self.sample_count, dtype=np.float32)
        self.write_position = 0
        self.samples_filled = 0
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self.image = None
        self.surface = None
        self.surface_axis = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Audio input status: %s", status)
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            logger.warning("Audio processing fell behind; dropping an input block.")

    def _append_samples(self, samples: np.ndarray) -> None:
        if samples.size >= self.sample_count:
            self.samples[:] = samples[-self.sample_count :]
            self.write_position = 0
            self.samples_filled = self.sample_count
            return

        end_position = self.write_position + samples.size
        if end_position <= self.sample_count:
            self.samples[self.write_position : end_position] = samples
        else:
            first_part_size = self.sample_count - self.write_position
            self.samples[self.write_position :] = samples[:first_part_size]
            self.samples[: end_position - self.sample_count] = samples[first_part_size:]
        self.write_position = end_position % self.sample_count
        self.samples_filled = min(self.samples_filled + samples.size, self.sample_count)

    def _recent_samples(self) -> np.ndarray:
        if self.samples_filled < self.sample_count:
            return self.samples[: self.samples_filled].copy()
        return np.concatenate(
            (self.samples[self.write_position :], self.samples[: self.write_position])
        )

    def _create_plot(self):
        figure = plt.figure(figsize=(14, 6))
        heatmap_axis = figure.add_subplot(1, 2, 1)
        self.surface_axis = figure.add_subplot(1, 2, 2, projection="3d")

        heatmap_axis.set_title("Live Spectrogram")
        heatmap_axis.set_xlabel("Time (seconds, rolling)")
        heatmap_axis.set_ylabel("Frequency (Hz)")
        heatmap_axis.set_xlim(-WINDOW_SECONDS, 0)
        heatmap_axis.set_ylim(0, SAMPLE_RATE / 2)
        self.image = heatmap_axis.imshow(
            np.full((FFT_SIZE // 2 + 1, 2), DB_FLOOR),
            origin="lower",
            aspect="auto",
            extent=(-WINDOW_SECONDS, 0, 0, SAMPLE_RATE / 2),
            vmin=DB_FLOOR,
            vmax=0,
            cmap="magma",
            interpolation="nearest",
        )
        figure.colorbar(self.image, ax=heatmap_axis, label="Amplitude (dBFS)")

        self.surface_axis.set_title("Live Spectrogram Surface")
        self.surface_axis.set_xlabel("Time (seconds, rolling)")
        self.surface_axis.set_ylabel("Frequency (Hz)")
        self.surface_axis.set_zlabel("Amplitude (dBFS)")
        self.surface_axis.set_xlim(-WINDOW_SECONDS, 0)
        self.surface_axis.set_ylim(0, SAMPLE_RATE / 2)
        self.surface_axis.set_zlim(DB_FLOOR, 0)
        figure.tight_layout()
        return figure

    def _update_plot(self, _frame):
        while True:
            try:
                self._append_samples(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        recent = self._recent_samples()
        if recent.size == 0:
            return (self.image,)

        frequencies, decibels = _compute_spectrogram(
            recent, SAMPLE_RATE, FFT_SIZE, HOP_SIZE
        )
        frame_times = (
            np.arange(decibels.shape[1]) * HOP_SIZE
            + FFT_SIZE / 2
        ) / SAMPLE_RATE - recent.size / SAMPLE_RATE
        time_step = HOP_SIZE / SAMPLE_RATE
        image_start = max(-WINDOW_SECONDS, frame_times[0] - time_step / 2)
        image_end = min(0, frame_times[-1] + time_step / 2)
        self.image.set_data(decibels)
        self.image.set_extent(
            (image_start, max(image_end, image_start + time_step), 0, SAMPLE_RATE / 2)
        )

        time_indices = np.linspace(
            0, decibels.shape[1] - 1,
            min(SURFACE_TIME_BINS, decibels.shape[1]),
            dtype=int,
        )
        frequency_indices = np.linspace(
            0, decibels.shape[0] - 1,
            min(SURFACE_FREQUENCY_BINS, decibels.shape[0]),
            dtype=int,
        )
        surface_times = frame_times[time_indices]
        surface_frequencies = frequencies[frequency_indices]
        surface_decibels = decibels[np.ix_(frequency_indices, time_indices)]
        time_grid, frequency_grid = np.meshgrid(surface_times, surface_frequencies)

        if self.surface is not None:
            self.surface.remove()
        self.surface = self.surface_axis.plot_surface(
            time_grid,
            frequency_grid,
            surface_decibels,
            cmap="magma",
            vmin=DB_FLOOR,
            vmax=0,
            linewidth=0,
            antialiased=False,
        )
        return (self.image, self.surface)

    def run(self) -> None:
        figure = self._create_plot()
        animation = FuncAnimation(
            figure,
            self._update_plot,
            interval=UPDATE_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_SIZE,
                callback=self._audio_callback,
            ):
                logger.info("Listening to the default microphone. Close the plot window to stop.")
                plt.show()
        finally:
            animation.event_source.stop()


def main() -> None:
    LiveSpectrogram().run()


if __name__ == "__main__":
    main()
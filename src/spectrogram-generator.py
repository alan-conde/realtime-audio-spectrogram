"""Plot a live microphone spectrogram in 2D and 3D."""

from __future__ import annotations

import argparse
import logging
import queue

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button


SAMPLE_RATE = 48_000
WINDOW_SECONDS = 5
FFT_SIZE = 4_096
HOP_SIZE = 512
UPDATE_INTERVAL_MS = 250
DB_FLOOR = -100.0
DB_DISPLAY_MIN = -80.0
DB_DISPLAY_MAX = -20.0
DB_DYNAMIC_RANGE = 35.0
DB_MIN_DYNAMIC_RANGE = 20.0
MIC_METER_FLOOR = -60.0
VOICE_MIN_FREQUENCY = 70.0
VOICE_MAX_FREQUENCY = 3_500.0
ACTIVE_FREQUENCY_DROP_DB = 24.0
FREQUENCY_RANGE_PADDING_HZ = 150.0
MIN_FREQUENCY_SPAN_HZ = 400.0
FREQUENCY_RANGE_SMOOTHING = 0.35
SURFACE_FREQUENCY_BINS = 160
SURFACE_TIME_BINS = 150

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


def _estimate_frequency_range(
    frequencies: np.ndarray, decibels: np.ndarray
) -> tuple[float, float]:
    """Estimate an active, speech-focused band while rejecting transient noise."""
    nyquist = float(frequencies[-1])
    spectrum = np.percentile(decibels, 90, axis=1)
    peak_db = float(np.max(spectrum))
    if peak_db <= DB_FLOOR + 3:
        return VOICE_MIN_FREQUENCY, min(VOICE_MAX_FREQUENCY, nyquist)

    active = (
        (frequencies >= VOICE_MIN_FREQUENCY)
        & (frequencies <= VOICE_MAX_FREQUENCY)
        & (
            spectrum
            >= max(DB_FLOOR + 6, peak_db - ACTIVE_FREQUENCY_DROP_DB)
        )
    )
    if not np.any(active):
        return VOICE_MIN_FREQUENCY, min(VOICE_MAX_FREQUENCY, nyquist)

    active_frequencies = frequencies[active]
    lower = max(
        VOICE_MIN_FREQUENCY,
        float(active_frequencies[0]) - FREQUENCY_RANGE_PADDING_HZ,
    )
    upper = min(
        VOICE_MAX_FREQUENCY,
        nyquist,
        float(active_frequencies[-1]) + FREQUENCY_RANGE_PADDING_HZ,
    )
    if upper - lower < MIN_FREQUENCY_SPAN_HZ:
        center = (lower + upper) / 2
        lower = max(VOICE_MIN_FREQUENCY, center - MIN_FREQUENCY_SPAN_HZ / 2)
        upper = min(
            VOICE_MAX_FREQUENCY,
            nyquist,
            lower + MIN_FREQUENCY_SPAN_HZ,
        )
        lower = max(VOICE_MIN_FREQUENCY, upper - MIN_FREQUENCY_SPAN_HZ)
    return lower, upper


def _downsample_surface(
    frequencies: np.ndarray,
    frame_times: np.ndarray,
    decibels: np.ndarray,
    frequency_min: float,
    frequency_max: float,
    frequency_bins: int,
    time_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average local power into surface cells to avoid sparse-bin artifacts."""
    visible_indices = np.flatnonzero(
        (frequencies >= frequency_min) & (frequencies <= frequency_max)
    )
    visible_decibels = decibels[visible_indices]
    frequency_count = min(frequency_bins, visible_indices.size)
    time_count = min(time_bins, frame_times.size)
    frequency_edges = np.linspace(
        0, visible_indices.size, frequency_count + 1, dtype=int
    )
    time_edges = np.linspace(0, frame_times.size, time_count + 1, dtype=int)
    frequency_starts = frequency_edges[:-1]
    time_starts = time_edges[:-1]

    power = np.power(10.0, visible_decibels / 10.0)
    cell_power = np.add.reduceat(power, frequency_starts, axis=0)
    cell_power = np.add.reduceat(cell_power, time_starts, axis=1)
    cell_sizes = np.diff(frequency_edges)[:, None] * np.diff(time_edges)[None, :]
    surface_decibels = 10.0 * np.log10(
        np.maximum(cell_power / cell_sizes, 10 ** (DB_FLOOR / 10))
    )
    surface_power = np.power(10.0, surface_decibels / 10.0)
    padded_power = np.pad(surface_power, ((1, 1), (0, 0)), mode="edge")
    surface_power = (
        padded_power[:-2] + 2 * padded_power[1:-1] + padded_power[2:]
    ) / 4
    padded_power = np.pad(surface_power, ((0, 0), (1, 1)), mode="edge")
    surface_power = (
        padded_power[:, :-2] + 2 * padded_power[:, 1:-1] + padded_power[:, 2:]
    ) / 4
    surface_decibels = 10.0 * np.log10(
        np.maximum(surface_power, 10 ** (DB_FLOOR / 10))
    )

    frequency_centers = np.array(
        [
            frequencies[
                visible_indices[start:end]
            ].mean()
            for start, end in zip(frequency_edges[:-1], frequency_edges[1:])
        ]
    )
    time_centers = np.array(
        [
            frame_times[start:end].mean()
            for start, end in zip(time_edges[:-1], time_edges[1:])
        ]
    )
    return frequency_centers, time_centers, surface_decibels


def select_input_device() -> int | None:
    """Show available microphones and preview the selected input level."""
    devices = [
        (index, device)
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]
    if not devices:
        raise RuntimeError("No audio input devices are available.")

    default_input = sd.default.device[0]
    initial_index = next(
        (position for position, (index, _) in enumerate(devices) if index == default_input),
        0,
    )
    columns = 2 if len(devices) > 6 else 1
    rows = (len(devices) + columns - 1) // columns
    figure_height = max(7.2, rows * 0.64 + 2.5)
    figure = plt.figure(figsize=(12, figure_height), facecolor="#f2f5f9")
    figure.text(
        0.055, 0.94, "AUDIO INPUT", color="#168c88", fontsize=10,
        fontweight="bold", va="center",
    )
    figure.text(
        0.055, 0.89, "Choose your microphone", color="#17263c",
        fontsize=23, fontweight="bold", va="center",
    )
    figure.text(
        0.055, 0.845, "Select a device and check its live level before continuing.",
        color="#637287", fontsize=11, va="center",
    )

    card_top = 0.79
    card_bottom = 0.17
    row_step = (card_top - card_bottom) / rows
    card_height = row_step * 0.82
    card_width = 0.275 if columns == 2 else 0.57
    card_axes = []
    card_details = []
    for position, (device_index, device) in enumerate(devices):
        row = position // columns
        column = position % columns
        x = 0.055 + column * 0.295
        y = card_top - (row + 1) * row_step + (row_step - card_height) / 2
        axis = figure.add_axes((x, y, card_width, card_height))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        card = FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#e5f5f3" if position == initial_index else "#ffffff",
            edgecolor="#168c88" if position == initial_index else "#dce3eb",
            linewidth=1.8 if position == initial_index else 1.0,
            transform=axis.transAxes,
            clip_on=False,
        )
        axis.add_patch(card)
        name = str(device["name"])
        axis.text(
            0.07, 0.62, name[:36] + ("..." if len(name) > 36 else ""),
            color="#17263c", fontsize=10, fontweight="bold",
            ha="left", va="center",
        )
        detail = f"Device {device_index}  ·  {device['max_input_channels']} channel(s)"
        if device_index == default_input:
            detail += "  ·  DEFAULT"
        axis.text(
            0.07, 0.30, detail, color="#637287", fontsize=8,
            ha="left", va="center",
        )
        card_axes.append(axis)
        card_details.append((card, axis, device))

    preview_axis = figure.add_axes((0.67, 0.39, 0.28, 0.4), facecolor="#17263c")
    preview_axis.set_xlim(MIC_METER_FLOOR, 0)
    preview_axis.set_ylim(0, 1)
    preview_axis.set_yticks([])
    preview_axis.set_xticks((-60, -40, -20, 0), labels=("−60", "−40", "−20", "0"))
    preview_axis.tick_params(axis="x", colors="#b7c6d9", labelsize=8, length=0)
    preview_axis.spines[:].set_visible(False)
    preview_axis.set_title("LIVE INPUT CHECK", loc="left", color="#8fd8d0",
                           fontsize=9, fontweight="bold", pad=16)
    preview_axis.text(
        0.02, 0.78, "Speak into your mic", transform=preview_axis.transAxes,
        color="#ffffff", fontsize=13, fontweight="bold", va="center",
    )
    level_bar = preview_axis.barh(
        0.4, 0, left=MIC_METER_FLOOR, height=0.12, color="#55d6c2"
    )[0]
    level_text = preview_axis.text(
        0.02, 0.06, "Starting preview...", transform=preview_axis.transAxes,
        color="#b7c6d9", fontsize=9, va="center",
    )
    selected_text = figure.text(
        0.67, 0.32, "", color="#637287", fontsize=9, va="top", wrap=True
    )
    start_axis = figure.add_axes((0.67, 0.16, 0.16, 0.085))
    cancel_axis = figure.add_axes((0.84, 0.16, 0.11, 0.085))
    start_button = Button(start_axis, "Start listening", color="#168c88",
                          hovercolor="#117672")
    cancel_button = Button(cancel_axis, "Cancel", color="#ffffff",
                           hovercolor="#e5ebf1")
    start_button.label.set_color("#ffffff")
    start_button.label.set_fontweight("bold")
    cancel_button.label.set_color("#44536a")
    cancel_button.label.set_fontweight("bold")

    state: dict[str, object] = {
        "device_index": devices[initial_index][0],
        "stream": None,
        "level_dbfs": MIC_METER_FLOOR,
        "preview_ready": False,
        "selected_position": initial_index,
    }
    result: list[int | None] = [None]

    def stop_preview() -> None:
        stream = state["stream"]
        state["stream"] = None
        state["preview_ready"] = False
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Microphone preview status: %s", status)
        rms = float(np.sqrt(np.mean(np.square(indata[:, 0]))))
        state["level_dbfs"] = 20 * np.log10(
            max(rms, 10 ** (MIC_METER_FLOOR / 20))
        )

    def start_preview(position: int) -> None:
        device_index, device = devices[position]
        stop_preview()
        state["device_index"] = device_index
        state["selected_position"] = position
        state["level_dbfs"] = MIC_METER_FLOOR
        update_cards()
        selected_text.set_text(f"Connecting to {device['name']}...")
        level_text.set_text("Waiting for input...")
        try:
            stream = sd.InputStream(
                device=device_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_SIZE,
                callback=audio_callback,
            )
            stream.start()
        except (sd.PortAudioError, ValueError) as error:
            logger.error(
                "Could not preview microphone '%s' (device %d): %s",
                device["name"],
                device_index,
                error,
            )
            selected_text.set_text(f"Could not preview {device['name']}:\n{error}")
            level_text.set_text("Preview unavailable")
            return

        state["stream"] = stream
        state["preview_ready"] = True
        selected_text.set_text(f"Previewing {device['name']}  ·  Device {device_index}")
        level_text.set_text("Waiting for sound...")

    def update_cards() -> None:
        for position, (card, axis, _device) in enumerate(card_details):
            selected = position == state["selected_position"]
            card.set_facecolor("#e5f5f3" if selected else "#ffffff")
            card.set_edgecolor("#168c88" if selected else "#dce3eb")
            card.set_linewidth(1.8 if selected else 1.0)

    def on_card_clicked(event) -> None:
        if event.inaxes in card_axes:
            start_preview(card_axes.index(event.inaxes))

    def update_meter() -> None:
        level_dbfs = float(state["level_dbfs"])
        level_bar.set_width(max(0.0, level_dbfs - MIC_METER_FLOOR))
        if state["preview_ready"]:
            level_text.set_text(
                f"{'● SIGNAL DETECTED' if level_dbfs > -55 else '● LISTENING'}"
                f"     {level_dbfs:.1f} dBFS"
            )
        figure.canvas.draw_idle()

    def on_start(_event) -> None:
        if state["preview_ready"]:
            result[0] = int(state["device_index"])
            plt.close(figure)

    def on_cancel(_event) -> None:
        plt.close(figure)

    start_button.on_clicked(on_start)
    cancel_button.on_clicked(on_cancel)
    figure.canvas.mpl_connect("button_press_event", on_card_clicked)
    figure.canvas.mpl_connect("close_event", lambda _event: stop_preview())
    meter_timer = figure.canvas.new_timer(interval=100)
    meter_timer.add_callback(update_meter)
    meter_timer.start()

    update_cards()
    start_preview(initial_index)
    plt.show()
    meter_timer.stop()
    stop_preview()
    return result[0]


class LiveSpectrogram:
    def __init__(self, device_index: int | None = None) -> None:
        self.device_index = device_index
        self.sample_count = SAMPLE_RATE * WINDOW_SECONDS
        self.samples = np.zeros(self.sample_count, dtype=np.float32)
        self.write_position = 0
        self.samples_filled = 0
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self.image = None
        self.surface = None
        self.heatmap_axis = None
        self.surface_axis = None
        self.mic_level_bar = None
        self.mic_level_text = None
        self.frequency_min = VOICE_MIN_FREQUENCY
        self.frequency_max = VOICE_MAX_FREQUENCY

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
        self.heatmap_axis = figure.add_subplot(1, 2, 1)
        self.surface_axis = figure.add_subplot(1, 2, 2, projection="3d")

        self.heatmap_axis.set_title("Live Spectrogram")
        self.heatmap_axis.set_xlabel("Time (seconds, rolling)")
        self.heatmap_axis.set_ylabel("Frequency (Hz)")
        self.heatmap_axis.set_xlim(-WINDOW_SECONDS, 0)
        self.heatmap_axis.set_ylim(self.frequency_min, self.frequency_max)
        self.image = self.heatmap_axis.imshow(
            np.full((FFT_SIZE // 2 + 1, 2), DB_FLOOR),
            origin="lower",
            aspect="auto",
            extent=(-WINDOW_SECONDS, 0, 0, SAMPLE_RATE / 2),
            vmin=DB_DISPLAY_MIN,
            vmax=DB_DISPLAY_MAX,
            cmap="magma",
            interpolation="nearest",
        )
        figure.colorbar(self.image, ax=self.heatmap_axis, label="Amplitude (dBFS)")

        self.surface_axis.set_title("Live Spectrogram Surface")
        self.surface_axis.set_xlabel("Time (seconds, rolling)")
        self.surface_axis.set_ylabel("Frequency (Hz)")
        self.surface_axis.set_zlabel("Amplitude (dBFS)")
        self.surface_axis.set_xlim(-WINDOW_SECONDS, 0)
        self.surface_axis.set_ylim(self.frequency_min, self.frequency_max)
        self.surface_axis.set_zlim(DB_DISPLAY_MIN, DB_DISPLAY_MAX)
        self.surface_axis.set_box_aspect((1.3, 1.1, 0.85))
        self.surface_axis.view_init(elev=28, azim=-55)
        figure.tight_layout(rect=(0, 0.14, 1, 1))
        meter_axis = figure.add_axes((0.2, 0.035, 0.6, 0.06))
        meter_axis.set_xlim(0, MIC_METER_FLOOR * -1)
        meter_axis.set_ylim(0, 1)
        meter_axis.set_yticks([])
        meter_axis.set_xticks((0, 20, 40, 60), labels=("-60", "-40", "-20", "0 dBFS"))
        meter_axis.set_title("Microphone input", fontsize=9)
        self.mic_level_bar = meter_axis.barh(0.5, 0, height=0.7, color="limegreen")[0]
        self.mic_level_text = meter_axis.text(
            1.01, 0.5, "Waiting for microphone...",
            transform=meter_axis.transAxes, va="center",
        )
        figure.canvas.mpl_connect(
            "key_press_event",
            lambda event: plt.close(figure) if event.key in ("q", "Q") else None,
        )
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
        rms = float(np.sqrt(np.mean(np.square(recent[-HOP_SIZE * 4 :]))))
        level_dbfs = 20 * np.log10(max(rms, 10 ** (MIC_METER_FLOOR / 20)))
        meter_width = max(0.0, min(-MIC_METER_FLOOR, level_dbfs - MIC_METER_FLOOR))
        self.mic_level_bar.set_width(meter_width)
        self.mic_level_text.set_text(f"{level_dbfs:.1f} dBFS")

        peak_dbfs = float(np.max(decibels))
        target_min, target_max = _estimate_frequency_range(frequencies, decibels)
        self.frequency_min += FREQUENCY_RANGE_SMOOTHING * (
            target_min - self.frequency_min
        )
        self.frequency_max += FREQUENCY_RANGE_SMOOTHING * (
            target_max - self.frequency_max
        )
        self.heatmap_axis.set_ylim(self.frequency_min, self.frequency_max)
        self.surface_axis.set_ylim(self.frequency_min, self.frequency_max)

        if peak_dbfs > DB_FLOOR + 3:
            display_max = min(0.0, peak_dbfs + 3.0)
            display_min = max(DB_FLOOR, display_max - DB_DYNAMIC_RANGE)
            if display_max - display_min < DB_MIN_DYNAMIC_RANGE:
                display_min = max(DB_FLOOR, display_max - DB_MIN_DYNAMIC_RANGE)
        else:
            display_min, display_max = DB_DISPLAY_MIN, DB_DISPLAY_MAX
        self.image.set_clim(display_min, display_max)
        self.surface_axis.set_zlim(display_min, display_max)

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

        surface_frequencies, surface_times, surface_decibels = _downsample_surface(
            frequencies,
            frame_times,
            decibels,
            self.frequency_min,
            self.frequency_max,
            SURFACE_FREQUENCY_BINS,
            SURFACE_TIME_BINS,
        )
        surface_decibels = np.clip(surface_decibels, display_min, display_max)
        time_grid, frequency_grid = np.meshgrid(surface_times, surface_frequencies)

        if self.surface is not None:
            self.surface.remove()
        self.surface = self.surface_axis.plot_surface(
            time_grid,
            frequency_grid,
            surface_decibels,
            cmap="magma",
            vmin=display_min,
            vmax=display_max,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        return (self.image, self.surface, self.mic_level_bar, self.mic_level_text)

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
            device_index = self.device_index
            if device_index is None:
                device_index = sd.default.device[0]
            if device_index is None or device_index < 0:
                raise RuntimeError(
                    "No default input device is set. Run with --list-devices, "
                    "then select one with --device DEVICE_ID."
                )
            device = sd.query_devices(device_index)
            if device["max_input_channels"] < 1:
                raise ValueError(f"Device {device_index} is not an audio input.")
            with sd.InputStream(
                device=device_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_SIZE,
                callback=self._audio_callback,
            ):
                logger.info(
                    "Listening to microphone '%s' (device %d) at %d Hz. Press q to quit.",
                    device["name"],
                    device_index,
                    SAMPLE_RATE,
                )
                plt.show()
        finally:
            if animation.event_source is not None:
                animation.event_source.stop()


def _print_input_devices() -> None:
    default_input = sd.default.device[0]
    devices = sd.query_devices()
    print("Available audio input devices:")
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            marker = " (default)" if index == default_input else ""
            print(f"  {index}: {device['name']}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list audio input device IDs and exit",
    )
    parser.add_argument(
        "--device",
        type=int,
        metavar="DEVICE_ID",
        help="audio input device ID from --list-devices (default: system default)",
    )
    args = parser.parse_args()

    if args.list_devices:
        _print_input_devices()
        return

    device_index = args.device
    if device_index is None:
        device_index = select_input_device()
    if device_index is not None:
        LiveSpectrogram(device_index=device_index).run()


if __name__ == "__main__":
    main()
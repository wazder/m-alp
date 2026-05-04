"""
ELMU2056 Signals and Systems — Option 3: Audio Signal Processing
Group Project

Overview:
  Three synthetic speech-like signals are generated to simulate different recording
  conditions (quiet environment, noisy environment, phone microphone). FIR and IIR
  filters are designed and applied. Time-domain, frequency-domain, and spectrogram
  analyses are performed. Six Signals & Systems concepts are demonstrated:
    1. Sampling Theorem
    2. DFT / FFT
    3. LTI Systems and Convolution
    4. FIR Filter Design (windowed-sinc, Hamming window)
    5. IIR Filter Design (Butterworth, SOS form)
    6. STFT and Spectrogram

  Extension (Creativity):
    A FIR window function comparison is performed to justify the choice of the
    Hamming window. Four windows are evaluated: Rectangular, Hann, Hamming, and
    Blackman. Their frequency responses, stopband attenuations, transition
    bandwidths, and achieved post-filtering SNR values are compared, revealing
    the fundamental trade-off between stopband attenuation and transition width.

Outputs:
  outputs/plots/  — 9 PNG figures (fig1–fig9)
  outputs/audio/  — 6 WAV files
"""

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf

# ── Constants ──────────────────────────────────────────────────────────────────
FS = 8000                          # sample rate (Hz); Nyquist limit = 4000 Hz
DURATION = 3.0                     # signal duration (seconds)
FREQS = [350, 500, 800, 1200, 2000, 3000]    # speech-like harmonic frequencies (all within 300–3400 Hz PSTN band)
AMPLITUDES = [1.0] * 6             # equal amplitude for each tone
SEED = 42                          # random seed for reproducibility
SNR_NOISY_DB = 5.0                 # combined SNR of HF-interference noisy signal (dB)
PHONE_SNR_DB = 8.0                 # combined SNR of phone noisy signal (dB)
FIR_CUTOFF = 3400.0                # FIR low-pass cutoff frequency (Hz)
FIR_NUMTAPS = 101                  # FIR filter length (odd → linear phase)
IIR_ORDER = 4                      # Butterworth filter order
IIR_LOWCUT = 300.0                 # IIR bandpass lower edge (Hz)
IIR_HIGHCUT = 3400.0               # IIR bandpass upper edge (Hz)
STFT_NPERSEG = 256                 # STFT window length (32 ms at 8 kHz)

PLOT_DIR = "outputs/plots"
AUDIO_DIR = "outputs/audio"


# ── Block A: Signal Generation ─────────────────────────────────────────────────

def generate_clean_signal(fs, duration, freqs, amplitudes):
    """Sum of sinusoids simulating a clean speech-like signal."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean = np.zeros(len(t))
    for f, a in zip(freqs, amplitudes):
        clean += a * np.sin(2 * np.pi * f * t)
    return t, clean


def add_hf_noise(signal, fs, target_snr_db=SNR_NOISY_DB):
    """
    Simulate a noisy environment with high-frequency interference:
      - Strong HF bandpass noise (3500–4000 Hz) — models electrical/RF interference
      - Weak broadband floor noise (SNR ≈ 28 dB) — models thermal/ambient noise
    The HF component dominates and is entirely outside the signal band, so the
    FIR low-pass filter can remove it effectively.
    """
    N = len(signal)
    p_signal = np.mean(signal ** 2)

    # HF interference: white noise bandpass-filtered to 3500–4000 Hz
    hf_white = np.random.normal(0, 1, N)
    h_hf = sig.firwin(FIR_NUMTAPS, [3500.0, 3900.0],
                      pass_zero=False, fs=fs, window='hamming')
    hf_noise = sig.lfilter(h_hf, [1.0], hf_white)

    # Weak broadband floor noise (fixed at SNR = 28 dB)
    floor_snr_linear = 10 ** (28.0 / 10)
    floor_noise = np.random.normal(0, np.sqrt(p_signal / floor_snr_linear), N)

    # Scale HF noise so that combined SNR ≈ target_snr_db
    p_floor = np.mean(floor_noise ** 2)
    target_total_noise_power = p_signal / (10 ** (target_snr_db / 10))
    p_hf_needed = target_total_noise_power - p_floor
    if p_hf_needed <= 0:
        p_hf_needed = target_total_noise_power * 0.9
    p_hf_raw = np.mean(hf_noise ** 2)
    if p_hf_raw > 0:
        hf_noise *= np.sqrt(p_hf_needed / p_hf_raw)

    total_noise = hf_noise + floor_noise
    return signal + total_noise, total_noise


def add_phone_noise(signal, fs, target_snr_db=PHONE_SNR_DB):
    """
    Simulate a phone microphone recording (signal tones are all within 300–3400 Hz):
      - Add low-frequency hum (50–200 Hz): models power-supply interference
      - Add HF hiss (3500–4000 Hz): models thermal/RF noise above the PSTN band
      - Add weak in-band floor noise
    Out-of-band components dominate, so the IIR bandpass removes most of the noise.
    """
    N = len(signal)
    p_signal = np.mean(signal ** 2)

    # Low-frequency hum (50–200 Hz)
    hum_white = np.random.normal(0, 1, N)
    h_hum = sig.firwin(FIR_NUMTAPS, [50.0, 200.0],
                       pass_zero=False, fs=fs, window='hamming')
    hum_noise = sig.lfilter(h_hum, [1.0], hum_white)

    # HF hiss (3500–4000 Hz)
    hiss_white = np.random.normal(0, 1, N)
    h_hiss = sig.firwin(FIR_NUMTAPS, [3500.0, 3900.0],
                        pass_zero=False, fs=fs, window='hamming')
    hiss_noise = sig.lfilter(h_hiss, [1.0], hiss_white)

    # Weak in-band floor noise (fixed at SNR = 25 dB)
    floor_noise = np.random.normal(0, np.sqrt(p_signal / (10 ** (25.0 / 10))), N)

    # Scale out-of-band components to share ~90 % of total noise budget
    target_total = p_signal / (10 ** (target_snr_db / 10))
    target_ooband = 0.9 * target_total
    p_hum_raw = np.mean(hum_noise ** 2)
    p_hiss_raw = np.mean(hiss_noise ** 2)
    if p_hum_raw > 0:
        hum_noise *= np.sqrt(0.5 * target_ooband / p_hum_raw)
    if p_hiss_raw > 0:
        hiss_noise *= np.sqrt(0.5 * target_ooband / p_hiss_raw)

    total_noise = hum_noise + hiss_noise + floor_noise
    return signal + total_noise, total_noise


# ── Block B: Filter Design ─────────────────────────────────────────────────────

def design_fir_lowpass(fs, cutoff, numtaps=FIR_NUMTAPS):
    """Windowed-sinc FIR low-pass filter using a Hamming window."""
    h = sig.firwin(numtaps, cutoff / (fs / 2), window='hamming')
    return h


def design_iir_bandpass(fs, lowcut=IIR_LOWCUT, highcut=IIR_HIGHCUT, order=IIR_ORDER):
    """Butterworth bandpass filter returned as second-order sections (SOS)."""
    sos = sig.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sos


# ── Block C: Analysis Utilities ────────────────────────────────────────────────

def apply_fir_filter(signal, h):
    """Apply FIR filter causally and zero-phase; return both outputs plus group delay."""
    causal = sig.lfilter(h, [1.0], signal)
    zerophase = sig.filtfilt(h, [1.0], signal)
    delay = (len(h) - 1) // 2
    return causal, zerophase, delay


def apply_iir_filter(signal, sos):
    """Apply IIR filter (SOS form) causally and zero-phase; return both."""
    causal = sig.sosfilt(sos, signal)
    zerophase = sig.sosfiltfilt(sos, signal)
    return causal, zerophase


def compute_fft(signal, fs):
    """One-sided FFT magnitude spectrum in dB, normalized to true amplitude."""
    N = len(signal)
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    magnitude = np.abs(X) / (N / 2)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    return freqs, magnitude_db


def compute_stft(signal, fs, nperseg=STFT_NPERSEG):
    """Short-Time Fourier Transform returning power in dB."""
    f, t, Zxx = sig.stft(signal, fs, window='hann',
                          nperseg=nperseg, noverlap=nperseg // 2)
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return f, t, Zxx_db


def compute_snr(signal, clean_ref):
    """Oracle SNR: signal vs. clean reference (dB)."""
    p_clean = np.sum(clean_ref ** 2)
    p_error = np.sum((signal - clean_ref) ** 2)
    if p_error == 0:
        return np.inf
    return 10 * np.log10(p_clean / p_error)


def demonstrate_lti_convolution(signal, h):
    """
    Formally prove that FIR filtering equals linear convolution.
    np.convolve(x, h) and lfilter(h, [1], x) must be numerically identical.
    """
    y_conv = np.convolve(signal, h)[:len(signal)]
    y_lfilter = sig.lfilter(h, [1.0], signal)
    assert np.allclose(y_conv, y_lfilter, atol=1e-10), \
        "LTI convolution equivalence check failed"
    return y_conv, y_lfilter


def save_wav(signal, path, fs):
    """Normalize to [-1, 1] and write as 16-bit PCM WAV."""
    peak = np.max(np.abs(signal))
    x_norm = signal / peak if peak > 0 else signal
    sf.write(path, x_norm, fs, subtype='PCM_16')


# ── Block D: Plotting ──────────────────────────────────────────────────────────

def plot_time_domain(t, signals, labels, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ['steelblue', 'tomato', 'seagreen']
    for ax, s, lbl, c in zip(axes, signals, labels, colors):
        ax.plot(t, s, color=c, linewidth=0.7)
        ax.set_title(lbl, fontsize=11)
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Fig. 1 — Time-Domain Signals', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_frequency_spectra(signals, fs, labels, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ['steelblue', 'tomato', 'seagreen']
    for ax, s, lbl, c in zip(axes, signals, labels, colors):
        freqs, mag_db = compute_fft(s, fs)
        ax.plot(freqs, mag_db, color=c, linewidth=0.8)
        ax.axvline(x=3400, color='gray', linestyle='--', linewidth=0.9,
                   label='3400 Hz cutoff')
        ax.set_title(lbl, fontsize=11)
        ax.set_ylabel('Magnitude (dB)')
        ax.set_ylim(-80, 30)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('Fig. 2 — Frequency Spectra (FFT)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_spectrograms(signals, fs, labels, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, s, lbl in zip(axes, signals, labels):
        f, t, Zxx_db = compute_stft(s, fs)
        mesh = ax.pcolormesh(t, f, Zxx_db, shading='gouraud',
                             cmap='inferno', vmin=-80, vmax=20)
        fig.colorbar(mesh, ax=ax, label='Power (dB)')
        ax.set_title(lbl, fontsize=11)
        ax.set_ylabel('Frequency (Hz)')
        ax.grid(False)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Fig. 3 — Spectrograms (STFT)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fir_filter_analysis(h, fs, noisy, fir_zerophase, t, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Impulse response
    axes[0].stem(np.arange(len(h)), h, markerfmt='C0.', linefmt='C0-',
                 basefmt='k-')
    axes[0].set_title('FIR Impulse Response h[n]', fontsize=11)
    axes[0].set_xlabel('Sample index n')
    axes[0].set_ylabel('h[n]')
    axes[0].grid(True, alpha=0.3)

    # Frequency response
    w, H = sig.freqz(h, [1.0], worN=4096, fs=fs)
    axes[1].plot(w, 20 * np.log10(np.abs(H) + 1e-12), color='steelblue')
    axes[1].axvline(x=FIR_CUTOFF, color='r', linestyle='--', label=f'Cutoff {FIR_CUTOFF} Hz')
    axes[1].set_title('FIR Frequency Response', fontsize=11)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_ylim(-80, 5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Before vs after (first 0.5 s)
    n05 = int(0.5 * fs)
    axes[2].plot(t[:n05], noisy[:n05], color='tomato', linewidth=0.7,
                 alpha=0.8, label='Noisy input')
    axes[2].plot(t[:n05], fir_zerophase[:n05], color='steelblue', linewidth=0.9,
                 label='FIR filtered output')
    axes[2].set_title('Noisy vs FIR Filtered (first 0.5 s)', fontsize=11)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Fig. 4 — FIR Filter Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_iir_filter_analysis(sos, fs, phone, iir_out, t, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Frequency response
    w, H = sig.sosfreqz(sos, worN=4096, fs=fs)
    axes[0].plot(w, 20 * np.log10(np.abs(H) + 1e-12), color='seagreen')
    axes[0].axvline(x=IIR_LOWCUT, color='r', linestyle='--',
                    label=f'Passband {IIR_LOWCUT}–{IIR_HIGHCUT} Hz')
    axes[0].axvline(x=IIR_HIGHCUT, color='r', linestyle='--')
    axes[0].set_title('IIR (Butterworth) Frequency Response', fontsize=11)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_ylim(-80, 5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phone signal (first 0.5 s)
    n05 = int(0.5 * fs)
    axes[1].plot(t[:n05], phone[:n05], color='tomato', linewidth=0.7)
    axes[1].set_title('Phone Signal (noisy, first 0.5 s)', fontsize=11)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # IIR filtered phone (first 0.5 s)
    axes[2].plot(t[:n05], iir_out[:n05], color='seagreen', linewidth=0.7)
    axes[2].set_title('IIR Filtered Phone Signal (first 0.5 s)', fontsize=11)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Fig. 5 — IIR Filter Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lti_convolution_demo(h, y_conv, y_lfilter, t, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].stem(np.arange(len(h)), h, markerfmt='C0.', linefmt='C0-',
                 basefmt='k-')
    axes[0].set_title('FIR Impulse Response h[n] (these coefficients ARE the filter)',
                      fontsize=11)
    axes[0].set_xlabel('Sample index n')
    axes[0].set_ylabel('h[n]')
    axes[0].grid(True, alpha=0.3)

    n02 = int(0.2 * FS)
    axes[1].plot(t[:n02], y_conv[:n02], color='steelblue', linewidth=2.0,
                 label='np.convolve(x, h)', alpha=0.9)
    axes[1].plot(t[:n02], y_lfilter[:n02], color='tomato', linewidth=1.0,
                 linestyle='--', label='lfilter(h, [1], x)', alpha=0.9)
    diff = np.max(np.abs(y_conv - y_lfilter))
    axes[1].set_title(
        f'Convolution vs lfilter — max|diff| = {diff:.2e}  (numerically identical)',
        fontsize=11)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Fig. 6 — LTI Convolution Equivalence Proof', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_snr_comparison(snr_data, path):
    """
    snr_data = {
        'AWGN Noisy': {'Before': x, 'After FIR': y, 'After IIR': z},
        'Phone':      {'Before': x, 'After FIR': y, 'After IIR': z},
    }
    """
    labels = list(snr_data.keys())
    conditions = ['Before', 'After FIR', 'After IIR']
    x = np.arange(len(labels))
    width = 0.25
    colors = ['#d9534f', '#5bc0de', '#5cb85c']

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        vals = [snr_data[lbl][cond] for lbl in labels]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=cond, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Fig. 7 — SNR Comparison Before and After Filtering', fontsize=12,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_noise_analysis(t, true_noise, estimated_noise, fs, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    n03 = int(0.3 * fs)
    axes[0].plot(t[:n03], true_noise[:n03], color='tomato', linewidth=0.8,
                 alpha=0.9, label='True noise (noisy − clean)')
    axes[0].plot(t[:n03], estimated_noise[:n03], color='steelblue',
                 linewidth=0.8, alpha=0.9, label='Residual (noisy − FIR output)')
    axes[0].set_title('True Noise vs FIR Residual (time domain, first 0.3 s)',
                      fontsize=11)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    f_true, mag_true = compute_fft(true_noise, fs)
    f_est, mag_est = compute_fft(estimated_noise, fs)
    axes[1].plot(f_true, mag_true, color='tomato', linewidth=0.8, alpha=0.9,
                 label='True noise spectrum')
    axes[1].plot(f_est, mag_est, color='steelblue', linewidth=0.8, alpha=0.9,
                 label='Residual spectrum')
    axes[1].set_title('True Noise vs FIR Residual (frequency domain)', fontsize=11)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_ylim(-80, 10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Fig. 8 — Noise Analysis: True Noise vs FIR Residual', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Block E: Extension — FIR Window Comparison ────────────────────────────────

WINDOWS = ['rectangular', 'hann', 'hamming', 'blackman']
WINDOW_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

# Theoretical peak stopband attenuation (dB) and transition width multiplier k,
# where ΔF ≈ k * fs / numtaps — from Harris (1978, Table 1).
# Used in the report to compare against measured values at +100 Hz probe.
WINDOW_THEORY = {
    'rectangular': {'attn_db': 13,  'k': 0.9},
    'hann':        {'attn_db': 44,  'k': 3.1},
    'hamming':     {'attn_db': 41,  'k': 3.3},
    'blackman':    {'attn_db': 74,  'k': 5.5},
}


def compare_fir_windows(noisy, clean, fs, cutoff, numtaps, path):
    """
    Design the same FIR low-pass filter with four different windows.
    For each window measure:
      - Frequency response (magnitude in dB)
      - First stopband sample attenuation at cutoff + 100 Hz
      - SNR of the zero-phase filtered output vs. the clean reference
    Return a list of dicts with per-window results.
    """
    results = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for window_name, color in zip(WINDOWS, WINDOW_COLORS):
        # scipy firwin uses 'boxcar' for the rectangular window
        scipy_name = 'boxcar' if window_name == 'rectangular' else window_name
        h = sig.firwin(numtaps, cutoff / (fs / 2), window=scipy_name)

        # Frequency response
        w, H = sig.freqz(h, [1.0], worN=8192, fs=fs)
        H_db = 20 * np.log10(np.abs(H) + 1e-12)

        # Stopband attenuation at cutoff + 100 Hz (first stopband sample)
        probe_freq = cutoff + 100.0
        idx = np.argmin(np.abs(w - probe_freq))
        atten_at_probe = H_db[idx]

        # Transition bandwidth: frequency where response crosses -6 dB
        idx6 = np.where(H_db <= -6.0)[0]
        f6 = w[idx6[0]] if len(idx6) > 0 else cutoff
        trans_bw = abs(f6 - cutoff)

        # Zero-phase filter and SNR
        zp_out = sig.filtfilt(h, [1.0], noisy)
        snr_val = compute_snr(zp_out, clean)

        results.append({
            'window':   window_name,
            'atten_db': atten_at_probe,
            'trans_bw': trans_bw,
            'snr':      snr_val,
        })

        label = f'{window_name.capitalize()} (SNR={snr_val:.1f} dB)'
        axes[0].plot(w, H_db, color=color, linewidth=1.2, label=label)

    axes[0].axvline(x=cutoff, color='gray', linestyle='--', linewidth=1.0,
                    label=f'Cutoff {cutoff:.0f} Hz')
    axes[0].axvline(x=cutoff + 100, color='black', linestyle=':', linewidth=0.9,
                    label='Probe (+100 Hz)')
    axes[0].set_xlim(2800, 4200)
    axes[0].set_ylim(-100, 5)
    axes[0].set_title('FIR Frequency Responses Near Cutoff (zoom)', fontsize=11)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Bar chart of post-filtering SNR
    snr_vals = [r['snr'] for r in results]
    bar_colors = WINDOW_COLORS
    bars = axes[1].bar(WINDOWS, snr_vals, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, snr_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3, f'{val:.2f} dB',
                     ha='center', va='bottom', fontsize=9)
    axes[1].set_title('Post-Filtering SNR Achieved by Each Window (HF Noisy Signal)',
                      fontsize=11)
    axes[1].set_xlabel('Window Function')
    axes[1].set_ylabel('SNR (dB)')
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Fig. 9 — Extension: FIR Window Function Comparison', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # ── Signal Generation ──────────────────────────────────────────────────────
    print("Generating signals...")
    t, clean = generate_clean_signal(FS, DURATION, FREQS, AMPLITUDES)
    noisy, hf_noise = add_hf_noise(clean, FS, SNR_NOISY_DB)
    phone, phone_noise = add_phone_noise(clean, FS, PHONE_SNR_DB)

    # ── Filter Design ──────────────────────────────────────────────────────────
    print("Designing filters...")
    h_fir = design_fir_lowpass(FS, FIR_CUTOFF, FIR_NUMTAPS)
    sos_iir = design_iir_bandpass(FS, IIR_LOWCUT, IIR_HIGHCUT, IIR_ORDER)

    # ── Filtering ──────────────────────────────────────────────────────────────
    print("Applying filters...")
    fir_causal, fir_zerophase, fir_delay = apply_fir_filter(noisy, h_fir)
    iir_phone_causal, iir_phone_zp = apply_iir_filter(phone, sos_iir)
    iir_noisy_causal, iir_noisy_zp = apply_iir_filter(noisy, sos_iir)
    _, fir_phone_zp, _ = apply_fir_filter(phone, h_fir)

    # LTI convolution proof
    y_conv, y_lfilter = demonstrate_lti_convolution(noisy, h_fir)
    print("  LTI convolution equivalence: PASSED")

    # ── SNR Analysis (zero-phase outputs for fair comparison) ──────────────────
    snr_noisy_before = compute_snr(noisy, clean)
    snr_noisy_fir    = compute_snr(fir_zerophase, clean)
    snr_noisy_iir    = compute_snr(iir_noisy_zp, clean)
    snr_phone_before = compute_snr(phone, clean)
    snr_phone_fir    = compute_snr(fir_phone_zp, clean)
    snr_phone_iir    = compute_snr(iir_phone_zp, clean)

    snr_data = {
        'HF Noisy': {
            'Before': snr_noisy_before,
            'After FIR': snr_noisy_fir,
            'After IIR': snr_noisy_iir,
        },
        'Phone': {
            'Before': snr_phone_before,
            'After FIR': snr_phone_fir,
            'After IIR': snr_phone_iir,
        },
    }

    print("\n── SNR Results ──────────────────────────────────────")
    print(f"{'Condition':<20} {'Before':>10} {'After FIR':>12} {'After IIR':>12}")
    print("-" * 56)
    for name, vals in snr_data.items():
        print(f"{name:<20} {vals['Before']:>9.2f} dB"
              f" {vals['After FIR']:>10.2f} dB"
              f" {vals['After IIR']:>10.2f} dB")
    print()

    # ── Save WAV Files ─────────────────────────────────────────────────────────
    print("Saving audio files...")
    save_wav(clean,         os.path.join(AUDIO_DIR, "01_clean_quiet.wav"),         FS)
    save_wav(noisy,         os.path.join(AUDIO_DIR, "02_noisy_hf_interference.wav"), FS)
    save_wav(phone,         os.path.join(AUDIO_DIR, "03_phone_noisy.wav"),         FS)
    save_wav(fir_zerophase,  os.path.join(AUDIO_DIR, "04_fir_filtered_noisy.wav"), FS)
    save_wav(iir_phone_causal, os.path.join(AUDIO_DIR, "05_iir_filtered_phone.wav"), FS)
    save_wav(hf_noise,      os.path.join(AUDIO_DIR, "06_hf_noise_component.wav"),   FS)

    # ── Generate Plots ─────────────────────────────────────────────────────────
    print("Generating figures...")

    plot_time_domain(
        t,
        [clean, noisy, phone],
        ['Clean Signal (Quiet Environment)',
         f'Noisy Signal (HF Interference, SNR={SNR_NOISY_DB} dB)',
         f'Phone Signal (Out-of-Band Noise, SNR={PHONE_SNR_DB} dB)'],
        os.path.join(PLOT_DIR, "fig1_time_domain_signals.png"))

    plot_frequency_spectra(
        [clean, noisy, phone], FS,
        ['Clean Signal — Frequency Spectrum',
         'Noisy Signal (HF Interference) — Frequency Spectrum',
         'Phone Signal — Frequency Spectrum'],
        os.path.join(PLOT_DIR, "fig2_frequency_spectra.png"))

    plot_spectrograms(
        [clean, noisy, phone], FS,
        ['Clean Signal — Spectrogram',
         'Noisy Signal (HF Interference) — Spectrogram',
         'Phone Signal — Spectrogram'],
        os.path.join(PLOT_DIR, "fig3_spectrograms.png"))

    plot_fir_filter_analysis(
        h_fir, FS, noisy, fir_zerophase, t,
        os.path.join(PLOT_DIR, "fig4_fir_filter_analysis.png"))

    plot_iir_filter_analysis(
        sos_iir, FS, phone, iir_phone_causal, t,
        os.path.join(PLOT_DIR, "fig5_iir_filter_analysis.png"))

    plot_lti_convolution_demo(
        h_fir, y_conv, y_lfilter, t,
        os.path.join(PLOT_DIR, "fig6_lti_convolution_demo.png"))

    plot_snr_comparison(
        snr_data,
        os.path.join(PLOT_DIR, "fig7_snr_comparison.png"))

    true_noise_arr = noisy - clean
    fir_residual = noisy - fir_zerophase
    plot_noise_analysis(
        t, true_noise_arr, fir_residual, FS,
        os.path.join(PLOT_DIR, "fig8_noise_analysis.png"))

    # ── Extension: FIR Window Comparison ──────────────────────────────────────
    print("Running FIR window comparison (extension)...")
    window_results = compare_fir_windows(
        noisy, clean, FS, FIR_CUTOFF, FIR_NUMTAPS,
        os.path.join(PLOT_DIR, "fig9_window_comparison.png"))

    print("\n── FIR Window Comparison Results ────────────────────")
    print(f"{'Window':<12} {'Atten@+100Hz':>14} {'Trans BW':>10} {'SNR':>10}")
    print("-" * 50)
    for r in window_results:
        print(f"{r['window']:<12} {r['atten_db']:>12.1f} dB"
              f" {r['trans_bw']:>8.1f} Hz"
              f" {r['snr']:>8.2f} dB")
    print()

    print("\nAll outputs written.")
    print(f"  Plots : {PLOT_DIR}/")
    print(f"  Audio : {AUDIO_DIR}/")


if __name__ == "__main__":
    main()

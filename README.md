# xdsp_rbj_filters
```python
#!/usr/bin/env python3
"""
xdsp_filters.py

Pure RBJ biquads + cascades + Butterworth-style + Linkwitz–Riley
All in one file.

Requires:
    numpy
    matplotlib
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt
from typing import Dict, List, Tuple, Literal, TypedDict


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

class BiquadCoeffs(TypedDict):
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


class BiquadState(TypedDict):
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    mode: str
    fs: float
    f0: float
    Q: float
    gain_db: float
    slope: float
    x1: float
    x2: float
    y1: float
    y2: float


# ---------------------------------------------------------------------
# RBJ Biquad Designer (Audio EQ Cookbook)
# ---------------------------------------------------------------------

def rbj_biquad_design(
    mode: str,
    f0: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> BiquadCoeffs:
    """
    Compute RBJ-style biquad filter coefficients.

    Supported modes:
        "lowpass", "highpass",
        "bandpass", "notch",
        "peak",
        "lowshelf", "highshelf"

    Returns:
        dict with keys: b0, b1, b2, a1, a2  (a0 normalized to 1).
    """
    if fs <= 0:
        raise ValueError("Sampling rate fs must be > 0.")
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist.")
    if Q <= 0:
        raise ValueError("Q must be > 0.")
    if slope <= 0:
        raise ValueError("slope must be > 0.")

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    cosw = cos(w0)
    sinw = sin(w0)

    # Default alpha (non-shelving)
    alpha = sinw / (2.0 * Q)

    # Shelf slope correction (RBJ: case S)
    if mode in ("lowshelf", "highshelf"):
        alpha = sinw / 2.0 * sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)

    # ---------------------------------------------------------------
    if mode == "lowpass":
        b0 = (1 - cosw) / 2.0
        b1 = 1 - cosw
        b2 = (1 - cosw) / 2.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "highpass":
        b0 = (1 + cosw) / 2.0
        b1 = -(1 + cosw)
        b2 = (1 + cosw) / 2.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "bandpass":
        # constant 0 dB peak gain variant
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "notch":
        b0 = 1.0
        b1 = -2 * cosw
        b2 = 1.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "peak":
        b0 = 1 + alpha * A
        b1 = -2 * cosw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw
        a2 = 1 - alpha / A

    elif mode == "lowshelf":
        sqrtA = sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha)
        a0 = (A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha

    elif mode == "highshelf":
        sqrtA = sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha)
        a0 = (A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha

    else:
        raise ValueError(f"Invalid rbj mode: '{mode}'")

    # Normalize to a0 = 1
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return {"b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2}


# ---------------------------------------------------------------------
# Biquad State Init / Processing
# ---------------------------------------------------------------------

def biquad_init(
    mode: str,
    f0: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> BiquadState:
    """Initialize a biquad filter with given design and zeroed state."""
    coeffs = rbj_biquad_design(mode, f0, fs, Q, gain_db, slope)
    return BiquadState(
        b0=coeffs["b0"],
        b1=coeffs["b1"],
        b2=coeffs["b2"],
        a1=coeffs["a1"],
        a2=coeffs["a2"],
        mode=mode,
        fs=fs,
        f0=f0,
        Q=Q,
        gain_db=gain_db,
        slope=slope,
        x1=0.0,
        x2=0.0,
        y1=0.0,
        y2=0.0,
    )


def biquad_tick(state: BiquadState, x: float) -> Tuple[float, BiquadState]:
    """Process one sample through a RBJ biquad."""
    b0, b1, b2 = state["b0"], state["b1"], state["b2"]
    a1, a2 = state["a1"], state["a2"]
    x1, x2 = state["x1"], state["x2"]
    y1, y2 = state["y1"], state["y2"]

    y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

    state["x2"] = x1
    state["x1"] = x
    state["y2"] = y1
    state["y1"] = y

    return y, state


def biquad_block(state: BiquadState, x: np.ndarray) -> Tuple[np.ndarray, BiquadState]:
    """Process a block through a RBJ biquad."""
    b0, b1, b2 = state["b0"], state["b1"], state["b2"]
    a1, a2 = state["a1"], state["a2"]
    x1, x2 = state["x1"], state["x2"]
    y1, y2 = state["y1"], state["y2"]

    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x, dtype=np.float64)

    for n, xn in enumerate(x):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn

    state["x1"], state["x2"] = x1, x2
    state["y1"], state["y2"] = y1, y2

    return y, state


# ---------------------------------------------------------------------
# Cascades
# ---------------------------------------------------------------------

def cascade_init(stage_params: List[dict]) -> List[BiquadState]:
    """Initialize a cascade of RBJ biquads from a list of biquad_init kwargs."""
    return [biquad_init(**p) for p in stage_params]


def cascade_block(states: List[BiquadState], x: np.ndarray) -> Tuple[np.ndarray, List[BiquadState]]:
    """Process a block through a cascade of RBJ biquads."""
    y = np.asarray(x, dtype=np.float64)
    for i, st in enumerate(states):
        y, states[i] = biquad_block(st, y)
    return y, states


# ---------------------------------------------------------------------
# Butterworth-style Qs (for RBJ-only higher order LP/HP)
# ---------------------------------------------------------------------

def butterworth_Qs(order: int) -> List[float]:
    """
    Return Q values for an even-order Butterworth-like low/high-pass cascade.

    This is the classic pole-angle formula. We only use it to choose
    RBJ Qs for cascaded biquads; no external libraries or ZPKs.
    """
    if order % 2 != 0 or order < 2:
        raise ValueError("Order must be an even integer >= 2.")

    n_sections = order // 2
    Qs: List[float] = []
    for k in range(1, n_sections + 1):
        theta = pi * (2 * k - 1) / (2 * order)
        Q = 1.0 / (2.0 * cos(theta))
        Qs.append(Q)
    return Qs


def butterworth_lowpass_stages(order: int, f0: float, fs: float) -> List[BiquadState]:
    """Nth-order low-pass via RBJ biquad cascade using Butterworth Qs."""
    Qs = butterworth_Qs(order)
    return [
        biquad_init("lowpass", f0=f0, fs=fs, Q=Q)
        for Q in Qs
    ]


def butterworth_highpass_stages(order: int, f0: float, fs: float) -> List[BiquadState]:
    """Nth-order high-pass via RBJ biquad cascade using Butterworth Qs."""
    Qs = butterworth_Qs(order)
    return [
        biquad_init("highpass", f0=f0, fs=fs, Q=Q)
        for Q in Qs
    ]


# ---------------------------------------------------------------------
# Higher-Order RBJ Filters (generic cascades)
# ---------------------------------------------------------------------

def rbj_higher_order(
    mode: str,
    fs: float,
    f0: float,
    order: int,
    Q_base: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
    bw_oct: float | None = None,
    spread_type: str = "none",
) -> List[BiquadState]:
    """
    Build higher-order RBJ filters by cascading RBJ biquads.

    - For lowpass/highpass: Qs from Butterworth-style distribution.
    - For others: default all Q = Q_base.
    - For peak/shelves: gain_db is split evenly across sections.
    - Optional bandwidth-based frequency spreading for peak/shelves.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("Order must be even and >= 2.")

    if mode not in (
        "lowpass", "highpass", "bandpass", "notch",
        "peak", "lowshelf", "highshelf"
    ):
        raise ValueError(f"Unsupported RBJ mode '{mode}'")

    n_sections = order // 2

    # Q distribution
    if mode in ("lowpass", "highpass"):
        Qs = butterworth_Qs(order)
    else:
        Qs = [Q_base] * n_sections

    # Gain distribution for gain-sensitive modes
    per_stage_gain = gain_db / n_sections if mode in ("peak", "lowshelf", "highshelf") else 0.0

    # Frequency spread for multi-section EQs
    if bw_oct is not None and n_sections > 1 and mode in ("peak", "lowshelf", "highshelf"):
        if spread_type == "log":
            # Log spacing around f0
            freqs = [
                f0 * 2 ** ((i - (n_sections - 1) / 2) * (bw_oct / (n_sections - 1)))
                for i in range(n_sections)
            ]
        elif spread_type == "linear":
            f_low = f0 / (2 ** (bw_oct / 2))
            f_high = f0 * (2 ** (bw_oct / 2))
            freqs = list(np.linspace(f_low, f_high, n_sections))
        else:
            freqs = [f0] * n_sections
    else:
        freqs = [f0] * n_sections

    stages: List[BiquadState] = []
    for i in range(n_sections):
        st = biquad_init(
            mode=mode,
            f0=freqs[i],
            fs=fs,
            Q=Qs[i],
            gain_db=per_stage_gain,
            slope=slope,
        )
        stages.append(st)

    return stages


# Convenience: high-order RBJ peaking EQ (Butterworth-style Qs)
def design_hpeq_rbj_butterworth(
    order: int,
    fs: float,
    f0: float,
    gain_db: float,
    Q_base: float = 1.0,
    bw_oct: float | None = None,
) -> List[BiquadState]:
    """
    High-order RBJ peaking EQ:
    - uses Butterworth Qs as a gentle shaping,
    - splits gain evenly across all sections,
    - optional BW-based frequency spreading.
    """
    return rbj_higher_order(
        mode="peak",
        fs=fs,
        f0=f0,
        order=order,
        Q_base=Q_base,
        gain_db=gain_db,
        bw_oct=bw_oct,
        spread_type="log" if bw_oct is not None else "none",
    )


# ---------------------------------------------------------------------
# Linkwitz–Riley (RBJ-only)
# ---------------------------------------------------------------------

def linkwitz_riley_stages(
    mode: Literal["lowpass", "highpass"],
    butter_order: int,
    f0: float,
    fs: float,
) -> List[BiquadState]:
    """
    Create Linkwitz–Riley stages using RBJ Butterworth-style cascades.

    butter_order:
        underlying Butterworth order (even).
        LR acoustic slope = 2 * butter_order.
    """
    if mode not in ("lowpass", "highpass"):
        raise ValueError("Linkwitz–Riley mode must be 'lowpass' or 'highpass'.")
    if butter_order % 2 != 0 or butter_order < 2:
        raise ValueError("butter_order must be even and >= 2.")

    Qs = butterworth_Qs(butter_order)

    stages: List[BiquadState] = []
    # LR: cascade two identical Butterworth filters
    for _ in range(2):
        for Q in Qs:
            stages.append(biquad_init(mode, f0, fs, Q=Q))
    return stages


def linkwitz_riley_block(states: List[BiquadState], x: np.ndarray) -> Tuple[np.ndarray, List[BiquadState]]:
    """Process a signal through a Linkwitz–Riley cascade."""
    return cascade_block(states, x)


# ---------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------

def compute_freq_response(b0, b1, b2, a1, a2, fs: float, n_fft: int = 2048):
    w = np.linspace(0, np.pi, n_fft)
    z = np.exp(1j * w)
    H = (b0 + b1 / z + b2 / (z ** 2)) / (1 + a1 / z + a2 / (z ** 2))
    f = w * fs / (2 * np.pi)
    return f, H


def plot_single_biquad_response(state: BiquadState, title: str):
    b0, b1, b2, a1, a2 = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    fs = state["fs"]
    f, H = compute_freq_response(b0, b1, b2, a1, a2, fs, 4096)
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.semilogx(f, mag_db)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cascade_response(states: List[BiquadState], title: str):
    fs = states[0]["fs"]
    w = np.linspace(0, np.pi, 4096)
    z = np.exp(1j * w)
    H_total = np.ones_like(w, dtype=complex)

    for s in states:
        b0, b1, b2, a1, a2 = [s[k] for k in ("b0", "b1", "b2", "a1", "a2")]
        H = (b0 + b1 / z + b2 / (z ** 2)) / (1 + a1 / z + a2 / (z ** 2))
        H_total *= H

    f = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.maximum(np.abs(H_total), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.semilogx(f, mag_db)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Example Demos (RBJ-only)
# ---------------------------------------------------------------------

def example_lowpass():
    fs = 48000
    f0 = 2000.0
    st = biquad_init("lowpass", f0, fs, Q=0.707)
    plot_single_biquad_response(st, f"RBJ Lowpass (f0={f0} Hz, Q=0.707)")


def example_highpass():
    fs = 48000
    f0 = 2000.0
    st = biquad_init("highpass", f0, fs, Q=0.707)
    plot_single_biquad_response(st, f"RBJ Highpass (f0={f0} Hz, Q=0.707)")


def example_bandpass():
    fs = 48000
    f0 = 1000.0
    st = biquad_init("bandpass", f0, fs, Q=5.0)
    plot_single_biquad_response(st, f"RBJ Bandpass (f0={f0} Hz, Q=5)")


def example_notch():
    fs = 48000
    f0 = 1000.0
    st = biquad_init("notch", f0, fs, Q=10.0)
    plot_single_biquad_response(st, f"RBJ Notch (f0={f0} Hz, Q=10)")


def example_peak():
    fs = 48000
    f0 = 1000.0
    st = biquad_init("peak", f0, fs, Q=2.0, gain_db=6.0)
    plot_single_biquad_response(st, f"RBJ Peaking EQ (+6 dB @ {f0} Hz, Q=2)")


def example_lowshelf():
    fs = 48000
    f0 = 200.0
    st = biquad_init("lowshelf", f0, fs, gain_db=6.0)
    plot_single_biquad_response(st, f"RBJ Low Shelf (+6 dB below {f0} Hz)")


def example_highshelf():
    fs = 48000
    f0 = 5000.0
    st = biquad_init("highshelf", f0, fs, gain_db=-6.0)
    plot_single_biquad_response(st, f"RBJ High Shelf (-6 dB above {f0} Hz)")


def example_filter_noise():
    """Apply an RBJ lowpass to white noise and show spectra."""
    fs = 48000
    f0 = 2000.0
    N = fs // 2

    x = np.random.randn(N)
    st = biquad_init("lowpass", f0, fs, Q=0.707)
    y, _ = biquad_block(st, x)

    win = np.hanning(N)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    mag_x = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(x * win)), 1e-9))
    mag_y = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(y * win)), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.semilogx(f, mag_x, label="Input Noise")
    plt.semilogx(f, mag_y, label="Filtered Output")
    plt.title("RBJ Lowpass on White Noise")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def example_nth_order_rbj_lowpass():
    fs = 48000
    f0 = 2000.0
    order = int(input("Enter even order for RBJ LPF (e.g., 4, 6, 8): ").strip())
    stages = butterworth_lowpass_stages(order, f0, fs)
    plot_cascade_response(stages, f"{order}th-Order RBJ Butterworth-style Lowpass")


def example_nth_order_rbj_highpass():
    fs = 48000
    f0 = 2000.0
    order = int(input("Enter even order for RBJ HPF (e.g., 4, 6, 8): ").strip())
    stages = butterworth_highpass_stages(order, f0, fs)
    plot_cascade_response(stages, f"{order}th-Order RBJ Butterworth-style Highpass")


def example_higher_order_rbj_peq():
    fs = 48000
    f0 = 1000.0
    order = int(input("Enter even order for RBJ PEQ (e.g., 4, 6): ").strip())
    gain_db = float(input("Enter peak gain in dB (e.g., 6.0): ").strip())
    bw_oct = float(input("Enter bandwidth in octaves (e.g., 1.0): ").strip())

    stages = design_hpeq_rbj_butterworth(order, fs, f0, gain_db, Q_base=1.0, bw_oct=bw_oct)
    plot_cascade_response(
        stages,
        f"{order}th-Order RBJ Peaking EQ ({gain_db} dB @ {int(f0)} Hz, BW={bw_oct} oct)"
    )


def example_linkwitz_riley_crossover():
    from numpy import unwrap, angle

    fs = 48000
    f0 = 2000.0
    butter_order = int(input("Enter base Butterworth order for LR (e.g., 2 or 4): ").strip())

    lp_stages = linkwitz_riley_stages("lowpass", butter_order, f0, fs)
    hp_stages = linkwitz_riley_stages("highpass", butter_order, f0, fs)

    w = np.linspace(0, np.pi, 4096)
    z = np.exp(1j * w)
    H_lp = np.ones_like(w, dtype=complex)
    H_hp = np.ones_like(w, dtype=complex)

    for s in lp_stages:
        b0, b1, b2, a1, a2 = [s[k] for k in ("b0", "b1", "b2", "a1", "a2")]
        H_lp *= (b0 + b1 / z + b2 / (z**2)) / (1 + a1 / z + a2 / (z**2))

    for s in hp_stages:
        b0, b1, b2, a1, a2 = [s[k] for k in ("b0", "b1", "b2", "a1", "a2")]
        H_hp *= (b0 + b1 / z + b2 / (z**2)) / (1 + a1 / z + a2 / (z**2))

    f = w * fs / (2 * np.pi)
    mag_lp = 20 * np.log10(np.maximum(np.abs(H_lp), 1e-9))
    mag_hp = 20 * np.log10(np.maximum(np.abs(H_hp), 1e-9))
    mag_sum = 20 * np.log10(np.maximum(np.abs(H_lp + H_hp), 1e-9))

    phase_lp = unwrap(angle(H_lp)) * 180.0 / pi
    phase_hp = unwrap(angle(H_hp)) * 180.0 / pi
    phase_diff = phase_lp - phase_hp

    plt.figure(figsize=(8, 7))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(f, mag_lp, label="LP")
    plt.semilogx(f, mag_hp, label="HP")
    plt.semilogx(f, mag_sum, "--", label="Sum")
    plt.title(f"Linkwitz–Riley (RBJ Butterworth {butter_order}) @ {f0:.0f} Hz")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    # Phase
    plt.subplot(2, 1, 2)
    plt.semilogx(f, phase_lp, label="LP phase")
    plt.semilogx(f, phase_hp, label="HP phase")
    plt.semilogx(f, phase_diff, "--", label="Phase diff")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    idx_cross = np.argmin(np.abs(f - f0))
    print(f"\nAt {f[idx_cross]:.2f} Hz:")
    print(f"  LP phase     = {phase_lp[idx_cross]:.2f}°")
    print(f"  HP phase     = {phase_hp[idx_cross]:.2f}°")
    print(f"  Phase diff   = {phase_diff[idx_cross]:.2f}°")
    print(f"  Sum magnitude= {mag_sum[idx_cross]:.2f} dB (target ≈ 0 dB)")


def example_lr_time_domain():
    fs = 48000
    f0 = 2000.0
    butter_order = 4
    N = fs

    x = np.random.randn(N)

    lp_stages = linkwitz_riley_stages("lowpass", butter_order, f0, fs)
    hp_stages = linkwitz_riley_stages("highpass", butter_order, f0, fs)

    y_lp, _ = linkwitz_riley_block(lp_stages, x)
    y_hp, _ = linkwitz_riley_block(hp_stages, x)
    y_sum = y_lp + y_hp

    err = y_sum - x
    rms_in = np.sqrt(np.mean(x**2))
    rms_err = np.sqrt(np.mean(err**2))
    rel_db = 20 * np.log10(max(rms_err / (rms_in + 1e-15), 1e-15))

    print(f"LR time-domain reconstruction error: {rms_err:.3e} (relative {rel_db:.2f} dB)")

    t = np.arange(0, min(N, 2000)) / fs
    plt.figure(figsize=(8, 3))
    plt.plot(t, x[:len(t)], label="Input", alpha=0.5)
    plt.plot(t, y_sum[:len(t)], label="LP+HP", alpha=0.7)
    plt.title("Linkwitz–Riley Time-Domain Reconstruction (zoom)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------

def main():
    print("\nRBJ Filters + Cascades + Linkwitz–Riley (PURE RBJ)")
    print("---------------------------------------------------")
    print(" 1: RBJ Lowpass")
    print(" 2: RBJ Highpass")
    print(" 3: RBJ Bandpass")
    print(" 4: RBJ Notch")
    print(" 5: RBJ Peaking EQ")
    print(" 6: RBJ Low Shelf")
    print(" 7: RBJ High Shelf")
    print(" 8: RBJ Lowpass on White Noise")
    print(" 9: Nth-Order RBJ Lowpass (Butterworth-style)")
    print("10: Nth-Order RBJ Highpass (Butterworth-style)")
    print("11: High-Order RBJ Peaking EQ (Butterworth-style Qs)")
    print("12: Linkwitz–Riley Crossover (mag + phase)")
    print("13: Linkwitz–Riley Time-Domain Test")

    choice = input("Select example (1–13): ").strip()

    if choice == "1":
        example_lowpass()
    elif choice == "2":
        example_highpass()
    elif choice == "3":
        example_bandpass()
    elif choice == "4":
        example_notch()
    elif choice == "5":
        example_peak()
    elif choice == "6":
        example_lowshelf()
    elif choice == "7":
        example_highshelf()
    elif choice == "8":
        example_filter_noise()
    elif choice == "9":
        example_nth_order_rbj_lowpass()
    elif choice == "10":
        example_nth_order_rbj_highpass()
    elif choice == "11":
        example_higher_order_rbj_peq()
    elif choice == "12":
        example_linkwitz_riley_crossover()
    elif choice == "13":
        example_lr_time_domain()
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

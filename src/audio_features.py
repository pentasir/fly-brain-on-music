"""Extract a small set of interpretable audio features, resampled to a fixed
number of frames so any track (any length) drives the simulation the same way."""
import numpy as np
import librosa


def extract_features(y: np.ndarray, sr: int, n_frames: int = 150) -> dict:
    duration = len(y) / sr
    hop_length = 512

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    bass = S[freqs < 250].mean(axis=0)
    mid = S[(freqs >= 250) & (freqs < 4000)].mean(axis=0)
    treble = S[freqs >= 4000].mean(axis=0)

    def resample(arr):
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n_frames)
        return np.interp(x_new, x_old, arr)

    def normalize(arr, lo=None, hi=None):
        lo = arr.min() if lo is None else lo
        hi = arr.max() if hi is None else hi
        span = hi - lo
        return np.clip((arr - lo) / span, 0, 1) if span > 0 else np.zeros_like(arr)

    bass_r, mid_r, treble_r = resample(bass), resample(mid), resample(treble)
    # Normalize bass/mid/treble jointly (shared min/max across all three), not
    # independently -- independent normalization stretches each band to its own
    # [0,1] range, which erases genuine magnitude differences between bands (real
    # music has far more bass energy than treble; independently normalizing hides
    # that and makes all three bands look similarly-shaped regardless of actual
    # relative loudness -- this was why bass/mid/treble activation barely diverged).
    shared_lo = min(bass_r.min(), mid_r.min(), treble_r.min())
    shared_hi = max(bass_r.max(), mid_r.max(), treble_r.max())

    features = {
        "times": np.linspace(0, duration, n_frames),
        "rms": normalize(resample(rms)),
        "onset": normalize(resample(onset_env)),
        "bass": normalize(bass_r, shared_lo, shared_hi),
        "mid": normalize(mid_r, shared_lo, shared_hi),
        "treble": normalize(treble_r, shared_lo, shared_hi),
        "tempo": float(tempo) if np.isscalar(tempo) else float(tempo[0]),
    }
    return features

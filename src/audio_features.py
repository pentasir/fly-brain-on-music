"""Extract a small set of interpretable audio features, resampled to a fixed
number of frames so any track (any length) drives the simulation the same way."""
import numpy as np
import librosa

# Number of mel-spaced frequency bands fed to the simulation as separate seed
# groups (see simulate.py's band_group = node % N_BANDS). Mel spacing puts more
# bands at low frequencies and fewer at high, the same way cochlear (and
# Johnston's Organ) frequency tuning is denser at the low end -- a cochlear-style
# front end, not a claim about the fly's real per-band frequency boundaries.
N_BANDS = 8


def extract_features(y: np.ndarray, sr: int, n_frames: int = 150) -> dict:
    duration = len(y) / sr
    hop_length = 512

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    # Mel spectrogram: frequency bins spaced to approximate cochlear tuning
    # (denser at low frequencies), then log-compressed (power_to_db) so loud
    # transients are compressed and quiet detail is expanded -- closer to real
    # auditory dynamic-range handling than raw linear STFT magnitude.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=N_BANDS * 4)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Group the N_BANDS*4 mel bins into N_BANDS coarser bands (mel bins are
    # already frequency-ordered, so this is a contiguous split, not a resample).
    band_edges = np.linspace(0, mel_db.shape[0], N_BANDS + 1).astype(int)
    band_slices = [mel_db[band_edges[i]:band_edges[i + 1]] for i in range(N_BANDS)]
    bands = [s.mean(axis=0) for s in band_slices]

    # Per-band onset: a transient (drum hit, pluck) in one frequency range
    # shouldn't have to wait for the whole-mix onset envelope below to notice it
    # -- e.g. a kick drum (low band) and a cymbal hit (high band) can now trigger
    # independently instead of both riding the same global onset bump. Reuses
    # the already-log-compressed mel_db slices (onset_strength's S expects a
    # log-power spectrogram) rather than recomputing a separate STFT per band.
    band_onsets = [librosa.onset.onset_strength(S=s, sr=sr, hop_length=hop_length) for s in band_slices]

    def resample(arr):
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n_frames)
        return np.interp(x_new, x_old, arr)

    def normalize(arr, lo=None, hi=None):
        lo = arr.min() if lo is None else lo
        hi = arr.max() if hi is None else hi
        span = hi - lo
        return np.clip((arr - lo) / span, 0, 1) if span > 0 else np.zeros_like(arr)

    bands_r = [resample(b) for b in bands]
    # Normalize all bands jointly (shared min/max across all of them), not
    # independently -- independent normalization stretches each band to its own
    # [0,1] range, which erases genuine magnitude differences between bands (real
    # music has far more low-frequency energy than high; independently normalizing
    # hides that and makes all bands look similarly-shaped regardless of actual
    # relative loudness -- this was why band activation barely diverged before).
    shared_lo = min(b.min() for b in bands_r)
    shared_hi = max(b.max() for b in bands_r)
    bands_norm = [normalize(b, shared_lo, shared_hi) for b in bands_r]

    # Per-band onset strengths get the same joint-normalization treatment as the
    # bands themselves, for the same reason: independent per-band normalization
    # would erase real differences in how "spiky" each band's transients are.
    onsets_r = [resample(o) for o in band_onsets]
    onset_lo = min(o.min() for o in onsets_r)
    onset_hi = max(o.max() for o in onsets_r)
    onsets_norm = [normalize(o, onset_lo, onset_hi) for o in onsets_r]

    features = {
        "times": np.linspace(0, duration, n_frames),
        "rms": normalize(resample(rms)),
        "onset": normalize(resample(onset_env)),
        "bands": bands_norm,  # list of N_BANDS arrays, low frequency -> high
        "band_onsets": onsets_norm,  # list of N_BANDS arrays, per-band transient strength
        "tempo": float(tempo) if np.isscalar(tempo) else float(tempo[0]),
    }
    return features

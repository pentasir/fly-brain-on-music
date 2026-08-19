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

# The 12 pitch classes (C, C#, D, ... B), independent of octave -- a second,
# separate seed grouping (pitch_group, see simulate.py) driven by which notes
# are sounding, not how loud or how sudden. Captures melody/chord content that
# a pure energy-per-band split can't (a C major chord and an F major chord can
# have identical loudness and spectral-band energy while differing entirely in
# which pitch classes are active).
N_PITCHES = 12
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# Minimum detected beats before beat-synced timing is trusted -- below this,
# librosa's beat tracker is usually either failing outright (silence, pure
# noise) or picking up a spurious/unstable pulse on material that has no real
# beat (e.g. free-tempo rubato piano), and forcing a grid onto it would distort
# timing rather than sync it. Falls back to fixed-rate framing in that case.
MIN_BEATS_FOR_SYNC = 8

# Hard cap on beat-synced frame count -- see the thinning step below for why.
MAX_BEAT_FRAMES = 300


def extract_features(y: np.ndarray, sr: int, n_frames: int = 150, beat_sync: bool = False, subdivisions: int = 2) -> dict:
    """beat_sync=True re-anchors simulation frames to the track's detected beat
    grid (subdivisions per beat) instead of a fixed wall-clock rate -- opt-in,
    since it only makes sense for material with a real, steady pulse. Falls
    back to fixed-rate framing (and reports why via "beat_sync_used"=False) if
    too few beats are detected."""
    duration = len(y) / sr
    hop_length = 512

    # Percussive/harmonic source separation (median-filtering based, Fitzgerald
    # 2010): splits the mix into a percussive component (drums, transients) and
    # a harmonic component (sustained tones, chords) before feature extraction,
    # so the two can drive structurally different response character downstream
    # (sharp/transient vs. smooth/sustained) instead of one blended signal.
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    beat_sync_used = beat_sync and len(beat_times) >= MIN_BEATS_FOR_SYNC
    if beat_sync_used:
        # subdivided beat grid: `subdivisions` frames per beat interval, plus the
        # track's start/end so nothing before the first beat or after the last
        # is silently dropped.
        grid = [0.0]
        for i in range(len(beat_times) - 1):
            grid.extend(np.linspace(beat_times[i], beat_times[i + 1], subdivisions, endpoint=False))
        grid.append(beat_times[-1])
        if beat_times[-1] < duration:
            grid.append(duration)
        times = np.array(sorted(set(grid)))
        # Bounds worst-case runtime/payload size for long and/or fast-tempo tracks
        # (same philosophy as simulate.py's MAX_SUBSTEPS_PER_FRAME) -- a 4.5 minute
        # track at a brisk tempo can produce 800+ beat-grid frames vs. fixed mode's
        # 150, which meant ~6x simulation time and a much larger embedded HTML
        # payload (proportionally more node snapshots). Thinning evenly keeps the
        # frames still beat-anchored, just coarser.
        if len(times) > MAX_BEAT_FRAMES:
            keep = np.linspace(0, len(times) - 1, MAX_BEAT_FRAMES).round().astype(int)
            times = times[np.unique(keep)]
    else:
        times = np.linspace(0, duration, n_frames)

    percussive_rms = librosa.feature.rms(y=y_percussive, hop_length=hop_length)[0]
    harmonic_rms = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]

    # Chroma (pitch-class energy) computed on the harmonic component only --
    # percussive transients have no clear pitch and would just add noise to
    # which pitch classes appear "active."
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)

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

    # Resample onto `times` (not a fixed n_frames linspace) so the same function
    # works for both fixed-rate and beat-synced framing -- fixed mode's `times`
    # IS a uniform linspace, so this is numerically identical to the old
    # behavior there; beat-synced mode's `times` is the irregular beat grid.
    time_frac = np.clip(times / duration, 0, 1) if duration > 0 else np.zeros_like(times)

    def resample(arr):
        x_old = np.linspace(0, 1, len(arr))
        return np.interp(time_frac, x_old, arr)

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

    # Percussive/harmonic ratio per frame, not independently-normalized absolute
    # energy -- what matters for shaping response character is which one
    # DOMINATES a given moment (a drum hit vs. a held chord), not their raw
    # levels (which are already reflected in rms/bands above).
    perc_r = resample(percussive_rms)
    harm_r = resample(harmonic_rms)
    total_r = perc_r + harm_r
    percussive_ratio = np.divide(perc_r, total_r, out=np.full_like(perc_r, 0.5), where=total_r > 0)

    # Same joint-normalization treatment as bands/onsets, for the same reason:
    # independent per-pitch-class normalization would erase which pitch classes
    # genuinely dominate a track vs. barely appear.
    chroma_r = [resample(chroma[p]) for p in range(N_PITCHES)]
    chroma_lo = min(c.min() for c in chroma_r)
    chroma_hi = max(c.max() for c in chroma_r)
    chroma_norm = [normalize(c, chroma_lo, chroma_hi) for c in chroma_r]

    features = {
        "times": times,
        "rms": normalize(resample(rms)),
        "onset": normalize(resample(onset_env)),
        "bands": bands_norm,  # list of N_BANDS arrays, low frequency -> high
        "band_onsets": onsets_norm,  # list of N_BANDS arrays, per-band transient strength
        "percussive_ratio": percussive_ratio,  # 0..1, how percussive- vs. harmonic-dominated each frame is
        "chroma": chroma_norm,  # list of N_PITCHES arrays, pitch-class energy (C, C#, D, ... B)
        "tempo": float(tempo) if np.isscalar(tempo) else float(tempo[0]),
        "beat_sync_used": beat_sync_used,  # False if beat_sync wasn't requested, or too few beats were detected
    }
    return features

# fly-music

Simulate fly (Drosophila) auditory-pathway activation in response to different music genres, using real FlyWire connectome data as the underlying graph.

## Status
Scaffolding only. Not yet pulling real data.

## Setup
```
source venv/bin/activate
pip install -r requirements.txt
python src/check_auth.py   # first run will prompt for a CAVE auth token
```

## Pipeline (planned)
1. `src/fetch_connectome.py` — pull Johnston's Organ -> AMMC -> downstream neurons via CAVEclient
2. `src/audio_features.py` — extract tempo/amplitude/frequency-band features per track
3. `src/simulate.py` — drive the connectome graph with audio features, run activation propagation
4. `src/visualize.py` — compare activation patterns across genres

## Notes
This is a speculative/artistic simulation, not a validated behavior predictor. The connectome data is real; the audio-to-neuron-drive mapping and resulting "behavior" are a modeling choice, not established science.

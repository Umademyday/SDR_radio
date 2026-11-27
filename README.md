# SDR_radio

Receiving broadcast FM radio with software, in Python. An RTL-SDR Blog V4 USB
dongle supplies raw radio samples; everything after that -- filtering,
demodulation, audio -- is NumPy and SciPy in this repository, written to be
read rather than to be fast.

Listening to Radio Nacional on 88.2 MHz, decoded sample by sample.

## The programs

| | |
|---|---|
| `fm_dsp.py` | The signal chain itself: filters, discriminator, de-emphasis. |
| `decode_fm.py` | Offline: a recorded I/Q capture becomes a mono WAV file. |
| `play_fm.py` | Live: dongle straight to the speakers, with an optional oscilloscope. |
| `audio_scope.py` | The waveform + spectrum window, used only by `play_fm.py --scope`. |

## How radio becomes sound

Both programs are thin front ends over the one chain in `fm_dsp.py` -- they
differ in where samples come from and where audio goes, not in the radio. The
tuner sits on the station, so the signal arrives already centred at 0 Hz:

```
1.8 MS/s complex I/Q   raw samples from the dongle
      | low-pass to the station's own bandwidth, then decimate
300 kS/s complex       one station, nothing either side of it
      | phase-difference discriminator: FM hides audio in the rate of phase change
300 kS/s real          frequency deviation in Hz -- this is the audio waveform
      | 15 kHz low-pass (drops the 19 kHz stereo pilot), 50 us de-emphasis, resample
48-50 kS/s real        mono audio -> a WAV file, or the speakers
```

## Getting started

```bash
brew install librtlsdr sox          # librtlsdr must be 2.0+ for the Blog V4
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

Listen live, with a window showing the waveform and spectrum of what you hear:

```bash
.venv/bin/python3 play_fm.py --station 88.2 --scope
```

Record ten seconds and decode it to a WAV instead:

```bash
rtl_sdr -f 88200000 -s 1800000 -g 33.8 -n 18000000 radio_nacional_88_2.cu8
.venv/bin/python3 decode_fm.py radio_nacional_88_2.cu8 radio_nacional_88_2.wav
afplay radio_nacional_88_2.wav
```

## Status

Mono FM works, offline and live, with a live audio oscilloscope.
Stereo is next: the 19 kHz pilot the mono filter currently discards is exactly
what a stereo decoder needs to recover the L-R subcarrier at 38 kHz.

# Capsule Python Package (pycapsule)

Read and convert proprietary .rec files from Neiry ecosystem and analyse it with MNE-python

## Reader
Read .rec file to mne.RawArray with `pycapsule.mne.io.read_raw_csr()`. Optionally, can return events (if present in data), event codes, timestamps and more

Read metadata from .dat file (info about device, recording parameters and more) with `record_reader.RecordReader.UnpackMetadata()`

## Converter

Use `pycap2bdf` script or `converter.py` to convert .rec format to 24-bit BDF. See `--help` for more.

if `-f` option with .rec file path not provided, falls back to interactive prompt mode where path to .rec file should be pasted.

In order for `pycap2bdf` to be accessible from console, make sure that Scripts folder for your Python distribution is in PATH. Alternatively, `converter.py` can be compiled with Pyinstaller and used in standalone mode. Note: as of MNE 1.2.2 you have to manually add "mne\icons" and "mne\report" folders to Pyinstaller to make it work

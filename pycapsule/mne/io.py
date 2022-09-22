from ..record_reader import RecordReader, RecordReaderVisitor, P300ProcessingUnit, BasicReaderVisitor
import numpy as np

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = np.searchsorted(array, value)
    return idx

def read_raw_csr(input_fname, outputTimestamps=False, rawEventCodes=False):
    import mne

    visitor = BasicReaderVisitor()
    RecordReader.Unpack(input_fname, visitor)

    metadata = RecordReader.UnpackMetadata(input_fname)
    deviceInfo = metadata["deviceInfo"]
    sfreq = deviceInfo["sampleRate"]

    if "channelNames" in deviceInfo:
        ch_names = deviceInfo["channelNames"]
    else:
        ch_names = ["O1", "C3", "CZ", "PZ", "O2", "C4", "FZ"]

    # MNE requires events to be > 0
    event_id = {
        "Undefined": 1,
        "Non-target": 2,
        "Target": 3
    }

    events = []

    if visitor.stimuliTimestamps is not None:
        for idx, stimulusTimestamp in enumerate(visitor.stimuliTimestamps):
            label = visitor.stimuliLabels[idx]

            eid = label + 2 # remap into event ids by shifting

            sampleIdx = find_nearest_idx(visitor.eegTimestamps, stimulusTimestamp)
            events.append([sampleIdx, 0, eid])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # events are converted to numpy array since MNE required it (due to indexing pattern)
    returnVals = mne.io.RawArray(data=visitor.eegData, info=info), np.array(events), event_id

    if outputTimestamps:
        returnVals = *returnVals, visitor.eegTimestamps
    
    if rawEventCodes:
        returnVals = *returnVals, visitor.rawStimuliCodes

    return returnVals
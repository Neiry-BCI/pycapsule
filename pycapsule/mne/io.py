from ..record_reader import RecordReader, RecordReaderVisitor, P300ProcessingUnit
import numpy as np

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = np.searchsorted(array, value)
    return idx

UNDEFINED_STIMULUS = -1
TARGET_STIMULUS = 1
NONTARGET_STIMULUS = 0

class Visitor(RecordReaderVisitor):
    eegData = []
    eegTimestamps = []
    stimuliTimestamps = []
    stimuliLabels = []
    stimuliIds = []

    def OnRawEEG(self, eegData:np.ndarray, eegTimestamps:np.array):
        self.eegData.append(eegData)
        self.eegTimestamps.append(eegTimestamps)
        
    def OnP300ProcessingUnit(self, p300unit:P300ProcessingUnit):
        targetStimulus = p300unit.targetStimulus

        stimuliTimestamps = []
        stimuliLabels = []
        stimuliIds = []

        for stimulusData in p300unit.stimuliData:
            self.stimuliTimestamps.append(stimulusData.timestamp)
            self.stimuliLabels.append(int(stimulusData.stimulusId == targetStimulus) if targetStimulus != UNDEFINED_STIMULUS else UNDEFINED_STIMULUS)
            self.stimuliIds.append(stimulusData.stimulusId)

def read_raw_csr(input_fname, outputTimestamps=False):
    import mne

    visitor = Visitor()
    RecordReader.Unpack(input_fname, visitor)
    visitor.eegData = np.hstack(visitor.eegData)
    visitor.eegTimestamps = np.ravel(visitor.eegTimestamps)
    visitor.stimuliTimestamps = np.ravel(visitor.stimuliTimestamps)
    visitor.stimuliLabels = np.ravel(visitor.stimuliLabels)
    visitor.stimuliIds = np.ravel(visitor.stimuliIds)

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

    returnVals = mne.io.RawArray(data=visitor.eegData, info=info), np.array(events), event_id

    if outputTimestamps:
        returnVals = *returnVals, visitor.eegTimestamps
        return returnVals

    return returnVals
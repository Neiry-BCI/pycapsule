from ..record_reader import RecordReader, RecordReaderVisitor, P300ProcessingUnit
import numpy as np

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

UNDEFINED_STIMULUS = -1
TARGET_STIMULUS = 1
NONTARGET_STIMULUS = 0

class Visitor(RecordReaderVisitor):
    eegData = None
    eegTimestamps = None
    stimuliTimestamps = None
    stimuliLabels = None
    stimuliIds = None

    def OnRawEEG(self, eegData:np.ndarray, eegTimestamps:np.array):
        self.eegData = np.append(self.eegData, eegData, axis=1) if self.eegData is not None else eegData
        self.eegTimestamps = np.append(self.eegTimestamps, eegTimestamps) if self.eegTimestamps is not None else eegTimestamps
        
    def OnP300ProcessingUnit(self, p300unit:P300ProcessingUnit):
        targetStimulus = p300unit.targetStimulus

        stimuliTimestamps = []
        stimuliLabels = []
        stimuliIds = []

        for stimulusData in p300unit.stimuliData:
            stimuliTimestamps.append(stimulusData.timestamp)
            stimuliLabels.append(int(stimulusData.stimulusId == targetStimulus) if targetStimulus != UNDEFINED_STIMULUS else UNDEFINED_STIMULUS)
            stimuliIds.append(stimulusData.stimulusId)

        self.stimuliTimestamps = np.append(self.stimuliTimestamps, stimuliTimestamps) if self.stimuliTimestamps is not None else np.array(stimuliTimestamps)
        self.stimuliLabels = np.append(self.stimuliLabels, stimuliLabels) if self.stimuliLabels is not None else np.array(stimuliLabels)
        self.stimuliIds = np.append(self.stimuliIds, stimuliIds) if self.stimuliIds is not None else stimuliIds

def read_raw_csr(input_fname):
    import mne

    visitor = Visitor()
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

    for idx, stimulusTimestamp in enumerate(visitor.stimuliTimestamps):
        label = visitor.stimuliLabels[idx]

        eid = label + 2 # remap into event ids by shifting

        sampleIdx = find_nearest_idx(visitor.eegTimestamps, stimulusTimestamp)
        events.append([sampleIdx, 0, eid])

    return mne.io.RawArray(visitor.eegData, mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')), np.array(events), event_id

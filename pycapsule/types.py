from dataclasses import dataclass
import numpy as np
import typing
from enum import Enum

@dataclass
class P300StimulusData:
    stimulusId: int
    sampleStartIndex: int           # index into eegData
    timestamp: float                # in seconds

@dataclass
class P300ProcessingUnit:
    unitId: int
    actId: int
    targetStimulus: int
    stimuliCount: int
    eegData: np.ndarray             # shape(channels, samples)
    eegTimestamps: np.array         # shape(samples)
    stimuliData: typing.List[P300StimulusData]
    shouldEndLearn: bool            # signals BCI system to train a model

class CSRPacketType(Enum):
    RawEEG = 0
    RawResistance = 1
    InterfaceData = 2
    P300ProcessingUnit = 3  

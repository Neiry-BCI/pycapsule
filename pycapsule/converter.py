import mne

import numpy as np
from pathlib import Path

import os.path
import sys
import argparse
import warnings

import pyedflib 
from datetime import datetime, timezone, timedelta
import os
from pycapsule.mne.io import read_raw_csr
import pycapsule

__version__ = 1.3

def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def write_mne_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, 
                  overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
        
    # static settings
    has_annotations = True if len(mne_raw.annotations)>0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = pyedflib.FILETYPE_EDFPLUS if has_annotations else pyedflib.FILETYPE_EDF
        dmin, dmax = -32768, 32767 
    else:
        file_type = pyedflib.FILETYPE_BDFPLUS if has_annotations else pyedflib.FILETYPE_BDF
        dmin, dmax = -8388608, 8388607
    
    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])
    
    if tmin:
        date += timedelta(seconds=tmin)
    # no conversion necessary, as pyedflib can handle datetime.
    #date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None

    
    # convert data
    channels = mne_raw.get_data(picks, 
                                start = first_sample,
                                stop  = last_sample)
    
    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)
    
    # create channel from this   
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels, 
                               file_type=file_type)
        
        channel_info = []
        
        ch_idx = range(n_channels) if picks is None else picks
        # keys = list(mne_raw._orig_units.keys())
        for i in ch_idx:
            try:
                ch_dict = {'label': mne_raw.ch_names[i], 
                           'dimension': 'µV', 
                           'sample_rate': mne_raw._raw_extras[0]['n_samps'][i], 
                           'physical_min': mne_raw._raw_extras[0]['physical_min'][i], 
                           'physical_max': mne_raw._raw_extras[0]['physical_max'][i], 
                           'digital_min':  mne_raw._raw_extras[0]['digital_min'][i], 
                           'digital_max':  mne_raw._raw_extras[0]['digital_max'][i], 
                           'transducer': '', 
                           'prefilter': ''}
            except:
                ch_dict = {'label': mne_raw.ch_names[i], 
                           'dimension': 'µV', 
                           'sample_rate': sfreq, 
                           'physical_min': channels.min(), 
                           'physical_max': channels.max(), 
                           'digital_min':  dmin, 
                           'digital_max':  dmax, 
                           'transducer': '', 
                           'prefilter': ''}
        
            channel_info.append(ch_dict)
        # f.setPatientCode(mne_raw._raw_extras[0]['subject_info'].get('id', '0'))
        # f.setPatientName(mne_raw._raw_extras[0]['subject_info'].get('name', 'noname'))
        # f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)
        for annotation in mne_raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            description = annotation['description']
            f.writeAnnotation(onset, duration, description)
        
    except Exception as e:
        raise e
    finally:
        f.close()
    return True


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Enter .rec file path.')
    parser.add_argument('-f', type=str, action='store', default=None, dest='filename', help='path to .rec file')
    parser.add_argument('-r', action='store_true', default=False, dest='raw_codes', help='wheter to output target/nontarget or raw stimuli ID (defaults to target/nontarget)')
    parser.add_argument('-v', '--version', action='store_true', default=False, dest='version', help='print version and exit')
    args = parser.parse_args()
    if args.version:
        print(f'pycaps2bdf {__version__} pycapsule {pycapsule.__version__}')
        sys.exit()
    if args.filename:
        filename = args.filename.strip('\"')
    else:
        filename = input('Enter .rec file path:')
        filename = filename.strip('\"')

    filename = Path(filename)
    filename.resolve()
    if not os.path.isfile(filename):
        print(f"No .rec file at {filename}")
        sys.exit()
    md = pycapsule.record_reader.RecordReader.UnpackMetadata(filename)
    raw, events, event_id, ts, ec = read_raw_csr(filename, outputTimestamps=True, rawEventCodes=True)
    raw.set_eeg_reference([])
    if events.size:
        if args.raw_codes:
            event_id = None
            events[:,-1] = ec
        else:
            event_id = {value:key for key, value in event_id.items()}
        
        annotations = mne.annotations_from_events(events, raw.info['sfreq'], 
            event_desc=event_id, first_samp=0, orig_time=None, verbose=None)
        raw.set_annotations(annotations)

    raw.set_meas_date(md['sessionInfo']['startUTCUnixTimestamp'])
    write_mne_edf(raw, str(filename.parent/f'{filename.stem}.bdf'), overwrite=True)

    print ('...all done!')



if __name__ == '__main__':
    main()
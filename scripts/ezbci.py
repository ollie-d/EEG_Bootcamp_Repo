# LM ollie-d 21Sep2023

import numpy as np
import pyxdf
import scipy.signal as signal
import copy
import re
import random
from itertools import chain


def loadxdf(fname, synthetic = False):
    """Loads an XDF containing an EEG stream and marker stream

    Args:
        fname (str): Full file location of the XDF
        synthetic (bool): Whether or not the file is a synthetic example (appends channels, zero-pads EEG data)
        
    Returns:
        dict: Creates an EEG structure as a dict containing:
            eeg_data: Time series EEG data
            eeg_time: Time points corresponding to EEG data
            event_data: Marker data (asynchronous)
            event_time: Time points corresponding to marker onset
            channels: Electrode locations as a dict
            fs: Sampling rate of EEG data
            fs_i: Initial sampling rate of EEG data (will not change when resampling)
            
    Unit Test:
        None
        
    Last Modified:
        Ollie 09Sep2019
    """
    # Load dataset from xdf and export eeg_raw, eeg_time, mrk_raw, mrk_time, channels
    streams, fileheader = pyxdf.load_xdf(fname, dejitter_timestamps=False) #ollie 9/11/2019
    
    # Create empty dict to be returned
    EEG = {}
    
    # Seperate streams
    eeg = None;
    mrk = None;
    for stream in streams:
        stream_type = stream['info']['type'][0]
        if stream_type.lower() == 'markers':
            mrk = stream
        if stream_type.lower() == 'eeg':
            eeg = stream
    if (eeg == None) or (mrk == None):
        print('ERROR, EEG AND MARKER STREAM NOT FOUND!')
        return
        
    # Create channel structure from stream info
    channel = {}
    try:    
        desc = eeg['info']['desc'][0]['channels'][0]['channel']
        for i in range(int(eeg['info']['channel_count'][0])):
            channel[desc[i]['label'][0]] = i
    except:
        print('Warning: Channel data not found. Using generic channels.')
        for i in range(int(eeg['info']['channel_count'][0])):
            channel[f'{i}'] = i

            
    # Let's also create time structures
    eeg_time = eeg['time_stamps']
    mrk_time = mrk['time_stamps']
    
    # Populate EEG with data
    EEG['eeg_data'] = eeg['time_series']
    EEG['eeg_time'] = eeg['time_stamps']
    EEG['event_data'] = mrk['time_series']
    EEG['event_time'] = mrk['time_stamps']
    EEG['channels'] = channel
    EEG['fs'] = int(eeg['info']['nominal_srate'][0])
    EEG['fs_i'] = int(eeg['info']['nominal_srate'][0]) # Note: This is a constant
    
    # If synthetic, pad data with ending 0's
    if synthetic:
        EEG['eeg_data'] = np.vstack((EEG['eeg_data'], np.zeros((10000, int(eeg['info']['channel_count'][0])))))

    # Clear un-needed objects
    streams = None
    fileheader = None
    eeg = None
    mrk = None
    desc = None
    channel = None
    
    return EEG

def firwin_bandpass(lowcut, highcut, fs, order=30):
    """ Create firwin, Hamming-window coefficients for bandpass filter

    Args:
        lowcut (float): Low cut threshold for coefficients
        highcut (float): High cut threshold for coefficients
        fs (float): Sampling rate
        order (int): Filter order (MUST BE EVEN)
        
    Returns:
        np.array: coefficients to be used in filter

    Unit Test:
        None:
        
    Last Modified:
        Ollie 27Apr2021
    """
   
    if order == None or order <= 0:
        print('ERROR: firwin order invalid. Using default value');
        order = 30
    elif order % 2 != 0:
        order -= 1
        print('Odd order given; new order is %d' % order)

    # Make sure low and high cut are in the correct order
    low = min(lowcut, highcut)
    high = max(lowcut, highcut)
   
    coeffs = signal.firwin(order+1, [low, high], fs=fs, pass_zero='bandpass', window='hamming')
    
    low = None
    high = None
    return coeffs

""" Create second order section coefficients for bandpass filter

Args:
    lowcut (float): Low cut threshold for coefficients
    highcut (float): High cut threshold for coefficients
    fs (float): Sampling rate
    order (int): Filter order
    
Returns:
    np.array: SOS coefficients to be used in filter

Unit Test:
    None:
    
Last Modified:
    Ollie 31Mar2020
"""
def butter_bandpass(lowcut, highcut, fs, order = 2):
        if lowcut > highcut:
            t = copy.deepcopy(highcut)
            highcut = copy.deepcopy(lowcut)
            lowcut = copy.deepcopy(t)
            t = None
        if order == None or order <= 0:
            order = 2
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog = False, btype = 'band', output = 'sos')
        return sos

""" Clone of butter_bandpass
"""
def bw_bp(lowcut, highcut, fs, order = 2):
        butter_bandpass(lowcut, highcut, fs, order)

""" Create coefficients and filter data (offline use only)
*** THIS IS TERRIBLY INEFFICIENT DO NOT USE ***
Args: 
    data (np.array): 1D float data of sampling rate fs to filter
    lowcut (float): Low threshold for bandpass coefficients
    highcut (float): High threshold for bandpass coefficients
    fs (float): Sampling rate of data and filter to be created
    order (int): SOS order
    type (str): 'filt' for causal and 'filtfilt' for zero-phase noncausal filter design
    
Returns:
    np.array: Filtered 1D signals equal to size of data
    
Unit Test:
    None
    
Last Modified:
    Ollie 01Jan2020 
"""
def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2, causal = False):
        sos = butter_bandpass(lowcut, highcut, fs, order = order)
        #zi = signal.sosfilt_zi(sos)
        if not causal:
            y = signal.sosfiltfilt(sos, data)
        else:
            y = signal.sosfilt(sos, data)
        return y
    
# Higher-level use of above filters to filter every channel
#TODO: Make more efficient (currently computes computations every channel
#TODO: Allow support for different styles of filters
def filter_eeg(EEG, low, high, order = None, chans = None, style = 'butter', causal = False):
    # Filter setup
    if style not in ['butter', 'firwin']:
        print('Invalid filter style. Valid choices are butter and firwin. Proceeding with butter.');
        style = 'butter'
    if style == 'butter':
        print('Designing IIR Butterworth filter...');
        coeffs = butter_bandpass(low, high, EEG['fs'], order)
    if style == 'firwin':
        print('Designing FIR Hamming-window filter...');
        coeffs = firwin_bandpass(low, high, EEG['fs'], order)
    if chans == None or isinstance(chans, list) == False:
        print('Channel definition not understood or not present. All channels will be filtered');
        chans = [x for x in range(EEG['eeg_data'].shape[1])];
        
    # Filtering
    print('Filtering data...')
    for i in chans:
        if style in ['butter']:
            if causal:
                EEG['eeg_data'][:, i] = signal.sosfilt(coeffs, EEG['eeg_data'][:, i])
            else:
                EEG['eeg_data'][:, i] = signal.sosfiltfilt(coeffs, EEG['eeg_data'][:, i])
        if style in ['firwin']:
            if causal:
                EEG['eeg_data'][:, i] = signal.lfilter(coeffs, 1.0, EEG['eeg_data'][:, i])
            else:
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfiltic.html#scipy.signal.lfiltic
                EEG['eeg_data'][:, i] = signal.filtfilt(coeffs, 1.0, EEG['eeg_data'][:, i])
    print('Filtering completed!');
    
    # DEBUG: old filtering method below (removed of branch/loop)
    #EEG['eeg_data'][:, i] = butter_bandpass_filter(EEG['eeg_data'][:, i], low, high, EEG['fs'], order, style)
    return EEG

# Function to downsample EEG data both the raw and time data
# Note: This does not make a deep copy and therefore the passed structure is modified
def downsample(EEG, fs_e):
    # If sampling rate is already desired, return fixed EEG
    if EEG['fs'] == fs_e:
        print('No need to downsample')
        return EEG
    
    # Downsample data from initial sampling rate fs_i to fs_e
    if fs_e > EEG['fs']:
        print('Invalid inputs detected')
        return None
    # If inputs are correct, slice accordingly
    slicer = round(EEG['fs'] / fs_e)
    EEG['eeg_data'] = EEG['eeg_data'][::slicer]
    EEG['eeg_time'] = EEG['eeg_time'][::slicer]
    EEG['fs'] = int(fs_e) # Update fs
    
    return EEG

# Epoch data into ERP structure
#TODO: Break this up into minor functions for photo and non-photo epoching?
#TODO: Include arg for tolerance in latency if event is not within dt
def epoch(EEG, epoch_s, epoch_e, bl_s, bl_e, chans, event, photosensor = None):
    # Initiate ERP structure (to be returned)
    ERP = {};
    
    # Handle channels. Currently only support locations
    if chans == '' or chans == None:
        channel = EEG['channels']
        chans = np.array(list(EEG['channels'].keys())).flatten()
    else:
        # If not all channels, create appropriate objects
        # First handle string input (put in a list)
        chans = chans if type(chans) == list else [chans]
        channel = {}
        for i in range(0, len(chans)):
            channel[chans[i]] = i
    
    # Get channel indices
    chans_ = [EEG['channels'][x] for x in chans]
    
    # Create epoched object
    fs = EEG['fs']
    fs_i = EEG['fs_i']
    dt = 1/fs
    dt_i = 1/fs_i
    epoch_len = int((abs(epoch_s) + abs(epoch_e)) * (fs / 1000));
    
    # We must store events manually
    events_df = []
    
    # Only keep events which we would like
    indices = []
    indices_bool = []
    for i in range(0, len(EEG['event_data'])):
        b = False
        t = EEG['event_data'][i][0]
        if re.match(event, t) != None:
            b = True
            events_df.append(t)
            indices.append(i)
        #indices.append(b)
        indices_bool.append(b)
        
    # If a photosensor channel is given (not None) use onset to epoch
    if photosensor != None and isinstance(photosensor, int):
        #print('Photosensor activate on chan ' + str(photosensor))
        onsetTimes = photosensorEpoch(EEG, photosensor)
        #print(len(onsetTimes))    
        #print(onsetTimes)
        
    # Epoch should be num_chans x samples x epochs
    epoch_df = np.zeros(shape = (len(chans_), epoch_len, sum(indices_bool)));
    # translate epoch_s and epoch_e into index space (iterate through num epochs)
    i = 0;
    t = EEG['eeg_time'];
    inds_ = ''
    
    # Initialize extended event structure to be populated within loop
    ERP['event_ext'] = {'label': [], 'onset_ix': [], 'epoch_s_ix': [], 'epoch_e_ix': []};
    
    for ix in indices:
        # Locate closet index to event
        t0 = EEG['event_time'][ix]
        inds = np.where((t >= (t0 - (2*dt))) & (t <= (t0 + (2*dt))))[0]
        if photosensor != None and isinstance(photosensor, int):
            inds_ = np.where((onsetTimes >= (t0 - 1)) & (onsetTimes <= (t0 + 1)))[0]

        # If multiple indices within threshold, find closest to event time
        if len(inds) == 0:
            print('WARNING -- EVENT NOT WITHIN dt OF ANY EEG TIMES' )
        elif len(inds) > 1:
            temp = EEG['eeg_time'][inds] - t0
            ix_0 = inds[np.argsort(abs(temp))[0]]
        else:
            ix_0 = inds[0]
        # Now epoch around this index
        e_i = int(ix_0 + (epoch_s / (1000 / fs)))
        e_e = int(ix_0 + (epoch_e / (1000 / fs)))
        b_i = int(ix_0 + (bl_s / (1000 / fs)))
        b_e = int(ix_0 + (bl_e / (1000 / fs)))
        
        # Update ERP structure
        ERP['event_ext']['label'].append(events_df[i])
        ERP['event_ext']['onset_ix'].append(ix_0)
        ERP['event_ext']['epoch_s_ix'].append(e_i)
        ERP['event_ext']['epoch_e_ix'].append(e_e)
        
        samples = EEG['eeg_data'][e_i:e_e, chans_]#[:, 0]#used for n channels
        epoch_df[:, :, i] = (samples - np.mean(EEG['eeg_data'][b_i:b_e, chans_], axis = 0).T).T # baseline correction
        samples = None
        i += 1;
        
    # Finish populating data structure
    ERP['bin_data'] = epoch_df;
    ERP['bin_times'] = np.arange(epoch_s, epoch_e, (1000 / fs))
    ERP['events'] = np.array(events_df);#EEG['event_data'][x][0] for x in range(0, len(EEG['event_data']))]
    ERP['fs'] = fs
    ERP['channels'] = channel

    # Clear memory
    epoch_len = None
    epoch_df = None
    events_df = None
    chans_ = None
    channel = None
    indices = None
    
    return ERP

# TODO: Confirm if a deep copy is auto created or not for this func and filter_eeg
def filter_erp(ERP, low, high, order = None, chans = None, style = 'butter', causal = False):
    # Filter setup
    if style not in ['butter', 'firwin']:
        print('Invalid filter style. Valid choices are butter and firwin. Proceeding with butter.');
        style = 'butter'
    if style == 'butter':
        print('Designing IIR Butterworth filter...');
        coeffs = butter_bandpass(low, high, ERP['fs'], order)
    if style == 'firwin':
        print('Designing FIR Hamming-window filter...');
        coeffs = firwin_bandpass(low, high, ERP['fs'], order)
    if chans == None or isinstance(chans, list) == False:
        print('Channel definition not understood or not present. All channels will be filtered');
        chans = [x for x in range(ERP['bin_data'].shape[0])];
        
    # Filtering
    print('Filtering data...')
    for i in chans:
        if style in ['butter']:
            if causal:
                ERP['bin_data'][i, :, :] = signal.sosfilt(coeffs, ERP['bin_data'][i, :, :], axis=0)
            else:
                ERP['bin_data'][i, :, :] = signal.sosfiltfilt(coeffs, ERP['bin_data'][i, :, :], axis=0)
        if style in ['firwin']:
            if causal:
                ERP['bin_data'][i, :, :] = signal.lfilter(coeffs, 1.0, ERP['bin_data'][i, :, :], axis=0)
            else:
                ERP['bin_data'][i, :, :] = signal.filtfilt(coeffs, 1.0, ERP['bin_data'][i, :, :], axis=0)
    print('Filtering completed!');
    
    # DEBUG: old filtering method below (removed of branch/loop)
    #EEG['eeg_data'][:, i] = butter_bandpass_filter(EEG['eeg_data'][:, i], low, high, EEG['fs'], order, style)
    return ERP

def svt_lite(data, threshold):
    min_ = np.min(data) <= (-1 * threshold)
    max_ = np.max(data) >= threshold
    return min_ or max_

def svt(EEG, chans, threshold_min, threshold_max, time_start_ix, time_end_ix):
    ''' Simple Voltage Threshold
        @params EEG: Epoched EEG data structure (3D np.array shape (nChans, nTimePoints, nEpochs))
        @params chans: Max chan num#Channels to perform operations upon (int array)
        @params threshold_min: Minimum boundary (in uV)
        @params threshold_max: Maximum boundary (in uV)
        @params time_start_ix: Lower temporal boundary to perform operations (in indices)
        @params time_end_ix: Upper temporal boundary to perform operations (in indices)
        
        @returns: good  - Structure with acceptable epochs
                  bad   - Structure with rejected epochs
                  flags - List len nEvents, 1 = good, 0 = bad
    '''
    # First allow empty chans to cycle through all available chans
    #if len(chans) == 0:
    #    chans = np.arange(EEG.shape[0])
    # Make sure only legal channels provided
    # todo: write this
    # I don't think chans is doing literally anything
    
    # Create empty good and bad dfs and loop through data
    good = np.zeros((EEG.shape[0], EEG.shape[1], 0))
    bad  = np.zeros((EEG.shape[0], EEG.shape[1], 0))
    flags = []
    for i in range(EEG.shape[-1]):
        min_ = np.min(EEG[:chans, time_start_ix:time_end_ix, i])
        max_ = np.max(EEG[:chans, time_start_ix:time_end_ix, i])
        if max_ >= threshold_max or min_ <= threshold_min:
            bad = np.concatenate((bad, EEG[:, :, i].reshape(EEG.shape[0], EEG.shape[1], 1)), axis = 2)
            flags.append(0)
        else:
            good = np.concatenate((good, EEG[:, :, i].reshape(EEG.shape[0], EEG.shape[1], 1)), axis = 2)
            flags.append(1)     
    min_ = None
    max_ = None
    return good, bad, flags

# Extract specific epochs from an ERP with an array of indices
def sub_epoch(ERP, arr_ix):
    # Make sure we were passed the correct second arg
    if not isinstance(arr_ix, list) and not isinstance(arr_ix, int):
        print('ERROR -- Valid inputs are list of integer indices or single integer index')
        return;
    for i in arr_ix:
        if not isinstance(i, int) and not isinstance(i, np.int32) and not isinstance(i, np.int64):
            print('ERROR -- Valid inputs are list of integer indices or single integer index')
            return;
        
    ERP_ = copy.deepcopy(ERP)
    ERP_['bin_data'] = ERP_['bin_data'][:, :, arr_ix]
    ERP_['events']   = ERP_['events'][arr_ix]
    return ERP_
    

# Bin ERPs into 1D arrays
# DISABLED, ALL FUNCTIONALITY ADDED TO 'epoch'
def bin_erp(ERP, event):
    # Currently assumes regexp or str events so crash if not
    indices = []
    for i in range(0, len(ERP['events'])):
        b = False
        t = tr_ERP['events'][i]
        if re.match(event, str(t)) != None:
            b = True
        indices.append(b)
        
    ERP_ = copy.deepcopy(ERP)
    ERP_['bin_data']  = ERP['bin_data'][:, :, indices]
    ERP_['events']    = ERP['events'][indices]
    return ERP_

# Windowed means of the data, return 2D windowed means 
def wm(ERP, w_i, w_e, points):
    # Take the windowed means of ERP data and return it as n_observations x dimensions (flat channels x points)
    t = ERP['bin_times']
    dt = 1/ERP['fs'];

    inds = np.where((t >= (w_i - dt)) & (t <= (w_i + dt)))[0]
    # If multiple indices within threshold, find closes to event time
    if len(inds) > 1:
        #temp = EEG['eeg_time'][inds] - w_i
        temp = t[inds] - w_i
        w_i_ix = inds[np.argsort(abs(temp))[0]]
    else:
        w_i_ix = inds[0]
        
    inds = np.where((t >= (w_e - dt)) & (t <= (w_e + dt)))[0]
    # If multiple indices within threshold, find closes to event time
    if len(inds) > 1:
        #temp = EEG['eeg_time'][inds] - w_e
        temp = t[inds] - w_e
        w_e_ix = inds[np.argsort(abs(temp))[0]]
    else:
        w_e_ix = inds[0]
        
    #w_i_ix = np.argwhere(ERP['bin_times'] == w_i)[0, 0];
    #w_e_ix = np.argwhere(ERP['bin_times'] == w_e)[0, 0];
    step = ((w_e_ix - w_i_ix) / points).astype(int)
    df = np.zeros(shape = (len(ERP['channels']), points, ERP['bin_data'].shape[2]))
    for i in range(0, points):
        df[:, i, :] = np.mean(ERP['bin_data'][:, w_i_ix+(i*step):w_i_ix+((i+1)*step), :], axis = 1);

    return df.reshape(-1, df.shape[-1]).T

def wm_lite(data, w_s_ix, step, points):
    # Shape: (nchan, ntimepoints, nobservations)
    df = np.zeros((data.shape[0], points, data.shape[2]))
    for o in range(data.shape[2]):
        for i in range(0, points):
            df[:, i, o] = np.mean(data[:, w_s_ix+(i*step):w_s_ix+((i+1)*step), o], axis = 1);
    return df.reshape(-1, df.shape[-1]).T

def balance(normal, oddball):
    # Balance normal to have as many as oddball
    z = normal['bin_data']
    z_ = normal['events']
    k = oddball['bin_data']
    k_ = oddball['events']
    if z.shape[2] < k.shape[2]:
        r = random.sample(range(0, k.shape[2]), z.shape[2])
        s = k[:, :, r]
        s_ = k_[r]
        oddball['bin_data'] = s;
        oddball['events'] = s_;
    elif z.shape[2] > k.shape[2]:
        r = random.sample(range(0, z.shape[2]), k.shape[2])
        s = z[:, :, r]
        s_ = z_[r]
        normal['bin_data'] = s;
        normal['events'] = s_;
    else:
        print('Target and non-targets have identical shape')
    z = None
    z_ = None
    k = None
    k_ = None
    r = None
    s = None
    return normal, oddball

def filter_eeg_3D(EEG, sos):
    global bl_s_ix
    global bl_e_ix
    df = np.zeros(EEG.shape)
    for i in range(0, EEG.shape[0]):
        for c in range(0, EEG.shape[1]):
            df[i, c, :] = signal.sosfiltfilt(sos, EEG[i, c, :]).flatten()
            df[i, c, :] = df[i, c, :] - np.mean(df[i, c, bl_s_ix:bl_e_ix])
    return df

def photosensorEpoch(data, photochan):
    fs = data['fs'];
    dt = 1.0 / fs
    dt_ms = int(np.round(1000.0 / fs))
    
    # Assume 8000:10,000 is baseline
    s0_len_ms = 1600; # ms
    s0_len = np.round(s0_len_ms / dt_ms).astype(int); # (samples) Use first s0_len samples to calculate s0 mean and std
    s0_m = 0;
    s0_sd = 0;
    
    # Now define variables for state 1 (white)
    s1_len_ms = 160; # ms
    s1_len = int(np.round(s1_len_ms / dt_ms)); # samples (sligh)
    s1_buffer = np.zeros((1, s1_len))
    s1_buffer_d = collections.deque(maxlen = s1_len);
    s1_counter = 0;
    s1_m = 0;
    s1_sd = 0;
    s0_m = np.mean(data['eeg_data'][4000:6000, photochan]);
    s0_sd = np.std(data['eeg_data'][4000:6000, photochan]);
    buffering = False;
    state = 0;
    sd_th = 5;
    s1_times = [];
    s1_counter = 0;
    s_i = 6000
    buffer_time = 35; # ms (roughly 2 frames at 60Hz)
    buffer_samples = int(np.round(buffer_time / dt_ms));
    #print('Buffer samples: ' + str(buffer_samples))
    #print('s1_len: ' + str(s1_len))
    for i in range(s_i, data['eeg_data'].shape[0]):
        if buffering:
            # If appropriate number of buffer samples is completed
            if i - s_i >= buffer_samples:
                buffering = False;
        else:
            if state == 0:
                if abs(data['eeg_data'][i, photochan] - s0_m) >= (sd_th * s0_sd):
                    state = 1;
                    s_i = i;
                    buffering = True;
                    s1_times.append(data['eeg_time'][i]);
                    #s1_times.append(i);
                    #print('Going into state 1 at ' + str(i) + '...'); sys.stdout.flush();
            elif state == 1:
                # In state 1, determine if we still need to set bl/sd or no
                if s1_counter < s1_len:
                    s1_buffer[0, s1_counter] = data['eeg_data'][i, photochan];
                    s1_counter += 1;
                    if s1_counter == s1_len:
                        s1_m = np.mean(s1_buffer);
                        s1_sd = np.std(s1_buffer);
                        #print('outtahere')
                        #print(s1_sd)
                else:
                    if abs(data['eeg_data'][i, photochan] - s1_m) >= (sd_th * s1_sd):
                        state = 0;
                        s_i = i;
                        buffering = True;
                        s1_counter = 0;
                        #print('Going into state 0 at ' + str(i) + '...'); sys.stdout.flush();
    return s1_times;

def reference(EEG, numEEG, currRef, newRef):
    # Re-reference EEG given the current reference and new reference(s)
    # Assumes labeled EEG structure
    # numEEG is number of EEG channels (do not re-ref aux)
    # currRef must be the channel name
    # newRef can be multiple channels
    
    # Determine if average reference or not
    avgRef = False
    if str(newRef).lower() == 'avg':
        avgRef = True
        
    
    # Create a list version of the EEG channel dict
    chanList = list(EEG['channels'])
    
    if currRef in chanList:
        print('ERROR -- Current reference is present in EEG. Doublecheck')
        return None;
    
    # Create new reference
    ref = np.zeros((EEG['eeg_data'].shape[0],))
    refData = []
    
    if avgRef:
        ref = -1*(np.sum(EEG['eeg_data'][:, 0:numEEG], 1) / (numEEG + 1)) # need to confirm this doesn't mess up pipeline
        #ref = np.mean(EEG['eeg_data'][:, 0:numEEG], 1)
        #print('Ref shape: ' + str(ref.shape))
        #print('EEG shape: ' + str(EEG['eeg_data'][:, 0:numEEG].shape))
        #refData = EEG['eeg_data'][:, 0:numEEG] - ref
        #refData = EEG['eeg_data'][:, 0:numEEG] - np.tile(ref, (numEEG, 1)).T
    else:
        for ch in newRef:
            ref = ref + EEG['eeg_data'][:, EEG['channels'][ch]]
        ref = ref / len(newRef)
    
    # Loop through the EEG channels and re-reference them
    for i in range(numEEG):
        ch = chanList[i]
        if ch not in newRef:
            refData.append(EEG['eeg_data'][:, i] - ref)
        else:
            if len(newRef) > 1:
                refs_ = [x for x in newRef if x is not ch]
                ref_ = np.zeros((EEG['eeg_data'].shape[0],))
                for ch_ in refs_:
                    ref_ = ref_ + EEG['eeg_data'][:, EEG['channels'][ch_]]
                ref_ = ref_ / len(refs_)
                refData.append(EEG['eeg_data'][:, i] - ref_)
            else:
                refData.append(EEG['eeg_data'][:, i] - ref)

    # Now we will re-create our channel which was our reference
    refData.append(-1 * ref)
    newChanList_ = chanList[0:numEEG] + [currRef] + chanList[numEEG:]

    # Drop our ref(s) from newChanList_
    newChanList = []
    for chan in newChanList_:
        if chan not in newRef:
            newChanList.append(chan)

    # Now let's return our new EEG structure
    newEEG = copy.deepcopy(EEG)
    newEEG['eeg_data'] = np.concatenate((np.array(refData).T, newEEG['eeg_data'][:, numEEG:]), axis = 1)
    newEEG['channels'] = {}
    for i in range(len(newChanList)):
        newEEG['channels'][newChanList[i]] = i
    return newEEG

""" Re-reference EEG data with average voltage

Note:
    This script will eventually replace the current reference script and will be renamed.
    The current reference script is inefficient and confusing.
    after avg_reference performs as expected (via comparison to EEGLAB) other references will be created

Args: 
    EEG (EEG struct)
    chans (channels within EEG struct (indices) to re-reference)
    currRef (label of current common reference(s))
    
Returns:
    EEG struct
    
Unit tests:
    None
    * Output tested against EEGLAB equivalent (pop_reref) and results were equivalent [Ollie 06Aug2020]
    
Last Modified:
    Ollie 06Aug2020
    
TODO:
    * Add ability to re-reference from multiple common references
    * Re-introduce functionality of re-referencing to any channel(s)
        * replace current reference function with this ones
    * Allow user to decide the index of re-created channel
"""
def avg_reference(EEG, chans, currRef, recreateChan = True):
    # First off, make a deep copy of EEG to make sure we're not modifying original object
    EEG = copy.deepcopy(EEG); # do I need a different variable name? (seems like no)
    
    # Calculate activity of our current reference
    ref = -1 * np.sum(EEG['eeg_data'][:, chans], axis=1) / (len(chans) + 1)
    
    # Create a matrix of our reference and add its activity to eeg_data
    EEG['eeg_data'][:, chans] += np.repeat([ref], len(chans), axis=0).T
    
    # If recreateChan, add our old reference back into dataset
    # Currently this will be at the last index included in chans + 1
    if recreateChan:
        # Add reference to EEG matrix
        EEG['eeg_data'] = np.insert(EEG['eeg_data'].T, max(chans)+1, ref, axis=0).T
        
        # Add reference to channel matrix
        channels = list(EEG['channels'])
        channels = channels[0:max(chans)+1] + [currRef] + channels[max(chans)+1:]
        EEG['channels'] = {}
        for i in range(len(channels)):
            EEG['channels'][channels[i]] = i
            
    channels = None
    ref      = None
    return EEG

def reference_lite(EEG, newRef_ix):
    ix0 = newRef_ix[0]
    ix1 = newRef_ix[1]
    newRef = (EEG[ix0, :] + EEG[ix1, :]) / 2
    r0 = EEG[ix0, :] - EEG[ix1, :]
    r1 = EEG[ix1, :] - EEG[ix0, :]
    for i in range(63):
        EEG[i, :] = EEG[i, :] - newRef
    EEG[ix0, :] = r0
    EEG[ix1, :] = r1
    return np.concatenate((EEG.T, (-1*newRef).reshape(-1, 1)), 1).T

def wm_lite(data, w_s_ix, step, points):
    df = np.zeros((data.shape[0], points));
    for i in range(0, points):
        df[:, i] = np.mean(data[:, w_s_ix+(i*step):w_s_ix+((i+1)*step)], axis = 1);
    return df

def filter_eeg_lite(EEG, sos):
    global bl_s_ix
    global bl_e_ix
    for i in range(0, EEG.shape[0]):
        #EEG[i, :] = signal.sosfilt(sos, EEG[i, :]) # unstable, massive ringing
        EEG[i, :] = signal.sosfiltfilt(sos, EEG[i, :])
        EEG[i, :] = EEG[i, :] - np.mean(EEG[i, bl_s_ix:bl_e_ix])
    return EEG

def balance_wm(wm, la):
    ix0 = np.where(la == 0)[0]
    ix1 = np.where(la == 1)[0]
    
    if len(ix1) < len(ix0): # should always be true
        r = random.sample(range(0, len(ix0)), len(ix1))
        wm0 = wm[ix0[r]]
        wm1 = wm[ix1]
        la_ = np.concatenate((np.ones((wm1.shape[0])),
                             np.zeros(wm0.shape[0])))
        wm_ = np.concatenate((wm1, wm0))
    else:
        print('Error balancing')
        return None, None
    return wm_, la_.astype(int)
    
def listFlatten(df):
    '''
        Flattens lists
    '''
    #t = []
    #for i in range(len(df)):
    #    for j in range(len(df[i])):
    #        t.append(df[i][j])
    #return t
    return list(chain.from_iterable(df))
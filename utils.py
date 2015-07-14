import tables
import numpy as np
from memorize import memorize
from tru.rec import foreach, rowstack, colstack
from matplotlib.backends.backend_pdf import PdfPages

import os

from tru.se_scenecam import screencapture_to_angles
from tru.rec import foreach, rowstack, colstack, rec_degroup, groupby_multiple, groupby, groupby_i, append_field

from constants import BENDS, DATA_FILE, AIKANAKSU, CORNERING, CONDITION_LAPS_13

def open_h5file(fname, group_name, mode="a"):
    """
        Opens a HDF file with group name.
    """
    h5file = tables.openFile(fname, mode=mode)
    if not hasattr(h5file.root, group_name):
        h5file.createGroup("/", group_name, group_name)
    return h5file

def append2table(h5file, group_name, table, data):
    """
        Table is name of table (string).
    """    
    
    group = getattr(h5file.root, group_name) 
    if not hasattr(group, table):
        h5file.createTable(group, table, data)    
    else:
        getattr(group, table).append(data)
        
    h5file.flush()

def set_table(h5file, group_name, table_name, data):
    """
        Table is name of table (string).
    """    
    group = getattr(h5file.root, group_name) 
    if hasattr(group, table_name):
        getattr(group, table_name).remove()
    h5file.createTable(group, table_name, data)
    h5file.flush()

def get_integrated(DATA_FILE):
    with tables.openFile(DATA_FILE, 'r') as h5file:
        cdata = h5file.root.ramppi11.rawdata.read()
        naksu_data = h5file.root.ramppi11.naksu_data.read()
    return cdata, naksu_data

def remove_cars_in_view(cdata):
    dir = AIKANAKSU
    dirlist = np.sort(np.array(os.listdir(dir)))
    dirlist = [f for f in dirlist if f[-4:] == '.csv']
    fnames = [os.path.join(dir, f) for f in dirlist]
    
    to_remove = np.array([False] * len(cdata))
    for fname in fnames:
        if os.path.exists(fname):
            sid = fname.split('/')[-1][:-4]
            sdata = cdata[cdata['session_id'] == int(sid)]
            
            with open(fname) as f:
                for line in f:
                    match = sdata[sdata['se_frame_number'] == int(line.strip())]
                    if (len(match) > 1):  # should do this better
                        match = match[0]
                    lap = match['lap']
                    dist = match['dist']
                    try:
                        selected = [bend for bend in BENDS if (dist[0] > bend[0]) & (dist[0] < bend[1])]
                    except IndexError:
                        selected = []
                    
                    print lap, dist, selected
                    if lap and dist and selected:
                        remove = ((cdata['session_id'] == int(sid)) &
                                 (cdata['lap'] == lap[0]) &
                                 (cdata['dist'] >= selected[0][0]) &
                                 (cdata['dist'] <= selected[0][1]))
                        remove = np.array(remove)
                        to_remove = to_remove + remove
                        #cdata = cdata[~remove]
                        #print len(cdata)
                        #print len(remove), len(np.flatnonzero(remove))
                    else:
                        print 'no such data'
                    
                    #print sid, lap#, selected
                    
        else:
            print "%s doesn't exist" % fname
    
    print len(cdata), np.sum(to_remove)
    cdata = cdata[~to_remove]        
    return cdata

def get_uncleaned_data(data=DATA_FILE):
    h5file = tables.openFile(data, 'r')
    cdata = h5file.root.ramppi11.rawdata.read()
    cdata = remove_cars_in_view(cdata)
    
    # argh! still need this (video missing)
    d_2011080504 = cdata['session_id'] == 2011080504
    # for 13...
    d_2011080302 = cdata['session_id'] == 2011080302
    d_2013070999 = cdata['session_id'] == 2013070999
    d_2013071500 = cdata['session_id'] == 2013071500
    d_2013071601 = cdata['session_id'] == 2013071601
    d_2013071703 = cdata['session_id'] == 2013071703
    d_2013101307 = cdata['session_id'] == 2013101307
    cdata = cdata[~d_2011080504 &
                  ~d_2011080302 &
                  ~d_2013070999 &
                  ~d_2013071500 &
                  ~d_2013071601 &
                  ~d_2013071703 &
                  ~d_2013101307]
    
    return cdata, h5file

def get_cleaned_data(data=DATA_FILE):
    h5file = tables.openFile(data, 'r')
    cdata = h5file.root.ramppi11.rawdata.read()
    
    cdata = remove_cars_in_view(cdata)
    
    bend1 = (cdata['dist'] >= BENDS[0][0]) & (cdata['dist'] <= BENDS[0][1])
    bend2 = (cdata['dist'] >= BENDS[1][0]) & (cdata['dist'] <= BENDS[1][1])
    bend3 = (cdata['dist'] >= BENDS[2][0]) & (cdata['dist'] <= BENDS[2][1])
    bend4 = (cdata['dist'] >= BENDS[3][0]) & (cdata['dist'] <= BENDS[3][1])
    
    # remove:
    
    # subjects
    d_2011080504 = cdata['session_id'] == 2011080504
    d_2011081508 = cdata['session_id'] == 2011081508
    d_2011081509 = cdata['session_id'] == 2011081509
    d_2011092220 = cdata['session_id'] == 2011092220
    
    # bends within subjects
    d_2011080302_b3 = (cdata['session_id'] == 2011080302) & bend3
    d_2011081710_b3 = (cdata['session_id'] == 2011081710) & bend3
    d_2011101321_b2 = (cdata['session_id'] == 2011101321) & bend2
    
    # laps within bends within subjects
    d_2011081107_b2_l = ((cdata['session_id'] == 2011081107) & 
                         (cdata['lap'] <= 2) & bend2)
    d_2011082512_b2_l = ((cdata['session_id'] == 2011082512) & 
                         (cdata['lap'] == 3) &
                         (cdata['lap'] == 5) &
                         (cdata['lap'] == 7) &
                         (cdata['lap'] == 9) & bend2)
    
    d_2011082512_b3_l = ((cdata['session_id'] == 2011082512) & 
                         (cdata['lap'] <= 9) & bend3)
    d_2011083115_b3_l = ((cdata['session_id'] == 2011083115) & 
                         (cdata['lap'] <= 7) & bend3)
    
    
    good_data = cdata[~d_2011080504 & 
                      ~d_2011081508 &
                      ~d_2011081509 &
                      ~d_2011092220 &
                      ~d_2011080302_b3 &
                      ~d_2011081710_b3 &
                      ~d_2011101321_b2 &
                      ~d_2011081107_b2_l &
                      ~d_2011082512_b2_l &
                      ~d_2011082512_b3_l &
                      ~d_2011083115_b3_l]
    
    return good_data, h5file
        
@memorize
def get_merged_data():
    # this is not needed unless TP location is an issue!
    # cdata, h5file = get_cleaned_data(DATA_FILE)
    cdata, h5file = get_uncleaned_data(DATA_FILE)
    naksu_raw = h5file.root.ramppi11.naksu_data.read()
    merged = []

    for d in foreach(cdata, ['session_id', 'lap']):
        naksu_x, naksu_y, lap_d = get_interp_naksu(d, naksu_raw)

        naksu_d = np.rec.fromarrays((naksu_x, naksu_y),
                names='naksu_x,naksu_y')
        merged.append(colstack(lap_d, naksu_d))
    
    merged = rowstack(merged)
    return merged

def get_segment_data(sid, range, cdata):
    session = cdata[cdata['session_id'] == sid]
    segment = session[(session['dist'] > range[0]) &
                      (session['dist'] < range[1])]
    segment = segment[segment['g_direction_q'] > 0.4]
    return segment

@memorize
def get_pursuit_fits(start, end):
    from segreg import piecewise_linear_regression_pseudo_em,\
        segmentation_to_table
    data = get_angled_range_data2(start, end)
    data = data[data['g_direction_q'] > 0.2]

    params = ((1.5, 1.5), 0.5)

    grouping = ('session_id', 'lap')
    grps = []
    for i, (grp, d) in enumerate(groupby_multiple(data, grouping)):
        t = d['ts']
        g = np.vstack((d['scenecam_x'], d['scenecam_y'])).T
        
        splits, valid, winner, params =\
            memorize(piecewise_linear_regression_pseudo_em)(
                t, g, *params)
        seg = segmentation_to_table(splits, t[valid], g[valid])
        print tuple(grp)
        grps.append((tuple(grp), seg))
        
    return grps

def get_angled_range_data(start, end):
    data = get_range_data(start, end)
    h, p = np.degrees(screencapture_to_angles(data['scenecam_x'], data['scenecam_y'], strict=False))
    data['scenecam_x'] = h
    data['scenecam_y'] = p
    data['naksu_x'], data['naksu_y'] = np.degrees(screencapture_to_angles(data['naksu_x'], data['naksu_y'], strict=False))
    return data

def get_condition_laps(sid):
    sid = str(sid)
    if CONDITION_LAPS_13.has_key(sid):
        laps = CONDITION_LAPS_13[sid]
    else:
        laps = CONDITION_LAPS_13['0']
    return laps

def filter_by_treatment(data, segs, control=True):
    if control == True:
        data = [x for x in data if x['lap'] not in get_condition_laps(x['session_id'])]
        segs = [x for x in segs if x[0][1] not in get_condition_laps(x[0][0])]
    else:
        data = [x for x in data if x['lap'] in get_condition_laps(x['session_id'])]
        segs = [x for x in segs if x[0][1] in get_condition_laps(x[0][0])]
    data = np.array(data).view(np.recarray)
    return data, segs

def get_pooled(cis):
    all_data = []
    all_segs = []
    for ci in cis:
        data = get_angled_range_data2(CORNERING[ci][0], CORNERING[ci][1])
        segs = get_pursuit_fits(CORNERING[ci][0], CORNERING[ci][1])
        all_data.append(data)
        all_segs.append(segs)
    d = np.hstack(all_data)
    s = np.vstack(all_segs)
    return d, s


#OMG!! Here just because of a hurry and a broken memoization
@memorize
def get_angled_range_data2(*args):
    return get_angled_range_data(*args)

@memorize
def get_range_data(bs, be):
    data = get_merged_data()
    in_bend = (data['dist'] > bs) & (data['dist'] < be)
    data = data[in_bend]
    return data

# returns interpolated naksu coords (x,y) and a slightly curbed lap
from scipy.interpolate import interp1d
def get_interp_naksu(lap_data, naksu_data):
    sid = np.unique(lap_data['session_id'])[0]
    naksu_session = naksu_data[naksu_data['session_id'] == sid]
      
    def find_nearest(arr, value):
        idx = (np.abs(arr-value)).argmin()
        return arr[idx]    

    print sid, len(naksu_session), len(lap_data)
    if len(naksu_session) == 0:
        zeros = [0]*len(lap_data)
        return zeros, zeros, lap_data

    start = np.min(lap_data['se_frame_number'])
    start = find_nearest(naksu_session['se_ts'], start)
    stop = np.max(lap_data['se_frame_number'])
    stop = find_nearest(naksu_session['se_ts'], stop)
    
    naksu_lap = naksu_session[(naksu_session['se_ts'] >= start) &
                               (naksu_session['se_ts'] <= stop)]
    
    # new definition of individual run, with new limits taken into account
    lap_curbed = lap_data[(lap_data['se_frame_number'] >= start) &
                          (lap_data['se_frame_number'] <= stop)]
    
    print sid
    print start, stop, len(lap_curbed)
    print np.min(naksu_lap['se_ts']), np.max(naksu_lap['se_ts']), len(naksu_lap)
    #print np.min(lap_curbed['se_frame_number']), np.max(lap_curbed['se_frame_number'])

    if (len(lap_curbed) < 2) or (len(naksu_lap) < 2):
        zeros = [0]*len(lap_data)
        return zeros, zeros, lap_data

    fx = interp1d(naksu_lap['se_ts'], naksu_lap['x'])
    fy = interp1d(naksu_lap['se_ts'], naksu_lap['y'])
    new_x = fx(lap_curbed['se_frame_number'])
    new_y = fy(lap_curbed['se_frame_number'])
    
    #print len(lap_curbed), len(new_x), len(new_y)
    
    return new_x, new_y, lap_curbed

def print_pdf(figs_to_print, path):
    pp = PdfPages(path)
    for fig in figs_to_print:
        pp.savefig(fig)
    pp.close()
    
def px2heading(x,y):
    slope = 0.11786904561521629 
    cx = -37.482356505638776
    cy = -29.46726140380407
    return x * slope + cx, y * slope + cy

def pxx2heading(x):
    slope = 0.11786904561521629 
    cx = -37.482356505638776
    return x * slope + cx

def pxy2heading(y):
    slope = 0.11786904561521629 
    cy = -29.46726140380407
    return y * slope + cy

import scipy
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf(1-((1-confidence)/2.0), n-1)
    return m, m-h, m+h
    

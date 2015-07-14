from cv2 import cv
import sys
import tables
import numpy as np
import matplotlib.pyplot as plt
import tru.memorize
from tru.numpygst import NumpyGst

from constants import DATA_FILE, BENDS
from utils import get_integrated, get_cleaned_data

video_speed = 200

def run_video():
    
    try:
        session = cdata[cdata['session_id'] == session_id]
        lap = session[session['lap'] == lap_nro]
        bend = lap[(lap['dist'] >= BENDS[which_bend-1][0]) & 
                   (lap['dist'] <= BENDS[which_bend-1][1])]
        
        naksu_session = naksu_data[naksu_data['session_id'] == session_id]
        naksu_bend = naksu_session[(naksu_session['se_ts'] >= np.min(bend['se_frame_number'])) &
                                   (naksu_session['se_ts'] <= np.max(bend['se_frame_number']))]
    except Exception:
        sys.exit("Data missing")
    
    vid = NumpyGst(video_file)
    vid.seek(naksu_bend['video_ts'][0])
    
    # naksu coordinates
    naksu_xs = (np.around(naksu_bend['x'])).astype(np.int)
    naksu_ys = (np.around(naksu_bend['y'])).astype(np.int)
    naksu_ts = naksu_bend['video_ts']
    
    # scene camera coordinates
    visualisation = np.searchsorted(bend['se_frame_number'], naksu_bend['se_ts'])
    visualisation = bend[visualisation]    
    scam_xs = (np.around(visualisation['scenecam_x'])).astype(np.int)
    scam_ys = (np.around(visualisation['scenecam_y'])).astype(np.int)
    
    while True:
        frame = vid.next()
        video_time = frame.timestamp
        print "videotime", video_time
        #print "frametime", frametime
        
        #i = np.searchsorted(naksu_ts, video_time)
        try:
            i = np.flatnonzero(naksu_ts == video_time)[0]
        except IndexError:
            sys.exit("Can't find matching timestamp")
            
        print "naksutime", naksu_ts[i]
        print "car_se", visualisation[i]['se_frame_number']
        print "naksu_se", naksu_bend[i]['se_ts']
        print "scam coords %f %f" %(scam_xs[i], scam_ys[i])
        
        frame = cv.fromarray(frame.copy())
        cv.Circle(frame, (naksu_xs[i], naksu_ys[i]), 3, (0,0,255,100), 3)
        cv.Circle(frame, (scam_xs[i], scam_ys[i]), 3, (0,255,0,100), 3)
        cv.ShowImage('frame1', frame)
        key = cv.WaitKey(0)


if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        sys.exit("naksuvisu.py [video_file] [session_id] [bend_nro] [lap_nro]")
    
    video_file = sys.argv[1]
    session_id = int(sys.argv[2])
    which_bend = int(sys.argv[3])
    lap_nro = int(sys.argv[4])
    
    data, h5file = get_cleaned_data(DATA_FILE)
    cdata = h5file.root.ramppi11.rawdata[:]
    naksu_data = h5file.root.ramppi11.naksu_data[:]
    
    run_video()
    

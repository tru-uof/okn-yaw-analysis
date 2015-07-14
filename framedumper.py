import numpy as np
import sys, os 
import tables
import sync_smarteye_video
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE

from constants import DATA_FILE, SESSIONS, BENDS
from utils import get_segment_data

naksudir = '/home/thitkone/Projects/tru/ramppi14/naksu/13/'

dirs = {2013071500: '/home/thitkone/Projects/tru/ramppi13/data/20130715-mt-pilotti',
        2013071601: '/home/thitkone/Projects/tru/ramppi13/data/20130716-vh-kh1',
        2013071602: '/home/thitkone/Projects/tru/ramppi14/data/13/20130716-jv-kh2',
        2013071703: '/home/thitkone/Projects/tru/ramppi13/data/20130717-mk-kh3',
        2013071704: '/home/thitkone/Projects/tru/ramppi14/data/13/20130717-ha-kh4',
        2013071805: '/home/thitkone/Projects/tru/ramppi14/data/13/20130718-rt-kh5',
        2013071806: '/home/thitkone/Projects/tru/ramppi14/data/13/20130718-ts-kh6',
        2013070999: '/home/thitkone/Projects/tru/ramppi13/data/20130709-ol',
        2013101307: '/home/thitkone/Projects/tru/ramppi13/data/20131013-jr-kh7',
        2013101308: '/home/thitkone/Projects/tru/ramppi14/data/13/20131013-ml-kh8',
        2013101909: '/home/thitkone/Projects/tru/ramppi14/data/13/20131019-mn-kh9',
        2013111110: '/home/thitkone/Projects/tru/ramppi14/data/13/201311110-ja-kh10'
        }

def dump():
    h5file = tables.openFile(DATA_FILE, mode='r')
    data = h5file.root.ramppi11.rawdata[:]
    sids = SESSIONS[-3:-1]
    #sids = np.unique(data['session_id'])#[-2:]
    sbends = [BENDS[1], BENDS[3]]
    
    for sid in sids:
        
        datapath = dirs[sid]
        ls = os.listdir(datapath)
        vids = [file for file in ls if file.lower().endswith('.mp4')]
        print sid, vids
        if len(vids) > 1:
            vid = os.path.join(datapath, vids[1])
        else:
            vid = os.path.join(datapath, vids[0])
        frame_to_vid = sync_smarteye_video.se_timestamp_fit(vid)
        print frame_to_vid
        
        dir1 = os.path.join(naksudir, str(sid))
        if not os.path.exists(dir1):
            os.mkdir(dir1)
            
        for bend in sbends:
            dir2 = os.path.join(dir1, 'bend'+str(BENDS.index(bend)+1))
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            
            bdata = get_segment_data(sid, bend, data)
            laps = np.unique(bdata['lap'])
            
            for lap in laps:
                dir3 = os.path.join(dir2, 'lap'+str(lap))
                if not os.path.exists(dir3):
                    os.mkdir(dir3)
                    
                ldata = bdata[ bdata['lap'] == lap ]
                videots = np.polyval(frame_to_vid, ldata['se_frame_number'])
                #plt.figure()
                #plt.plot(ldata['se_frame_number'], videots, ',b')
                #plt.show()
                print lap, videots[0], videots[-1]
                duration = videots[-1] - videots[0]
                
                Popen("avconv -i " + vid + " -qscale 1 -ss " + str(videots[0]) + " -t " + 
                  str(duration) + " " + os.path.join(dir3, "image%5d.jpg"), 
                  stdout=PIPE, shell=True).communicate()[0]
            
    
    
    h5file.close()
    

if __name__ == '__main__':
    dump()

import sys, os
import numpy as np
import tru.rec
import tables

from utils import open_h5file, append2table, set_table
from constants import DATA_FILE_11, DATA_FILE_13, SCENE_CALIB_11, SCENE_CALIB_13

def add_scenecam(fname, cal_path):
    data_table = tables.openFile(fname, mode='r')
    data = data_table.root.ramppi11.rawdata[:]

    coords = []
    sid = np.nan
    for row in data:
        # change calibration file according to session
        direction = [ row['g_direction_x'], row['g_direction_y'], row['g_direction_z'] ]
        if sid != row['session_id']:
            sid = row['session_id']
            print sid
            
            fpath = os.path.join(cal_path, str(sid) + '.cal')
            
            if not os.path.exists(fpath):
                sys.exit('missing cal file: %s' %fpath)
            
            calibration = loadCalibration(fpath)
            
        result = LinearMappingXYZ2RK(calibration, direction)
        coords.append(result)

    if len(data) == len(coords):
        print 'writing...'
        x, y = zip(*coords)
        
        new_rec = tru.rec.append_field(data, 'scenecam_x', x, np.float)
        new_rec = tru.rec.append_field(new_rec, 'scenecam_y', y, np.float)
        
        newname = fname[:-5] + '_sc.hdf5'
        new_h5 = open_h5file(newname, 'ramppi11', mode="w")
        new_h5.createTable('/ramppi11', 'rawdata', new_rec)
        new_h5.close()
    else:
        print 'some data is missing - no output'
   
    data_table.close()
        

def loadCalibration(filename):
    with open(filename, 'r') as f:
        data = [float(x) for x in f.read().split()]
    return data
        
def LinearMappingXYZ2RK(calibration, direction):
    x = (calibration[0] * direction[0]
        + calibration[1] * direction[1]
        + calibration[2] * direction[2]
        + calibration[3])
    y = (calibration[4] * direction[0]
        + calibration[5] * direction[1]
        + calibration[6] * direction[2]
        + calibration[7])
    
    # add offset for screen capture
    x += 2
    y += 25
    
    # if coords 242 and 106, then signal stuck ....
    
    return (x,y)

if __name__ == '__main__':
    
    add_scenecam(DATA_FILE_13, SCENE_CALIB_13)

import numpy as np
import tables
import os
import sys
import subprocess

from tru.rec import append_field, rowstack
from PIL import Image, ImageOps
from StringIO import StringIO

from constants import DATA_FILE
#DATA_FILE = '/home/thitkone/trustore/Ramppi11/repo/integrated_plus.hdf5'

from utils import open_h5file, set_table

def add_to_h5(naksu_data, h5path):
    h5file = open_h5file(h5path, 'ramppi11', mode='a')
    set_table(h5file, 'ramppi11', 'naksu_data', naksu_data)
    h5file.close()    

def integrate_manual(fpath):
    sid_paths = os.listdir(fpath)
    sid_paths = np.sort(sid_paths)

    description = [('video_ts',np.float32), ('x', np.float32), ('y', np.float32), 
                    ('se_ts', np.int), ('session_id', np.int)]
    naksu_data = np.array([], dtype=description)

    for sid in sid_paths:        
        bend_paths = os.listdir(os.path.join(fpath,sid))
        for b in bend_paths:
            print sid, b
            lap_paths = os.listdir(os.path.join(fpath, sid, b))
            for l in lap_paths:
                files = os.listdir(os.path.join(fpath, sid, b, l))
                files = [os.path.join(fpath, sid, b, l, x) for x in files if x.lower().endswith('.mrk')]
                for f in files:
                    with open(f) as file:
                        fdata = np.genfromtxt(file, dtype=[('point', np.int), 
                                   ('x', np.float), ('y', np.float)])
                        if np.size(fdata) == 0: continue
                        fdata = fdata if np.size(fdata['x']) == 1 else fdata[0] 
                        
                        impath = f[:-4] + ".jpg"
                        img = Image.open(impath)
                        imstr = StringIO()
                        pos, size = (3, 24), (80, 24)
                        img = img.crop((pos[0], pos[1], pos[0]+size[0], pos[1]+size[1]))
                        img = ImageOps.invert(img)
                        img.save(imstr, format="PPM")
                        imstr = imstr.getvalue()

                        cmd = "gocr -C 0-9. -"
                        proc = subprocess.Popen(cmd, shell=True,
                            stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE)
                        timestamp = proc.communicate(imstr)[0]
                        timestamp = timestamp.strip()
                        timestamp = timestamp.strip('_')
                        try:
                            timestamp = float(timestamp)
                        except ValueError:
                            continue


                        d = np.array((0, fdata['x'], fdata['y'], timestamp, sid), dtype=description)
                        naksu_data = rowstack((naksu_data, d))
    
    naksu_data.sort(order=['session_id', 'se_ts'])                    
    return naksu_data
 

def integrate(fpath, poly):
        
    fnames = os.listdir(fpath)
    fnames = np.sort(fnames)
    
    description = [('video_ts',np.float32), ('x', np.float32), ('y', np.float32)]
    
    for i, fname in enumerate(fnames):
        session_file = os.path.join(fpath,fname)
        with open(session_file) as file:
            
            sid = int(fname[:-4])
            data = np.genfromtxt(file, dtype=description, delimiter=',', names=None)
            
            se_ts = (data['video_ts'] - poly[i][1]) / poly[i][0]
            se_ts = np.around(se_ts)
            se_ts = se_ts.astype(np.int)
            
            data = append_field(data, 'se_ts', se_ts)
            data = append_field(data, 'session_id', sid)
            
            if i == 0:
                naksu_data = data
            else:
                naksu_data = rowstack((naksu_data, data))
                #naksu_data = np.vstack((naksu_data, data))

    return naksu_data

# from carviz.apps.sync_smarteye_video
vts11 = [[0.01668534, 2.96801069], 
        [1.66843839e-02, -2.46916964e+03], 
        [0.01668573, 2.88136877], 
        [0.01668517, 2.97778914], 
        [0.01668551, 2.95642598], 
        [0.01668459, 3.25443409],
        [0.0166853, 2.94671961],
        [0.01668517, 2.97256686],
        [0.01668541, 3.21398337], 
        [0.01668437, 3.30599417],
        [0.01668526, 2.95370766], 
        [0.01668428, 3.34937475],
        [0.01668511, 3.07647741], 
        [0.01668516, 3.22689175],
        [0.01668523, 3.02151738], 
        [0.0166853, 4.08961772],
        [1.66851999e-02, -8.57353029e+02], 
        [0.01668558, 2.90329217],
        [0.01668578, 2.98952651],
        [0.01668503, 4.38129802]]

# in the same order as the sids
vts13 = [#[  1.66852128e-02,   4.86067501e+02], #01
        [  1.66849572e-02,  -9.44847247e+02], #02
        #[ 0.01668523,  3.08578862], #03
        [ 0.01668483,  4.42412374], #04
        [ 0.01668523,  3.14917173], #05
        [ 0.01668489,  4.46630083], #06, 
        #07 missing
        [ 1.66851598e-02,  -2.76282366e+01], #08
        [ 0.01668519,  3.30959399], #09
        [ 0.01668526,  3.98676738]] #10 

if __name__ == '__main__':
    
    # location of taikanaksu/naksu data (see ramppi.sh from lanedetect)
    fpath = sys.argv[1]
    h5path = DATA_FILE
    
    #naksu_data = integrate(fpath, vts13)
    naksu_data = integrate_manual(fpath)
    add_to_h5(naksu_data, h5path)

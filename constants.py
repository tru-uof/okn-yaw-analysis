
REPO11 = '/home/thitkone/Projects/tru/ramppi11/repo'
REPO1314 = '/home/thitkone/Projects/tru/ramppi14/repo'
DATA_FILE_11 = '/home/thitkone/Projects/tru/ramppi11/repo/integrated.hdf5'
DATA_FILE_13 = '/home/thitkone/Projects/tru/ramppi14/repo/13.hdf5'
SCENE_CALIB_11 = '/home/thitkone/Projects/tru/ramppi14/cal/11/'
SCENE_CALIB_13 = '/home/thitkone/Projects/tru/ramppi14/cal/13/'
AIKANAKSU_11 = '/home/thitkone/Projects/tru/ramppi14/aikanaksu/11/'
AIKANAKSU_13 = '/home/thitkone/Projects/tru/ramppi14/aikanaksu/13/'

DATA_FILE = DATA_FILE_13
AIKANAKSU = AIKANAKSU_13

# by distance
BENDS = [(258.35, 564.56), (929.78, 1270.18), (1319.62,1651.00), (2047.94, 2255.00)]

APPROACH = [(BENDS[0][0], 0), (BENDS[1][0], 1012.0), 
            (BENDS[2][0], 1404.0), (BENDS[3][0], 2088.0)]

ENTRY = [(APPROACH[0][1], 0), (APPROACH[1][1], 1062.0), 
         (APPROACH[2][1], 1454.0), (APPROACH[3][1], 2138.0)]

CORNERING = [(ENTRY[0][1], 0), (ENTRY[1][1], 1229.0), 
             (ENTRY[2][1], 1605.0), (ENTRY[3][1], 2224.0)]

EXIT = [(CORNERING[0][1], BENDS[0][1]), (CORNERING[1][1], BENDS[1][1]), 
        (CORNERING[2][1], BENDS[2][1]), (CORNERING[3][1], BENDS[3][1])]

# see utils.py get_cleaned_data()
SESSIONS11 = [2011080101, 
              2011080302, # huom: mutka 3: taikanaksu kokonaan huono
              2011080503,
              2011080504, # debut video missing 
              2011080805, 
              2011081006, 
              2011081107, 
              2011081508, # raining
              2011081509, # raining
              2011081710, # huom: mutka 3: taikanaksu kokonaan huono
              2011082411, 
              2011082512, # huom: mutka 3: 9 ekaa kierrosta pois (taikanaksu)
              2011082913, 
              2011082914, 
              2011083115, # huom: mutka 3: 7 ekaa kierrosta pois (taikanaksu)
              2011090216, 
              2011090817, 
              2011090918,
              2011091519, 
              2011092220, #raining
              2011101321]
           
SESSIONS13 = [#2013071601, # huono
              2013071602, 
              #2013071703, # huono
              2013071704,
              2013071805,
              2013071806, 
              #2013101307, # huono
              2013101308, # aurinko
              2013101909, # aurinko
              2013111110] # aurinko

SESSIONS = SESSIONS13

CONDITION_LAPS_13 = { '0': [5,6,7,8,13,14,15,16],
                      '2013101308': [5,6,7,8,13,14,15,16,17],
                      '2013111110': [5,6,7,8,9,14,15,16,17] }

PROTO_SESSION = 2011080302

import os
import re
import numpy as np
import pandas as pd

def read_plates_data(input_dir, ext='csv', skip=10, verbose=False):
    
    print('Read Data')
    
    if ext == 'csv':
        sep = ','
    else:
        sep = '\t'
    
    plates = {}
    for filename in os.listdir(input_dir):
        filename = re.sub(f"[.]{ext}", "", filename)
        plates[filename] = os.path.join(input_dir, filename + f'.{ext}')
    
    plates_dict = {}
    rownames = {}
    for plate in plates.keys():
        if verbose:
            print(plate)
        plates_dict[plate]=[]
        rownames[plate]=[]
        plate_rowcol=[]
        with open(plates[plate]) as fin:
            for _ in range(skip):
                linePre=fin.readline()
                
            plate_colnames = linePre.strip().split(sep) #capture colname line
            # check for trailing separator
            ncol0 = len(plate_colnames)
            n_left_padding = 0
            while plate_colnames[0] == '':
                n_left_padding += 1
                plate_colnames = plate_colnames[1:]
            n_right_padding = 0
            while plate_colnames[-1] == '':
                n_right_padding += 1
                plate_colnames = plate_colnames[:-1]

            while True:
                line=fin.readline()
                if line in '\n':
                    break
                ls = line.strip().split(sep)
                if ls[0] in '':
                    break
                out = [int(item) for item in ls[n_left_padding:len(ls)-n_right_padding]]
                plates_dict[plate].extend(out)
                rownames[plate].extend([ls[0]])
                plate_rowcol.extend([f'{ls[0]}-{item}' for item in plate_colnames])
    
    print('Merge Plates Data')
    
    plates_df = pd.DataFrame.from_dict(plates_dict)
    plates_df.index = plate_rowcol

    return plates_df



def read_plates_annot(input_dir, rowcol=None, ext='txt', skip=0, verbose=False):
    
    print('Read Annotation')
    
    if ext == 'csv':
        sep = ','
    else:
        sep = '\t'
    
    plates = {}
    for filename in os.listdir(input_dir):
        filename = re.sub(f"[.]{ext}", "", filename)
        plates[filename] = os.path.join(input_dir, filename + f'.{ext}')
    
    plates_dict = {}
    rownames = {}
    for plate in plates.keys():
        if verbose:
            print(plate)
        plates_dict[plate]=[]
        rownames[plate]=[]
        plate_rowcol=[]
        with open(plates[plate]) as fin:
            for _ in range(skip):
                next(fin)

            while True:
                line=fin.readline()
                if line in '\n':
                    break
                ls = line.strip().split(sep)
                plates_dict[plate].extend(ls)
        
    
    print('Merge Plates Data')
    plates_df = pd.DataFrame.from_dict(plates_dict)
    if rowcol is not None:
        plates_df.index = rowcol

    return plates_df
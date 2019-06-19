import numpy as np
import pandas as pd

resolution = 10 ** 4
loops_info = pd.read_csv('GSE63525_GM12878_primary+replicate_HiCCUPS_looplist.txt', sep='\t',
                             dtype = {'x1': np.long, 'x2': np.long, 'y1': np.long, 'y2': np.long})
loops_info[['x1', 'x2', 'y1', 'y2', 'centroid1', 'centroid2', 'radius']] = loops_info[['x1', 'x2', 'y1', 'y2', 'centroid1', 'centroid2', 'radius']].apply(
    lambda x : (x / resolution).astype('int64')
)

def filter_loops(loops, from_percentile = 0, to_percentile = 90):
	loops_from = np.percentile((loops['y1'] - loops['x1']), from_percentile)
	loops_to = np.percentile((loops['y1'] - loops['x1']), to_percentile)
	l = loops[(loops['y1'] - loops['x1']) > loops_from]
	loops_d = l[(l['y1'] - l['x1']) < loops_to]

	return loops_d

def prepare_data(image_size, chromosomes_idx = [1], is_log = False, is_norm_over_diagonals = True):
    images = []
    masks = []

    loops = filter_loops(loops_info)

    for chr_num in chromosomes_idx:
        chr = 'chr' + str(chr_num)
        
        path = 'arrs/10kb_' + chr
        if is_norm_over_diagonals:
            path += '_norm'
        if is_log:
            path += '_log'
        path += '.npy'

        arr = np.load(path)
   
        loops_positions = loops[loops_info.chr1 == str(chr_num)]

        # create mask array
        mask = np.zeros_like(arr)
        
        for index, row in loops_positions.iterrows():
            x1, x2, y1, y2 = row[['x1', 'x2', 'y1', 'y2']]
            mask[x1:y1, x2:y2] = 1

        i = image_size
        step = image_size
        while (i < arr.shape[0]):
            start, end = i - step, i

            area = arr[start:end, start:end]
            area[np.isnan(area)] = 0
            area[area == -np.inf] = 0
            area_mask = mask[start:end, start:end]

            i += step
            images.append(np.reshape(area, (image_size,image_size,1)))
            masks.append(np.reshape(area_mask, (image_size,image_size,1)))

        i -= step

        #get the last part of an array
        if i < arr.shape[0]:
            start, end = arr.shape[0] - step, arr.shape[0]

            area = arr[start:end, start:end]
            area[np.isnan(area)] = 0
            area[area == -np.inf] = 0
            area_mask = mask[start:end, start:end]

            images.append(np.reshape(area, (image_size,image_size,1)))
            masks.append(np.reshape(area_mask, (image_size,image_size,1)))

    X = np.array(images)
    y = np.array(masks)
        
    return X,y
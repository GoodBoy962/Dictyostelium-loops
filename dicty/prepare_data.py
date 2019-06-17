import numpy as np
import pandas as pd

loops_info = pd.read_excel('dicty_loop_positions_Chr1_Chr6.xlsx')
loops_x = loops_info['Genomic bin, Left base'].values - 1
loops_y = loops_info['Genomic bin, Right base'].values - 1

def substact_are_and_mask_and_save_to_tiff(start, end):
    '''
        got start and end position of an image
        substracting corresponding area from raw Hi-C array and loops mask and save it to tiff images
    '''
    area = arr[start:end, start:end]
    area_mask = mask[start:end, start:end]
    
    area_im = Image.fromarray(area)
    area_im.save('dataset/images/' + chr + '_' + str(start) + '_' + str(end) + '.png')
    
    area_mask_im = Image.fromarray(area_mask)    
    area_mask_im.save('dataset/masks/' + chr + '_' + str(start) + '_' + str(end) + 'label.png')

def prepare_data(image_size, isLog = False, isNormOverDiagonals = True):
    images = []
    masks = []

    for chr_num in [1, 6]:
        chr = 'chr' + str(chr_num)
        
        path = 'arrs/2kb_' + chr
        if isNormOverDiagonals:
            path += '_norm'
        if isLog:
            path += '_log'
        path += '.npy'

        arr = np.load(path)
        loops_positions = loops_info[loops_info.Chr == chr_num]

        # create mask array
        mask = np.zeros_like(arr)
        for idx, x in enumerate(loops_x):
            mask[x-1:loops_y[idx]-1, x-1:loops_y[idx]-1] = 1  

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
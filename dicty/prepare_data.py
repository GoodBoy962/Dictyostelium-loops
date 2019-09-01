import numpy as np
import pandas as pd

loops_info = pd.read_excel('dicty_loop_positions_Chr1_Chr5_Chr6.xlsx')
loops_info['x'] = loops_info['x'].apply(lambda x: int(x - 1))
loops_info['y'] = loops_info['y'].apply(lambda x: int(x - 1))


def substact_are_and_mask_and_save_to_tiff(start, end):
    """
        got start and end position of an image
        substracting corresponding area from raw Hi-C array and loops mask and save it to tiff images
    """
    area = arr[start:end, start:end]
    area_mask = mask[start:end, start:end]

    area_im = Image.fromarray(area)
    area_im.save('dataset/images/' + chr + '_' + str(start) + '_' + str(end) + '.png')

    area_mask_im = Image.fromarray(area_mask)
    area_mask_im.save('dataset/masks/' + chr + '_' + str(start) + '_' + str(end) + 'label.png')


def prepare_data(image_size, chromosomes=[1],
                 is_log=False,
                 is_norm_over_diagonals=True,
                 is_mask_corner_peak=False,
                 mask_window=1):
    images = []
    masks = []

    for chr_num in chromosomes:
        chr = 'chr' + str(chr_num)

        path = 'arrs/2kb_' + chr
        if is_norm_over_diagonals:
            path += '_norm'
        if is_log:
            path += '_log'
        path += '.npy'

        arr = np.load(path)
        loops = loops_info[loops_info.chr == chr_num]

        # create mask array
        mask = np.zeros_like(arr)

        for idx, loop in loops.iterrows():
            if not is_mask_corner_peak:
                mask[int(loop.x):int(loop.y), int(loop.x):int(loop.y)] = 1
            else:
                mask[int(loop.x - mask_window):int(loop.x + mask_window),
                int(loop.y - mask_window):int(loop.y + mask_window)] = 1
                mask[int(loop.y - mask_window):int(loop.y + mask_window),
                int(loop.x - mask_window):int(loop.x + mask_window)] = 1

        i = image_size
        step = image_size
        while i < arr.shape[0]:
            start, end = i - step, i

            area = arr[start:end, start:end]
            area[np.isnan(area)] = 0
            area[area == -np.inf] = 0
            area_mask = mask[start:end, start:end]

            i += int(step / 2)
            images.append(np.reshape(area, (image_size, image_size, 1)))
            masks.append(np.reshape(area_mask, (image_size, image_size, 1)))

        i -= step

        # get the last part of an array
        if i < arr.shape[0]:
            start, end = arr.shape[0] - step, arr.shape[0]

            area = arr[start:end, start:end]
            area[np.isnan(area)] = 0
            area[area == -np.inf] = 0
            area_mask = mask[start:end, start:end]

            images.append(np.reshape(area, (image_size, image_size, 1)))
            masks.append(np.reshape(area_mask, (image_size, image_size, 1)))

    images = np.array(images)
    masks = np.array(masks)

    return images, masks

# coding=utf-8
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mirnylib import numutils


##################################
# Parsing ########################
##################################

class CoolerParser:

    def __init__(self, cooler):
        self.cooler = cooler

    def generate_numpy_array_from_cooler_for_concrete_chromosome(self, chr, is_balanced):
        """
            Parse cooler to numpy array for the specific chromosome
            Get bins corresponding to the specific chromosome ch and translate then to array
        """
        bins = self.cooler.bins()[:]
        indexes = bins[bins.chrom == chr].index.values
        start = indexes[0]
        end = indexes[-1]
        mat = self.cooler.matrix(balance=is_balanced, sparse=True)[start:end, start:end]
        return mat.toarray()

    def generate_and_save_arrays_for_chromosome(self, path, chr,
                                                is_balanced=True,
                                                is_create_log=True,
                                                is_create_norm=True):
        """
            Parse and save cooler to numpy array for the specific chromosome
        """
        mat = self.generate_numpy_array_from_cooler_for_concrete_chromosome(chr, is_balanced)
        mat_norm = numutils.observedOverExpected(mat)

        resolution = str(int(self.cooler.info['bin-size'] / 10 ** 3)) + 'kb'

        is_not_balanced_arr = ''
        if not is_balanced:
            is_not_balanced_arr = '_not_balanced'

        np.save(path + resolution + '_' + chr + is_not_balanced_arr, mat)
        if is_create_log:
            np.save(path + resolution + '_' + chr + '_log' + is_not_balanced_arr, np.log(mat))
        if is_create_norm:
            np.save(path + resolution + '_' + chr + '_norm' + is_not_balanced_arr, mat_norm)
        if is_create_log and is_create_norm:
            np.save(path + resolution + '_' + chr + '_norm_log' + is_not_balanced_arr, np.log(mat_norm))


##################################
# Plots ##########################
##################################

def plot_prediction_HiC(raw, raw_mask, pred, pred_t, image_size, is_log_HiC=True, figsize=(10, 10), name=None):
    """
        Plot 4 figures
        raw - raw HiC
        raw_mask - mask of raw HiC
        pred - predicted probabilities
        pred_t - predicted with threshold
        image_size - size of the image
        is_log_hic - do or not to do log with raw HiC
    """

    raw = np.reshape(raw, (image_size, image_size))

    if is_log_HiC:
        raw = np.log(raw)

    raw_mask = np.reshape(raw_mask, (image_size, image_size))
    pred = np.reshape(pred, (image_size, image_size))
    pred_t = np.reshape(pred_t, (image_size, image_size))

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)

    im1 = axs[0, 0].matshow(raw, cmap='RdBu_r')
    axs[0, 0].set_title('Hi-C')
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].matshow(raw_mask, cmap='RdBu_r')
    axs[0, 1].set_title('Mask')
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[1, 0].matshow(pred, cmap='RdBu_r')
    axs[1, 0].set_title('Predicted probabilities')
    fig.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].matshow(pred_t, cmap='RdBu_r')
    axs[1, 1].set_title('Predicted probabilities with threshold')
    fig.colorbar(im4, ax=axs[1, 1])

    if name is not None:
        fig.savefig('pictures/' + name + '.png')


def plot_HiC(arr, figsize=(15, 15), name=None, is_loop=False, is_loop_window=False):
    """
        Plot Hi-С map in blue-red colormap
        Function returns ax to add smth in the figure if needed
        function add anchors on plot if loop is plotted
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.matshow(arr, cmap='RdBu_r')
    fig.colorbar(im)

    if is_loop:
        ax.axvline(x=math.floor(arr.shape[0] / 2), color='k', linestyle='--', lw=3)
        plt.text(-1.6, math.floor(arr.shape[0] / 2) - 1.8, 'left anchor', rotation=90, size=20)
        ax.axhline(y=math.floor(arr.shape[1] / 2), color='k', linestyle='--', lw=3)
        plt.text(math.floor(arr.shape[1] / 2) - 1.8, arr.shape[1] + 0.5, 'right anchor', size=20)

    if is_loop_window:
        ax.axvline(x=math.floor(arr.shape[0] / 3 + 1), color='k', linestyle='--', lw=3)
        ax.axvline(x=2 * math.floor(arr.shape[0] / 3 + 1), color='k', linestyle='--', lw=3)
        plt.text(-8, math.floor(arr.shape[0] / 3 + 1) - 5, 'left anchor', rotation=90, size=20)
        ax.axhline(y=math.floor(arr.shape[1] / 3 + 1), color='k', linestyle='--', lw=3)
        ax.axhline(y=2 * math.floor(arr.shape[1] / 3 + 1), color='k', linestyle='--', lw=3)
        plt.text(-8, 2 * math.floor(arr.shape[0] / 3 + 1) - 5, 'right anchor', rotation=90, size=20)

    if name is not None:
        fig.savefig('pictures/' + name + '.png')

    return ax


def get_loop_with_window(arr, loop_x_centroid, loop_y_centroid, window_size=13):
    """
        Substract loop as an array with center in loop_x_centroid, loop_y_centroid and window_size around it
        If loop cordinates are outside the array return zero array
    """
    # loop_window_size = math.ceil(window_size/2) + 1
    loop_window = np.zeros((window_size, window_size))

    if loop_x_centroid - math.floor(window_size / 2) > 0 and \
            loop_y_centroid - math.floor(window_size / 2) > 0 and \
            loop_x_centroid + math.floor(window_size / 2) < arr.shape[0] and \
            loop_y_centroid + math.floor(window_size / 2) < arr.shape[0]:
        loop_window = arr[loop_y_centroid - math.floor(window_size / 2):loop_y_centroid + math.ceil(window_size / 2),
                      loop_x_centroid - math.floor(window_size / 2):loop_x_centroid + math.ceil(window_size / 2)]

        ##loop_window[loop_window == -np.inf] = 0
        ##loop_window = np.nan_to_num(loop_window) 

    return loop_window


def z_norm_zero_middle(arr):
    mean = np.mean(arr)
    std = np.std(arr)

    z_normed_arr = (arr - mean) / std

    norm_arr = z_normed_arr - (np.max(z_normed_arr) + np.min(z_normed_arr)) / 2

    return norm_arr


def resize_image_arr(original_image, width, height):
    """
        resizing original image to image with (width, height) size
    """
    resized_image = np.zeros(shape=(width, height))
    for W in range(width):
        for H in range(height):
            new_width = int(W * original_image.shape[0] / width)
            new_height = int(H * original_image.shape[1] / height)
            resized_image[W][H] = original_image[new_width][new_height]

    # resized_image[resized_image == -np.inf] = 0
    resized_image = np.nan_to_num(resized_image)

    return resized_image


##################################
# Loop utils #####################
##################################
class LoopChromosomeContainer:

    def __init__(self, arr, loops_info):
        self.arr = arr
        self.loops_info = loops_info

    def sum_loop(self, window_size=13):
        """
            Get average of all loops by extracting each loop and summing them into one
        """
        loop_sum = np.zeros((window_size, window_size))
        for idx, loop in self.loops_info.iterrows():
            loop_window = np.nan_to_num(self.get_loop_with_window(idx))
            loop_sum = loop_sum + loop_window

        return loop_sum

    def avg_loop(self, window_size=13):
        """
            Get sum of all loops by extracting each loop and summing them into one and dividing by loops number
        """
        return self.sum_loop(window_size) / self.loops_info.shape[0]

    def sum_loop_resized(self, loop_new_size, is_window=False):
        """
            Get sum resized of all loops by extracting each loop and summing them into one and dividing by loops number
        """
        width, height = loop_new_size, loop_new_size
        resized_image_sum = np.zeros(shape=(width, height))

        for idx, loop in self.loops_info.iterrows():
            window = 0
            if is_window:
                window = int(loop.y - loop.x)

            if loop.x - window > 0 and loop.y + window < self.arr.shape[1]:
                original_image = np.nan_to_num(
                    self.arr[int(loop.x) - window:int(loop.y) + window, int(loop.x) - window:int(loop.y) + window])
                resized_image = resize_image_arr(original_image, width, height)
                resized_image_sum = resized_image_sum + resized_image

        return resized_image_sum

    def avg_loop_resized(self, loop_new_size, is_window=False):
        """
            Get average resized of all loops by extracting each loop and summing them into one and dividing by loops number
        """
        return self.sum_loop_resized(loop_new_size, is_window) / self.loops_info.shape[0]

    def get_loop_with_window(self, index, window_size=13):
        """
            Substract loop as an array with center in loop_x_centroid, loop_y_centroid and window_size around it
            If loop cordinates are outside the array return zero array
        """
        # loop_window_size = math.ceil(window_size/2) + 1
        loop_window = np.zeros((window_size, window_size))

        loop_x_centroid = self.loops_info.x[index]
        loop_y_centroid = self.loops_info.y[index]

        if loop_x_centroid - math.floor(window_size / 2) > 0 and \
                loop_y_centroid - math.floor(window_size / 2) > 0 and \
                loop_x_centroid + math.floor(window_size / 2) < self.arr.shape[0] and \
                loop_y_centroid + math.floor(window_size / 2) < self.arr.shape[0]:
            loop_window = self.arr[
                          loop_x_centroid - math.floor(window_size / 2):loop_x_centroid + math.ceil(window_size / 2),
                          loop_y_centroid - math.floor(window_size / 2):loop_y_centroid + math.ceil(window_size / 2)]

            ##loop_window[loop_window == -np.inf] = 0
            ##loop_window = np.nan_to_num(loop_window)

        return loop_window

    def get_mean_scaling_values(self):
        scaling_values = []

        for idx, loop in self.loops_info.iterrows():
            scaling = calc_scaling_mean(self.arr[int(loop.x):int(loop.y), int(loop.x):int(loop.y)])
            scaling_values.append(scaling)

        return scaling_values

    def get_sum_scaling_values(self):
        scaling_values = []

        for idx, loop in self.loops_info.iterrows():
            scaling = calc_scaling_sum(self.arr[int(loop.x):int(loop.y), int(loop.x):int(loop.y)])
            scaling_values.append(scaling)

        return scaling_values

    def get_interloops_mean_scaling_values(self):
        scaling_values = []

        begin = 0

        for idx, loop in self.loops_info.iterrows():
            if begin < loop.x:
                a = self.arr[int(begin):int(loop.x), int(begin):int(loop.x)]
                begin = loop.y
                scaling = calc_scaling_mean(a)
                scaling_values.append(scaling)

        return scaling_values

    def get_interloops_sum_scaling_values(self):
        scaling_values = []

        begin = 0

        for idx, loop in self.loops_info.iterrows():
            if begin < loop.x:
                a = self.arr[int(begin):int(loop.x), int(begin):int(loop.x)]
                begin = loop.y
                scaling = calc_scaling_sum(a)
                scaling_values.append(scaling)

        return scaling_values

    def calc_scaling_loop_plus_add_mean(self, idx, max_loop_size):
        x = self.loops_info.x[idx]
        y = self.loops_info.y[idx]

        loop = self.arr[x:y, x:y]
        scaling = calc_scaling_mean(loop)
        add = max_loop_size - (y - x) + 10

        if x - add > 0 and y + add < self.arr.shape[0]:
            after_loop = np.transpose(self.arr[y:y + add, x - add:x])
            after_scaling = np.flip(calc_scaling_mean(after_loop))

            s = np.append(scaling, after_scaling)

            return s
        else:
            return None

    def get_mean_scaling_values_plus_area(self, max_loop_size):
        scaling_values = []

        for idx, loop in self.loops_info.iterrows():
            d = self.calc_scaling_loop_plus_add_mean(idx, max_loop_size)
            if d is not None:
                scaling_values.append(d)

        mean = np.nanmean(scaling_values, axis=0)

        return scaling_values, mean

    def get_max_interloop(self):
        begin = 0
        max_interloop = 0

        for idx, loop in self.loops_info.iterrows():
            x = loop.x
            if begin < x and max_interloop < x - begin:
                max_interloop = x - begin
            begin = loop.y

        return max_interloop

    def calc_scaling_interloops_plus_add_mean(self, begin, end, max_interloop):

        loop = self.arr[int(begin):int(end), int(begin):int(end)]
        scaling = calc_scaling_mean(loop)

        add = max_interloop - (end - begin) + 10

        if begin - add > 0 and end + add < self.arr.shape[0]:
            after_a = np.transpose(self.arr[int(end):int(end + add), int(begin - add):int(begin)])
            after_scaling = np.flip(calc_scaling_mean(after_a))
            s = np.append(scaling, after_scaling)
            return s
        else:
            return None

    def get_interloops_mean_scaling_values_plus_area(self, max_interloop):
        scaling_values = []

        begin = 0

        for idx, loop in self.loops_info.iterrows():
            x = loop.x
            if begin < x and x - begin < 200:
                scaling = self.calc_scaling_interloops_plus_add_mean(begin, x, max_interloop)
                if scaling is not None:
                    scaling_values.append(scaling)
            begin = loop.y

        mean = np.nanmean(scaling_values, axis=0)

        return scaling_values, mean

    def get_loops_number(self):
        return self.loops_info.shape[0]


##################################
# Scaling ########################
##################################

def calc_scaling_mean(arr):
    """
        calculate mean of each diagonal
        mean of each diagonal corresponds for the scaling for each genomic size step
    """
    scaling = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        scaling[i] = np.nanmean(np.diagonal(arr, i))

    return scaling


def calc_scaling_sum(arr):
    """
        calculate sum of each diagonal
        mean of each diagonal corresponds for the scaling for each genomic size step
    """
    scaling = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        scaling[i] = np.nansum(np.diagonal(arr, i))

    return scaling


def plot_scaling_values(scaling_values, name=None):
    """
        plot all scaling values on a one plot in double-log coordinates
    """
    fig = plt.figure()

    for scaling in scaling_values:
        plt.plot(np.log(range(scaling.shape[0])), np.log(scaling), 'b')

    if name is not None:
        fig.savefig('pictures/scaling/' + name + '.png')

    plt.show()

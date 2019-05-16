import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mirnylib import numutils

##################################
## Parsing
##################################

def generate_numpy_array_from_cooler_for_concrete_chromosome(cooler, chr, is_balanced):
    '''
        Parse cooler to numpy array for the specific chromosome
        Get bins corresponding to the specific chromosome ch and translate then to array
    '''
    bins = cooler.bins()[:]

    bins_num = bins[bins.chrom == chr].shape[0]
    indx = bins[bins.chrom == chr].index.values
    start = indx[0]
    end = indx[-1]
    mat = cooler.matrix(balance=is_balanced, sparse=True)[start:end, start:end]
    return mat.toarray()

def generate_and_save_arrays_for_chromosome(path, cooler, chr, is_balanced = True):
    '''
        Parse and save cooler to numpy array for the specific chromosome
    '''
    mat = generate_numpy_array_from_cooler_for_concrete_chromosome(cooler, chr, is_balanced)
    mat_norm = numutils.observedOverExpected(mat)
    
    resolution = str(int(cooler.info['bin-size']/10**3)) + 'kb'

    is_not_balanced_arr = ''
    if not is_balanced:
        is_not_balanced_arr = '_not_balanced'
    
    np.save(path + resolution + '_' + chr + is_not_balanced_arr, mat)
    np.save(path + resolution + '_' + chr + '_log' + is_not_balanced_arr, np.log(mat))
    np.save(path + resolution + '_' + chr + '_norm' + is_not_balanced_arr, mat_norm)
    np.save(path + resolution + '_' + chr + '_norm_log' + is_not_balanced_arr, np.log(mat_norm))

##################################
## Plots
##################################

def plot_HiC(arr, figsize=(15,15), name=None):
    '''
        Plot Hi-С map in blue-red colormap
        Function returns ax to add smth in the figure if needed
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.matshow(arr, cmap='RdBu_r')
    fig.colorbar(im)

    if name is not None:
        fig.savefig('pictures/' + name + '.png')

    return ax

def get_loop_with_window(arr, loop_x_centroid, loop_y_centroid, window_size=13):
    '''
        Substract loop as an array with center in loop_x_centroid, loop_y_centroid and window_size around it
        If loop cordinates are outside the array return zero array
    '''     
    loop_window_size = math.ceil(window_size/2) + 1
    loop_window = np.zeros((loop_window_size, loop_window_size))
    
    if loop_x_centroid-math.floor(window_size/2) > 0 and loop_y_centroid-math.floor(window_size/2) > 0:

        loop_window = arr[loop_y_centroid-math.floor(window_size/2):loop_y_centroid+math.ceil(window_size/2), 
                          loop_x_centroid-math.floor(window_size/2):loop_x_centroid+math.ceil(window_size/2)]
        
        ##loop_window[loop_window == -np.inf] = 0
        ##loop_window = np.nan_to_num(loop_window) 

    return loop_window

def z_norm_zero_middle(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    
    z_normed_arr = (arr - mean)/std
    
    norm_arr = z_normed_arr - (np.max(z_normed_arr) + np.min(z_normed_arr))/2
    
    return norm_arr

def resize_image_arr(original_image, width, height):
    '''
        resizing original image to image with (width, height) size
    '''
    resized_image = np.zeros(shape=(width,height))
    for W in range(width):
        for H in range(height):
            new_width = int( W * original_image.shape[0] / width )
            new_height = int( H * original_image.shape[1] / height )
            resized_image[W][H] = original_image[new_width][new_height]
            
    #resized_image[resized_image == -np.inf] = 0
    resized_image = np.nan_to_num(resized_image) 

    return resized_image

##################################
## Scaling
##################################

def calc_scaling_mean(arr):
    '''
        calculate mean of each diagonal
        mean of each diagonal corresponds for the scaling for each genomic size step
    '''
    scaling = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        scaling[i] = np.nanmean(np.diagonal(arr, i))
        
    return scaling

def calc_scaling_sum(arr):
    '''
        calculate sum of each diagonal
        mean of each diagonal corresponds for the scaling for each genomic size step
    '''
    scaling = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        scaling[i] = np.nansum(np.diagonal(arr, i))
        
    return scaling

def plot_scaling_values(scaling_values, name=None):
    '''
    plot all scaling values on a one plot in double-log coordinates
    '''
    fig = plt.figure()
    ax = plt.axes()

    for scaling in scaling_values:
        plt.plot(np.log(range(scaling.shape[0])), np.log(scaling), 'b')

    if name is not None:
        fig.savefig('pictures/' + name + '.png')

    plt.show()
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cooler

from mirnylib import numutils

def generate_numpy_array_from_cooler_for_concrete_chromosome(cooler, ch):
    '''
        Parse cooler to numpy array for concrete chromosome
        Get bins corresponding to concrete chromosome ch and translate then to array
    '''
    bins = cooler.bins()[:]

    bins_num = bins[bins.chrom == chr].shape[0]
    indx = bins[bins.chrom == chr].index.values
    start = indx[0]
    end = indx[-1]
    mat = cooler.matrix(balance=True, sparse=True)[start:end, start:end]
    return mat.toarray()


def generate_and_save_arrays_for_chromosome(cooler, chr):
    mat = generate_numpy_array_from_cooler_for_concrete_chromosome(c, chr)
    mat_norm = numutils.observedOverExpected(mat)
    
    resolution = str(int(c.info['bin-size']/10**3)) + 'kb'
    
    np.save(resolution + '_' + chr, mat)
    np.save(resolution + '_' + chr + '_log', np.log(mat))
    np.save(resolution + '_' + chr + '_norm', mat_norm)
    np.save(resolution + '_' + chr + '_norm_log', np.log10(mat_norm))


resolution = 2000
filepath = 'Dicty_0A.1000.cool.multires::resolutions/' + str(resolution)
c = cooler.Cooler(filepath)


for chr in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6']:
    generate_and_save_arrays_for_chromosome(c, chr)


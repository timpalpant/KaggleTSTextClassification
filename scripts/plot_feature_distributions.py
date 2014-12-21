#!/usr/bin/env python

'''
Plot distribution of each feature,
conditioned on its bfeature type
'''

import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import *
from information import utils
from scipy.stats import itemfreq

nbins = 100

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=load_npz,
        help='Training data features (npz)')
    parser.add_argument('output',
        help='Output file with plots (pdf)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    pdf = PdfPages(args.output)

    dfs = args.features['ifeatures']
    cfs = args.features['ffeatures']
           
    print "Plotting float features"
    bfs = args.features['bfeatures']
    u = utils.unique_rows(bfs)
    indices = [np.all(bfs==ui, axis=-1) for ui in u]
    for j, f in enumerate(cfs.T):
        print "...ffeature %d" % j
        fig = plt.figure()
        h = np.zeros(nbins)
        not_nan = f[np.logical_not(np.isnan(f))]
        f_min = not_nan.min()
        f_max = not_nan.max()
        x = np.linspace(f_min, f_max, nbins)
        dx = (f_max - f_min) / nbins
        for idx in indices:
            h_new, bins = np.histogram(f[idx], range=(f_min, f_max), bins=nbins)
            plt.bar(x, h_new, bottom=h, width=dx)
            h += h_new
        plt.xlim(f_min, f_max)
        plt.xlabel('f')
        plt.ylabel('P(f)')
        plt.title('FFeature %d. # NaN = %d' % (j, np.sum(np.isnan(f))))
        pdf.savefig(fig)
        plt.close()
        
    print "Plotting integer features"
    for j, x in enumerate(dfs.T):
        print "...dfeature %d" % j
        freq = itemfreq(x)
        fig = plt.figure()
        xu = np.sort(np.unique(x))
        h = np.zeros_like(xu)
        for idx in indices:
            f = itemfreq(x[idx])
            h_new = np.zeros_like(h)
            h_new[f[:,0]] = f[:,1]
            plt.bar(xu, h_new, bottom=h)
            h += h_new
        plt.xlabel('f')
        plt.ylabel('P(f)')
        plt.title('DFeature %d' % j)
        pdf.savefig(fig)
        plt.close()
    
    pdf.close()

#!/usr/bin/env python

'''
Compute mutual information between individual features
and labels
'''

import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import *

def opts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=load_npz,
        help='Training data features (npz)')
    parser.add_argument('labels', type=load_npz,
        help='Training data labels (npz)')
    parser.add_argument('output',
        help='Output file with plots (pdf)')
    return parser

if __name__ == "__main__":
    args = opts().parse_args()
    print "Loading labels"
    labels = args.labels['labels']
    header = args.labels['header']
    pdf = PdfPages(args.output)
    
    #print "Plotting boolean features conditioned on labels"
    #bf = args.features['bfeatures']
    #n = bf.shape[1]
    #m = np.zeros((n,11))
    #m[:,0] = np.sum(bf==-1, axis=0)
    #m[:,1] = np.sum(bf==0, axis=0)
    #m[:,2] = np.sum(bf==1, axis=0)
    #fig = plt.figure()
    #pdf.savefig(fig)
    #plt.close()
        
    print "Plotting float features conditioned on labels"
    ff = args.features['ffeatures']
    n = ff.shape[1]
    x = np.arange(n)
    for i, l in enumerate(labels.T):
        print "label %d" % i
        for j, f in enumerate(ff.T):
            print "...ffeature %d" % j
            fig = plt.figure()
            plt.hist(f[l], normed=True, label='P(f | l)',
                     color='blue', alpha=0.4, 
                     range=(f.min(),f.max()), bins=25)
            plt.hist(f[np.logical_not(l)], normed=True, label='P(f | ~l)',
                     color='green', alpha=0.4, 
                     range=(f.min(),f.max()), bins=25)
            plt.xlim(f.min(), f.max())
            plt.xlabel('f')
            plt.ylabel('P(f)')
            plt.title('FFeature %d, Label %s' % (j, header[i]))
            pdf.savefig(fig)
            plt.close()
    
    pdf.close()
import numpy as np
import matplotlib.pyplot as plt
import collections


def sel_idx_summary(selectivity_idx, bins=10):
    # TODO: change the map color of bars
    # TODO: check what kind of index it is. For example, for symmetry plot only the avg

    for k, v in selectivity_idx.items():
        N = len(v)
        pos = 0

        for l in v:
            num_f = len(l)
            counts, bins = np.histogram(l, bins=bins, range=(0, 1))
            prc = np.zeros(len(counts))

            for i in xrange(len(counts)):
                prc[i] = float(counts[i])/num_f*100.

            y_offset = 0

            bars = []
            for i in xrange(len(prc)):
                p = plt.bar(pos, prc[i], bottom=y_offset, width=0.35)
                bars.append(p)
                y_offset = y_offset+prc[i]
            pos += 1

        xticks = []
        for i in xrange(N):
            xticks.append('Layer ' + str(i + 1))
        plt.xticks(np.arange(N), xticks)
        plt.yticks(np.arange(0, 101, 10))

        labels = [str(bins[i]) + ':' + str(bins[i+1]) for i in xrange(len(prc))]

        plt.ylabel('% of Neurons')
        plt.title(k + ' selectivity')
        plt.legend(bars, labels, bbox_to_anchor=(1.02, 1.02), loc=2)
        plt.subplots_adjust(right=0.75)
        plt.show()




def main():
    sel_idx = dict()
    sel_idx['color'] = []
    sel_idx['color'].append(np.random.rand(96))
    sel_idx['color'].append(np.random.rand(128))

    sel_idx['symmetry'] = []
    a = []
    for i in xrange(96):
        a.append(np.random.rand(5))
    sel_idx['symmetry'].append(a)
    a = []
    for i in xrange(128):
        a.append(np.random.rand(5))
    sel_idx['symmetry'].append(a)


    sel_idx_summary(sel_idx)



if __name__ == '__main__':
    main()
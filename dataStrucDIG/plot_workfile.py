#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

path_file = sys.argv[1]

with open(path_file) as f:
    data = f.read()

    data = data.split('\n')[1:]
    data2 = data.pop()

    y = [row.split(' ')[0] for row in data]
    x = [row.split(' ')[1] for row in data]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.set_title("Density on DIG2")    
    ax1.set_xlabel('number of intervals')
    ax1.set_ylabel('density')
    ax1.set_ylim([0,1])

    ax1.plot(x,y, c='r', label='the data')

    leg = ax1.legend()

    plt.show()

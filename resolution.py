#!/usr/bin/env python
"""
Read drs4 files and do their waveforms

Usage:
    scifi-resolution --output_dir=PATH --method=STR [--debug] <INPUT>

Options:
    -h -help                    Show this screen.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --method=STR                method of time interpolation : time_of_max, first_peak, rising, const_frac, template
    -v --debug                  Enter the debug mode.
"""

from drs4 import DRS4BinaryFile
from readout import read_data
from readout import discard_event

import os
import re
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import time

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from docopt import docopt
from digicampipe.utils.docopt import convert_text
from histogram.histogram import Histogram1D


def gaussian_func(x, A, mu, sigma):
    gauss_x = A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) / (np.sqrt(2) * sigma)

    return gauss_x


def fit_gaussian(data, bin_centers, entries):

    mu, sigma = norm.fit(data)
    max_entry = np.max(entries)

    popt, pcov = curve_fit(gaussian_func, xdata=bin_centers, ydata=entries, p0=[max_entry, mu, sigma])
    p_sigma = np.sqrt(np.diag(pcov))

    y_fit = gaussian_func(data, *popt)
    chisq = np.sum((data - y_fit)**2 / y_fit)

    print("A : {} ± {}".format(popt[0], p_sigma[0]))
    print("mu : {} ± {}".format(popt[1], p_sigma[1]))
    print("sigma : {} ± {}".format(popt[2], p_sigma[2]))
    print("chi2 : {}".format(chisq))

    # return popt, p_sigma, chisq
    return popt, p_sigma


def get_label(method):

    if method == "time_of_max":
        label = "Time of maximum"
    elif method == "first_peak":
        label = "First peak"
    elif method == "rising":
        label = "Rising linear fit"
    elif method == "const_frac":
        label = "Constant fraction"
    elif method == "template":
        label = "Template"
    else:
        label = "none"
        print('Wrong method, chose among :')
        print('time_of_max, first_peak, rising, const_frac, template')
        exit(-1)

    return label


def draw_timing_histograms(timing_vector, output_dir, method):

    label = get_label(method)
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    histo_attribute = []
    binning = "auto"
    # Triggers
    entries, bins, patches = axs[0, 0].hist(timing_vector[1], bins=binning, color='tab:red', label="Ch. 0 : Trigger", histtype='step')
    histo_attribute.append([entries, bins, patches])
    axs[0, 0].legend()
    axs[0, 0].set_title(label)
    entries, bins, patches = axs[0, 1].hist(timing_vector[3], bins=binning, color='tab:red', label="Ch. 2 : Trigger", histtype='step')
    histo_attribute.append([entries, bins, patches])
    axs[0, 1].set_title(label)
    axs[0, 1].legend()
    # Signals
    entries, bins, patches = axs[1, 0].hist(timing_vector[0], bins=binning, color='tab:green', label="Ch. 1 : Signal", histtype='step')
    histo_attribute.append([entries, bins, patches])
    axs[1, 0].set_title(label)
    axs[1, 0].legend()
    entries, bins, patches = axs[1, 1].hist(timing_vector[2], bins=binning, color='tab:green', label="Ch. 3 : Signal", histtype='step')
    histo_attribute.append([entries, bins, patches])
    axs[1, 1].set_title(label)
    axs[1, 1].legend()

    axs[1, 0].set_xlabel(r'Time [ns]')
    axs[1, 1].set_xlabel(r'Time [ns]')
    axs[0, 0].set_ylabel('Counts')
    axs[1, 0].set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig('{}/timing_dist_{}.png'.format(output_dir, method))

    return fig, axs, histo_attribute


def draw_delta_histograms(delta_vector, output_dir, method):

    label = get_label(method)
    fig, axs = plt.subplots(1, 2, sharey=True)

    binning = 125
    color_list = ['tab:red', 'tab:green']
    label_list = [r"$\Delta t_{Trigger} = t_{Ch.0} - t_{Ch.2}$", r"$\Delta t_{Signal} = t_{Ch.1} - t_{Ch.3}$"]
    n = 6
    # the histogram attribute vector is : histo_attribute = [entries, bins, patches]
    histo_attribute = []
    for k in range(len(delta_vector)):
        mu, sigma = norm.fit(delta_vector[k])
        entries, bins, patches = axs[k].hist(delta_vector[k], bins=binning, color=color_list[k], label=label_list[k], histtype='step', range=(mu - n * sigma, mu + n * sigma))
        histo_attribute.append([entries, bins, patches])
        axs[k].set_title(label)
        axs[k].legend()
        axs[k].set_xlabel(r'$\Delta$t [ns]')
        if k == 0:
            axs[k].set_ylabel('Counts')

    plt.savefig('{}/timing_ts_dist_{}.png'.format(output_dir, method))

    return fig, axs, histo_attribute


def entry():

    args = docopt(__doc__)
    file = args['<INPUT>']
    output_dir = convert_text(args['--output_dir'])
    method = convert_text(args['--method'])
    debug = args['--debug']

    print('Output dir : ', output_dir)
    print('File :', file)
    print('Debug :', debug)

    start_time = time.time()
    channels = 4
    event_counter = 0
    discarded_events = 0
    waveforms = read_data(file=file, debug=False)

    if method == 'time_of_max':

        # Waveform was set from -0.05 to 0.95 mV
        max_timing = []
        for time_widths, times, adc_data, event_rc in waveforms:
            if event_counter % 10000 == 0:
                print("event number : ", event_counter)
            temp_max_timing = []

            if discard_event(amplitude_matrix=adc_data, min_threshold=0.015) is True:
                discarded_events += 1
                continue
            else:
                for k in range(channels):
                    temp_time = times[k][np.argmax(adc_data[k])]
                    temp_max_timing.append(temp_time)
                if any(x > 180 for x in temp_max_timing) or any(x < 150 for x in temp_max_timing):
                    discarded_events += 1
                    continue

            max_timing.append(temp_max_timing)

            # if event_counter == 100000:
            if event_counter == 1000:
                break
            event_counter += 1

        max_timing = np.array(max_timing).T

        delta_signal = max_timing[2] - max_timing[0]
        delta_trigger = max_timing[3] - max_timing[1]
        delta_vector = [delta_trigger, delta_signal]
        _, _, _ = draw_timing_histograms(timing_vector=max_timing, output_dir=output_dir, method=method)
        fig, axs, histo_delta_attributes = draw_delta_histograms(delta_vector=delta_vector, output_dir=output_dir, method=method)

        for k in range(len(delta_vector)):
            entries = histo_delta_attributes[k][0]
            bins = histo_delta_attributes[k][1]
            bin_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
            popt, pcov = fit_gaussian(delta_vector[k], bin_centers, entries)
            time_v = np.linspace(bins.min(), bins.max(), 100)

            axs[k].plot(time_v, gaussian_func(time_v, *popt), 'b--', linewidth=1)

        axs[0].set_xlabel(r'Time [ns]')
        axs[1].set_xlabel(r'Time [ns]')
        axs[0].set_ylabel('Counts')
        plt.savefig('{}/timing_ts_dist_{}.png'.format(output_dir, method))
        plt.close(fig)

        print("events : {}".format(event_counter - discarded_events))

    end_time = time.time()
    print("Total execution time : ", end_time - start_time)


if __name__ == '__main__':
    entry()
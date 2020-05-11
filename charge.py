#!/usr/bin/env python
"""
Computed charge related computations

Usage:
    scifi-charge --output_dir=PATH --n_channels=INT [--debug] <INPUT>

Options:
    -h -help                    Show this screen.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --n_channels=INT            Number of total channels used in the DRS4 configuration. [Default: 4]
    -v --debug                  Enter the debug mode.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText

from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text
from histogram.histogram import Histogram1D

from readout import read_data
from readout import get_float_waveform
from readout import draw_waveforms


def sum_waveforms(file, n_channels):
    print('Get Summed Waveforms')

    waveforms = read_data(file)
    wave_sum = np.zeros((n_channels, 1024))
    n_events = 0
    for time_widths, times, adc_data, event_range_center in waveforms:

        if n_events % 10000 == 0:
            print('Event {}'.format(n_events))

        adc_data = get_float_waveform(adc_data, event_range_center)
        adc_data = np.array(adc_data)
        wave_sum += adc_data

        n_events += 1

    print('Total number of waveforms : ', n_events)

    return times, wave_sum, n_events


def get_average_baseline(file, n_channels):
    print('Compute Average Baseline')

    times, wave_sum, n_events = sum_waveforms(file, n_channels)
    # Getting the index of the maximum
    idx = np.argmax(wave_sum, axis=1)
    baseline = []
    for channel in range(len(wave_sum)):
        baseline.append(np.mean(wave_sum[channel][: idx[channel] - 100] / n_events))
        print('Ch {} : baseline = {} and idx of maximum summed = {}'.format(channel, baseline[channel], idx[channel]))
    baseline = np.array(baseline)

    return baseline, idx, n_events


def compute_amplitude_charge(file, baseline, idx_of_max, total_events):
    print('Compute Amplitude Charge')

    n_channels = len(baseline)
    charge = np.zeros((n_channels, total_events))
    window = [50, 50]
    waveforms = read_data(file)
    n_events = 0
    for time_widths, times, adc_data, event_range_center in waveforms:

        if n_events % 10000 == 0:
            print('Event {}'.format(n_events))

        adc_data = np.array(get_float_waveform(adc_data, event_range_center))

        for k in range(len(baseline)):
            adc_data[k] -= baseline[k]
            amplitude = np.max(adc_data[k][idx_of_max[k] - window[0]: idx_of_max[k] + window[1]])
            charge[k][n_events] = amplitude

        n_events += 1

    return charge


def compute_integral_charge(file, baseline, idx_of_max, total_events):
    print('Compute Integral Charge')

    n_channels = len(baseline)
    charge = np.zeros((n_channels, total_events))
    window = [200, 200]
    waveforms = read_data(file)
    n_events = 0
    for time_widths, times, adc_data, event_range_center in waveforms:

        if n_events % 10000 == 0:
            print('Event {}'.format(n_events))

        adc_data = np.array(get_float_waveform(adc_data, event_range_center))

        for k in range(len(baseline)):
            adc_data[k] -= baseline[k]
            integral = np.sum(adc_data[k][idx_of_max[k] - window[0]: idx_of_max[k] + window[1]])
            charge[k][n_events] = integral

        n_events += 1

    return charge


def make_and_draw_charge_histo(data, bins, label):

    bin_edges = np.linspace(np.min(data), np.max(data) + 1, bins)
    histogram = Histogram1D(bin_edges=bin_edges)
    histogram.fill(data)

    # Histogram display
    fig, ax = plt.subplots()
    histogram.draw(axis=ax, label=label, legend=False)
    text = histogram._write_info(())
    anchored_text = AnchoredText(text, loc=5)
    ax.add_artist(anchored_text)

    # Formatting to use in the digicampipe fitting single mpe method
    histogram.data = histogram.data.reshape((1, 1, -1))
    histogram.overflow = histogram.overflow.reshape((1, -1))
    histogram.underflow = histogram.underflow.reshape((1, -1))

    return fig, ax, histogram


def entry():

    args = docopt(__doc__)
    file = args['<INPUT>']
    output_dir = convert_text(args['--output_dir'])
    n_channels = convert_int(args['--n_channels'])
    debug = args['--debug']

    print('output_dir : ', output_dir)
    print('File :', file)

    baseline, idx, total_events = get_average_baseline(file, n_channels)
    integral_charge = compute_integral_charge(file, baseline, idx, total_events)
    amplitude_charge = compute_amplitude_charge(file, baseline, idx, total_events)

    pdf_integral_charge = PdfPages('{}/integral_charge.pdf'.format(output_dir))
    pdf_amplitude_charge = PdfPages('{}/amplitude_charge.pdf'.format(output_dir))

    bins = 2000
    for channel in range(n_channels):
        fig, ax, histo = make_and_draw_charge_histo(data=integral_charge[channel],
                                                    bins=bins,
                                                    label='Ch. {} : Integral charge'.format(channel))
        histo.save(os.path.join(output_dir, 'integral_charge_ch{}.fits'.format(channel)))
        pdf_integral_charge.savefig(fig)
        if debug:
            plt.show()
        plt.close(fig)
        del fig, ax, histo

        fig, ax, histo = make_and_draw_charge_histo(data=amplitude_charge[channel],
                                                    bins=bins,
                                                    label='Ch. {} : Amplitude charge'.format(channel))
        histo.save(os.path.join(output_dir, 'amplitude_charge_ch{}.fits'.format(channel)))
        pdf_amplitude_charge.savefig(fig)
        if debug:
            plt.show()
        plt.close(fig)
        del fig, ax, histo

    pdf_integral_charge.close()
    pdf_amplitude_charge.close()

    # Draw summed waveforms
    times, wave_sum, total_events = sum_waveforms(file, n_channels)
    for channel in range(n_channels):
        if channel == 0:
            labels_sum = ['Ch. {} : Sum'.format(channel)]
        else:
            labels_sum.append('Ch. {} : Sum'.format(channel))

    fig, ax = draw_waveforms(times, wave_sum, labels_sum)
    pdf_summed_waveforms = PdfPages('{}/summed_waveforms.pdf'.format(output_dir))
    pdf_summed_waveforms.savefig(fig)
    pdf_summed_waveforms.close()
    if debug:
        plt.show()
    plt.close(fig)

    print('End charge')


if __name__ == '__main__':
    entry()

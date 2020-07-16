#!/usr/bin/env python
"""
Read drs4 files and do their waveforms

Usage:
    scifi-readout --output_dir=PATH [--debug] <INPUT>

Options:
    -h -help                    Show this screen.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    -v --debug                  Enter the debug mode.
"""

import numpy as np
import matplotlib.pyplot as plt
from drs4 import DRS4BinaryFile

import os
import re
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from docopt import docopt
from digicampipe.utils.docopt import convert_text


def give_list_of_file(input_dir):

    files = [f for f in os.listdir(input_dir) if
             os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]

    files.sort()

    for i, file in enumerate(files):
        files[i] = os.path.join(input_dir, file)

    return files


def read_data(file, debug=False):

    with DRS4BinaryFile(file) as f:
        board_id = f.board_ids[0]
        active_channels = f.channels

        time_widths = np.array([f.time_widths[board_id][1],
                                f.time_widths[board_id][2],
                                f.time_widths[board_id][3],
                                f.time_widths[board_id][4]])

        times = np.array([np.cumsum(time_widths[0]),
                          np.cumsum(time_widths[1]),
                          np.cumsum(time_widths[2]),
                          np.cumsum(time_widths[3])])

        # times = np.insert(times, 0, 0, axis=1)

        for event in f:

            event_id = event.event_id
            event_timestamp = event.timestamp
            event_rc = event.range_center
            event_scalers = np.array([event.scalers[board_id][1],
                                      event.scalers[board_id][2],
                                      event.scalers[board_id][3],
                                      event.scalers[board_id][4]])
            event_trigger_cell = event.trigger_cells[board_id]

            adc_data = np.array([event.adc_data[board_id][1],
                                 event.adc_data[board_id][2],
                                 event.adc_data[board_id][3],
                                 event.adc_data[board_id][4]])

            # adc_data = np.insert(adc_data, 0, 0, axis=1)

            if debug is True:
                print('board id : ', board_id)
                print('active channels : ', active_channels[board_id])
                print('event id :', event_id)
                print('event timestamp :', event_timestamp)
                print('event range center :', event_rc)
                print('event scaler :', event_scalers)
                print('event trigger cell :', event_trigger_cell)

            #yield board_id, active_channels, time_widths, event_id, event_timestamp, event_range_center, event_scalers, event_trigger_cell, adc_data
            yield time_widths, times, adc_data, event_rc


def get_float_waveform(adc_waveform_matrix, range_center):
    """
    :param adc_waveform_matrix:     Matrix containing the ADC counts as amplitude of the n channels of the DRS4 board.
                                    Each row is an ADC waveform.
    :param range_center:            Range center given by the board in mV (needs to be divided by 1000 be in V)
    :return:                        Matrix containing the amplitudes of the n channels of the DRS4 board in Volts
    """
    # Get the range according to the range center defined
    amplitude_low_edge = -0.5 + range_center/1000.
    # Amplitude in volts
    amplitude = adc_waveform_matrix/(np.power(2, 16)-1) + amplitude_low_edge

    return amplitude


def draw_waveforms(times, adc_data, labels):

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # Triggers
    axs[0, 0].plot(times[1], adc_data[1], color='tab:red', label=labels[1])
    axs[0, 0].legend()
    axs[0, 1].plot(times[3], adc_data[3], color='tab:red', label=labels[3])
    axs[0, 1].legend()
    # Signals
    axs[1, 0].plot(times[0], adc_data[0], color='tab:green', label=labels[0])
    axs[1, 0].legend()
    axs[1, 1].plot(times[2], adc_data[2], color='tab:green', label=labels[2])
    axs[1, 1].legend()

    axs[1, 0].set_xlabel(r'Time [ns]')
    axs[1, 1].set_xlabel(r'Time [ns]')
    axs[0, 0].set_ylabel('Amplitude [V]')
    axs[1, 0].set_ylabel('Amplitude [V]')

    return fig, axs


def entry():

    args = docopt(__doc__)
    file = args['<INPUT>']
    output_dir = convert_text(args['--output_dir'])
    debug = args['--debug']

    print('Output dir : ', output_dir)
    print('File :', file)
    print('Debug :', debug)

    start_time = time.time()

    waveforms = read_data(file=file, debug=False)
    # Waveform was set from -0.05 to 0.95 mV
    bins_y = np.linspace(-0.10, 1.0, 221)
    bin_edges = []

    event_counter = 0

    for time_widths, times, adc_data, event_rc in waveforms:

        if event_counter is 0:
            times = np.insert(times, 0, 0, axis=1)
            times = np.delete(times, -1, axis=1)
            bin_edges = time_widths / 2 + times

        if event_counter % 1000 == 0:
            print('Event id :', event_counter)

        adc_data = get_float_waveform(adc_data, event_rc)

        threshold = 15./1000.
        if (all(x < threshold for x in adc_data[1]) or all(x < threshold for x in adc_data[3])) is True:
            continue
        else:
            temp_histo2d_ch0, xedges_ch0, yedges_ch0 = np.histogram2d(times[0], adc_data[0], bins=[bin_edges[0], bins_y])
            temp_histo2d_ch1, xedges_ch1, yedges_ch1 = np.histogram2d(times[1], adc_data[1], bins=[bin_edges[1], bins_y])
            temp_histo2d_ch2, xedges_ch2, yedges_ch2 = np.histogram2d(times[2], adc_data[2], bins=[bin_edges[2], bins_y])
            temp_histo2d_ch3, xedges_ch3, yedges_ch3 = np.histogram2d(times[3], adc_data[3], bins=[bin_edges[3], bins_y])

        if event_counter == 0:
            histo2d_ch0 = np.zeros_like(temp_histo2d_ch0)
            histo2d_ch1 = np.zeros_like(temp_histo2d_ch1)
            histo2d_ch2 = np.zeros_like(temp_histo2d_ch2)
            histo2d_ch3 = np.zeros_like(temp_histo2d_ch3)
        else:
            histo2d_ch0 += temp_histo2d_ch0
            histo2d_ch1 += temp_histo2d_ch1
            histo2d_ch2 += temp_histo2d_ch2
            histo2d_ch3 += temp_histo2d_ch3

        event_counter += 1

    histo2d = [histo2d_ch0, histo2d_ch1, histo2d_ch2, histo2d_ch3]
    xedges = [xedges_ch0, xedges_ch1, xedges_ch2, xedges_ch3]
    yedges = [yedges_ch0, yedges_ch1, yedges_ch2, yedges_ch3]

    for k, histo in enumerate(histo2d):
        text = 'channel {}'.format(k)
        anchored_text = AnchoredText(text, loc=1)

        fig, ax = plt.subplots()
        ax.imshow(histo.T, origin='low', norm=LogNorm(),
                  extent=[xedges[k][0], xedges[k][-1], yedges[k][0], yedges[k][-1]])
        ax.set_aspect(200)
        ax.add_artist(anchored_text)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Amplitude [V]'.format(k))
        plt.savefig('{}/waveforms_histo2d_ch{}.png'.format(output_dir, k))
        plt.close(fig)

    end_time = time.time()
    print("time of waveform process : ", end_time - start_time)

    if debug:

        pdf_waveforms_plots = PdfPages('{}/waveforms_by_4.pdf'.format(output_dir))

        waveforms = read_data(file, debug=False)
        event_counter = 0
        for time_widths, times, adc_data, event_rc in waveforms:
            if event_counter is 0:
                times = np.insert(times, 0, 0, axis=1)
                times = np.delete(times, -1, axis=1)
                bin_edges = time_widths / 2 + times

                fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
                # Triggers
                ax[0, 0].hist(time_widths[1], 100, color='tab:red', label="Ch. 1 : Trigger", histtype='step')
                ax[0, 0].legend()
                ax[0, 1].hist(time_widths[3], 100, color='tab:red', label="Ch. 3 : Trigger", histtype='step')
                ax[0, 1].legend()
                # Signals
                ax[1, 0].hist(time_widths[0], 100, color='tab:green', label="Ch. 0 : Signal", histtype='step')
                ax[1, 0].legend()
                ax[1, 1].hist(time_widths[2], 100, color='tab:green', label="Ch. 2 : Signal", histtype='step')
                ax[1, 1].legend()

                ax[1, 0].set_xlabel(r'$\Delta$ Time [ns]')
                ax[1, 1].set_xlabel(r'$\Delta$ Time [ns]')
                ax[0, 0].set_ylabel('Counts')
                ax[1, 0].set_ylabel('Counts')
                plt.tight_layout()
                plt.savefig('{}/time_width_dist.png'.format(output_dir))
                plt.close(fig)

                fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
                # Triggers
                ax[0, 0].hist(times[1], bins=bin_edges[1], color='tab:red', label="Ch. 1 : Trigger", histtype='step')
                ax[0, 0].legend()
                ax[0, 1].hist(times[3], bins=bin_edges[3], color='tab:red', label="Ch. 3 : Trigger", histtype='step')
                ax[0, 1].legend()
                # Signals
                ax[1, 0].hist(times[0], bins=bin_edges[0], color='tab:green', label="Ch. 0 : Signal", histtype='step')
                ax[1, 0].legend()
                ax[1, 1].hist(times[2], bins=bin_edges[2], color='tab:green', label="Ch. 2 : Signal", histtype='step')
                ax[1, 1].legend()

                ax[1, 0].set_xlabel(r'Time [ns]')
                ax[1, 1].set_xlabel(r'Time [ns]')
                ax[0, 0].set_ylabel('Counts')
                ax[1, 0].set_ylabel('Counts')
                plt.tight_layout()
                plt.savefig('{}/times_dist.png'.format(output_dir))
                plt.close(fig)

            adc_data = get_float_waveform(adc_data, event_rc)
            if event_counter % 1000 == 0:
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
                # Triggers
                axs[0, 0].plot(np.cumsum(time_widths[1]), adc_data[1], linestyle="solid", color='tab:red', label='signal : event {}'.format(event_counter))
                axs[0, 0].legend()
                axs[0, 0].set_ylim(-0.10, 1.00)
                axs[0, 1].plot(np.cumsum(time_widths[3]), adc_data[3], linestyle="solid", color='tab:red', label='signal : event {}'.format(event_counter))
                axs[0, 1].legend()
                axs[0, 1].set_ylim(-0.10, 1.00)
                # Signals
                axs[1, 0].plot(np.cumsum(time_widths[0]), adc_data[0], linestyle="solid", color='tab:green', label='trigger')
                axs[1, 0].legend()
                axs[1, 0].set_ylim(-0.10,1.00)
                axs[1, 1].plot(np.cumsum(time_widths[2]), adc_data[2], linestyle="solid", color='tab:green', label='trigger')
                axs[1, 1].legend()
                axs[1, 1].set_ylim(-0.10,1.00)

                axs[1, 0].set_xlabel(r'Time [ns]')
                axs[1, 1].set_xlabel(r'Time [ns]')
                axs[0, 0].set_ylabel('Amplitude [V]')
                axs[1, 0].set_ylabel('Amplitude [V]')
                pdf_waveforms_plots.savefig(fig)
                plt.close(fig)

            event_counter += 1
        pdf_waveforms_plots.close()

        print('End debug')


if __name__ == '__main__':
    entry()

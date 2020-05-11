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


def read_data(file, min_amplitude=0.005, debug=False):

    with DRS4BinaryFile(file) as f:
        board_id = f.board_ids[0]
        active_channels = f.channels

        time_widths = [f.time_widths[board_id][1],
                       f.time_widths[board_id][2],
                       f.time_widths[board_id][3],
                       f.time_widths[board_id][4]]

        times = np.array([np.cumsum(time_widths[0]),
                          np.cumsum(time_widths[1]),
                          np.cumsum(time_widths[2]),
                          np.cumsum(time_widths[3])])

        for event in f:

            # for k in range(len(adc_data)):
            #     if k==0:
            #         value = adc_data[k] < threshold
            #     else:
            #         value *= adc_data[k] < threshold
            # if value is False:
            #     continue
            # threshold = min_amplitude
            # if k==0:
            #         value = adc_data[k] < threshold
            #     else:
            #         value *= adc_data[k] < threshold

            event_id = event.event_id
            event_timestamp = event.timestamp
            event_range_center = event.range_center
            event_scalers = [event.scalers[board_id][1],
                             event.scalers[board_id][2],
                             event.scalers[board_id][3],
                             event.scalers[board_id][4]]
            event_trigger_cell = event.trigger_cells[board_id]

            adc_data = [event.adc_data[board_id][1],
                        event.adc_data[board_id][2],
                        event.adc_data[board_id][3],
                        event.adc_data[board_id][4]]

            # adc_data = np.insert(adc_data, 0, 0, axis=1)

            if debug is True:
                print('board id : ', board_id)
                print('active channels : ', active_channels[board_id])
                print('event id :', event_id)
                print('event timestamp :', event_timestamp)
                print('event range center :', event_range_center)
                print('event scaler :', event_scalers)
                print('event trigger cell :', event_trigger_cell)

            #yield board_id, active_channels, time_widths, event_id, event_timestamp, event_range_center, event_scalers, event_trigger_cell, adc_data
            yield time_widths, times, adc_data, event_range_center


def get_float_waveform(adc_waveform_matrix, range_center):
    """
    :param adc_waveform_matrix:     Matrix containing the ADC counts as amplitude of the n channels of the DRS4 board.
                                    Each row is an ADC waveform.
    :param range_center:            Range center given by the board
    :return:                        Matrix containing the amplitudes of the n channels of the DRS4 board in Volts
    """
    # Get the range according to the range center defined
    amplitude_range = -0.5 + range_center/1000.
    # Amplitude in volts
    amplitude = adc_waveform_matrix/(np.power(2, 16)) + amplitude_range

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

    print('output_dir : ', output_dir)
    print('File :', file)

    waveforms = read_data(file, debug=False)
    bins_y = np.linspace(-0.05, 1.00, 200)
    event_counter = 0

    for time_widths, times, adc_data, event_range_center in waveforms:
        if event_counter % 1000 == 0:
            print('Event id :', event_counter)
        adc_data = get_float_waveform(adc_data, event_range_center)

        temp_histo2d_ch0, xedges_ch0, yedges_ch0 = np.histogram2d(times[0], adc_data[0], bins=[times[0], bins_y])
        temp_histo2d_ch1, xedges_ch1, yedges_ch1 = np.histogram2d(times[1], adc_data[1], bins=(times[1], bins_y))
        temp_histo2d_ch2, xedges_ch2, yedges_ch2 = np.histogram2d(times[2], adc_data[2], bins=(times[2], bins_y))
        temp_histo2d_ch3, xedges_ch3, yedges_ch3 = np.histogram2d(times[3], adc_data[3], bins=(times[3], bins_y))

        if event_counter != 0:

            histo2d_ch0 += temp_histo2d_ch0
            histo2d_ch1 += temp_histo2d_ch1
            histo2d_ch2 += temp_histo2d_ch2
            histo2d_ch3 += temp_histo2d_ch3

        else:
            histo2d_ch0 = temp_histo2d_ch0
            histo2d_ch1 = temp_histo2d_ch1
            histo2d_ch2 = temp_histo2d_ch2
            histo2d_ch3 = temp_histo2d_ch3

        event_counter += 1

    histo2d = [histo2d_ch0, histo2d_ch1, histo2d_ch2, histo2d_ch3]
    xedges = [xedges_ch0, xedges_ch1, xedges_ch2, xedges_ch3]
    yedges = [yedges_ch0, yedges_ch1, yedges_ch2, yedges_ch3]

    for k, histo in enumerate(histo2d):

        pdf_waveforms_histo2d = PdfPages('{}/waveforms_histo2d_ch{}.pdf'.format(output_dir, k))

        text = 'channel {}'.format(k)
        anchored_text = AnchoredText(text, loc=1)

        fig, ax = plt.subplots()
        ax.imshow(histo.T, origin='low', norm=LogNorm(),
                  extent=[xedges[k][0], xedges[k][-1], yedges[k][0], yedges[k][-1]])
        ax.set_aspect(200)
        ax.add_artist(anchored_text)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Amplitude [V]'.format(k))
        pdf_waveforms_histo2d.savefig(fig)
        pdf_waveforms_histo2d.close()
        plt.close(fig)

    if debug:

        pdf_waveforms_plots = PdfPages('{}/waveforms_by_4.pdf'.format(output_dir))

        waveforms = read_data(file, debug=False)
        event_counter = 0
        for time_widths, times, adc_data, event_range_center in waveforms:
            if event_counter % 1000 == 0:
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
                # Triggers
                axs[0, 0].plot(np.cumsum(time_widths[1]), adc_data[1], color='tab:red', label='signal : event {}'.format(event_counter))
                axs[0, 0].legend()
                axs[0, 1].plot(np.cumsum(time_widths[3]), adc_data[3], color='tab:red', label='signal : event {}'.format(event_counter))
                axs[0, 1].legend()
                # Signals
                axs[1, 0].plot(np.cumsum(time_widths[0]), adc_data[0], color='tab:green', label='trigger')
                axs[1, 0].legend()
                axs[1, 1].plot(np.cumsum(time_widths[2]), adc_data[2], color='tab:green', label='trigger')
                axs[1, 1].legend()

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



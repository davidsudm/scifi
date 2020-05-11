#!/usr/bin/env python
"""
Fitter of the charge and related computations

Usage:
    scifi-fitter digicampipe --output_dir=PATH --initial_values_dir=PATH --n_channels=INT [--debug] <INPUT_DIR>
    scifi-fitter gauss --output_dir=PATH --n_channels=INT [--debug] <INPUT_DIR>
    scifi-fitter poisson --output_dir=PATH --n_channels=INT [--debug] <INPUT_DIR>

Options:
    -h --help                    Show this screen.
    --output_dir=PATH           Path to the output directory, where the outputs (pdf files) will be saved.
    --initial_values_dir=PATH   Path to the initial values directory, inside one dictionary file per channel
    --n_channels=INT            Number of total channels used in the DRS4 configuration. [Default: 4]
    -v --debug                  Enter the debug mode.

Commands:
    gauss                       Multi gauss peak fitter method
    digicampipe                 digicampipe fitter method for multi-photon spectra
    poisson                     Poisson fitter for the envelope
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text
from histogram.histogram import Histogram1D
from digicampipe.scripts.mpe import fit_single_mpe

from readout import give_list_of_file


def gauss(x, A, sigma, mu):
    """
    Gaussian function

    :param x:           x values
    :param A:           Amplitude (actually, number of entries, area under the curve)
    :param sigma:       standard deviation of the gaussian
    :param mu:          mean value of the gaussian
    :return:            Gaussian pdf
    """
    norm = sigma * np.sqrt(2 * np.pi)

    return A * (1 / norm) * np.exp(-(((x - mu) / sigma) ** 2) / 2)


def entry():

    args = docopt(__doc__)
    input_dir = args['<INPUT_DIR>']
    output_dir = convert_text(args['--output_dir'])
    initial_values = convert_text(args['--initial_values_dir'])
    n_channels = convert_int(args['--n_channels'])
    debug = args['--debug']

    if args['digicampipe']:
        print('Fitting charge with digicampipe')

        print('input directory :', input_dir)
        print('output directory : ', output_dir)

        file_list = give_list_of_file(input_dir)
        values_file_list = give_list_of_file(initial_values)

        print('Histogram files :', file_list)
        print('Initial values files : ', values_file_list)

        for channel, file in enumerate(file_list):

            previous_file = os.path.join(input_dir, 'fitted_parameters_ch{}.yml'.format(channel))

            if os.path.isfile(previous_file):
                os.remove(previous_file)

            print('file', file_list[channel])
            print('channel', channel)

            with open(values_file_list[channel]) as f:
                init_parameters = yaml.load(f, Loader=yaml.FullLoader)

            print('Initial values file :', values_file_list[channel])
            print('Initial fitting parameters', init_parameters)

            # We need this new format to make work our fit function, it was built that way
            temp_dict = {}
            for key, value in init_parameters.items():
                temp_dict[key] = np.array([[value]])
            init_parameters = temp_dict
            del temp_dict

            data = fit_single_mpe(file, ac_levels=[0], pixel_ids=[0], init_params=init_parameters, debug=True)

            temp_data = {}
            for key, value in data.items():
                if key is not 'pixel_ids':
                    temp_data[key] = (value[0][0]).tolist()
            fitted_parameters = temp_data

            print('Fitted parameters : ', fitted_parameters)
            fitted_parameters_path = os.path.join(output_dir, 'fitted_parameters_ch{}.yml'.format(channel))

            with open(fitted_parameters_path, 'w') as f:
                yaml.dump(fitted_parameters, f)
            print('END for fitter in channel {}'.format(channel))

        print('END of the digicampipe fitter')

    if args['gauss']:
        print('Fitting with gaussian method')

        file_list = give_list_of_file(input_dir)
        print('input directory :', input_dir)
        print('output directory : ', output_dir)
        print('Histogram files :', file_list)

        for channel, file in enumerate(file_list):

            file = os.path.join(input_dir, file)

            charge_histogram = Histogram1D.load(file)
            data = charge_histogram.data
            x_data = charge_histogram.bin_centers

            if debug:
                fig, ax = plt.subplots()
                ax.plot(x_data, data, 'D', label='plot data')
                charge_histogram.draw(axis=ax, legend=False)
                ax.set_xlabel('[LSB]')
                plt.show()

                fig, ax = plt.subplots()
                ax.plot(data, label='plot data vs indexes', marker='o', markerfacecolor='tab:red')
                ax.set_xlabel('[Indexes]')
                plt.show()

            # Automatizing the initial values guess
            # Find peaks: Gain
            # First approx, it will set us to the separation of to peaks
            idx_peak, _ = find_peaks(data, prominence=0.8, height=150, width=0.6)

            delta_idx = np.diff(idx_peak)

            idx_peak, _ = find_peaks(data, distance=delta_idx[0], height=150)
            print('idx of peaks found', idx_peak)
            print('Peaks found : ', len(idx_peak))

            if debug:
                fig, ax = plt.subplots()
                ax.plot(x_data, data, label='plot data', color='tab:green')
                ax.plot(x_data[idx_peak], data[idx_peak], 'D', label='first peak', color='tab:orange')
                ax.set_xlabel('[LSB]')
                ax.legend()
                plt.show()

                fig, ax = plt.subplots()
                ax.plot(x_data, data, label='plot data', color='tab:green')
                ax.plot(x_data[idx_peak], data[idx_peak], 'D', label='first peak', color='tab:orange')
                ax.set_xlabel('[Indexes]')
                ax.legend()
                plt.show()

            # Initial values for one peak
            for i, id_peak in enumerate(idx_peak):

                print('Peak', i)
                lower_bound = int(np.round(id_peak - 0.5 * delta_idx[i]))
                if lower_bound < 0:
                    lower_bound = int(0)

                #right_size =
                temp = np.argemin(x_data[id_peak, id_peak + 0.5 * delta_idx[i]])
                upper_bound = int(np.round(id_peak + 0.5 * delta_idx[i]))

                x_red_data = x_data[lower_bound: upper_bound]
                y_red_data = data[lower_bound: upper_bound]

                if debug:
                    fig, ax = plt.subplots()
                    ax.plot(x_red_data, y_red_data, 'x', label='peak {} data'.format(i), color='tab:green')
                    ax.plot(x_red_data[id_peak], y_red_data[id_peak], 'D', label='first peak', color='tab:orange')
                    plt.show()

                    fig, ax = plt.subplots()
                    ax.plot(x_red_data, y_red_data, 'x', label='peak {} data'.format(i), color='tab:green')
                    ax.plot(x_red_data[id_peak], y_red_data[id_peak], 'D', label='first peak', color='tab:orange')
                    plt.show()

    if args['poisson']:
        print('Fitting with gaussian method')

        file_list = give_list_of_file(input_dir)
        print('input directory :', input_dir)
        print('output directory : ', output_dir)
        print('Histogram files :', file_list)

        for channel, file in enumerate(file_list):

            file = os.path.join(input_dir, file)

            charge_histogram = Histogram1D.load(file)
            data = charge_histogram.data
            x_data = charge_histogram.bin_centers

            # Smooth data to Find peaks for the envelopes
            yy_data = savgol_filter(x=data, window_length=len(data), polyorder=5, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0)
            idx_peak, _ = find_peaks(data, prominence=1.0)

            if debug:
                fig, ax = plt.subplots()
                ax.plot(data, label='data', linestyle='None', color='tab:green')
                ax.plot(yy_data, label='smooth data', linestyle='None', color='tab:red')
                ax.plot(idx_peak, data[idx_peak], label='peaks', marker='v', color='tab:purple')
                plt.show()

            print('hello')






if __name__ == '__main__':
    entry()

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import peakutils as peakutils
from scipy.optimize import curve_fit
from scipy.special import factorial
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages

from docopt import docopt
from digicampipe.utils.docopt import convert_int, convert_text
from histogram.histogram import Histogram1D

channel = 3

file = '/Users/lonewolf/Desktop/scifi/integral_charge/integral_charge_ch{}.fits'.format(channel)
charge_histogram = Histogram1D.load(file)
y_data = charge_histogram.data
x_data = charge_histogram.bin_centers

y = y_data[:1300].astype(float)
x = np.arange(len(y))

# 1st stage : Find estimates of the peaks, high threshold
indexes = peakutils.peak.indexes(y, thres=0.5, thres_abs=False)
estimated_gain = np.average(np.diff(indexes), weights=y[indexes[:-1]])
# 2nd stage : Define a threshold for selecting peak (1/15 of the highest peak)
idx_highest_peak = np.argmax(y)
threshold = y[idx_highest_peak] / 15
# 3rd stage : Find all the true peak
indexes = peakutils.peak.indexes(y, thres=threshold, min_dist=estimated_gain, thres_abs=True)

# 4th stage : scaling
gain = np.mean(np.diff(indexes))
xx = (x - indexes[0])/gain

pdf_scifi_peak_finder = PdfPages(os.path.join('/Users/lonewolf/Desktop/scifi', 'ch{}_peak_finder.pdf'.format(channel)))

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, y, label='channel {} : raw data'.format(channel), color='tab:green')
ax1.plot(x[indexes], y[indexes], label='peaks', marker='v', linestyle='None', color='tab:red')
ax1.legend()
ax1.set_xlabel('Indexes')

ax2.plot(xx, y, label='channel {} : scaled data'.format(channel), color='tab:green')
ax2.plot(xx[indexes], y[indexes], label='peaks', marker='v', linestyle='None', color='tab:red')
ax2.legend()
ax2.set_xlabel('Number of p.e')

pdf_scifi_peak_finder.savefig(fig)
pdf_scifi_peak_finder.close()
#plt.show()
plt.close(fig)

# 5th stage : Get envelope from peaks and fit
x_env = xx[indexes]
x_env = x_env[1:]
y_env = y[indexes]
y_env = y_env[1:]


# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)


# fit with curve_fit
parameters, cov_matrix = curve_fit(poisson, x_env, y_env, p0=[8])
#popt, pcov = curve_fit(gauss, x, y, p0=[init_amplitude, init_mu, init_sigma])

fig, ax = plt.subplots()
x_fit = np.linspace(x_env[0], x_env[-1], 1000)
y_fit = poisson(x_env, *parameters) * 6e3

ax.plot(x_env, y_env, label='data peaks')
ax.plot(x_env, y_fit, label='fit')
plt.show()


# peakutils.baseline.baseline(y, deg=None, max_it=None, tol=None)
# peakutils.baseline.envelope(y, deg=None, max_it=None, tol=None)
# peakutils.peak.centroid(x, y)
# peakutils.peak.gaussian(x, ampl, center, dev)
# peakutils.peak.gaussian_fit(x, y, center_only=True)
# peakutils.peak.indexes(y, thres=0.3, min_dist=1, thres_abs=False)
# peakutils.peak.interpolate(x, y, ind=None, width=10, func=<function gaussian_fit>)


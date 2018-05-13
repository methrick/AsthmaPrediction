import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange


def plot_frequency(data):
    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    data = np.sin(2 * np.pi * f * x / Fs)
    plt.plot(x, data)
    signal_spectrum = (np.fft.fft(data))
    time_step = 1 / 30
    freqs = (np.fft.fftfreq(signal_spectrum.size, d=time_step))
    plt.plot(freqs, np.abs(signal_spectrum))


def plot_power(x_k):
    ps = np.abs(x_k) ** 2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(x_k.size, time_step)
    idx = np.argsort(freqs)
    plot(freqs[idx], ps[idx])


def plot_spectrum(y, fs, extitle=None, path_to_save=''):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y)  # length of the signal
    cut = np.floor(n / 2).astype(int)
    Y = abs(fft(y))  # fft computing and normalization
    Y = Y[0:cut] / cut
    if extitle is None:
        extitle = ''

    title('Frequency Spectrum of the signal  %s' % (str(extitle)))
    dk = 1 / n
    freq = linspace(0, 0.5 - dk, cut) * fs
    plt.clf()
    plt.plot(freq, Y, 'r')  # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    axes = plt.gca()
    axes.set_ylim([0, Y.max()])
    axes.set_xlim([0, 2000])
    # plt.show()
    plt.savefig('%s/Frequency Spectrum of the signal  %s' % (path_to_save, str(extitle)) + '.png', bbox_inches='tight')


def plot_time(y, fs, extitle=None, path_to_save=''):
    n = len(y)  # length of the signal
    T = n / fs
    t = np.linspace(0, T, n, endpoint=False)
    if extitle is None:
        extitle = ''

    title('Time Domain of the signal  %s' % (str(extitle)))
    plt.clf()
    plt.plot(t, y)
    xlabel('Time')
    ylabel('Amplitude')
    # plt.show()
    plt.savefig('%s/Time Domain of the signal  %s' % (path_to_save, str(extitle)) + '.png', bbox_inches='tight')


def plot_time_freq(x_t, fs, title='', path_to_save=None):
    plot_time(x_t, fs, title, path_to_save)
    plot_spectrum(x_t, fs, title, path_to_save)

'''
ASSIGNMENT 4 Python
Final Version
Owners: Giovanni & Ana
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spl
import matplotlib.pyplot as plt

class VoltageData:
    '''
    Class to handle a sequence of voltage measurements at different times
    '''
    def __init__(self, times, voltages, errors):
        '''
        Constructor
        '''
        self._times = np.array(times, dtype = np.float64)
        self._voltages = np.array(voltages, dtype = np.float64)
        if len(errors) == len(self._times):
            self._errors = np.array(errors, dtype = np.float64)
        else:
            self._errors = np.zeros(len(self._times))
        self._values = np.stack((self._times, self._voltages, self._errors), axis = 1)
    #Read times
    @property
    def times(self):
        '''Time attribute'''
        return self._times
    #Read voltages
    @property
    def voltages(self):
        '''Voltage attribute'''
        return self._voltages
    #Read errors
    @property
    def errors(self):
        '''Error attribute'''
        return self._errors
    #Read file
    @classmethod
    def parse_line(cls, path):
        '''Alternative constructor'''
        times, voltages, errors = np.loadtxt(path, unpack = True)
        return cls(times, voltages, errors)
    # Get each item
    def __getitem__(self, index):
        return self._values[index]
    #Read length
    def __len__(self):
        return len(self._values)
    #Iterable
    def __iter__(self):
        return iter(self._values)
    #Calling interpolation
    def __call__(self, time, order):
        spline = spl(self._times, self._voltages, k = order)
        return spline(time)
    #Defining printing style
    def __repr__(self):
        table = ''
        for i in range(len(self._values)):
            table += (f'{i}: {self._values[i, 0]} {self._values[i, 1]} {self._values[i, 2]}\n')
        return table
    #Show values
    def debug_repr(self):
        '''Printing all the values'''
        for value in self._values:
            print(value)
    #Plotting
    def plot(self, axs = None, lab = '', tit = '', rangex = [None, None], rangey = [None, None],
        colour = None, plot_style = 'o'):
        '''Plot'''
        plt.axes(arg = axs,  xlabel = 'times', title = tit, xlim = (rangex[0], rangex[1]),
            ylabel = 'voltages', ylim = (rangey[0], rangey[1]))
        plt.errorbar(self._times, self._voltages, yerr = self._errors, ecolor = colour,
            fmt = plot_style, label = lab)
        plt.legend()


if __name__ == '__main__':
    v = VoltageData.parse_line('sample_data_file_with_errs.txt')
    print(v(5., 3))
    v.plot(tit = 'Prova')

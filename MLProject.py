#!/usr/bin/env python

from sklearn import svm
# to manipulate data
import numpy as np
# to plot things
import matplotlib.pyplot as plt
# for spectrogram
from scipy import signal
# for LPC
import audiolazy
# to do plenty of preprocessing
import librosa
# to treat the .csv
import csv
import wave

#
# PART I: Preprocessing
#


class Configuration():

    def __init__(self,
            sample_rate,
            max_length,
            sample_directory,
            csv_path
            ):

        # initialises
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.sample_directory = sample_directory
        self.csv_path = csv_path

        # default values
        self.meta_data = {}
        self.keys = []
        self.nSample = 0
        self.duration = 0
        self.sample = []
        self.N = 0
        self.data = []

        # features
        self.MFCC = []
        self.MFCC_DELTA = []
        self.MEL_SPECTROGRAM = []
        self.SPECTRO = []
        self.LPC =[]
        self.FFT = []
        self.CWT = []

    def ExtractData(self):
        reader = csv.reader(open(self.csv_path, 'r'))
        self.N = len(reader)
        self.data = np.zeros((self.N, self.max_length))
        for row in reader:
            k, v, c = row
            self.meta_data[k] = v
        self.keys = self.meta_data.keys()
        self.nSample = len(self.keys)

    def PreprocessSample(self, sample_num):

        # get sample_path
        sample_name = self.keys[sample_num]
        sample_path = self.sample_directory + sample_name

        # LOAD SAMPLE
        self.sample, _ = librosa.core.load(sample_path, sr=self.sample_rate, res_type='kaiser_fast')  # returns a np.ndarray
        assert self.sample_rate == _

        # RESIZING
        delta = len(self.sample) - self.max_length
        if delta > 0:
            offset = np.random.randint(delta)
            self.sample = self.sample[offset:offset+self.max_length]
        elif delta < 0:
            delta = abs(delta)
            offset = np.random.randint(delta)
            self.sample = np.pad(self.sample, (offset, delta - offset), "constant")

        self.data[i,:]=self.sample

    def GetDuration(self, sample_num):

        # get sample_path
        sample_name = self.keys[sample_num]
        sample_path = self.sample_directory + sample_name

        # CALCULATE DURATION
        wv = wave.open(sample_path)
        nFrames = wv.getnframes()
        frameRate = wv.getframerate()
        self.duration = nFrames*1./frameRate

    def ExtractFeatures(self):
        """exctracts features from datas"""

        self.MFCC = librosa.feature.mfcc(self.sample, sr=self.sample_rate, n_mfcc=13)
        self.MFCC_DELTA = librosa.feature.delta(self.MFCC)
        self.MEL_SPECTROGRAM = librosa.feature.melspectrogram(self.sample, sr=self.sample_rate)
        f, t, SPECTRO = signal.spectrogram(self.sample)
        self.SPECTRO
        self.LPC = np.array(audiolazy.lazy_lpc.lpc.autocor(self.sample, 2).numerator)
        self.FFT = np.fft.fft(self.sample)
        widths = np.arange(1, 31)
        self.CWT = signal.cwt(self.sample, signal.ricker, widths)



def SVM(sample_rate, max_length, sample_directory, csv_path):
    """Run SVM ML"""
    # Configuration
    config = Configuration(sample_rate, max_length, sample_directory, csv_path)
    config.ExtractData()

    # PREPARING VECTORS
    X = [0]*config.nSample
    Y = [0]*config.nSample

    # FEATURES EXTRACTION
    for sample_num in range(10):

        print("calculating features...")
        config.PreprocessSample(sample_num)
        config.GetDuration(sample_num)

    config.data
        # config.ExtractFeatures()
        #
        # print("nFrames:",int(config.duration*config.sample_rate))
        # # FILLING VECTORS
        # features = [config.MFCC, config.FFT]
        # np.hstack(features[0])
        # np.hstack(features[1])


if __name__ == '__main__':
    SVM(16000, 80000, 'audio_train/', 'train.csv')

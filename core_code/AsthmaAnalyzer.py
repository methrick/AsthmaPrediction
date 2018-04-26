import string

import numpy as np

from pydub import AudioSegment
from pydub.utils import which
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
import os

from core_code.FeatureExtractor import FeatureExtractor

AudioSegment.converter = which("ffmpeg")


class AsthmaAnalyzer(object):
    audio_file_path: string = ''
    fs_signal: int = 8 * 10 ** 3

    audio_duration: float = 0
    Fs_audio: int = 0
    signal_length: int = 1024
    number_of_signals_in_file: np.int64 = 0
    input_signal: np.ndarray
    resampled_input_signal: np.ndarray
    extracted_features = []

    # Saving paths variables
    file_name: string = ''
    general_path_to_save = '../experiments'
    result_path = ''
    screen_shot_path = ''

    def __init__(self, audio_file_path: string, file_name: string):
        self.file_name = file_name
        self.audio_file_path = audio_file_path
        self.__extract_data()

    def __extract_data(self):
        fs, data = wavfile.read(self.audio_file_path)
        # data = np.sin(data)
        if data.ndim > 1:
            data = data[:, 0]  # Taking the first channel only
        data = data.astype(np.float64)
        data = (1 - (-1)) * (data - data.min(0)) / data.ptp(0) - 1  # normalize the data between -1 and
        # data = np.random.rand(301) - 0.5
        self.audio_duration = data[:].size / fs
        self.Fs_audio = fs
        self.input_signal = data[:]
        self.create_required_directories()
        # plot_time_freq(self.input_signal, self.Fs_audio, 'Pure Signal', self.screen_shot_path)

    def get_signal_features(self):
        # 1. Pre process the signal
        self.__pre_process_signal()
        # 2. Loop through all signals
        # sd.play(self.input_signal, self.Fs_audio)

        #  plot_time_freq(self.resampled_input_signal, self.fs_signal, ' Resembled Signal', self.screen_shot_path)
        file_obj = open(self.result_path + "/data.txt", "w+")
        file_obj.write("Signal Name : " + self.file_name + "\n")
        current_time: float = 0
        max_ase = 0
        max_Ti = -10000
        for i in range(self.number_of_signals_in_file - 1):
            current_signal_obj = self.resampled_input_signal[i * self.signal_length:(i + 1) * self.signal_length]
            file_obj.write('------------- New Frame : %d----------------\n' % (i + 1))
            end_time, start_time = self.calculate_current_frame_time(current_time)
            # plot_time_freq(current_signal_obj, self.Fs_audio,
            # 'Segment : %d ; start =  %.3f ; End = %.3f' % (i, start_time, end_time), self.screen_shot_path)
            current_feature_extractor_obj = FeatureExtractor(current_signal_obj, self.signal_length, start_time,
                                                             end_time, i)
            ASE_val, Ti_val = current_feature_extractor_obj.calculate_signal_features()
            file_obj.write(
                'start Time=%f \nEnd time = %f \nASEd = %f \nTi= %f \n' % (start_time, end_time, ASE_val, Ti_val))
            max_ase = max(max_ase, ASE_val)
            max_Ti = max(max_Ti, Ti_val)

        file_obj.write('Max Values  :  \n ASE = %f  \n Ti = %f \n' % (max_ase, max_Ti))
        file_obj.close()

    def calculate_current_frame_time(self, current_time):
        start_time = current_time
        current_time += self.signal_length / self.fs_signal
        end_time = current_time
        return end_time, start_time

    def __pre_process_signal(self):
        resampled_length = np.floor(self.fs_signal * self.audio_duration)
        resampled_length = resampled_length.astype(np.int64)
        self.resampled_input_signal = signal.resample(self.input_signal, resampled_length)
        self.resampled_input_signal_length = resampled_length
        self.number_of_signals_in_file = np.floor(self.resampled_input_signal_length / self.signal_length).astype(
            np.int64)

    def create_required_directories(self):
        self.general_path_to_save += '/' + self.file_name
        if not os.path.exists(self.general_path_to_save):
            os.makedirs(self.general_path_to_save)
        self.screen_shot_path = self.general_path_to_save + '/screenshot/'
        if not os.path.exists(self.screen_shot_path):
            os.makedirs(self.screen_shot_path)
        self.result_path = self.general_path_to_save + '/result/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)



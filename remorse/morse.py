from __future__ import annotations
from pysine import sine
from scipy.fftpack import fft, ifft, fftfreq
from sklearn.cluster import KMeans
from remorse.utils import clamp, is_close, wpm_to_spu
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import soundfile
import sys
import time

class MorseCharacterIterator:
    def __init__(self, morse_character: MorseCharacter):
        self._index = 0
        self._morse_character = morse_character

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._morse_character._string):
            char = self._morse_character._string[self._index]
            self._index += 1
            return char
        raise StopIteration

class MorseCharacter:
    """ A Morse character consisting of one or multiple dits and dahs, e.g. `--.` (representing `G`) """
    def __init__(self, string: str):
        self._string = ""
        self.set(string)

    def set(self, string: str):
        new_string = ""
        for char in string:
            if char == "." or char == "-":
                new_string += char
            else:
                raise Exception("string contains not allowed characters; allowed are '.' and '-'")
        self._string = new_string

    def __repr__(self):
        return repr(self._string)

    def __len__(self) -> int:
        return len(self._string)

    def __str__(self) -> str:
        return self._string

    def __iter__(self):
        return MorseCharacterIterator(self)

    def __eq__(self, string: MorseCharacter | str) -> bool:
        return self._string == string._string if string is MorseCharacter else self._string == string

    def __hash__(self) -> int:
        return hash(self._string)

class MorseWordPause:
    """ Representation of a pause between words, that is a pause with a length of 7 dits. Does not hold any
        functionality; only used for differentiation with Morse characters within Morse strings. """
    pass

class MorseStringIterator:
    def __init__(self, morse_string: MorseString):
        self._index = 0
        self._morse_string = morse_string

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._morse_string._chars):
            char = self._morse_string._chars[self._index]
            self._index += 1
            return char
        raise StopIteration

class MorseString:
    """ A Morse string consisting of none, one or multiple Morse characters as well as pauses,
        e.g. `--. -` (representing `GT`) """
    def __init__(self, input: MorseCharacter | list[MorseCharacter | MorseWordPause] | MorseString = None):
        self._chars: list[MorseCharacter | MorseWordPause] = []
        if input is not None:
            if isinstance(input, MorseCharacter):
                self._chars.append(input)
            elif isinstance(input, list[MorseCharacter | MorseWordPause]):
                self._chars.extend(input)
            elif isinstance(input, MorseString):
                self._chars = input._chars.copy()

    def __str__(self):
        result = ""
        last_char = False
        for char in self._chars:
            if isinstance(char, MorseCharacter):
                if last_char:
                    result += " "
                result += str(char)
                last_char = True
            elif isinstance(char, MorseWordPause):
                result += "/"
                last_char = False
        return result

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self._chars)

    def __iter__(self):
        return MorseStringIterator(self)

    def __iadd__(self, input: MorseCharacter | list[MorseCharacter | MorseWordPause] | MorseString):
        if isinstance(input, MorseCharacter) or isinstance(input, MorseWordPause):
            self._chars.append(input)
        elif isinstance(input, MorseString):
            self._chars.extend(input._chars)
        elif isinstance(input, list):
            for element in input:
                if isinstance(element, MorseCharacter) or isinstance(element, MorseWordPause):
                    self._chars.append(input)
        return self

class MorseEmitter:
    """ An abstract base class that allows to emit Morse strings in some shape or form. """
    def emit(self, morse_string: MorseString):
        self.pre_emit()
        last = None
        for morse_char in morse_string:
            if isinstance(morse_char, MorseCharacter):
                if not last or isinstance(last, MorseWordPause):
                    self.pre_emit_word()
                self.pre_emit_character()
                for index, char in enumerate(morse_char):
                    if index > 0:
                        self.emit_intra_character_pause()
                    self.pre_emit_symbol()
                    if char == '.':
                        self.emit_dit()
                    elif char == '-':
                        self.emit_dah()
                    self.post_emit_symbol()
                self.post_emit_character()
                if isinstance(last, MorseWordPause):
                    self.post_emit_word()
                self.emit_inter_character_pause()
            elif isinstance(morse_char, MorseWordPause):
                self.emit_inter_word_pause()
            last = morse_char
        self.post_emit()

    def pre_emit(self):
        pass

    def pre_emit_word(self):
        pass

    def pre_emit_character(self):
        pass

    def pre_emit_symbol(self):
        pass

    def post_emit(self):
        pass

    def post_emit_word(self):
        pass

    def post_emit_character(self):
        pass

    def post_emit_symbol(self):
        pass

    def emit_dit(self):
        pass

    def emit_dah(self):
        pass

    def emit_intra_character_pause(self):
        pass

    def emit_inter_character_pause(self):
        pass

    def emit_inter_word_pause(self):
        pass

class MorseMultiEmitter(MorseEmitter):
    """ A Morse emitter that emits Morse strings to multiple Morse emitters at the same time. """
    def __init__(self, simultaneous: bool, *morse_emitters: MorseEmitter):
        self._simultaneous = simultaneous
        self._emitters = []
        for emitter in morse_emitters:
            if issubclass(type(emitter), MorseEmitter) and emitter not in self._emitters:
                self._emitters.append(emitter)

    def add_emitter(self, emitter: MorseEmitter):
        if issubclass(type(emitter), MorseEmitter) and emitter not in self._emitters:
            self._emitters.append(emitter)

    def remove_emitter(self, emitter: MorseEmitter):
        if emitter in self._emitters:
            self._emitters.remove(emitter)

    def emit(self, morse_string: MorseString):
        if not self._simultaneous:
            for emitter in self._emitters:
                emitter.emit(morse_string)
        else:
            for emitter in self._emitters:
                emitter.pre_emit()
            last = None
            for morse_char in morse_string:
                if isinstance(morse_char, MorseCharacter):
                    if not last or isinstance(last, MorseWordPause):
                        for emitter in self._emitters:
                            emitter.pre_emit_word()
                    for emitter in self._emitters:
                        emitter.pre_emit_character()
                    for index, char in enumerate(morse_char):
                        if index > 0:
                            for emitter in self._emitters:
                                emitter.emit_intra_character_pause()
                        for emitter in self._emitters:
                            emitter.pre_emit_symbol()
                        if char == '.':
                            for emitter in self._emitters:
                                emitter.emit_dit()
                        elif char == '-':
                            for emitter in self._emitters:
                                emitter.emit_dah()
                        for emitter in self._emitters:
                            emitter.post_emit_symbol()
                    for emitter in self._emitters:
                        emitter.post_emit_character()
                    if isinstance(last, MorseWordPause):
                        for emitter in self._emitters:
                            emitter.post_emit_word()
                    for emitter in self._emitters:
                        emitter.emit_inter_character_pause()
                elif isinstance(morse_char, MorseWordPause):
                    for emitter in self._emitters:
                        emitter.emit_inter_word_pause()
                last = morse_char
            for emitter in self._emitters:
                emitter.post_emit()

class MorseStream:
    pass

class Color:
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7

class ColorizationMode:
    NONE = 0
    CHARACTERS = 1
    WORDS = 2
    SYMBOLS = 3

class MorsePrinter(MorseEmitter):
    """ A Morse printer that prints Morse strings to the given output device (e.g. console or file). """
    def __init__(self, output_device: sys.TextIO = sys.stdout, dit_symbol: str = '.', dah_symbol: str = '-',
                 pause_char: str = ' ', word_pause_char: str = '/'):
        self._output_device = output_device
        self._dit_symbol = dit_symbol
        self._dah_symbol = dah_symbol
        self._pause_char = pause_char
        self._word_pause_char = word_pause_char

    def emit_dit(self):
        self._output_device.write(self._dit_symbol)
        self._output_device.flush()

    def emit_dah(self):
        self._output_device.write(self._dah_symbol)
        self._output_device.flush()

    def emit_inter_character_pause(self):
        self._output_device.write(self._pause_char)
        self._output_device.flush()

    def emit_inter_word_pause(self):
        self._output_device.write(self._word_pause_char)
        self._output_device.flush()

class MorseVisualizer(MorseEmitter):
    """ A Morse visualizer that prints Morse characters and strings to the given output device (e.g. console or file)
        by visualizing the duration of the signals and the pauses. """
    def __init__(self, output_device: sys.TextIO = sys.stdout):
        self._output_device = output_device
        self._colors = [ Color.RED ]
        self._color_index = 0
        self._enable_colored_output = False

    def pre_emit_word(self):
        if self._colorization_mode == ColorizationMode.WORDS:
            self.set_next_color()

    def pre_emit_character(self):
        if self._colorization_mode == ColorizationMode.CHARACTERS:
            self.set_next_color()

    def pre_emit_symbol(self):
        if self._colorization_mode == ColorizationMode.SYMBOLS:
            self.set_next_color()

    def post_emit(self):
        self._output_device.write('\x1b[0m')
        self._output_device.flush()

    def emit_dit(self):
        self._output_device.write('▄')
        self._output_device.flush()

    def emit_dah(self):
        self._output_device.write('▄▄▄')
        self._output_device.flush()

    def emit_pause(self, num_instances: int = 1):
        num_instances = num_instances if num_instances in { 1, 3, 7 } else 1
        self._output_device.write('\u2003' * num_instances)
        self._output_device.flush()

    def emit_intra_character_pause(self):
        self.emit_pause(num_instances = 1)

    def emit_inter_character_pause(self):
        self.emit_pause(num_instances = 3)

    def emit_inter_word_pause(self):
        self.emit_pause(num_instances = 7)

    def enable_colored_output(self):
        self._enable_colored_output = True

    def disable_colored_output(self):
        self._enable_colored_output = False

    def set_colorization_mode(self, colorization_mode: ColorizationMode):
        self._colorization_mode = colorization_mode

    def set_colors(self, *colors: int):
        new_colors = []
        for color in colors:
            if 0 <= color <= 7:
                new_colors.append(color)
        if len(new_colors):
            self._colors = new_colors

    def set_next_color(self):
        if not self._enable_colored_output:
            return
        self._output_device.write(f"\x1b[3{self._colors[self._color_index]}m")
        self._color_index = (self._color_index + 1) % len(self._colors)

class MorsePlayer(MorseEmitter):
    """ A Morse player that plays Morse characters and strings as sounds on the default audio device. """
    def __init__(self, frequency: float = 800.0, speed: float = 20.0):
        self._frequency = frequency
        self._words_per_minute = min(max(speed, 1), 60)
        self._seconds_per_unit = wpm_to_spu(self._words_per_minute)

    def emit_dit(self):
        sine(frequency = self._frequency, duration = self._seconds_per_unit)

    def emit_dah(self):
        sine(frequency = self._frequency, duration = self._seconds_per_unit * 3)

    def emit_pause(self, num_instances: int = 1):
        num_instances = num_instances if num_instances in { 1, 3, 7 } else 1
        sine(frequency = 0, duration = self._seconds_per_unit * num_instances)

    def emit_intra_character_pause(self):
        self.emit_pause(num_instances = 1)

    def emit_inter_character_pause(self):
        self.emit_pause(num_instances = 3)

    def emit_inter_word_pause(self):
        self.emit_pause(num_instances = 7)

class MorseReader:
    """ An abstract base class that allows to receive Morse strings from some shape or form. """
    def read(self) -> MorseString:
        pass

class MonoReduceFunction:
    SPECIFIC_CHANNEL = 0
    AVERAGE = 1
    MAXIMUM = 2

class MorseSoundFileReader(MorseReader):
    """ A Morse sound receiver for extracting Morse strings from sound files. """
    def __init__(self, file_path: str, volume_threshold: float = 0.35, normalize_volume: bool = True,
                 use_multiprocessing: bool = True, kernel_seconds: float = 0.001, min_signal_seconds: float = 0.01,
                 low_cut_frequency: float = None, high_cut_frequency: float = None, show_plots: bool = False):
        self._file_path = file_path
        self._volume_threshold = clamp(volume_threshold, 0.1, 0.9)
        self._normalize_volume = normalize_volume
        self._use_multiprocessing = use_multiprocessing
        self._kernel_seconds = max(kernel_seconds, 0.0001)
        self._min_signal_seconds = max(min_signal_seconds, 0.01)
        self._low_cut_frequency = low_cut_frequency
        self._high_cut_frequency = high_cut_frequency
        self._show_plots = show_plots

    def set_show_plots(self, show_plots: bool):
        self._show_plots = show_plots

    def show_plots(self) -> bool:
        return self._show_plots

    def multi_channel_to_absolute_mono(array_of_channels: list[tuple] | list[list],
                                       mono_reduce_function: MonoReduceFunction = MonoReduceFunction.MAXIMUM,
                                       specific_channel: int = None) -> list[float]:
        if array_of_channels is None or len(array_of_channels) == 0:
            return []

        num_channels = len(array_of_channels[0])
        if num_channels == 1:
            return array_of_channels

        if mono_reduce_function == MonoReduceFunction.SPECIFIC_CHANNEL:
            if specific_channel is None or specific_channel >= num_channels:
                return []
            return [channels[specific_channel] for channels in array_of_channels]

        elif mono_reduce_function == MonoReduceFunction.AVERAGE:
            result = []
            for channels in array_of_channels:
                value = 0
                for channel in channels:
                    value += abs(channel)
                result.append(value / num_channels)
            return result

        elif mono_reduce_function == MonoReduceFunction.MAXIMUM:
            result = []
            for channels in array_of_channels:
                max_value = 0
                for channel in channels:
                    value = abs(channel)
                    if value > max_value:
                        max_value = value
                result.append(max_value)
            return result

        else:
            raise Exception(f"Mono reduce function {mono_reduce_function} is not supported")

    def detect_signals(self, samples: list, start: int, end: int) -> list[int]:
        """ Detects `on` and `off` signals in the given samples in the range [start; end), counts the lengths of the
            signals and returns them in a list that always starts with an `on` signal. """
        num_samples = len(samples)
        signal_lengths = []
        consecutive_on_signal_samples = 0
        consecutive_off_signal_samples = 0
        previous_signal = None

        # Analyze on and off signals
        for i in range(start, end):
            is_last_sample = i == end - 1

            kernel_left_pos = max(i - self._half_kernel_samples, 0)
            kernel_right_pos = min(i + self._half_kernel_samples, num_samples)

            max_kernel_sample = max(samples[kernel_left_pos:kernel_right_pos])
            current_signal = max_kernel_sample >= self._volume_threshold

            # Change of signal from 'on' to 'off' or vice versa
            if (previous_signal is not None and previous_signal != current_signal) or is_last_sample:
                if previous_signal == True:
                    signal_lengths.append(consecutive_on_signal_samples)
                    consecutive_on_signal_samples = 0
                else:
                    if len(signal_lengths) == 0:
                        # Make sure the signal lengths array begins with the length of an `on` signal
                        signal_lengths.append(0)
                    signal_lengths.append(consecutive_off_signal_samples)
                    consecutive_off_signal_samples = 0

            if current_signal == True:
                consecutive_on_signal_samples += 1
            else:
                consecutive_off_signal_samples += 1

            previous_signal = current_signal

        return signal_lengths

    def detect_signals_mp(args) -> list[int]:
        """ Detects `on` and `off` signals in the given samples in the range [start; end), counts the lengths of the
            signals and returns them in a list that always starts with an `on` signal. """
        samples, half_kernel_samples, volume_threshold = args

        num_samples = len(samples)
        signal_lengths = []
        consecutive_on_signal_samples = 0
        consecutive_off_signal_samples = 0
        previous_signal = None

        # Analyze on and off signals
        for i in range(0, num_samples):
            is_last_sample = i == num_samples - 1

            kernel_left_pos = max(i - half_kernel_samples, 0)
            kernel_right_pos = min(i + half_kernel_samples, num_samples)

            max_kernel_sample = max(samples[kernel_left_pos:kernel_right_pos])
            current_signal = max_kernel_sample >= volume_threshold

            # Change of signal from 'on' to 'off' or vice versa
            if (previous_signal is not None and previous_signal != current_signal) or is_last_sample:
                if previous_signal == True:
                    signal_lengths.append(consecutive_on_signal_samples)
                    consecutive_on_signal_samples = 0
                else:
                    if len(signal_lengths) == 0:
                        # Make sure the signal lengths array begins with the length of an `on` signal
                        signal_lengths.append(0)
                    signal_lengths.append(consecutive_off_signal_samples)
                    consecutive_off_signal_samples = 0

            if current_signal == True:
                consecutive_on_signal_samples += 1
            else:
                consecutive_off_signal_samples += 1

            previous_signal = current_signal

        return signal_lengths

    def merge_lengths(array_of_lengths: list[list]) -> list:
        """ Merges multiple arrays of lengths to a single array of lengths. Expects each given array to contain lengths
            of alternating type (e.g. `on` and `off` signal lengths). Expects all given arrays of lengths to start with
            the same type of length (e.g. `on` signal length). """
        merged = array_of_lengths[0]

        for array in array_of_lengths[1:]:
            if array is None or not len(array):
                continue

            # If `merged` contains an odd number of elements then it has to end with an `on` signal
            if len(merged) % 2 == 1:
                if array[0] > 0:
                    merged[-1] += array[0]
                merged.extend(array[1:])
            else:
                if array[0] == 0 and len(array) > 1:
                    merged[-1] += array[1]
                merged.extend(array[2:])

        return merged

    def patch_length_holes(lengths: list, min_length: int):
        """ Patches holes that undercut the given minimum length in the given array of lengths in place. This algorithm
            keeps elements that will not be coalesced at their even/odd index, that is, the coalescence of lengths
            causes the following elements to always move up (towards the beginning) a multiple of 2 indices. """
        i = 0
        front = lengths[:2]
        min_index = 0
        # Iterate through all lengths and coalesce them if the minimum length is undercut
        while i < len(lengths):
            if lengths[i] < min_length:
                if i == 0:
                    min_index = 2
                    i += 2
                else:
                    # Make sure we do not try to access beyond the first element
                    left = max(i - 1, min_index)
                    right = min(i + 2, len(lengths))

                    # Sum up the (up to) three elements
                    lengths[i - 1] = sum(lengths[left:right])

                    # Purge elements
                    del lengths[i:right]
            else:
                i += 1

    def find_unit_duration(data) -> float:
        """ Finds and returns the unit duration derived from the given input data using k-means clustering. """
        data = np.array(data).reshape(-1, 1)

        # Find two factors with the approximate ratio 1:3 using k-means clustering
        kmeans = KMeans(n_init = 10, n_clusters = 2)
        kmeans.fit(data)
        durations = sorted(kmeans.cluster_centers_.flatten())
        factors = durations / durations[0]

        if not (2.0 < factors[1] < 3.9):
            raise Exception("Error: The data cannot be fitted into a Morse signal frame")

        # Return the average of both single and triple unit durations as "the" unit duration
        return sum(durations) / 4

    def read(self) -> MorseString:
        result = MorseString()
        samples, sample_rate = soundfile.read(self._file_path)
        num_samples = len(samples)

        def samples_to_seconds(sample: int) -> float:
            return sample / sample_rate

        def seconds_to_samples(seconds: float) -> int:
            return int(seconds * sample_rate)

        self._kernel_samples = seconds_to_samples(self._kernel_seconds)
        self._kernel_samples = max(self._kernel_samples, 6)
        self._half_kernel_samples = self._kernel_samples // 2

        # The minimum signal length in samples to be considered (signals of shorter length will be ignored)
        self._min_signal_samples = int(self._min_signal_seconds * sample_rate)

        # Return empty Morse string if the audio file is too short
        if num_samples < self._kernel_samples:
            return result

        # Un-tuple the raw samples and enforce mono
        if samples.ndim > 1:
            samples = [tuple[0] for tuple in samples]

        samples_abs = np.array(samples)
        perform_filtering = self._low_cut_frequency is not None or self._high_cut_frequency is not None

        # Analyze the frequency spectrum of the audio signal using FFT
        if perform_filtering:
            fft_result = fft(samples_abs)
            freq_axis = fftfreq(num_samples, d = 1 / sample_rate)

            # Find indices that correspond to minimum and maximum frequency in the frequency axis
            if self._low_cut_frequency is not None:
                low_cut_frequency_index = np.argmax(freq_axis >= self._low_cut_frequency)
            if self._high_cut_frequency is not None:
                high_cut_frequency_index = np.argmax(freq_axis >= self._high_cut_frequency)

            # Perform band-pass filtering using inverse FFT
            filtered_fft_result = fft_result.copy()
            filtered_fft_result[:low_cut_frequency_index] = 0
            filtered_fft_result[high_cut_frequency_index:] = 0
            filtered_samples = ifft(filtered_fft_result)
            filtered_samples = np.real(filtered_samples).astype(samples_abs.dtype)
        else:
            fft_result = None
            freq_axis = None
            filtered_samples = samples
            filtered_samples_abs = samples_abs

        # Normalize amplitude
        if self._normalize_volume:
            amplitude_factor = 1 / max(filtered_samples)
            filtered_samples *= amplitude_factor
            filtered_samples_abs = [abs(sample) for sample in filtered_samples]

        # Prepare data for plotting
        if self._show_plots:
            plot_selection = None
            #plot_selection = (11200, 14400) # test_2.wav
            #plot_selection = (0, 250000) # test_old_recording_denoise.wav

            if plot_selection is None:
                plot_selection = (0, len(filtered_samples))
            t = [samples_to_seconds(sample) for sample in range(plot_selection[0], plot_selection[1])]

        # Plot frequency domain of the waveform
        if self._show_plots and freq_axis is not None and fft_result is not None:
            magnitude_spectrum = np.abs(fft_result)
            plt.figure(num = "Original frequency domain")
            plt.plot(freq_axis, magnitude_spectrum)
            plt.title(f"Original frequency domain of {self._file_path}")
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.xlim(left = 0)
            plt.ion()
            plt.show(block = False)

        # Plot original and filtered waveform
        if self._show_plots:
            plt.figure(num = "Original vs filtered waveform", figsize = (20, 6))
            plt.title(f"Original vs filtered waveform of {self._file_path}")
            plt.plot(t, samples[plot_selection[0]:plot_selection[1]], label = 'Original')
            plt.plot(t, filtered_samples[plot_selection[0]:plot_selection[1]], label = 'Filtered')
            plt.xlabel('Time [s]')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.show(block = False)

        # t0 = time.time()

        # Detect signals from the samples
        if self._use_multiprocessing:
            num_processes = 4
            pool = multiprocessing.Pool(processes = num_processes)
            samples_per_group = num_samples // num_processes
            sample_group_start = 0
            sample_group_end = samples_per_group
            detect_signals_args_array = []

            # Divide samples into sample groups and prepare task arguments
            for i in range(num_processes):
                detect_signals_args = (filtered_samples_abs[sample_group_start:sample_group_end],
                                       self._half_kernel_samples, self._volume_threshold)
                detect_signals_args_array.append(detect_signals_args)
                sample_group_start = sample_group_end
                sample_group_end = num_samples if i == num_processes - 1 else sample_group_end + samples_per_group

            # Submit sample group tasks to process pool
            partial_signal_lengths = pool.map(MorseSoundFileReader.detect_signals_mp, detect_signals_args_array)
            pool.close()

            # Merge all partial signal lengths
            signal_lengths = MorseSoundFileReader.merge_lengths(partial_signal_lengths)
        else:
            signal_lengths = self.detect_signals(filtered_samples_abs, 0, len(filtered_samples_abs))

        # t1 = time.time()
        # print(f"Detecting signals took {t1 - t0:0.1f} seconds")

        # Patch holes in the signal lengths in place
        MorseSoundFileReader.patch_length_holes(signal_lengths, self._min_signal_samples)

        # Plot waveform and signals
        if self._show_plots:
            signals = []
            # Build array of individual signals for plotting
            for index, signal_length in enumerate(signal_lengths):
                signal = 1 if index % 2 == 0 else 0
                if signal_length > 0:
                    signals.extend([signal] * signal_length)

            len_samples = len(samples)
            len_signals = len(signals)
            if len_samples > len_signals:
                last_signal = signals[-1]
                signals.extend([last_signal] * (len_samples - len_signals))

            plt.figure(num = "Waveform and signals", figsize = (20, 6))
            plt.title(f"Waveform and signals of {self._file_path}")
            plt.plot(t, samples[plot_selection[0]:plot_selection[1]], label = 'Raw')
            plt.plot(t, signals[plot_selection[0]:plot_selection[1]], label = 'Processed')
            plt.xlabel('Time [s]')
            plt.ylabel('Magnitude')
            plt.ylim(ymax = 1.05, ymin = 0)
            plt.legend()
            plt.show()

        # Extract only the `on` signal lengths from the signal lengths
        on_signal_lengths = [signal_lengths[i] for i in range(0, len(signal_lengths), 2)]

        # Determine number of segments for which to determine Morse unit durations separately
        morse_unit_durations: list[tuple[int, int]] = []
        morse_unit_segments = max(num_samples // seconds_to_samples(10), 1)
        on_signal_lengths_per_segment = len(on_signal_lengths) // morse_unit_segments
        last_morse_unit_duration = None

        # Determine Morse unit durations for one or more segments
        for i in range(morse_unit_segments):
            start = i * on_signal_lengths_per_segment
            end = ( i + 1 ) * on_signal_lengths_per_segment if i < morse_unit_segments - 1 else len(on_signal_lengths)
            morse_unit_duration = int(MorseSoundFileReader.find_unit_duration(on_signal_lengths[start:end]))

            # Only add new Morse unit duration if it is substantially different to the last one added
            if last_morse_unit_duration is None or not is_close(morse_unit_duration, last_morse_unit_duration, 0.01):
                morse_unit_durations.append((end * 2, morse_unit_duration))
                last_morse_unit_duration = morse_unit_duration

        morse_character = ""
        morse_unit_durations_index = 0

        # Detect Morse characters from the signals and string them together.
        # Iterate over the signals and check the length of 'on' and 'off' signals with respect to the determined Morse
        # unit length; we classify dits and dahs as well as the three kinds of pauses as forgiving and generous as
        # possible, because it is always better to return the most likely Morse character, even if it is incorrect.
        for index, signal_length in enumerate(signal_lengths):
            if signal_length == 0:
                continue

            # Find the correct Morse unit duration according to the signal length index
            while morse_unit_durations_index < len(morse_unit_durations) and index > morse_unit_durations[morse_unit_durations_index][0]:
                morse_unit_durations_index += 1
            morse_unit_durations_index = min(morse_unit_durations_index, len(morse_unit_durations) - 1)
            morse_unit_duration = morse_unit_durations[morse_unit_durations_index][1]

            signal_units = signal_length / morse_unit_duration

            # Signal is an `on` signal
            if index % 2 == 0:
                morse_character += "." if signal_units < 2 else "-"
            elif 2.2 <= signal_units < 5 and morse_character:
                result += MorseCharacter(morse_character)
                morse_character = ""
            elif 5 <= signal_units and morse_character:
                result += MorseCharacter(morse_character)
                result += MorseWordPause()
                morse_character = ""

        if morse_character:
            result += MorseCharacter(morse_character)

        return result

class MorseWriter:
    def write(self, input: MorseString):
        pass

class MorseSoundFileWriter(MorseWriter):
    def __init__(self, file_path: str, volume: float = 0.8, frequency: float = 800.0, speed: float = 20.0,
                 sample_rate: int = 8000):
        self._sound_file = None
        self._file_path = os.path.expanduser(file_path)
        self._volume = clamp(volume, 0, 1)
        self._frequency = frequency
        self._words_per_minute = clamp(speed, 1, 60)
        self._seconds_per_unit = wpm_to_spu(self._words_per_minute)
        self._sample_rate = clamp(sample_rate, 1000, 192000)

        self.open()

    def __del__(self):
        self.close()

    def open(self):
        """ Opens sound file for writing. """
        if self._sound_file is None:
            directory_path = os.path.dirname(self._file_path)
            os.makedirs(directory_path, exist_ok = True)
            self._sound_file = soundfile.SoundFile(self._file_path, mode = 'w', samplerate = self._sample_rate,
                                                   channels = 1)

    def close(self):
        """ Closes sound file for writing. """
        if self._sound_file is not None:
            self._sound_file.close()
            self._sound_file = None

    def write_durations(self, durations: list[float]):
        """ Writes audible `on` signals and silent `off` signals according to the given durations (even indices denote
            `on`, odd indices denote `off` signals) to the open sound file. """
        if self._sound_file is None:
            return;

        # Start off with an empty array
        waveform = np.array([])

        # Iterate over all durations and generate waveform data
        for i, duration in enumerate(durations):
            # On signals (audible)
            if i % 2 == 0:
                t = np.linspace(0, duration, int(duration * self._sample_rate), endpoint = False)
                audible_wave = self._volume * np.sin(2 * np.pi * self._frequency * t)
                waveform = np.concatenate([waveform, audible_wave])

            # Off signals (silent)
            else:
                silent_wave = np.zeros(int(duration * self._sample_rate))
                waveform = np.concatenate([waveform, silent_wave])

        # Write waveform data to the given file
        self._sound_file.write(waveform)

    def write(self, input: MorseString | MorseCharacter | MorseWordPause):
        """ Writes the given input Morse string, character or word pause to the open sound file. """
        if self._sound_file is None:
            return;

        string = None
        durations = []

        if isinstance(input, MorseCharacter) or isinstance(input, MorseWordPause):
            string = MorseString(input)
        elif isinstance(input, MorseString):
            string = input

        # Iterate over the given Morse string and build an array of signal durations
        for morse_char in string:
            if isinstance(morse_char, MorseCharacter):
                for index, char in enumerate(morse_char):
                    if index > 0:
                        durations.append(self._seconds_per_unit)
                    if char == '.':
                        durations.append(self._seconds_per_unit)
                    elif char == '-':
                        durations.append(self._seconds_per_unit * 3)
                durations.append(self._seconds_per_unit * 3)
            elif isinstance(morse_char, MorseWordPause) and len(durations) > 0:
                durations[-1] = self._seconds_per_unit * 7

        self.write_durations(durations)

LATIN_MORSE_TREE_LINEARIZED = ('etianmsurwdkgohvf#l#pjbxcyzq##54#3#¿#2&#+####16=/###(#7###8#90'
                               '############?_####"##.####@###\'##-########;!#)###¡#,####:#######')

def layer_at_index(index: int, m: int) -> int:
    """ Returns the 0-based `layer` of a node (internal or leaf) given the consecutive 0-based `index` in its linearized
        full `m`-ary tree. See https://www.desmos.com/calculator/7yynaybzzs for graphs and https://oeis.org/A064099 """
    assert int(index) == index and index >= 0, "index must be integer and greater or equal to 0"
    assert int(m) == m and m >= 2, "m must be integer and greater of equal to 2"
    return math.ceil(math.log((m - 1) * (index + 1) + 1, m)) - 1

def build_symbol_mapping_from_linearized_tree(in_symbols: list[str] | str,
                                              out_symbols: list[str] | str) -> dict[str, str]:
    """ Builds and returns a symbol mapping from the given `in_symbols` to the given `out_symbols`. `out_symbols` must
        contain at least 2 symbols so that an encoding of recurring prefixed patterns can be created. Can also be used
        to create a 1:1 mapping for scrambling and unscrambling texts. """
    assert len(out_symbols) >= 2, "out_symbols must contain at least 2 symbols"
    BASE = len(out_symbols)
    result = {}
    layer_index = 0
    last_node_layer = 1
    for index, letter in enumerate(in_symbols):
        code = ''
        node_layer = layer_at_index(index + 1, BASE)
        if node_layer != last_node_layer:
            layer_index = 0
            last_node_layer = node_layer
        val = layer_index
        layer_index += 1
        if letter in {'', '#'}:
            continue
        code = out_symbols[val % BASE]
        while (node_layer - 1) > 0:
            val = val // BASE
            code = out_symbols[val % BASE] + code
            node_layer -= 1
        result[letter.upper()] = MorseCharacter(code)
    return result

LATIN_TO_MORSE = build_symbol_mapping_from_linearized_tree(LATIN_MORSE_TREE_LINEARIZED, ['.', '-'])
MORSE_TO_LATIN = { v: k for k, v in LATIN_TO_MORSE.items() }

def text_to_morse(text: str, symbol_separator: str = ' ', word_separator: str = '/',
                  dit_symbol: str = '.', dah_symbol: str = '-', width_equals_time: bool = False) -> str:
    """ Converts text into Morse code. """
    words = []
    morse = []
    morse_string = MorseString()
    symbol_space = symbol_separator if not width_equals_time else '\u2003\u2003\u2003'
    for text_symbol in text.upper():
        if text_symbol == ' ':
            morse_string += MorseWordPause()
            words.append(''.join(morse).rstrip())
            morse = []
        elif text_symbol in LATIN_TO_MORSE:
            # Lookup Morse symbol in forward map
            morse_character = LATIN_TO_MORSE[text_symbol]
            morse_string += morse_character
            morse_symbol = str(morse_character)
            # Undo print representation or symbol substitution
            if width_equals_time:
                morse_symbol = morse_symbol.replace('.', '▄ ').replace('-', '▄▄▄ ').rstrip().replace(' ', '\u2003')
            else:
                morse_symbol = morse_symbol.replace('.', dit_symbol).replace('-', dah_symbol)
            morse.append(morse_symbol + symbol_space)
        else:
            morse.append('#' + symbol_space)
    if morse:
        words.append(''.join(morse).rstrip())
        morse = []
    return morse_string
    #return ('\u2003\u2003\u2003\u2003\u2003\u2003\u2003' if width_equals_time else word_separator).join(words)

def morse_to_text(morse: str, symbol_separator: str = ' ', word_separator: str = '/',
                  dit_symbol: str = '.', dah_symbol: str = '-', width_equals_time: bool = False) -> str:
    """ Converts Morse code into text. """
    text = []
    for morse_word in morse.split(word_separator if not width_equals_time else '\u2003\u2003\u2003\u2003\u2003\u2003\u2003'):
        for morse_symbol in morse_word.rstrip().split(symbol_separator if not width_equals_time else '\u2003\u2003\u2003'):
            # Undo print representation or symbol substitution
            if width_equals_time:
                morse_symbol = morse_symbol.replace('▄▄▄', '-').replace('▄', '.').replace(' ', '').replace('\u2003', '')
            else:
                morse_symbol = morse_symbol.replace(dit_symbol, '.').replace(dah_symbol, '-')
            # Lookup text symbol in reverse map
            if morse_symbol in MORSE_TO_LATIN:
                text.append(MORSE_TO_LATIN[morse_symbol])
            else:
                text.append('#')
        text.append(' ')
    return ''.join(text).rstrip()

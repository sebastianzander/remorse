from __future__ import annotations
from pysine import sine
from scipy.fftpack import fft, ifft, fftfreq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import sys

def clamp(value: int | float, minimum: int | float, maximum: int | float):
    return min(max(value, minimum), maximum)

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
    def __init__(self, parallel: bool, *morse_emitters: MorseEmitter):
        self._parallel = parallel
        self._emitters = []
        for emitter in morse_emitters:
            if issubclass(type(emitter), MorseEmitter):
                self._emitters.append(emitter)

    def emit(self, morse_string: MorseString):
        if not self._parallel:
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
    def __init__(self, frequency: float = 800.0, words_per_minute: float = 20.0):
        self._frequency = frequency
        self._words_per_minute = min(max(words_per_minute, 1), 60)
        self._seconds_per_unit = MorsePlayer.wpm_to_spu(self._words_per_minute)

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

    def wpm_to_spu(words_per_minute: float) -> float:
        """ Converts the given 'words per minute' into 'seconds per unit'. """
        return 60 / (50 * words_per_minute)

    def spu_to_wpm(seconds_per_unit: float) -> float:
        """ Converts the given 'seconds per unit' into 'words per minute'. """
        return 60 / (50 * seconds_per_unit)

class MorseReceiver:
    """ An abstract base class that allows to receive Morse strings from some shape or form. """
    def receive(self) -> MorseString:
        pass

class MonoReduceFunction:
    SPECIFIC_CHANNEL = 0
    AVERAGE = 1
    MAXIMUM = 2

class MorseSoundReceiver(MorseReceiver):
    """ A Morse sound receiver for extracting Morse strings from sound files. """
    def __init__(self, file_path: str, volume_threshold: float = 0.35, kernel_seconds: float = 0.001,
                 min_signal_seconds: float = 0.01, min_frequency: float = None, max_frequency: float = None):
        self._file_path = file_path
        self._volume_threshold = clamp(volume_threshold, 0.1, 0.9)
        self._kernel_seconds = max(kernel_seconds, 0.0001)
        self._min_signal_seconds = max(min_signal_seconds, 0.01)
        self._min_frequency = min_frequency
        self._max_frequency = max_frequency
        self._show_plots = False

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

    def receive(self) -> MorseString:
        result = MorseString()
        samples, sample_rate = soundfile.read(self._file_path)
        num_samples = len(samples)

        def samples_to_seconds(sample: int) -> float:
            return sample / sample_rate

        def seconds_to_samples(seconds: float) -> int:
            return int(seconds * sample_rate)

        kernel_samples = int(self._kernel_seconds * sample_rate)
        kernel_samples = max(kernel_samples, 6)
        half_kernel_samples = kernel_samples // 2

        # The minimum signal length in samples to be considered (signals of shorter length will be ignored)
        min_signal_samples = int(self._min_signal_seconds * sample_rate)

        # Return empty Morse string if the audio file is too short
        if num_samples < kernel_samples:
            return result

        # Un-tuple the raw samples and force them into mono
        if samples.ndim > 1:
            samples = [tuple[0] for tuple in samples]

        samples_abs = np.array(samples)
        perform_filtering = self._min_frequency is not None or self._max_frequency is not None

        # Analyze the frequency spectrum of the audio signal with the help of FFT
        if perform_filtering:
            fft_result = fft(samples_abs)
            freq_axis = fftfreq(num_samples, d = 1 / sample_rate)

            # Identify major frequency in the audio signal
            magnitude_spectrum = np.abs(fft_result)
            max_magnitude_index = np.argmax(magnitude_spectrum)
            major_frequency = freq_axis[max_magnitude_index]

            # Find indices that correspond to minimum and maximum frequency in the frequency axis
            if self._min_frequency is not None:
                min_freq_index = np.argmax(freq_axis >= self._min_frequency)
            if self._max_frequency is not None:
                max_freq_index = np.argmax(freq_axis >= self._max_frequency)

            # Perform band-pass filtering with the help of inverse FFT
            filtered_fft_result = fft_result.copy()
            filtered_fft_result[:min_freq_index] = 0
            filtered_fft_result[max_freq_index:] = 0
            filtered_samples = ifft(filtered_fft_result)
            filtered_samples = np.real(filtered_samples).astype(samples_abs.dtype)
        else:
            filtered_samples = samples
            filtered_samples_abs = samples_abs

        # Normalize amplitude
        amplitude_factor = 1 / max(filtered_samples)
        filtered_samples *= amplitude_factor
        filtered_samples_abs = [abs(sample) for sample in filtered_samples]

        plot_selection = None
        #plot_selection = (11200, 14400) # test_2.wav
        #plot_selection = (0, 250000) # test_old_recording_denoise.wav

        if plot_selection is None:
            plot_selection = (0, len(filtered_samples))
        t = [samples_to_seconds(sample) for sample in range(plot_selection[0], plot_selection[1])]

        if self._show_plots:
            # Plot frequency domain of the audio signal
            plt.plot(freq_axis, magnitude_spectrum)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.xlim(left = 0, right = max(freq_axis))
            plt.title(f"FFT frequency domain of {self._file_path}")
            plt.show()

            # Plot original and filtered audio signal
            plt.figure(figsize = (20, 6))
            plt.plot(t, samples[plot_selection[0]:plot_selection[1]], label = 'Raw')
            plt.plot(t, filtered_samples[plot_selection[0]:plot_selection[1]], label = 'Filtered')
            plt.xlabel('Time [s]')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.title(f"Raw vs filtered waveform of {self._file_path}")
            plt.show()

        signals = []
        num_on_signals = 0
        num_off_signals = 0
        consecutive_on_signal_samples = 0
        consecutive_off_signal_samples = 0
        previous_signal = None
        num_signal_failures = 0

        # Contains in alternating order the lengths in samples of consecutive 'on' and 'off' signals and must always
        # start with the length of an 'on' signal
        signal_durations = []
        max_kernel_sample = max(samples[0:kernel_samples])

        # If the audio file starts with an 'off' signal immediately, prepend an initial 'on' signal with no length
        # if max_value < self._volume_threshold:
        #     signal_lengths.append(0)

        # Analyze on and off signals
        for i in range(num_samples):
            is_last_sample = i == num_samples - 1

            kernel_left_pos = max(i - half_kernel_samples, 0)
            kernel_right_pos = min(i + half_kernel_samples, num_samples)

            max_kernel_sample = max(filtered_samples_abs[kernel_left_pos:kernel_right_pos])
            current_signal = max_kernel_sample >= self._volume_threshold

            # Change of signal from 'on' to 'off' or vice versa
            if (previous_signal is not None and previous_signal != current_signal) or is_last_sample:
                # Previous was 'on', so process all previous 'on' signals
                if previous_signal == True:
                    if consecutive_on_signal_samples < min_signal_samples:
                        consecutive_off_signal_samples += consecutive_on_signal_samples
                        consecutive_on_signal_samples = 0
                        num_signal_failures += 1
                    elif consecutive_on_signal_samples >= min_signal_samples or is_last_sample:
                        signals.extend([1] * consecutive_on_signal_samples)
                        signal_durations.append(consecutive_on_signal_samples)
                        consecutive_on_signal_samples = 0
                # Previous was 'off', so process all previous 'off' signals
                else:
                    if consecutive_off_signal_samples < min_signal_samples:
                        consecutive_on_signal_samples += consecutive_off_signal_samples
                        consecutive_off_signal_samples = 0
                        num_signal_failures += 1
                    elif consecutive_off_signal_samples >= min_signal_samples or is_last_sample:
                        signals.extend([0] * consecutive_off_signal_samples)
                        consecutive_off_signal_samples = 0

            if current_signal:
                num_on_signals += 1
                consecutive_on_signal_samples += 1
            else:
                num_off_signals += 1
                consecutive_off_signal_samples += 1

            previous_signal = current_signal

        # Determine Morse unit duration in samples and seconds
        morse_unit_samples = int(MorseSoundReceiver.find_unit_duration(signal_durations))
        morse_unit_seconds = samples_to_seconds(morse_unit_samples)

        # Iterate over the signals and check the length of 'on' and 'off' signals with respect to the determined Morse
        # unit length; we classify dits and dahs as well as the three kinds of pauses as forgiving and generous as
        # possible, because it is always better to return the most likely Morse character, even if it is incorrect.

        i = 0
        signal = signals[i]
        len_signals = len(signals)
        while signal == 0 and i < len_signals:
            signal = signals[i]
            i += 1

        last_signal = 1
        last_signal_change_pos = i
        signal_index = 0
        morse_character = ""

        while i < len_signals - 1:
            i += 1
            if signals[i] == last_signal:
                continue

            signal_length = i - last_signal_change_pos
            signal_units = signal_length / morse_unit_samples

            if last_signal == 1:
                morse_character += "." if signal_units < 2 else "-"
            elif 2.2 <= signal_units < 5:
                result += MorseCharacter(morse_character)
                morse_character = ""
            elif 5 <= signal_units:
                result += MorseCharacter(morse_character)
                result += MorseWordPause()
                morse_character = ""

            last_signal = signals[i]
            last_signal_change_pos = i
            signal_index += 1

        if morse_character:
            result += MorseCharacter(morse_character)

        if self._show_plots:
            len_samples = len(samples)
            len_signals = len(signals)
            if len_samples > len_signals:
                last_signal = signals[-1]
                signals.extend([last_signal] * (len_samples - len_signals))

            # Plot waveform and signals
            plt.figure(figsize = (20, 6))
            plt.plot(t, samples[plot_selection[0]:plot_selection[1]], label = 'Raw')
            plt.plot(t, signals[plot_selection[0]:plot_selection[1]], label = 'Processed')
            plt.xlabel('Time [s]')
            plt.ylabel('Magnitude')
            plt.ylim(ymax = 1.05, ymin = 0)
            plt.legend()
            plt.title(f"Waveform and signals of {self._file_path}")
            plt.show()

        return result

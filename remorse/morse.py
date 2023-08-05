from __future__ import annotations
from scipy.fftpack import fft, ifft, fftfreq
from sklearn.cluster import KMeans
from remorse.args import parse_frequency, parse_morse_frequency, parse_sample_rate, parse_speed, parse_time_samples, parse_time_seconds
from remorse.utils import color_to_ansi_escape, clamp, remap, wpm_to_spu, spu_to_wpm, preprocess_input_morse, preprocess_input_text, nwise, Color, ColorizationMode, TextCase, SimpleMovingAverage, StringVerifier
from typing import BinaryIO
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import multiprocessing
import numpy as np
import os
import pyaudio
import random
import re
import scipy.ndimage as ndimage
import scipy.signal as signal
import soundfile
import sys
import time

MORSE_CODE_SPLIT_PATTERN = re.compile(r'( ) *| *(\/)(?: *\/*)*')

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
        return self._string

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
    def __str__(self):
        return '/'

    def __repr__(self):
        return '/'

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
    def __init__(self, input: str | MorseCharacter | list[MorseCharacter | MorseWordPause] | MorseString = None):
        self._chars: list[MorseCharacter | MorseWordPause] = []
        if input is not None:
            if isinstance(input, str):
                parts = MORSE_CODE_SPLIT_PATTERN.split(input)
                for index, part in enumerate(parts):
                    if index % 3 == 0 and part != '':
                        self._chars.append(MorseCharacter(part))
                    elif index % 3 == 2 and part == '/' and len(self._chars) > 0:
                        self._chars.append(MorseWordPause())
                if isinstance(self._chars[-1], MorseWordPause):
                    self._chars = self._chars[:-1]
            elif isinstance(input, MorseCharacter):
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

    def __eq__(self, other):
        return self.__str__() == other.__str__()

class FlushTriggerMode:
    UNSPECIFIED = 0
    AFTER_EACH_RECEIVE = 1
    AFTER_EACH_CHARACTER = 2
    AFTER_EACH_SIGNAL = 3

class StreamReceiver:
    """ An abstract base class that allows to receive data from a stream. """
    def __init__(self):
        self._num_emitted = 0
        self._last_was_pause = False
        self._is_strict = False
        self._verifier = None

    def receive(self, data):
        pass

    def is_strict(self):
        """ Returns `True` if this receiver is strict, that is it only accepts chunks of data that are entirely in the
            correct format and/or contain only supported characters. Each implementation decides for itself what exactly
            strict means and what its constraints are. """
        return self._is_strict

    def set_is_strict(self, is_strict: bool):
        self._is_strict = is_strict

    def set_verifier(self, verifier: StringVerifier):
        self._verifier = verifier

    def verify(self, data) -> tuple[bool | None, str, str]:
        if self._verifier is not None:
            return self._verifier.verify(data, additive = True)
        return (None, data, '')

class Stream:
    def __init__(self):
        self._subscribers: set[StreamReceiver] = set()

    def num_subscribers(self) -> int:
        return len(self._subscribers)

    def subscribe(self, subscriber):
        self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber):
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def send(self, data):
        for subscriber in self._subscribers:
            subscriber.receive(data)

class TextStreamer:
    def __init__(self):
        self._text_stream = Stream()

    def text_stream(self):
        return self._text_stream

class MorseStreamer:
    def __init__(self):
        self._morse_stream = Stream()

    def morse_stream(self):
        return self._morse_stream

class TextPrinter(StreamReceiver):
    """ A simple printer class that implements a stream receiver and prints received text to the specified output
        device, standard output by default. """
    def __init__(self, output_device: sys.TextIO = sys.stdout, text_case: TextCase = TextCase.SENTENCE,
                 flush_after_receive: bool = True):
        super().__init__()
        self._output_device = output_device
        self._text_case = text_case
        self._flush_after_receive = flush_after_receive
        self._new_sentence = True
        self._color_escape = ''

    def set_color(self, color: int | str):
        if isinstance(color, int):
            if 0 <= color <= 7:
                self._color_escape = f'\x1b[3{color}m'
            elif 9 <= color <= 255:
                self._color_escape = f'\x1b[38;5;{color}m'
        elif isinstance(color, tuple) and len(color) == 3:
            r, g, b = color
            self._color_escape = f'\x1b[38;2;{r};{g};{b}m'
        elif isinstance(color, str):
            self._color_escape = color if color.startswith('\x1b[') else color_to_ansi_escape(color)

    def text_case(self):
        return self._text_case

    def set_text_case(self, text_case: TextCase):
        self._text_case = text_case

    def receive(self, data: str):
        if self._output_device is None:
            return
        if not isinstance(data, str):
            data = str(data)
        if len(data) == 0:
            return

        matching, diff_string, _ = self.verify(data)
        output = diff_string if matching == False else data

        if self._text_case == TextCase.LOWER:
            output = output.lower()
        elif self._text_case == TextCase.SENTENCE:
            new_output = ''
            for char in output:
                if self._new_sentence and char.isalpha():
                    new_output += char.upper()
                    self._new_sentence = False
                else:
                    new_output += char.lower()
                    self._new_sentence |= char == '.'
            output = new_output

        if output[-1] == '.':
            self._new_sentence = True

        self._output_device.write(self._color_escape)
        self._output_device.write(output)

        if self._flush_after_receive:
            self._output_device.flush()

class MorseEmitter(StreamReceiver):
    """ An abstract base class that allows to emit Morse strings in some shape or form. """
    def __init__(self):
        super().__init__()

    def emit(self, morse_string: MorseString):
        self.pre_emit()
        last = None
        for morse_char in morse_string:
            if isinstance(morse_char, MorseCharacter):
                if last and isinstance(last, MorseCharacter):
                    self.emit_inter_character_pause()
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
            elif isinstance(morse_char, MorseWordPause):
                self.emit_inter_word_pause()
            last = morse_char
        self.post_emit()

    def receive(self, data):
        if not isinstance(data, str):
            data = str(data)

        for char in data:
            if char == '.':
                if self._num_emitted > 0 and not self._last_was_pause:
                    self.emit_intra_character_pause()
                elif self._num_emitted == 0:
                    self.pre_emit_word()
                    self.pre_emit_character()
                self.pre_emit_symbol()
                self.emit_dit()
                self.post_emit_symbol()
                self._num_emitted += 1
                self._last_was_pause = False

            elif char == '-':
                if self._num_emitted > 0 and not self._last_was_pause:
                    self.emit_intra_character_pause()
                elif self._num_emitted == 0:
                    self.pre_emit_word()
                    self.pre_emit_character()
                self.pre_emit_symbol()
                self.emit_dah()
                self.post_emit_symbol()
                self._num_emitted += 1
                self._last_was_pause = False

            elif char == ' ':
                self.post_emit_character()
                self.emit_inter_character_pause()
                self.pre_emit_character()
                self._num_emitted += 1
                self._last_was_pause = True

            elif char == '/':
                self.post_emit_character()
                self.post_emit_word()
                self.emit_inter_word_pause()
                self.pre_emit_word()
                self.pre_emit_character()
                self._num_emitted += 1
                self._last_was_pause = True

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

class MorsePrinter(MorseEmitter):
    """ A Morse printer that prints Morse strings to the given output device (e.g. console or file). """
    def __init__(self, output_device: sys.TextIO = sys.stdout, color: int | str = 7, dit_symbol: str = '.',
                 dah_symbol: str = '-', pause_char: str = ' ', word_pause_char: str = ' / ',
                 word_pause_color: int | str = '#808080', strip_escape_sequences: bool = False):
        super().__init__()
        self._output_device = output_device
        self._color = color_to_ansi_escape(color, foreground = True)
        self._dit_symbol = dit_symbol
        self._dah_symbol = dah_symbol
        self._pause_char = pause_char
        self._word_pause_char = word_pause_char
        self._strip_escape_sequences = strip_escape_sequences

        self.set_color(color)
        self.set_word_pause_color(word_pause_color)

    def set_color(self, color: int | str):
        self._color = color_to_ansi_escape(color, foreground = True)

    def set_word_pause_color(self, word_pause_color: int | str = 233):
        self._word_pause_color = color_to_ansi_escape(word_pause_color, foreground = True)

    def emit_dit(self):
        matching, diff_string, _ = self.verify('.')
        if matching == False and not self._strip_escape_sequences:
            self._output_device.write(diff_string)
        elif not self._strip_escape_sequences and self._color is not None:
            self._output_device.write(f'{self._color}{self._dit_symbol}')
        else:
            self._output_device.write(self._dit_symbol)
        self._output_device.flush()

    def emit_dah(self):
        matching, diff_string, _ = self.verify('-')
        if matching == False and not self._strip_escape_sequences:
            self._output_device.write(diff_string)
        elif not self._strip_escape_sequences and self._color is not None:
            self._output_device.write(f'{self._color}{self._dah_symbol}')
        else:
            self._output_device.write(self._dah_symbol)
        self._output_device.flush()

    def emit_inter_character_pause(self):
        matching, diff_string, _ = self.verify(' ')
        if matching == False and not self._strip_escape_sequences:
            self._output_device.write(diff_string)
        else:
            self._output_device.write(self._pause_char)
        self._output_device.flush()

    def emit_inter_word_pause(self):
        matching, diff_string, _ = self.verify('/')
        if matching == False and not self._strip_escape_sequences:
            self._output_device.write(diff_string)
        elif not self._strip_escape_sequences and self._word_pause_color is not None:
            self._output_device.write(f'{self._word_pause_color}{self._word_pause_char}')
        else:
            self._output_device.write(self._word_pause_char)
        self._output_device.flush()

class MorseVisualizer(MorseEmitter):
    """ A Morse visualizer that prints Morse characters and strings to the given output device (e.g. console or file)
        by visualizing the duration of the signals and the pauses. """
    def __init__(self, output_device: sys.TextIO = sys.stdout):
        super().__init__()
        self._output_device = output_device
        self._colors = [ Color.RED ]
        self._color_index = 0
        self._colorization_mode = ColorizationMode.NONE

    def normal_to_visual(normal: str):
        visual = normal.replace(' ', '\u2003' * 3).replace('/', '\u2003' * 7)
        visual = visual.replace('.', '▄\u2003').replace('-', '▄▄▄\u2003')
        if visual.endswith('▄\u2003'):
            visual = visual[:-1]
        return visual

    def pre_emit_word(self):
        if self._colorization_mode == ColorizationMode.WORDS:
            self._set_color_code()

    def pre_emit_character(self):
        if self._colorization_mode == ColorizationMode.CHARACTERS:
            self._set_color_code()

    def pre_emit_symbol(self):
        if self._colorization_mode == ColorizationMode.SYMBOLS:
            self._set_color_code()

    def post_emit(self):
        self._output_device.write('\x1b[0m')
        self._output_device.flush()

    def emit_dit(self):
        matching, _, expected = self.verify('.')
        if matching == False:
            expected = MorseVisualizer.normal_to_visual(expected)
            self._output_device.write(f'\x1b[31m▄\x1b[32m{expected}')
            self._set_color_code(False)
        else:
            self._output_device.write('▄')
        self._output_device.flush()

    def emit_dah(self):
        matching, _, expected = self.verify('-')
        if matching == False:
            expected = MorseVisualizer.normal_to_visual(expected)
            self._output_device.write(f'\x1b[31m▄▄▄\x1b[32m{expected}')
            self._set_color_code(False)
        else:
            self._output_device.write('▄▄▄')
        self._output_device.flush()

    def emit_intra_character_pause(self):
        self._output_device.write('\u2003')
        self._output_device.flush()

    def emit_inter_character_pause(self):
        pause = '\u2003' * 3
        matching, _, expected = self.verify(' ')
        if matching == False:
            expected = MorseVisualizer.normal_to_visual(expected)
            pause = '⎵' * 3
            self._output_device.write(f'\x1b[31m{pause}\x1b[32m{expected}')
            self._set_color_code(False)
        else:
            self._output_device.write(pause)
        self._output_device.flush()

    def emit_inter_word_pause(self):
        pause = '\u2003' * 7
        matching, _, expected = self.verify('/')
        if matching == False:
            expected = MorseVisualizer.normal_to_visual(expected)
            pause = '⎵' * 7
            self._output_device.write(f'\x1b[31m{pause}\x1b[32m{expected}')
            self._set_color_code(False)
        else:
            self._output_device.write(pause)
        self._output_device.flush()

    def set_colorization_mode(self, colorization_mode: ColorizationMode):
        self._colorization_mode = colorization_mode

    def set_colors(self, *colors: int):
        new_colors = []
        for color in colors:
            if isinstance(color, list):
                new_colors.extend(color)
            else:
                new_colors.append(color)
        if len(new_colors):
            self._colors = new_colors

    def _set_color_code(self, advance_color_index: bool = True):
        if self._colorization_mode == ColorizationMode.NONE:
            return
        new_color = self._colors[self._color_index]
        if isinstance(new_color, int):
            if 0 <= new_color <= 7:
                self._output_device.write(f'\x1b[3{new_color}m')
            elif 9 <= new_color <= 255:
                self._output_device.write(f'\x1b[38;5;{new_color}m')
        elif isinstance(new_color, tuple) and len(new_color) == 3:
            r, g, b = new_color
            self._output_device.write(f'\x1b[38;2;{r};{g};{b}m')
        elif isinstance(new_color, str):
            ansi_escape = new_color if new_color.startswith('\x1b[') else color_to_ansi_escape(new_color)
            self._output_device.write(ansi_escape)

        if advance_color_index:
            self._color_index = (self._color_index + 1) % len(self._colors)

class MorsePlayer(MorseEmitter):
    """ A Morse player that plays Morse characters and strings as sounds on the default audio device. """
    def __init__(self, frequency: int | float | str = 800.0, speed: int | float | str = 20.0, volume: float = 0.9,
                 sample_rate: int | float | str = 8000):
        super().__init__()
        self._muted = False
        self._pyaudio = None
        self._stream = None
        self._sample_rate = None
        self.set_frequency(frequency)
        self.set_speed(speed)
        self.set_volume(volume)
        self.set_sample_rate(sample_rate)

        self.open()

    def __del__(self):
        self.close()

    def frequency(self):
        return self._frequency

    def set_frequency(self, frequency: int | float | str):
        if isinstance(frequency, str):
            if parsed := parse_morse_frequency(frequency):
                self._frequency = parsed
        else:
            self._frequency = clamp(frequency, 100, 10000)

    def speed(self):
        return self._words_per_minute

    def set_speed(self, speed: int | float | str):
        if isinstance(speed, str):
            if parsed := parse_speed(speed):
                self._words_per_minute = parsed
        else:
            self._words_per_minute = clamp(speed, 2, 60)
        self._seconds_per_unit = wpm_to_spu(self._words_per_minute)

    def volume(self):
        return self._volume

    def set_volume(self, volume: float):
        self._volume = clamp(volume, 0, 1)

    def sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, sample_rate: int | float | str):
        old_sample_rate = self._sample_rate
        new_sample_rate = old_sample_rate
        if isinstance(sample_rate, str):
            if parsed := parse_sample_rate(sample_rate):
                new_sample_rate = parsed
        else:
            new_sample_rate = clamp(sample_rate, 1000, 192000)

        if new_sample_rate != old_sample_rate:
            self._sample_rate = new_sample_rate

            # Re-initialize the audio stream with the new sample rate
            if self._pyaudio is not None:
                self.close()
                self.open()

    def muted(self):
        return self._muted

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False

    def open(self):
        if self._pyaudio is not None:
            return
        self._pyaudio = pyaudio.PyAudio()
        try:
            self._stream = self._pyaudio.open(format = pyaudio.paFloat32, channels = 1, rate = self._sample_rate,
                                              output = True)
        except:
            print("Error: Could not establish connection to default audio device", file = sys.stderr)
            self._pyaudio = None

    def close(self):
        if self._pyaudio is None:
            return
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()
        self._stream = None
        self._pyaudio = None

    def play_sine_wave(self, duration: float, volume_multiplier: float = 1.0):
        if self._stream is None:
            return

        points = int(self._sample_rate * duration)

        if self._muted:
            data = np.zeros(points).astype(np.float32)
        else:
            times = np.linspace(0, duration, points, endpoint = False)
            data = np.sin(times * self._frequency * 2 * np.pi).astype(np.float32)

        bytes = (data * self._volume * clamp(volume_multiplier, 0, 1)).tobytes()
        self._stream.write(bytes)

    def emit_dit(self):
        self.play_sine_wave(self._seconds_per_unit)

    def emit_dah(self):
        self.play_sine_wave(self._seconds_per_unit * 3)

    def emit_pause(self, num_instances: int = 1):
        num_instances = num_instances if num_instances in { 1, 3, 7 } else 1
        self.play_sine_wave(self._seconds_per_unit * num_instances, volume_multiplier = 0)

    def emit_intra_character_pause(self):
        self.emit_pause(num_instances = 1)

    def emit_inter_character_pause(self):
        self.emit_pause(num_instances = 3)

    def emit_inter_word_pause(self):
        self.emit_pause(num_instances = 7)

class MorseReader:
    """ An abstract base class that allows to receive Morse strings from some shape or form. """
    def __init__(self):
        pass

    def read(self) -> MorseString:
        pass

class MorseStringStreamer(MorseReader, MorseStreamer, TextStreamer):
    """ A Morse string receiver for extracting text and Morse from an input string. """
    def __init__(self, data: str, data_is_morse: bool):
        MorseReader.__init__(self)
        MorseStreamer.__init__(self)
        TextStreamer.__init__(self)
        self._data = preprocess_input_morse(data) if data_is_morse else preprocess_input_text(data)
        self._data_is_morse = data_is_morse

    def read(self):
        if self._data_is_morse:
            self.read_morse()
        else:
            self.read_text()

    def split_morse_characters(morse_string: str):
        iterator = MORSE_CODE_SPLIT_PATTERN.finditer(morse_string)
        start_index = 0

        for match in iterator:
            delimiter = match.group(1)
            end_index = match.start()
            yield morse_string[start_index:end_index], delimiter
            start_index = match.end()

        yield morse_string[start_index:], None

    def read_morse(self):
        def decode_morse_char(morse_char):
            if morse_char in MORSE_TO_LATIN:
                text_char = MORSE_TO_LATIN[morse_char]
                self.text_stream().send(text_char)
            else:
                self.text_stream().send('#')

        morse_char = ''
        for morse_symbol in self._data:
            self.morse_stream().send(morse_symbol)
            if morse_symbol in { ' ', '/' }:
                decode_morse_char(morse_char)
                if morse_symbol == '/':
                    self.text_stream().send(' ')
                morse_char = ''
            else:
                morse_char += morse_symbol

        if len(morse_char) > 0:
            decode_morse_char(morse_char)
            morse_char = ''

    def read_text(self):
        last_was_char = False
        for text_char in self._data:
            self.text_stream().send(text_char)

            if last_was_char and text_char not in { ' ', '\n', '\r' }:
                self.morse_stream().send(' ')

            if text_char in LATIN_TO_MORSE:
                morse_char = LATIN_TO_MORSE[text_char]
                for morse_symbol in morse_char:
                    self.morse_stream().send(morse_symbol)
                last_was_char = True
            elif text_char in { '\n', '\r' }:
                self.morse_stream().send(text_char)
                last_was_char = False
            elif text_char == ' ':
                self.morse_stream().send('/')
                last_was_char = False
            else:
                self.morse_stream().send('#')
                last_was_char = True

class MorseSoundStreamer(MorseReader, MorseStreamer, TextStreamer):
    """ A Morse sound receiver for extracting Morse strings from microphone or line input. """

    # Corresponds to 20 words per minute
    INITIAL_UNIT_DURATION = 0.06

    # The maximum duration of samples and signals that can be plotted (given in seconds)
    MAXIMUM_PLOTTABLE_DURATION = 30

    def __init__(self, device: str = 'microphone', input: bool = True, output: bool = True, open: bool = True,
                 threshold: float = 0.35, sample_rate: int | float | str = 8000, min_signal_size: str = '0.01s',
                 filtering_mode: str = 'none', filtering_args: str | list[str] = [], noise_reduction_mode: str = 'none',
                 normalization_mode: str = 'scale', buffer_size: str = '2s', plot: bool = False,
                 output_filtered_sound_file: bool = False, debug_args = {}):
        MorseReader.__init__(self)
        MorseStreamer.__init__(self)
        TextStreamer.__init__(self)
        self._device = device
        self._device_name = device
        self._device_is_file = None
        self._input = input
        self._output = output
        self._threshold = clamp(threshold, 0.1, 0.9) if threshold is not None else 0.35
        self._sample_rate = None
        self._min_signal_size = min_signal_size or '0.01s'
        self._min_signal_samples = None
        self._filtering_mode = filtering_mode
        self._filtering_args = filtering_args
        self._noise_reduction_mode = noise_reduction_mode
        self._normalization_mode = normalization_mode
        self._buffer_size = buffer_size
        self._buffer_samples = None
        self._dont_use_file_buffer = True
        self._plot = plot
        self._output_filtered_sound_file = output_filtered_sound_file
        self._pyaudio = None
        self._pyaudio_stream = None
        self._data_buffer = np.array([])
        self._signals_backlog = np.array([])
        self._maximum_backlog = np.array([])
        self._filtered_backlog = np.array([])
        self._data_backlog = np.array([])
        self._filtered_samples = np.array([])
        self._input_file_sample_rate = None
        self._input_file = None
        self._output_file = None
        self._kmeans = KMeans(n_init = 10, n_clusters = 2)
        self._unit_duration = SimpleMovingAverage(4)
        self._current_character = ""
        self._current_string = ""
        self._current_decoded_string = ''
        self._num_characters_emitted = 0
        self._num_symbols_emitted = 0
        self._num_samples_analyzed = 0
        self._num_chunks_analyzed = 0
        self._sample_position = 0
        self._last_pause_samples = 0
        self._total_read_duration = 0
        self._total_analyze_duration = 0
        self._debug_args = debug_args
        self._parallelize_io = 'dont-parallelize-io' not in self._debug_args and not self._plot

        self.set_sample_rate(sample_rate)
        self.initialize()

        if open:
            self.open()

    def __del__(self):
        self.deinitialize()

    def sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, sample_rate: int | float | str):
        old_sample_rate = self._sample_rate
        new_sample_rate = old_sample_rate
        if isinstance(sample_rate, str):
            if parsed := parse_sample_rate(sample_rate):
                new_sample_rate = int(parsed)
        else:
            new_sample_rate = int(clamp(sample_rate, 1000, 192000))

        if new_sample_rate != old_sample_rate:
            self._sample_rate = new_sample_rate
            buffer_samples = parse_time_samples(self._buffer_size, self._sample_rate)
            if buffer_samples == 0:
                self._dont_use_file_buffer = True
            self._buffer_samples = int(max(buffer_samples, self._sample_rate))
            self._min_signal_samples = int(max(parse_time_samples(self._min_signal_size, self._sample_rate),
                                               0.01 * self._sample_rate))

            if self._unit_duration.empty():
                # Reset the unit duration to the number of samples corresponding to the initial unit duration
                self._unit_duration.reset(self._sample_rate * MorseSoundStreamer.INITIAL_UNIT_DURATION)

            elif old_sample_rate is not None:
                # Update the unit duration that we already found by multiplying with the sample rate factor
                factor = new_sample_rate / old_sample_rate
                self._unit_duration.multiply(factor)

            # Re-initialize the audio stream with the new sample rate
            if self._pyaudio_stream is not None:
                self.close()
                self.open()

    def nyquist_frequency(self):
        return 0.5 * self._sample_rate

    def do_plot(self):
        if not self._plot:
            return False
        elif 'plot-chunks' in self._debug_args:
            plot_chunks_str = set(self._debug_args['plot-chunks'].split(','))
            plot_chunks = { int(chunk_id) for chunk_id in plot_chunks_str }
            return self._num_chunks_analyzed in plot_chunks
        else:
            return True

    def initialize(self):
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

    def deinitialize(self):
        self.close()
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def open(self):
        if self._pyaudio_stream is not None:
            return

        self._device_is_file = self._device is not None and os.path.isfile(self._device)

        # If device is a file
        if self._device_is_file:
            self._input_file = soundfile.SoundFile(self._device)
            self.set_sample_rate(self._input_file.samplerate)

            if self._output_filtered_sound_file and self._filtering_mode.lower() != 'none':
                file_name, _ = os.path.splitext(os.path.basename(self._device))
                output_file_path = os.path.join(os.path.dirname(self._device), f'{file_name}_filtered.wav')
                self._output_file = soundfile.SoundFile(output_file_path, 'w', samplerate = self._input_file.samplerate,
                                                        channels = 1, subtype = 'PCM_16', format = 'WAV',
                                                        endian = self._input_file.endian)

        if self._buffer_samples > 0: frames_per_buffer = self._buffer_samples
        else: frames_per_buffer = pyaudio.paFramesPerBufferUnspecified

        if self._input:
            device_index = None
            if not self._device_is_file:

                # Print available input devices and ask for device index
                if self._device is None or self._device == '':
                    info = self._pyaudio.get_host_api_info_by_index(0)
                    numdevices = info.get('deviceCount')
                    devices = {}

                    for i in range(numdevices):
                        if self._pyaudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                            device_name = self._pyaudio.get_device_info_by_host_api_device_index(0, i).get('name')
                            print(f"{i}: {device_name}")
                            devices[i] = device_name

                    idx = None
                    while idx is None or not idx.isdigit() or int(idx) not in devices:
                        if idx is not None: prompt = "Please insert a valid index from the list: "
                        else: prompt = "Select input device index: "
                        idx = input(prompt)

                    device_index = int(idx)
                    self._device_name = devices[device_index]
            try:
                callback = self.read_callback if self._parallelize_io else None
                self._pyaudio_stream = self._pyaudio.open(input_device_index = device_index, format = pyaudio.paFloat32,
                                                          input = True, output = self._output, channels = 1,
                                                          rate = self._sample_rate, frames_per_buffer = frames_per_buffer,
                                                          stream_callback = callback)
            except:
                print(f"Error: Could not establish connection to audio device '{self._device}'", file = sys.stderr)
                return
        else:
            try:
                callback = self.write_callback if self._parallelize_io else None
                self._pyaudio_stream = self._pyaudio.open(format = pyaudio.paFloat32, output = True, channels = 1,
                                                          rate = self._sample_rate,
                                                          frames_per_buffer = frames_per_buffer,
                                                          stream_callback = callback)
            except:
                print(f"Error: Could not establish connection to audio device '{self._device}'", file = sys.stderr)
                return

    def close(self):
        if self._pyaudio_stream is not None:
            self._pyaudio_stream.stop_stream()
            self._pyaudio_stream.close()
            self._pyaudio_stream = None

        if self._input_file:
            self._input_file.close()
            self._input_file = None

        if self._output_file:
            self._output_file.close()
            self._output_file = None

    def emit_rest(self):
        if self._current_character:
            self._current_string += self._current_character
            decoded = MORSE_TO_LATIN[self._current_character] if self._current_character in MORSE_TO_LATIN else "#"
            if 'debug-print' in self._debug_args:
                print(decoded + '\x1b[0m', end = '', flush = True)
            for morse_symbol in self._current_character:
                self.morse_stream().send(morse_symbol)
            self.text_stream().send(decoded)
            self._current_character = ""

    def read(self):
        if self._pyaudio is None or self._pyaudio_stream is None:
            return

        if self._parallelize_io:
            self._pyaudio_stream.start_stream()

        # If device is a file
        if self._device_is_file:
            if self._parallelize_io:
                while self._pyaudio_stream.is_active():
                    time.sleep(0.1)
            else:
                while True:
                    _, return_code = self.read_callback(None, None, None, None)
                    if return_code == pyaudio.paComplete:
                        break
            self.emit_rest()

        # If device is the name of an audio device
        else:
            print(f"Recording from {self._device_name}")
            input("Press 'Return' to stop recording...\n")
            self.emit_rest()
            if 'debug-print' in self._debug_args:
                print("\x1b[0m", end = '')

        if self._parallelize_io:
            self._pyaudio_stream.stop_stream()

        # Debug argument guided analysis
        if 'dont-analyze-in-callback' in self._debug_args:
            if 'analyze-in-chunks' in self._debug_args:
                if 'analyze-chunk-size' in self._debug_args:
                    chunk_samples = max(int(parse_time_samples(self._debug_args['analyze-chunk-size'],
                                                               self._sample_rate)), self._sample_rate)
                else:
                    chunk_samples = self._buffer_samples
                for i in range(0, len(self._data_buffer), chunk_samples):
                    start = i
                    end = min(start + chunk_samples, len(self._data_buffer))
                    self.analyze(self._data_buffer[start:end], data_is_chunk = chunk_samples < len(self._data_buffer))
                self.emit_rest()
            else:
                self.analyze(self._data_buffer, data_is_chunk = False)

        return self._current_string

    def read_callback(self, data, frame_count, time_info, flags):
        complete = False
        if self._device_is_file:
            # Stream audio data from a file
            samples = self._input_file.read(frames = -1 if self._dont_use_file_buffer else self._buffer_samples,
                                            always_2d = True)
            samples = np.mean(samples, axis = 1)
            frames_read = len(samples)

            if self._buffer_samples > 0 and frames_read == 0:
                complete = True

            elif self._buffer_samples == 0:
                self._data_buffer = samples
                if 'dont-analyze-in-callback' not in self._debug_args:
                    self.analyze(self._data_buffer, data_is_chunk = False)
                return samples, pyaudio.paComplete

            data = samples
        else:
            # Stream audio data from a sound device
            samples = np.fromstring(data, dtype = np.float32)

        self._data_buffer = np.append(self._data_buffer, samples)

        if 'dont-analyze-in-callback' not in self._debug_args:
            self.analyze(samples, data_is_chunk = True)

        return data, pyaudio.paComplete if complete else pyaudio.paContinue

    def write_callback(self, data, frame_count, time_info, flags):
        return data, pyaudio.paContinue

    def remove_short_signals(signals, minimum_signal_length: int):
        """ Removes sequences in signals that are shorter than the given minimum signal lengths by merging them into
            their neighboring signals. """
        transitions = np.where(signals[:-1] != signals[1:])[0] + 1

        # Remove all short 1's spikes
        for i0, i1 in nwise(transitions, 2):
            if signals[i0] == 1.0 and i1 - i0 < minimum_signal_length:
                signals[i0:i1] = -1.0

        # Remove all short 0's spikes
        for i0, i1 in nwise(transitions, 2):
            if signals[i0] == 0.0 and i1 - i0 < minimum_signal_length:
                signals[i0:i1] = 1.0

        signals = np.clip(signals, 0.0, 1.0, out = signals)

    def find_unit_duration(self, data, hint) -> float:
        """ Finds and returns the unit duration derived from the given input data using k-means clustering. """
        data_set = set(data)
        if len(data_set) == 1:
            # Over simplification: If we only have one unique duration in the data, assume it as the unit duration;
            # this is still better than returning no estimate at all
            return list(data_set)[0]
        if len(data_set) < 2:
            return None

        data = np.array(data).reshape(-1, 1)

        # Find two factors with the approximate ratio 1:3 using k-means clustering
        self._kmeans.fit(data)
        durations = sorted(self._kmeans.cluster_centers_.flatten())

        # Return the average of both 1-unit and 3-unit durations as "the" unit duration
        unit_duration = sum(durations) / 4

        # Use the hint (e.g. previous unit duration) to detect and avoid outliers
        if hint is not None:
            adjust = False

            if unit_duration > hint * 1.3:
                # The new unit duration is noticeably higher than the hint; the reason may be that we only received
                # 3-unit durations in the given data; add a single 1-unit durations and adjust the result
                data = np.append(data, hint)
                adjust = True
            elif unit_duration < hint * 0.77:
                # The new unit duration is noticeably lower than the hint; the reason may be that we only received
                # 1-unit durations in the given data; add a single 3-unit durations and adjust the result
                data = np.append(data, hint * 3)
                adjust = True

            if adjust:
                data = np.array(data).reshape(-1, 1)

                self._kmeans.fit(data)
                durations = sorted(self._kmeans.cluster_centers_.flatten())

                # Return the average of both single and triple unit durations as "the" unit duration
                unit_duration = sum(durations) / 4

        return unit_duration

    def perform_fft(self, samples):
        """ Performs the fast Fourier transform required by subsequent algorithms. """
        self._fft_result = fft(samples)
        self._fft_magnitudes = np.abs(self._fft_result)[:len(samples) // 2 + 1]
        freq_space = np.linspace(0, self.nyquist_frequency(), len(self._fft_magnitudes))

        if self.do_plot():
            fig, ax = plt.subplots()
            ax.set_title('Fast Fourier Transform', pad = 20, size = 17, fontweight = 'bold')
            ax.set_ylabel('Magnitude')
            ax.set_xlabel('Frequency [Hz]')
            ax.plot(freq_space, self._fft_magnitudes)
            plt.show(block = True)

    def reduce_noise(self, samples):
        """ Reduces background noise in the given samples. """
        # Compute power spectral density and find the greatest power value
        psd = self._fft_result * np.conjugate(self._fft_result) / len(samples)
        max_power = np.max(psd)

        if self._noise_reduction_mode.lower() in { 'h', 'high' }:
            power_threshold = max_power * 0.1
        elif self._noise_reduction_mode.lower() in { 'm', 'medium' }:
            power_threshold = max_power * 0.05
        else: #self._noise_reduction_mode.lower() in { 'l', 'low', 'a', 'auto' }:
            power_threshold = max_power * 0.02

        # Create a power mask and set those indices to 1 that are greater than the power threshold
        power_mask = psd > power_threshold

        # Multiply by the power mask to zero out anything that is not powerful enough
        freq_domain = power_mask * self._fft_result

        # Perform inverse transform
        time_domain = ifft(freq_domain)
        return np.real(time_domain).astype(samples.dtype)

    def filter_samples(self, samples):
        """ Filters certain frequencies from the given samples. """
        # Perform band pass filtering based on given frequency range
        if self._filtering_mode.lower() in { 'b', 'band', 'bandpass' }:
            lowpass_arg = self._filtering_args[0] if isinstance(self._filtering_args, list) else self._filtering_args
            lowpass_freq = parse_frequency(lowpass_arg, 50, 192000, False)
            lowpass_freq_normalized = lowpass_freq / self.nyquist_frequency()
            highpass_arg = self._filtering_args[1] if isinstance(self._filtering_args, list) else self._filtering_args
            highpass_freq = parse_frequency(highpass_arg, 50, 192000, False)
            highpass_freq_normalized = highpass_freq / self.nyquist_frequency()
            b, a = signal.butter(4, [lowpass_freq_normalized, highpass_freq_normalized], btype = 'bandpass')
            return signal.lfilter(b, a, samples)

        # Perform low pass filtering based on a given threshold frequency
        elif self._filtering_mode.lower() in { 'l', 'low', 'lowpass' }:
            lowpass_arg = self._filtering_args[0] if isinstance(self._filtering_args, list) else self._filtering_args
            lowpass_freq = parse_frequency(lowpass_arg, 50, 192000, False)
            lowpass_freq_normalized = lowpass_freq / self.nyquist_frequency()
            b, a = signal.butter(4, lowpass_freq_normalized, btype = 'lowpass')
            return signal.lfilter(b, a, samples)

        # Perform high pass filtering based on a given threshold frequency
        elif self._filtering_mode.lower() in { 'h', 'high', 'highpass' }:
            highpass_arg = self._filtering_args[0] if isinstance(self._filtering_args, list) else self._filtering_args
            highpass_freq = parse_frequency(highpass_arg, 50, 192000, False)
            highpass_freq_normalized = highpass_freq / self.nyquist_frequency()
            b, a = signal.butter(4, highpass_freq_normalized, btype = 'highpass')
            return signal.lfilter(b, a, samples)

        # Perform advanced automatic filtering
        elif self._filtering_mode.lower() in { 'a', 'auto' }:
            # Perform fast Fourier transform
            num_bins = len(self._fft_result)

            # Calculate the frequency occurrence count (number of times each frequency appears)
            # You can use a threshold to determine if a frequency is present or not
            # For example, if the magnitude is above a certain threshold, consider the frequency present
            freq_threshold = 0.5
            freq_presence = self._fft_magnitudes > freq_threshold
            freq_occurrence = np.sum(freq_presence, axis = 0)

            # Calculate the prominence metric by multiplying the magnitudes with the occurrence count
            prominence_metric = self._fft_magnitudes * freq_occurrence

            # Sort the frequencies based on the prominence metric in descending order
            sorted_indices = np.argsort(prominence_metric)[::-1]

            # Get the top frequencies
            frequencies = fftfreq(num_bins, d = 1 / self._sample_rate)
            top_frequency = frequencies[sorted_indices[0]]

            low_cutoff_normalized = (top_frequency - 25) / self.nyquist_frequency()
            high_cutoff_normalized = (top_frequency + 25) / self.nyquist_frequency()
            b, a = signal.butter(4, [low_cutoff_normalized, high_cutoff_normalized], btype = 'bandpass')
            return signal.lfilter(b, a, samples)

        # Unsupported filter type
        elif self._filtering_mode.lower() not in { 'none', 'auto' }:
            pass

        return samples

    def normalize_samples(self, samples, strength: float = 0.5):
        """ Normalizes the given samples (amplitudes) and returns the resulting magnitudes. """
        if self._normalization_mode.lower() in { 'r', 'remap' }:
            magnitudes = np.abs(samples)

            # Calculate the finite derivative of the magnitudes
            diff = np.abs(np.diff(magnitudes, 3))
            diff = np.append(diff, [1, 1, 1])

            # Filter only the plateaus so we get the different absolute amplitudes from the samples
            plateaus = magnitudes[diff < 0.001]
            reshaped = np.array(plateaus).reshape(-1, 1)
            strength = clamp(strength, 0, 1)

            # We use k-means clustering here with two clusters since our Morse signal is assumed to be groupable into
            # two loudness bins
            means = KMeans(n_init = 10, n_clusters = 2)
            means.fit(reshaped)

            # Find out if there really are just two loudness bins in the samples
            inertia = means.inertia_

            # Extract lower and upper magnitudes that become the new lower and upper extremes
            lower, upper = sorted(means.cluster_centers_.flatten())

            # Adjust lower and upper magnitudes with respect to the given strength
            lower = lower * strength
            upper = 1 - ((1 - upper) * strength)

            # Since the lower of the two loudness bins is considered to represent the abscence of a signal we use it as
            # the lower bound of our normalization
            return remap(magnitudes, lower, upper, 0, 1, True)

        elif self._normalization_mode.lower() in { 's', 'scale', 'a', 'auto' }:
            return samples / np.max(np.abs(samples))

        return samples

    def analyze(self, data, data_is_chunk: bool):
        if data is None or len(data) == 0:
            return

        if 'analyze-sleep' in self._debug_args:
            analyze_sleep = max(parse_time_seconds(self._debug_args['analyze-sleep'], None), 0)
            time.sleep(analyze_sleep)

        symbol_sleep = 0
        if 'symbol-sleep' in self._debug_args:
            symbol_sleep = max(parse_time_seconds(self._debug_args['symbol-sleep'], None), 0)

        character_sleep = 0
        if 'character-sleep' in self._debug_args:
            character_sleep = max(parse_time_seconds(self._debug_args['character-sleep'], None), 0)

        simulated_error_percentage = 0
        if 'simulated-error-percentage' in self._debug_args:
            simulated_error_percentage = clamp(float(self._debug_args['simulated-error-percentage']), 0, 1)

        # Perform fast Fourier transform (required by subsequent algorithms)
        self.perform_fft(data)

        # Perform noise reduction if requested
        cleaned = self.reduce_noise(data) if self._noise_reduction_mode.lower() != 'none' else data

        # Perform filtering if requested
        filtered = self.filter_samples(cleaned) if self._filtering_mode.lower() != 'none' else cleaned

        # Normalize the filtered samples if requested
        if self._normalization_mode.lower() != 'none':
            filtered = self.normalize_samples(filtered)

        maximum_filter_size = 30
        maximum = ndimage.maximum_filter1d(np.abs(filtered), size = maximum_filter_size)

        signals = np.copy(maximum)
        signals[maximum < self._threshold] = 0
        signals[maximum >= self._threshold] = 1

        signals_before = np.copy(signals)

        # Remove outlier signals
        MorseSoundStreamer.remove_short_signals(signals, self._min_signal_samples)

        if len(self._signals_backlog):
            signals = np.insert(signals, 0, self._signals_backlog)
            filtered = np.insert(filtered, 0, self._filtered_backlog)
            data = np.insert(data, 0, self._data_backlog)
            self._signals_backlog = np.array([])
            self._maximum_backlog = np.array([])
            self._filtered_backlog = np.array([])
            self._data_backlog = np.array([])

        def chunk_sample_to_total_sample(chunk_sample: int) -> int:
            return self._sample_position + chunk_sample

        num_edge_values = 10
        last_values = signals[-num_edge_values:]
        ones_in_last_values = np.greater_equal(last_values, 1.0)

        if np.any(ones_in_last_values):
            # A signal seems to be split between two chunks of samples; split the current chunk samples right before it
            # and move its last bit into the backlog

            # Find the last signal transition
            for i in range(len(signals) - num_edge_values, -1, -1):
                if signals[i] == 0.0:
                    index = i + 1
                    remains = signals[:index]
                    signals[-num_edge_values:] = 1.0
                    self._signals_backlog = np.append(self._signals_backlog, signals[index:])
                    self._maximum_backlog = np.append(self._maximum_backlog, maximum[index:])
                    self._filtered_backlog = np.append(self._filtered_backlog, filtered[index:])
                    self._data_backlog = np.append(self._data_backlog, data[index:])
                    signals = remains
                    signals_before = signals_before[:index]
                    filtered = filtered[:index]
                    maximum = maximum[:index]
                    data = data[:index]
                    break

        if self._output_file:
            self._output_file.write(filtered)

        first_values = signals[:num_edge_values]
        ones_in_first_values = np.greater_equal(first_values, 1.0)

        if np.all(signals[0]) == 0.0 and np.any(ones_in_first_values):
            signals[:num_edge_values] = 1.0

        # Describes the indices of changes in the signals (from token to pause or vice versa)
        transitions = np.insert(np.where(signals[:-1] != signals[1:])[0], 0, 0)
        transitions = np.append(transitions, len(signals) - 1)

        # Describes what remainder tokens have within the signals (as compared to pauses)
        signal_token_remainder = 0 if np.all(signals[0]) == 1.0 else 1

        plot_offset = 0

        # Adjust the transitions if we have to prepend/extend a pause at the beginning, that was at the end of the
        # previous chunk and that could not be resolved; since the boundaries of chunks may lie in the middle of pauses
        # we cannot decide whether we have to decode a inter character or inter word pause; the decision has to be post-
        # poned until the following chunk is analyzed.
        if self._last_pause_samples > 0:
            transitions[1:] += self._last_pause_samples
            if signal_token_remainder == 0:
                # If the signals in this chunk start with a token we have to prepend a pause and change the remainder
                transitions = np.insert(transitions, 1, self._last_pause_samples)
                signal_token_remainder = 1
            plot_offset = self._last_pause_samples
            self._last_pause_samples = 0

        # Describes the lengths of signals; whether this starts with tokens or pauses is defined by the signal token
        # remainder
        signal_lengths = [transitions[i + 1] - transitions[i] for i in range(0, len(transitions) - 1)]

        # Describes the lengths of all tokens (signals without pauses); used to determine the Morse unit duration
        token_lengths = [signal_lengths[i] for i in range(signal_token_remainder, len(signal_lengths), 2)]

        if len(token_lengths) > 1:
            unit_duration = self.find_unit_duration(token_lengths, self._unit_duration.sma())
            if unit_duration is not None and unit_duration != 0:
                self._unit_duration.update(unit_duration)

        if self._unit_duration.sma() is None or self._unit_duration.sma() == 0:
            self._sample_position += len(data)
            self._num_chunks_analyzed += 1
            return

        plot_signals = self.do_plot() and len(data) < self._sample_rate * MorseSoundStreamer.MAXIMUM_PLOTTABLE_DURATION

        # Plot absolute and filtered samples as well as signals
        if plot_signals:
            sample_space = np.arange(self._sample_position, self._sample_position + len(data))

            mpl.rcParams['agg.path.chunksize'] = 10000

            # Plot and axes settings
            fig, ax1 = plt.subplots(figsize = (24, 6))
            plt.subplots_adjust(left = 0.04, right = 0.96, top = 0.85, bottom = 0.17)
            average_unit_samples = int(self._unit_duration.sma())
            average_unit_milliseconds = int(self._unit_duration.sma() / self._sample_rate * 1000)
            ax1.set_title(f'Detected Signals (1 unit ≈ {average_unit_samples} smp ≈ {average_unit_milliseconds} ms)',
                          pad = 20, size = 17, fontweight = 'bold')
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(15, min_n_ticks = 15))
            ax1.ticklabel_format(useOffset = False)
            ax1.set_ylabel('Magnitude')
            ax1.set_ylim(ymax = 1.1, ymin = 0)
            ax1.set_xlabel('Time [smp]')
            ax1.set_xlim(left = sample_space[0], right = sample_space[-1] + 1)

            def samples_to_seconds(smp):
                return smp / self._sample_rate
            def seconds_to_samples(s):
                return s * self._sample_rate

            ax2 = ax1.secondary_xaxis('top', functions = (samples_to_seconds, seconds_to_samples))
            ax2.ticklabel_format(useOffset = False)
            ax2.set_xlabel('Time [s]')
            ax2.get_xaxis().set_major_locator(ticker.MaxNLocator(15, min_n_ticks = 15))

            # Plots
            ax1.plot(sample_space, data, label = 'Original Waveform', color = '#f0c47e')
            if self._filtering_mode.lower() != 'none':
                ax1.plot(sample_space, filtered, label = 'Filtered Waveform', color = '#2bc49a')
            ax1.plot(sample_space, maximum, label = 'Maximum Filtered', color = '#0a8751')
            ax1.plot(sample_space, signals_before, label = 'Raw Signals', color = '#fcbdc1', alpha = 0.4, zorder = 0)
            ax1.plot(sample_space, signals, label = 'Filtered Signals', color = '#ff5f5f')
            ax1.plot(sample_space, np.full((len(data), 1), self._threshold), label = 'Signal Threshold',
                     linestyle = 'dotted', linewidth = 1, color = '#ff2200')

            # Fills
            ax1.fill_between(sample_space, signals, 0, color = '#ff5f5f', alpha = 0.2)
            ax1.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.13), ncol = 6)

        elif self.do_plot():
            print('\x1b[0mWarning: Unable to plot audio files or chunks longer than '
                  f'{MorseSoundStreamer.MAXIMUM_PLOTTABLE_DURATION} seconds', file = sys.stderr)

        if 'debug-print' in self._debug_args:
            print('\x1b[34m', end = '')

        set_character_start_position = True
        character_start_position = 0
        character_end_position = 0
        plot_character = False
        current_position = 0

        def signal_position_to_total_sample(signal_position: int) -> int:
            return chunk_sample_to_total_sample(signal_position - plot_offset)

        # Used to indicate within the plot that a character is split between chunks
        unfinished_character = len(self._current_character) > 0

        # Detect Morse characters from the tokens and pauses, string them together and put them in the stream.
        # Iterate over all the tokens and pauses and check their lengths with respect to the determined Morse unit
        # duration; we classify dits and dahs as well as the three kinds of pauses as forgiving and generous as
        # possible, because it is always better to return the most likely Morse character, even if it's not correct.
        for index, signal_length in enumerate(signal_lengths):
            if signal_length == 0:
                continue

            if symbol_sleep > 0:
                time.sleep(symbol_sleep)

            # Whether this is the last signal length in the given data
            is_last = index == len(signal_lengths) - 1

            # Calculate the duration of the current signal in Morse units
            signal_units = signal_length / self._unit_duration.sma()

            # Simulate transmission/decoding errors but only for tokens
            if simulated_error_percentage > 0:
                simulate_error = random.random() < simulated_error_percentage
                if simulate_error and index % 2 == signal_token_remainder:
                    if signal_units > 2.2: signal_units /= 3
                    else: signal_units *= 3

            if index % 2 == signal_token_remainder:
                # If the signal is a token
                token = "." if signal_units < 2 else "-"
                self.morse_stream().send(token)
                self._current_character += token
                self._current_string += token
                if set_character_start_position:
                    character_start_position = signal_position_to_total_sample(current_position)
                    set_character_start_position = False

            elif signal_units > 2.2 or (is_last and not data_is_chunk):
                # If the signal is at least an inter character pause emit the finished character
                if self._current_character:
                    if self._current_character in MORSE_TO_LATIN:
                        decoded = MORSE_TO_LATIN[self._current_character]
                        self.text_stream().send(decoded)
                        self._current_decoded_string += decoded
                        if 'debug-print' in self._debug_args:
                            print(decoded, end = '', flush = True)
                    else:
                        decoded = '#'
                        self._current_decoded_string += decoded
                        self.text_stream().send(decoded)
                        if 'debug-print' in self._debug_args:
                            print(f'\x1b[41m{decoded}\x1b[0;34m', end = '', flush = True)
                    self._num_characters_emitted += 1
                    self._num_symbols_emitted += 1
                    character_end_position = signal_position_to_total_sample(current_position)
                    set_character_start_position = True
                    plot_character = True
                    self._current_character = ""

                    if character_sleep > 0:
                        time.sleep(character_sleep)

                if signal_units > 5 and not is_last and self._num_characters_emitted > 0:
                    # If the signal is an inter word pause (i.e. ideally 7 units long)
                    self._current_decoded_string += ' '
                    if 'debug-print' in self._debug_args:
                        print(' ', end = '', flush = True)
                    self._num_symbols_emitted += 1

                # Send pause to the stream
                if not is_last and self._num_characters_emitted > 0:
                    if signal_units > 5:
                        self.morse_stream().send('/')
                        self.text_stream().send(' ')
                        self._current_string += '/'
                    else:
                        self.morse_stream().send(' ')
                        self._current_string += ' '

            if index % 2 != signal_token_remainder and data_is_chunk and is_last:
                # If the signal is a pause and the last in this chunk, remember and process it in the following
                # chunk, but at the latest after all chunks were processed
                self._last_pause_samples = signal_length

            # Add the decoded character to the plot
            if plot_signals and plot_character:
                character_length = character_end_position - character_start_position
                character_center_position = character_start_position + character_length // 2

                box_color = '#ff5f5f'
                rect = patches.Rectangle((character_start_position, 1.02), character_length, 0.06,
                                         linewidth = 1, facecolor = box_color, edgecolor = box_color, fill = True)
                ax1.add_patch(rect)
                text = decoded
                if unfinished_character:
                    text = '..' + text
                    unfinished_character = False
                ax1.text(character_center_position, 1.033, text, fontweight = 'bold', fontsize = 'large',
                         fontfamily = 'Ubuntu Condensed', horizontalalignment = 'center', color = '#ffffff')
                plot_character = False

            current_position += signal_length

        # Present the prepared plot
        if plot_signals:
            plt.show(block = True)

        # Emit the reset of the Morse characters if we already have all the data
        if not data_is_chunk:
            self.emit_rest()

        self._sample_position += len(data)
        self._num_chunks_analyzed += 1

class MorseWriter:
    def write(self, input: MorseString):
        pass

class MorseSoundFileWriter(MorseWriter, StreamReceiver):
    def __init__(self, file: str | int | BinaryIO, volume: float = 0.9, frequency: float = 800.0,
                 speed: float = 20.0, sample_rate: int = 8000,
                 flush_trigger_mode: FlushTriggerMode = FlushTriggerMode.AFTER_EACH_RECEIVE):
        super().__init__()
        self._sound_file = None
        self._file = os.path.expanduser(file) if isinstance(file, str) else file
        self._volume = clamp(volume, 0, 1)
        self._frequency = frequency
        self._words_per_minute = clamp(speed, 1, 60)
        self._seconds_per_unit = wpm_to_spu(self._words_per_minute)
        self._sample_rate = clamp(sample_rate, 1000, 192000)
        self._flush_trigger_mode = flush_trigger_mode
        self._last_was_audible = False

        self.open()

    def __del__(self):
        self.close()

    def open(self):
        """ Opens sound file for writing. """
        if self._sound_file is None:
            if isinstance(self._file, str):
                directory_path = os.path.dirname(self._file)
                os.makedirs(directory_path, exist_ok = True)
                self._sound_file = soundfile.SoundFile(self._file, mode = 'w', samplerate = self._sample_rate,
                                                       channels = 1)
            else:
                self._sound_file = soundfile.SoundFile(self._file, mode = 'w', samplerate = self._sample_rate,
                                                       channels = 1, format = 'WAV', subtype = 'PCM_16',
                                                       closefd = False)

    def close(self):
        """ Closes sound file for writing. """
        if self._sound_file is not None:
            self._sound_file.close()
            self._sound_file = None

    def flush(self):
        if self._sound_file is not None:
            self._sound_file.flush()

    def flush_trigger_mode(self):
        return self._flush_trigger_mode

    def set_flush_trigger_mode(self, flush_trigger_mode: FlushTriggerMode):
        self._flush_trigger_mode = flush_trigger_mode

    def get_audible_waveform(self, waveform: np.array, duration):
        t = np.linspace(0, duration, int(duration * self._sample_rate), endpoint = False)
        audible_wave = self._volume * np.sin(2 * np.pi * self._frequency * t)
        waveform = np.concatenate([waveform, audible_wave])
        self._last_was_audible = True
        return waveform

    def get_silent_waveform(self, waveform: np.array, duration):
        silent_wave = np.zeros(int(duration * self._sample_rate))
        waveform = np.concatenate([waveform, silent_wave])
        self._last_was_audible = False
        return waveform

    def write_durations(self, durations: list[float]):
        """ Writes audible `on` signals and silent `off` signals according to the given durations (even indices denote
            `on`, odd indices denote `off` signals) to the open sound file. """
        if self._sound_file is None:
            return;

        # Start off with an empty array
        waveform = np.array([])

        # Iterate over all durations and generate waveform data
        for i, duration in enumerate(durations):
            if i % 2 == 0: # On signals (audible)
                waveform = self.get_audible_waveform(waveform, duration)
            else: # Off signals (silent)
                waveform = self.get_silent_waveform(waveform, duration)

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

    def receive(self, data: str):
        if self._sound_file is None:
            return

        if not isinstance(data, str):
            data = str(data)

        for char in data:
            waveform = np.array([])

            if self._last_was_audible:
                waveform = self.get_silent_waveform(waveform, self._seconds_per_unit)
            if char == '.':
                waveform = self.get_audible_waveform(waveform, self._seconds_per_unit)
            elif char == '-':
                waveform = self.get_audible_waveform(waveform, self._seconds_per_unit * 3)
            elif char == ' ':
                waveform = self.get_silent_waveform(waveform, self._seconds_per_unit * 3)
            elif char == '/':
                waveform = self.get_silent_waveform(waveform, self._seconds_per_unit * 7)

            # Write waveform data to the given file
            self._sound_file.write(waveform)

            if self._flush_trigger_mode == FlushTriggerMode.AFTER_EACH_SIGNAL:
                self._sound_file.flush()

            if self._flush_trigger_mode == FlushTriggerMode.AFTER_EACH_CHARACTER and char == '/':
                self._sound_file.flush()

        if self._flush_trigger_mode == FlushTriggerMode.AFTER_EACH_RECEIVE:
            self._sound_file.flush()

LATIN_MORSE_TREE_LINEARIZED = ('etianmsurwdkgohvfüläpjbxcyzqö#54#3#¿#2&#+####16=/###(#7###8#90'
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

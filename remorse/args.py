from remorse.utils import clamp, spu_to_wpm, dual_split, color_to_ansi_escape, ColorizationMode, TextCase
import argparse
import os
import re
import remorse.version as version
import sys

ALLOWED_INPUT_FORMATS = { 't', 'text', 'c', 'code', 's', 'sound', 'f', 'file' }
ALLOWED_OUTPUT_FORMATS = { 't', 'text', 'c', 'code', 'n', 'nicecode', 's', 'sound', 'f', 'file' }

INPUT_OUTPUT_PATTERN = re.compile(r'^(?P<format>[a-z]+):(?P<value>.+)$')
TIME_UNIT_VALUE_PATTERN = re.compile(r'^(?P<value>[-+]?(?:\d*)(?:[\.,](?(1)\d*|\d+))?([eE\^][+-]?\d+)?)(?P<unit>s|ms|smp)$', re.IGNORECASE)
SPEED_UNIT_VALUE_PATTERN = re.compile(r'^(?P<value>[-+]?(?:\d*)(?:[\.,](?(1)\d*|\d+))?([eE\^][+-]?\d+)?)(?P<unit>wpm|spu)$', re.IGNORECASE)
FREQUENCY_UNIT_VALUE_PATTERN = re.compile(r'^(?P<value>[-+]?(?:\d*)(?:[\.,](?(1)\d*|\d+))?([eE\^][+-]?\d+)?)(?P<unit>hz|khz)$', re.IGNORECASE)

def check_time(input: str) -> float:
    """ Checks a time value from the given input string for validity. Returns `True` if it is valid. """
    if input and (match := TIME_UNIT_VALUE_PATTERN.match(input)):
        unit = match.group('unit')
        value = float(match.group('value'))
        return value >= 0 and unit.lower() in { 's', 'ms', 'smp' }
    return False

def parse_time_seconds(input: str, sample_rate: int) -> float:
    """ Parses a time value from the given input string, clamps it and returns it in seconds [s]. Returns `None` if the
        input string does not contain a valid time value. Supported units are seconds [s], milliseconds [ms] and samples
        [smp]. Minimum is 0 seconds. """
    if input and (match := TIME_UNIT_VALUE_PATTERN.match(input)):
        unit = match.group('unit')
        value = float(match.group('value'))
        if unit.lower() == 'ms':
            value /= 1000
        elif unit.lower() == 'smp':
            value /= sample_rate
        return max(value, 0)
    return None

def parse_time_samples(input: str, sample_rate: int) -> float:
    """ Parses a time value from the given input string, clamps it and returns it in samples [smp]. Returns `None` if
        the input string does not contain a valid time value. Supported units are seconds [s], milliseconds [ms] and
        samples [smp]. Minimum is 0 samples. """
    if input and (match := TIME_UNIT_VALUE_PATTERN.match(input)):
        unit = match.group('unit')
        value = float(match.group('value'))
        if unit.lower() == 's':
            value *= sample_rate
        elif unit.lower() == 'ms':
            value *= sample_rate / 1000
        return max(value, 0)
    return None

def parse_speed(input: str) -> float:
    """ Parses a speed value from the given input string, clamps it and returns it in words per minute [wpm]. Returns
        `None` if the input string does not contain a valid speed value. Supported units are words per minute [wpm] and
        seconds per unit [spu]. Valid range is [2 wpm; 60 wpm]. """
    if input and (match := SPEED_UNIT_VALUE_PATTERN.match(input)):
        unit = match.group('unit')
        value = float(match.group('value'))
        if unit.lower() == 'spu':
            value = spu_to_wpm(value)
        return clamp(value, 2, 60)
    return None

def parse_frequency(input: str, minimum: float, maximum: float) -> float:
    """ Parses a frequency value from the given input string, clamps it to the given range and returns it in
        Hertz [Hz]. Returns `None` if the input string does not contain a valid frequency value. Supported units are
        Hertz [Hz] and Kilohertz [kHz]. """
    if input and (match := FREQUENCY_UNIT_VALUE_PATTERN.match(input)):
        unit = match.group('unit')
        value = float(match.group('value'))
        if unit.lower() == 'khz':
            value *= 1000
        return clamp(value, minimum, maximum)
    return None

def parse_morse_frequency(input: str) -> float:
    """ Parses a Morse frequency value from the given input string, clamps it and returns it in Hertz [Hz]. Returns
        `None` if the input string does not contain a valid frequency value. Supported units are Hertz [Hz] and
        Kilohertz [kHz]. Valid range is [100 Hz; 10 kHz]. """
    return parse_frequency(input, 100, 10000)

def parse_sample_rate(input: str) -> float:
    """ Parses a sample rate value from the given input string, clamps it and returns it in Hertz [Hz]. Returns `None`
        if the input string does not contain a valid frequency value. Supported units are Hertz [Hz] and Kilohertz
        [kHz]. Valid range is [1 kHz; 192 kHz]. """
    return parse_frequency(input, 1000, 192000)

def parse_color(input: str) -> str:
    """ Parses an ANSI color escape sequence from the given input string. Prefixes `fg:` and `bg:` for foreground and
        background respectively may be specified; if the prefix is omitted the color is thought to be used in the
        foreground. Supported formats are 4-bit terminal color indices (`0-7`), 8-bit terminal color indices (`0-255`),
        tuples that hold three 8-bit integers for red, green and blue (`0-255`) and hexadecimal color values with or
        without leading hash symbols, e.g. `#ff5f5f`. """
    foreground = True
    if input.lower().startswith('fg:'):
        foreground = True
        input = input[3:]
    elif input.lower().startswith('bg:'):
        foreground = False
        input = input[3:]
    return color_to_ansi_escape(input, foreground = foreground)

def parse_colorization_mode(input: str) -> ColorizationMode | None:
    """ Parses a colorization mode from the given input string. Returns a `ColorizationMode` if the string represents a
        valid colorization mode, `None` otherwise. """
    if input.lower() == 'none':
        return ColorizationMode.NONE
    elif input.lower() == 'words' or input.lower() == 'word':
        return ColorizationMode.WORDS
    elif input.lower() in { 'characters', 'character', 'chars', 'char' }:
        return ColorizationMode.CHARACTERS
    elif input.lower() == 'symbols' or input.lower() == 'symbol':
        return ColorizationMode.SYMBOLS
    return None

def parse_text_case(input: str) -> TextCase | None:
    """ Parses a text case from the given input string. Returns a `TextCase` if the string represents a valid text case,
        `None` otherwise. """
    if input.lower() == 'none':
        return TextCase.NONE
    elif input.lower() == 'upper' or input.lower() == 'uc':
        return TextCase.UPPER
    elif input.lower() == 'lower' or input.lower() == 'lc':
        return TextCase.LOWER
    elif input.lower() == 'sentence':
        return TextCase.SENTENCE
    return None

def parse_args():
    debug_args = {}
    argi = 1
    argc = len(sys.argv)

    # Perform some manual argument extraction, especially debug arguments that are not registered with argparse
    while argi < argc:
        arg = sys.argv[argi]
        if arg == '--version':
            print(version.version_string_full)
            exit(0)

        # Extract debug arguments
        if arg.startswith('-D'):
            debug_arg, debug_value = dual_split(arg[2:], '=')
            debug_args[debug_arg] = debug_value
            del sys.argv[argi]
            argc -= 1
            continue

        argi += 1

    parser = argparse.ArgumentParser(prog = 'remorse', usage = '%(prog)s <input> -o <format> [options..]')

    parser.add_argument('input', type = str, nargs = 1, help = 'Input string or file to be converted')
    parser.add_argument('-b', '--buffer-size', metavar = '\x1b[3m<time>\x1b[0m', type = str, default = '2s',
                        action = 'store', help = 'Length of the audio sample buffer, e.g. 1.5s, 900ms or 2000smp')
    parser.add_argument('-c', '--color', metavar = '\x1b[3m<color>\x1b[0m', type = str, default = [],
                        action = 'append', help = 'Colors used for alternating and distinguished output')
    parser.add_argument('-C', '--colorization', metavar = '\x1b[3m<mode>\x1b[0m', type = str, default = 'words',
                        action = 'store', help = 'Alternating colorization for none, words, characters or symbols')
    parser.add_argument('-f', '--frequency', metavar = '\x1b[3m<frequency>\x1b[0m', type = str, default = '800hz',
                        action = 'store', help = 'Frequency used to play Morse sounds in, e.g. 800hz or 1.2kHz')
    parser.add_argument('-m', '--min-signal-size', metavar = '\x1b[3m<time>\x1b[0m', type = str, default = '0.01s',
                        action = 'store', help = 'Minimum required length of a valid signal, e.g. 0.01s or 80smp')
    parser.add_argument('-o', '--output', metavar = '\x1b[3m<format>\x1b[0m', type = str, required = True,
                        action = 'append', help = 'Output format into which shall be converted')
    parser.add_argument('-p', '--plot', action = 'store_true',
                        help = 'Plot graphs that visualize frequency spectrums and signal data from sound files')
    parser.add_argument('-r', '--sample-rate', metavar = '\x1b[3m<rate>\x1b[0m', type = str, default = '8kHz',
                        action = 'store', help = 'Sample rate used to generate Morse sounds and sound files')
    parser.add_argument('-s', '--speed', metavar = '\x1b[3m<speed>\x1b[0m', type = str, default = '20wpm',
                        action = 'store', help = 'Speed in which to generate Morse sounds, e.g. 20wpm or 0.06spu')
    parser.add_argument('-v', '--volume', metavar = '\x1b[3m<volume>\x1b[0m', type = float, default = 0.9,
                        action = 'store', help = 'Volume for generated Morse sounds and sound files')
    parser.add_argument('-t', '--volume-threshold', metavar = '\x1b[3m<threshold>\x1b[0m', type = float, default = 0.35,
                        action = 'store', help = 'Threshold in volume for distinguishing audible from silent signals')
    parser.add_argument('-T', '--test-against-text', metavar = '\x1b[3m<file>\x1b[0m', type = str, action = 'store',
                        help = 'File containing expected conversion text result to test against')
    parser.add_argument('-M', '--test-against-morse', metavar = '\x1b[3m<file>\x1b[0m', type = str, action = 'store',
                        help = 'File containing expected conversion Morse result to test against')
    parser.add_argument('--text-case', metavar = '\x1b[3m<case>\x1b[0m', type = str, default = 'upper',
                        action = 'store', help = 'Text case used for displaying decoded text: upper, lower, sentence')
    parser.add_argument('--version', action = 'store_true', help = 'Prints the version and legal information')

    result = parser.parse_args()

    # Parse input format
    result.input = result.input[0]
    if (match := INPUT_OUTPUT_PATTERN.match(result.input)):
        result.input_format = match.group('format')
        if result.input_format not in ALLOWED_INPUT_FORMATS:
            print(f"Error: Positional input argument specified an invalid format: {result.input_format}. "
                  f"Supported formats are {', '.join(ALLOWED_INPUT_FORMATS)}")
            exit(1)
        result.input_value = match.group('value')
    else:
        result.input_format = 'text'
        result.input_value = result.input

    # Parse output formats
    output_formats = {}
    for elem in result.output:
        for format_args in elem.split(','):
            format, args = dual_split(format_args, ':')
            if format[0] not in output_formats:
                output_formats[format[0]] = args.split(':') if args is not None else []
    result.output = output_formats

    # Parse colors
    colors = []
    for elem in result.color:
        if color := parse_color(elem):
            colors.append(color)
        else:
            print(f"Error: '{elem}' is not a valid value to argument --color")
    if len(colors) == 0:
        # Add default colors
        colors.append('#ff5f5f')
        colors.append('#ff1f9f')
    result.color = colors

    # Parse colorization
    if (colorization := parse_colorization_mode(result.colorization)) is not None:
        result.colorization = colorization
    else:
        print("Error: Argument --colorization must be either one of 'none', 'words', 'characters' or 'symbols'")
        exit(1)

    # Parse text case
    if (text_case := parse_text_case(result.text_case)) is not None:
        result.text_case = text_case
    else:
        print("Error: Argument --text-case must be either one of 'none', 'upper', 'lower' or 'sentence'")
        exit(1)

    # Parse speed
    if (speed := parse_speed(result.speed)) is not None:
        result.speed = speed
    else:
        print("Error: Argument --speed is given in wrong format")
        exit(1)

    # Parse frequency
    if (frequency := parse_morse_frequency(result.frequency)) is not None:
        result.frequency = frequency
    else:
        print("Error: Argument --frequency is given in wrong format")
        exit(1)

    # Parse sample rate
    if (sample_rate := parse_sample_rate(result.sample_rate)) is not None:
        result.sample_rate = int(sample_rate)
    else:
        print("Error: Argument --sample-rate is given in wrong format")
        exit(1)

    # Check buffer size
    if result.buffer_size and not check_time(result.buffer_size):
        print("Error: Argument --buffer-size is given in wrong format")
        exit(1)

    # Check minimum signal size
    if result.min_signal_size and not check_time(result.min_signal_size):
        print("Error: Argument --min-signal-size is given in wrong format")
        exit(1)

    # Parse volume
    if result.volume:
        result.volume = clamp(result.volume, 0.1, 1)

    # Parse volume threshold
    if result.volume_threshold:
        result.volume_threshold = clamp(result.volume_threshold, 0.1, 0.9)

    # Test against (file containing expected conversion text result)
    if result.test_against_text and not os.path.isfile(result.test_against_text):
        print("Error: Argument --test-against-text refers to a file that does not exist")
        exit(1)

    # Test against (file containing expected conversion Morse result)
    if result.test_against_morse and not os.path.isfile(result.test_against_morse):
        print("Error: Argument --test-against-morse refers to a file that does not exist")
        exit(1)

    # Assign debug arguments
    result.debug_args = debug_args
    return result

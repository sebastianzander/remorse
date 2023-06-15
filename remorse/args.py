from remorse.utils import clamp, spu_to_wpm
import argparse
import re

ALLOWED_INPUT_FORMATS = { 't', 'text', 'c', 'code', 'f', 'file' }
ALLOWED_OUTPUT_FORMATS = { 't', 'text', 'c', 'code', 'n', 'nicecode', 's', 'sound', 'f', 'file' }

INPUT_OUTPUT_PATTERN = re.compile(r'^(?P<format>[a-z]+):(?P<value>.+)$')
SPEED_UNIT_VALUE_PATTERN = re.compile(r'^(?P<value>[-+]?(?:\d*)(?:[\.,](?(1)\d*|\d+))?([eE\^][+-]?\d+)?)(?P<unit>wpm|spu)$', re.IGNORECASE)
FREQUENCY_UNIT_VALUE_PATTERN = re.compile(r'^(?P<value>[-+]?(?:\d*)(?:[\.,](?(1)\d*|\d+))?([eE\^][+-]?\d+)?)(?P<unit>hz|khz)$', re.IGNORECASE)

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

def parse_args():
    parser = argparse.ArgumentParser(prog = 'remorse', usage = '%(prog)s <input> -o <format> [options..]')

    parser.add_argument('input', type = str, nargs = 1, help = 'Input string or file to be converted')
    parser.add_argument('-o', '--output', metavar = '\x1b[3m<format>\x1b[0m', type = str, required = True,
                        action = 'append', help = 'Output format into which shall be converted')
    parser.add_argument('-s', '--speed', metavar = '\x1b[3m<speed>\x1b[0m', type = str, default = '20wpm',
                        action = 'store', help = 'Speed in which to generate Morse sounds, e.g. 20wpm or 0.06spu')
    parser.add_argument('-f', '--frequency', metavar = '\x1b[3m<frequency>\x1b[0m', type = str, default = '800hz',
                        action = 'store', help = 'Frequency used to play Morse sounds in, e.g. 800hz or 1.2kHz')
    parser.add_argument('-r', '--sample-rate', metavar = '\x1b[3m<rate>\x1b[0m', type = str, default = '8kHz',
                        action = 'store', help = 'Sample rate used to generate and save Morse sounds to files')
    parser.add_argument('-p', '--plot', action = 'store_true',
                        help = 'Plot graphs that visualize frequency spectrums and signal data')
    parser.add_argument('--simultaneous', action = 'store_true',
                        help = 'Outputs eligible formats simultaneously (e.g. text and sound character by character)')
    parser.add_argument('-v', '--volume-threshold', metavar = '\x1b[3m<threshold>\x1b[0m', type = float,
                        default = 0.35, action = 'store',
                        help = 'Threshold in volume for distinguishing on from off signals')

    result = parser.parse_args()

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

    # Parse speed
    if (speed := parse_speed(result.speed)):
        result.speed = speed
    else:
        print("Error: Argument --speed is given in wrong format")
        exit(1)

    # Parse frequency
    if (frequency := parse_morse_frequency(result.frequency)):
        result.frequency = frequency
    else:
        print("Error: Argument --frequency is given in wrong format")
        exit(1)

    # Parse sample rate
    if (sample_rate := parse_sample_rate(result.sample_rate)):
        result.sample_rate = sample_rate
    else:
        print("Error: Argument --sample-rate is given in wrong format")
        exit(1)

    output_formats = []
    for elem in result.output:
        for format in elem.split(','):
            if format not in output_formats:
                output_formats.append(format)
    result.output = output_formats

    return result

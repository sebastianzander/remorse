from remorse.args import ALLOWED_INPUT_FORMATS, parse_args
from remorse.morse import *
from remorse.utils import *
import re

MORSE_INPUT_PATTERN = re.compile(r'^[\.\-\/ ]+$')

def main():
    args = parse_args()

    assert args.input is not None, "Input must be given"
    assert args.output is not None, "Output format must be given"

    # Prepare input format and value
    input_split = args.input.split(':', 1)

    if len(input_split) > 1:
        if input_split[0] in ALLOWED_INPUT_FORMATS:
            # Input is in the form `<format>:<value>`
            input_format = input_split[0]
            input_value = input_split[1]
        else:
            input_format = 't'
            input_value = args.input
    else:
        # Input is the actual value and format must either be `text` or `code` (try to guess it)
        input_value = input_split[0]
        input_format = 'code' if MORSE_INPUT_PATTERN.fullmatch(input_value) else 'text'
        if os.path.isfile(input_value):
            input_format = 'file'

    # Prepare output formats
    output_formats = args.output

    if input_format in output_formats:
        del output_formats[input_format]

    if len(output_formats) == 0:
        print("Error: output format must not be the same as the input format!")
        exit(1)
    elif 'c' in output_formats and 'n' in output_formats:
        print("Error: output formats 'c/code' and 'n/nicecode' must not be given together!")
        exit(1)

    test_against_text = None
    if args.test_against_text is not None and os.path.isfile(args.test_against_text):
        with open(args.test_against_text, 'r') as file:
            test_against_text = file.read()

    test_against_morse = None
    if args.test_against_morse is not None and os.path.isfile(args.test_against_morse):
        with open(args.test_against_morse, 'r') as file:
            test_against_morse = file.read()

    streamer = None

    # Create the input device
    if input_format in { 't', 'text' }:
        streamer = MorseStringStreamer(data = input_value, data_is_morse = False)
    elif input_format in { 'c', 'code' }:
        streamer = MorseStringStreamer(data = input_value, data_is_morse = True)
    elif input_format in { 'f', 'file', 's', 'sound' }:
        streamer = MorseSoundStreamer(device = input_value, output = False, plot = args.plot,
                                      buffer_size = args.buffer_size, min_signal_size = args.min_signal_size,
                                      filtering_mode = args.filtering_mode, filtering_args = args.filtering_args,
                                      noise_reduction_mode = args.noise_reduction, threshold = args.threshold,
                                      normalization_mode = args.normalization,
                                      debug_args = args.debug_args)

    # This should not be possible: Invalid input formats should be catched above
    if streamer is None:
        print("Error: no streamer available")
        exit(1)

    # Create the output devices
    for output_format, output_args in output_formats.items():
        # Output is code
        if output_format == 'c':
            morse_printer = MorsePrinter(word_pause_char = ' / ')
            if len(args.color) > 0:
                morse_printer.set_color(args.color[0])
            streamer.morse_stream().subscribe(morse_printer)

            if test_against_morse is not None:
                verifier = StringVerifier(expected = test_against_morse, grace_width = 2)
                morse_printer.set_verifier(verifier)

        # Output is nicely formatted code
        elif output_format == 'n':
            morse_visualizer = MorseVisualizer()
            if args.colorization is not None:
                morse_visualizer.set_colorization_mode(args.colorization)
            if len(args.color) > 0:
                morse_visualizer.set_colors(args.color)
            streamer.morse_stream().subscribe(morse_visualizer)

            if test_against_morse is not None:
                verifier = StringVerifier(expected = test_against_morse, grace_width = 2)
                morse_visualizer.set_verifier(verifier)

        # Output is decoded text
        elif output_format == 't':
            text_printer = TextPrinter()
            if len(args.color) > 0:
                text_printer.set_color(args.color[0])
            if args.text_case is not None:
                text_printer.set_text_case(args.text_case)
            streamer.text_stream().subscribe(text_printer)

            if test_against_text is not None:
                verifier = StringVerifier(expected = test_against_text, grace_width = 2)
                text_printer.set_verifier(verifier)

        # Output is Morse sound
        elif output_format == 's':
            morse_player = MorsePlayer(frequency = args.frequency, speed = args.speed, volume = args.volume,
                                       sample_rate = args.sample_rate)
            streamer.morse_stream().subscribe(morse_player)

        # Output is a sound file (can be used to convert noisy and old sound files to sterile and clean ones or simply
        # to change frequency, speed and sample rate)
        elif output_format == 'f':
            if output_args is None or len(output_args) == 0:
                print("Error: output formats 'f/file' requires an output file name as a first argument! "
                    "Example: file:path/to/file.mp3")
                exit(1)

            file_path = output_args[0]
            file_writer = MorseSoundFileWriter(file = file_path, volume = args.volume, frequency = args.frequency,
                                               speed = args.speed, sample_rate = args.sample_rate)
            streamer.morse_stream().subscribe(file_writer)
            print(f"Writing to sound file {file_path}")

    # Start the stream if there are subscribers
    if streamer.morse_stream().num_subscribers() or streamer.text_stream().num_subscribers():
        streamer.read()

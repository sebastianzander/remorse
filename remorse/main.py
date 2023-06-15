from remorse.args import parse_args
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
        # Input is in the form `<format>:<value>`
        input_format = input_split[0]
        input_value = input_split[1]
    else:
        # Input is the actual value and format must either be `text` or `code` (try to guess it)
        input_value = input_split[0]
        input_format = 'code' if MORSE_INPUT_PATTERN.fullmatch(input_value) else 'text'

    # Prepare output formats
    output_formats = [output_format[0] for output_format in args.output if output_format[0] != input_format[0]]

    if len(output_formats) == 0:
        print("Error: output format must not be the same as the input format!")
        exit(1)
    elif 'c' in output_formats and 'n' in output_formats:
        print("Error: output formats 'c/code' and 'n/nicecode' must not be given together!")
        exit(1)

    # TODO: Create MorseSound; refactor MorseEmitters and MorseReceivers and let them work with instances of MorseString
    # and MorseSound; offer all possible conversion directions here

    # Input is plain text
    if input_format in { 't', 'text' }:
        original_text = preprocess_input_text(input_value)
        morse = text_to_morse(original_text)

        # Create an outer multi emitter that holds all individual emitters
        multi_emitter = MorseMultiEmitter(simultaneous = args.simultaneous)

        for output_format in output_formats:
            # Output is code
            if output_format == 'c':
                printer = MorsePrinter()
                multi_emitter.add_emitter(printer)

            # Output is nicely formatted code
            elif output_format == 'n':
                visualizer = MorseVisualizer()
                # TODO: Set options from command line arguments
                visualizer.enable_colored_output()
                visualizer.set_colorization_mode(ColorizationMode.CHARACTERS)
                visualizer.set_colors(Color.RED, Color.CYAN)
                multi_emitter.add_emitter(visualizer)

            # Output is sound (played on default audio device)
            elif output_format == 's':
                player = MorsePlayer(frequency = args.frequency, speed = args.speed)
                # TODO: Set options from command line arguments
                multi_emitter.add_emitter(player)

            # Output is a sound file
            elif output_format == 'f':
                print("Note: output format 'f/file' is currently not supported.")
                # TODO: Read arguments from output formats in the form `<format>:<args>`
                # Example usage: remorse '... --- ...' -o file:~/sounds/sos.mp3

        multi_emitter.emit(morse)
        print()

    # Input is Morse code in form of text
    elif input_format in { 'c', 'code' }:
        text = morse_to_text(input_value)
        print(f"\x1b[34m{text}\x1b[0m")

    # Input is Morse code in form of an audio file
    elif input_format in { 'f', 'file' }:
        sound_receiver = MorseSoundReceiver(input_value, kernel_seconds = 0.01, use_multiprocessing = False)

        if args.plot:
            sound_receiver.set_show_plots(True)

        extracted_morse = sound_receiver.receive()
        text = morse_to_text(str(extracted_morse))
        print(f"\x1b[34m{text}\x1b[0m")

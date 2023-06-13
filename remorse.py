from morse import *
from utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog = 'remorse', usage = '%(prog)s [options..] <input>')

    parser.add_argument('input', type = str, nargs = 1, help = 'Input string or audio file to be converted')
    parser.add_argument('-w', '--words-per-minute', type = int, default = 20, action = 'store',
                        help = 'Speed in which to play Morse audio given in words per minute')
    parser.add_argument('--frequency', type = float, default = 800.0, action = 'store',
                        help = 'Frequency used to play Morse audio given in Hertz')
    parser.add_argument('-f', '--from', metavar = '\x1b[3m<format>\x1b[0m', action = 'store',
                        help = 'What format the input is in; one of: text, morse, audio')
    parser.add_argument('-t', '--to', metavar = '\x1b[3m<format>\x1b[0m', action = 'store',
                        help = 'What format the output shall be; one or multiple of: text, morse, audio')
    parser.add_argument('-p', '--plot', action = 'store_true',
                        help = 'Plot graphs that visualize frequency spectrums and signal data')

    result = parser.parse_args()
    result.words_per_minute = clamp(result.words_per_minute, 2, 60)
    result.frequency = clamp(result.frequency, 100.0, 2000.0)

    return result

if __name__ == '__main__':
    args = parse_args()

    original_text = preprocess_input_text(args.input[0])
    # print(f"\x1b[33m{original_text}\x1b[0m")

    morse = text_to_morse(original_text)

    printer = MorsePrinter()
    visualizer = MorseVisualizer()
    player = MorsePlayer(frequency = args.frequency, words_per_minute = args.words_per_minute)

    visualizer_and_player = MorseMultiEmitter(True, visualizer, player)

    visualizer.enable_colored_output()
    visualizer.set_colorization_mode(ColorizationMode.CHARACTERS)
    visualizer.set_colors(Color.RED, Color.CYAN) #, Color.YELLOW, Color.GREEN

    # visualizer_and_player.emit(morse)
    # print()

    # unmorse = morse_to_text(morse_encoded, symbol_separator = symbol_separator, word_separator = word_separator,
    #                         dit_symbol = dit_symbol, dah_symbol = dah_symbol)
    # assert original_text == unmorse, "Morse roundtrip conversion failed"

    sound_receiver = MorseSoundReceiver("audio/simple_8khz_32kbps.mp3", kernel_seconds = 0.01, use_multiprocessing = False)
    # sound_receiver = MorseSoundReceiver("audio/230516_30wpm_8khz_32kbps.mp3", kernel_seconds = 0.01, use_multiprocessing = False)
    # sound_receiver = MorseSoundReceiver("audio/analog_old_recording_8khz_32kbps.mp3", kernel_seconds = 0.01, use_multiprocessing = False, min_signal_seconds = 0.015)

    if args.plot:
        sound_receiver.set_show_plots(True)

    extracted_morse = sound_receiver.receive()
    unmorse = morse_to_text(str(extracted_morse))

    print(f"\x1b[34m{unmorse}\x1b[0m")

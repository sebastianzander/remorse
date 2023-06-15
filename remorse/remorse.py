from args import parse_args
from morse import *
from utils import *

if __name__ == '__main__':
    args = parse_args()

    original_text = preprocess_input_text(args.input)
    # print(f"\x1b[33m{original_text}\x1b[0m")

    morse = text_to_morse(original_text)

    printer = MorsePrinter()
    visualizer = MorseVisualizer()
    player = MorsePlayer(frequency = args.frequency, speed = args.speed)

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

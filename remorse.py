import argparse
import math
from morse import *

LATIN_MORSE_TREE_LINEARIZED = ('etianmsurwdkgohvf#l#pjbxcyzq##54#3#¿#2&#+####16=/###(#7###8#90'
                               '############?_####"##.####@###\'##-########;!#)###¡#,####:#######')

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

#LATIN_TO_TRIMORSE = build_symbol_mapping_from_linearized_tree(LATIN_MORSE_TREE_LINEARIZED, ['.', '-', '*'])

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

def preprocess_input_text(text: str) -> str:
    return text.upper().replace('Ä', 'AE').replace('Ö', 'OE').replace('Ü', 'UE').replace('ß', 'SS')

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

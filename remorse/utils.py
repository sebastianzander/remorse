from itertools import tee, islice
import re

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

class TextCase:
    NONE = 0
    UPPER = 1
    LOWER = 2
    SENTENCE = 3

def hexcolor_to_rgb(hex_color: str):
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def hexcolor_to_ansi_escape_8bit(hex_color: str, foreground: bool = True) -> str:
    r, g, b = hexcolor_to_rgb(hex_color)
    r, g, b = round(r / 255 * 5), round(g / 255 * 5), round(b / 255 * 5)
    ansi_code = 16 + 36 * r + 6 * g + b
    prefix = 38 if foreground else 48
    return f'\x1b[{prefix};5;{ansi_code}m'

def hexcolor_to_ansi_escape_24bit(hex_color: str, foreground: bool = True) -> str:
    r, g, b = hexcolor_to_rgb(hex_color)
    prefix = 38 if foreground else 48
    return f'\x1b[{prefix};2;{r};{g};{b}m'

def color_to_ansi_escape(color: int | tuple[int, int, int] | str, foreground: bool = True) -> str:
    p = 3 if foreground else 4
    if isinstance(color, str) and color.isdigit():
        color = int(color)
    if isinstance(color, int):
        if 0 <= color <= 7:
            return f'\x1b[{p}{color}m'
        elif 9 <= color <= 255:
            return f'\x1b[{p}8;5;{color}m'
    elif isinstance(color, tuple) and len(color) == 3:
        r, g, b = color
        return f'\x1b[{p}8;2;{r};{g};{b}m'
    elif isinstance(color, str):
        parts = color.split(',')
        if len(parts) == 3:
            r = parts[0].strip()
            g = parts[1].strip()
            b = parts[2].strip()
            if r.isdigit() and g.isdigit() and b.isdigit():
                rgb = (clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255))
                return hexcolor_to_ansi_escape_24bit(rgb, foreground = foreground)
        elif len(parts) == 1 and color.startswith('\x1b['):
            return color
        elif len(parts) == 1:
            return hexcolor_to_ansi_escape_24bit(color, foreground = foreground)
    return None

def scramble(clear_text: str, scramble_map: dict[str, str]) -> str:
    """ Scrambles the given `text` using `scramble_map`; requires `scramble_map` to contain single character keys and
        values exclusively in order to generate a reversible 1:1 text scrambling. """
    result = ""
    for symbol in clear_text:
        assert len(symbol) == 1 and (symbol not in scramble_map or len(scramble_map[symbol]) == 1), "Scramble map must only contain single character keys and values"
        result += scramble_map[symbol] if symbol in scramble_map else symbol
    return result

def unscramble(scrambled_text: str, scramble_map: dict[str, str]) -> str:
    """ Simply inverses `scramble_map` previously used for scrambling and uses it to unscramble the given
        `scrambled_text`; requires `scramble_map` to contain single character keys and values exclusively in order to
        reverse a 1:1 text scrambling. """
    unscramble_map = { v: k for k, v in scramble_map.items() }
    return scramble(scrambled_text, unscramble_map)

def clamp(value: int | float, minimum: int | float, maximum: int | float):
    return min(max(value, minimum), maximum)

def is_close(test_number: float, standard: float, percentual_deviation: float) -> bool:
    """ Returns `true` if the given test number is within +/- a given percentual deviation around a given standard. """
    low = standard - standard * percentual_deviation
    high = standard + standard * percentual_deviation
    return low <= test_number <= high

def wpm_to_spu(words_per_minute: float) -> float:
    """ Converts the given 'words per minute' into 'seconds per unit'. """
    return 60 / (50 * words_per_minute)

def spu_to_wpm(seconds_per_unit: float) -> float:
    """ Converts the given 'seconds per unit' into 'words per minute'. """
    return 60 / (50 * seconds_per_unit)

def preprocess_input_text(text: str) -> str:
    """ Converts all occurrences of eszetts to its two-letter equivalent and all backslashes to forward slashes. """
    return text.upper().replace('ß', 'SS').replace('\\', '/')

CONSECUTIVE_SPACES_PATTERN = re.compile(r'\s{2,}')
CONSECUTIVE_WORD_PAUSES_PATTERN = re.compile(r'\s*(\/)(?:\s*\/*)*')

def preprocess_input_morse(morse: str) -> str:
    """ Strips leading and trailing white spaces, reduces multiple inner white spaces to single white spaces. """
    result = morse.strip()
    result = CONSECUTIVE_WORD_PAUSES_PATTERN.sub('/', result)
    return CONSECUTIVE_SPACES_PATTERN.sub(' ', result)

def dual_split(input: str, separator: str) -> list:
    """ Splits the given input string into two parts: the part before the given separator and the part after the given
        separator. If no separator can be found, returns `input` as the first part and `None` as the second part. """
    elements = input.split(sep = separator, maxsplit = 1)
    if len(elements) == 1:
        elements.append(None)
    return elements

def nwise(iterable, n):
    """ Returns tuples of `n` consecutive elements from `iterable` in overlapping fashion,
        thus returns `len(iterable) - n + 1` tuples. """
    iters = tee(iterable, max(n, 1))
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return zip(*iters)

def overlapped(iterable, group_size):
    """ Returns tuples of `group_size` consecutive elements from `iterable` in overlapping fashion,
        thus returns `len(iterable) - group_size + 1` tuples. Alias of `nwise`. """
    return nwise(iterable, group_size)

class tuplewise:
    """ Generates an iterable range that returns tuples of the given size from the given list without repetition, i.e.
        each element appears in only one tuple. The argument `strict` makes sure that only complete tuples are
        returned. If you instead want the last returned tuple to be padded with `None` values (if necessary) set
        `strict = False`.

        Example 1: `tuplewise(list = [1, 2, 3, 4, 5, 6, 7], tuple_size = 2)` -> `(1, 2)`, `(3, 4)`, `(5, 6)`

        Example 2: `tuplewise(list = [1, 2, 3, 4, 5, 6, 7], tuple_size = 3)` -> `(1, 2, 3)`, `(4, 5, 6)`

        Example 3a: `tuplewise(list = ['A', 'B'], tuple_size = 3, strict = True)` -> no iteration

        Example 3b: `tuplewise(list = ['A', 'B'], tuple_size = 3, strict = False)` -> `('A', 'B', None)` """

    def __init__(self, list: list, tuple_size: int, strict: bool = True):
        self._list = list
        self._tuple_size = max(tuple_size, 1)
        self._strict = strict
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        len_list = len(self._list)
        underflow = self._index > len_list - self._tuple_size
        if self._index >= len_list or (underflow and self._strict):
            raise StopIteration
        if underflow:
            t = self._list[self._index:]
            t.extend([None] * (self._tuple_size - len(t)))
            self._index += self._tuple_size
            return tuple(t)
        else:
            t = self._list[self._index:self._index + self._tuple_size]
            self._index += self._tuple_size
            return tuple(t)

class SimpleMovingAverage:
    def __init__(self, window_size: int, initial_value = None):
        self._window_size = max(window_size, 2)
        self._data_points = []
        if initial_value is not None:
            self._data_points.append(initial_value)

    def reset(self, value):
        """ Resets this moving average by setting it to the given value. """
        self._data_points.clear()
        self._data_points.append(value)

    def update(self, value):
        """ Adds the given new data point to the list of observed data points. """
        self._data_points.append(value)
        if len(self._data_points) > self._window_size:
            self._data_points.pop(0)

    def sma(self):
        """ Returns the simple moving average of the collected data points. If there are no data points recorded this
            method returns `None`. """
        if not self._data_points:
            return None
        return sum(self._data_points) / len(self._data_points)

    def empty(self):
        return len(self._data_points) == 0

    def multiply(self, factor: float):
        """ Multiplies all values in this moving average by the given factor. """
        new_data_points = []
        for data_point in self._data_points:
            new_data_points.append(data_point * factor)
        self._data_points = new_data_points

class StringVerifier:
    """ A class for verifying a growing string against an expected string. """
    def __init__(self, expected: str, grace_width: int, highlight_background: bool = True):
        super().__init__()
        self._expected = expected
        self._expected_index = 0
        self._received = str()
        self._received_index = 0
        self._received_offset = 0
        self._grace_width = max(grace_width, 0)
        self._p = 4 if highlight_background else 3
        self._num_mismatches = 0
        self._last_matching = True
        self._last_matching_indices = None
        self._last_expected = None
        self._debug = False

    def num_mismatches(self):
        """ Returns the number of mismatches detected in the currently received string. """
        return self._num_mismatches

    def reset(self, expected: str = None, grace_width: int = None, highlight_background: bool = None):
        """ Resets the state of this verifier, also allows to set a new expected string. """
        if expected is not None:
            self._expected = expected
        self._expected_index = 0
        self._received = str()
        self._received_index = 0
        self._received_offset = 0
        if grace_width is not None:
            self._grace_width = max(grace_width, 0)
        if highlight_background is not None:
            self._p = 4 if highlight_background else 3
        self._num_mismatches = 0
        self._last_matching = True
        self._last_matching_indices = None
        self._last_expected = None

    def verify(self, string: str, additive: bool = True) -> tuple[bool | None, str, str, str]:
        """ Verifies that the newly received string matches what is expected.
            Returns a tuple of `(matching, diff, expected)`, where `matching` is `True` if the string matches the
            expected string since the last call to `verify()`, or `matching` is `False` if the string does not match or
            exceeds what is expected. `matching` may be `None` if nothing was yet received. `diff` contains `string` if
            matching, otherwise `diff` contains what would be expected highlighted red (i.e. removed in the received
            string) and unexpected characters highlighted green (i.e. added in the received string). `expected` contains
            the expected character or substring but only if `matching` is `False`. """
        if additive:
            self._received += string
        else:
            self._received = string
            self._received_index = -1
            self._received_offset = 0
            self._expected_index = 0

        matching = True
        diff_string = ''
        len_received = len(self._received)
        len_expected = len(self._expected)

        if len_received == 0 or len(string) == 0:
            return (None, '', '')

        if self._expected_index >= len_expected:
            return (False, f'\x1b[{self._p}2m{string}\x1b[0m', '')

        total_expected = ''

        while self._expected_index < len_expected and self._received_index < len_received:
            expected = self._expected[self._expected_index]
            received = self._received[self._received_index]
            matching = expected == received
            fwd_matching = None

            if not matching:
                self._num_mismatches += 1
                self._last_expected = expected
                fwd_matching = False

                if self._last_matching:
                    diff_string += f'\x1b[{self._p}1m{expected}\x1b[0m'
                    diff_string_plus = ''

                    # Look forward into expected and check if received is only missing characters
                    fwd_index = self._expected_index + 1
                    while fwd_index < len_expected and (fwd_index - self._expected_index) <= self._grace_width:
                        expected_forward = self._expected[fwd_index]
                        if expected_forward == received:
                            self._expected_index = fwd_index
                            fwd_matching = True
                            diff_string_plus += expected_forward
                            break
                        else:
                            diff_string_plus += f'\x1b[{self._p}1m{expected_forward}\x1b[0m'
                        fwd_index += 1

                    if fwd_matching == False:
                        diff_string += f'\x1b[{self._p}2m{received}\x1b[0m'
                        #self._expected_index -= 1
                    elif fwd_matching == True:
                        diff_string += diff_string_plus
                else:
                    diff_string += f'\x1b[{self._p}2m{received}\x1b[0m'
                    self._expected_index -= 1
            else:
                self._last_matching_indices = (self._expected_index, self._received_index)
                diff_string += received

            total_expected += expected

            self._expected_index += 1
            self._received_index += 1
            self._last_matching = matching

        return (matching, diff_string, total_expected)

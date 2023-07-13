from itertools import tee, islice
import re

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
    """ Converts all occurrences of German umlauts and eszetts to their two-letter equivalent. """
    return text.upper().replace('Ä', 'AE').replace('Ö', 'OE').replace('Ü', 'UE').replace('ß', 'SS')

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

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

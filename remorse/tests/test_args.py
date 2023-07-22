import remorse.args as args
import remorse.utils as utils
import unittest

class ArgsTests(unittest.TestCase):
    def test_check_time(self):
        self.assertTrue(args.check_time("1s"))
        self.assertTrue(args.check_time("1.3s"))
        self.assertTrue(args.check_time("750ms"))
        self.assertTrue(args.check_time("8000smp"))
        self.assertFalse(args.check_time("21"))
        self.assertFalse(args.check_time("21ns"))
        self.assertFalse(args.check_time("21h"))

    def test_parse_time_seconds(self):
        # Test that values are correctly parsed and converted into seconds
        self.assertEqual(1, args.parse_time_seconds("1s", None))
        self.assertEqual(2.3, args.parse_time_seconds("2300ms", None))
        self.assertEqual(3, args.parse_time_seconds("24000smp", 8000))

    def test_parse_time_samples(self):
        # Test that values are correctly parsed and converted into samples
        self.assertEqual(8000, args.parse_time_samples("1s", 8000))
        self.assertEqual(18400, args.parse_time_samples("2300ms", 8000))
        self.assertEqual(24000, args.parse_time_samples("24000smp", None))

    def test_parse_speed(self):
        # Test that values are correctly parsed and converted into words per minute
        self.assertEqual(20, args.parse_speed("20wpm"))
        self.assertEqual(20, args.parse_speed("20WPM"))
        self.assertEqual(20, args.parse_speed("0.06spu"))

        # Test that parsed values are clamped
        self.assertEqual(2, args.parse_speed("1wpm"))
        self.assertEqual(60, args.parse_speed("61wpm"))

        # Test that None is returned if invalid value or unit
        self.assertEqual(None, args.parse_speed("13mb"))

    def test_parse_frequency(self):
        # Test that values are correctly parsed and converted into Hertz
        self.assertEqual(20, args.parse_frequency("20hz", 0, 100000))
        self.assertEqual(2000, args.parse_frequency("2khz", 0, 100000))

    def test_parse_morse_frequency(self):
        # Test that values are correctly parsed, clamped and converted into Hertz
        self.assertEqual(100, args.parse_morse_frequency("90hz"))
        self.assertEqual(10000, args.parse_morse_frequency("11khz"))

    def test_parse_sample_rate(self):
        # Test that values are correctly parsed, clamped and converted into Hertz
        self.assertEqual(1000, args.parse_sample_rate("0.1khz"))
        self.assertEqual(192000, args.parse_sample_rate("193khz"))

    def test_parse_color(self):
        # Test that values are correctly parsed and prefixes considered
        self.assertEqual("\x1b[38;2;255;95;95m", args.parse_color("ff5f5f"))
        self.assertEqual("\x1b[38;2;255;95;95m", args.parse_color("#ff5f5f"))
        self.assertEqual("\x1b[38;2;95;255;95m", args.parse_color("fg:#5fff5f"))
        self.assertEqual("\x1b[48;2;95;95;255m", args.parse_color("bg:#5f5fff"))
        self.assertEqual("\x1b[48;2;255;95;95m", args.parse_color("bg:255,95,95"))

    def test_parse_colorization_mode(self):
        # Test that values are correctly parsed and converted to a colorization mode
        self.assertEqual(utils.ColorizationMode.NONE, args.parse_colorization_mode('none'))
        self.assertEqual(utils.ColorizationMode.WORDS, args.parse_colorization_mode('word'))
        self.assertEqual(utils.ColorizationMode.WORDS, args.parse_colorization_mode('words'))
        self.assertEqual(utils.ColorizationMode.CHARACTERS, args.parse_colorization_mode('Characters'))
        self.assertEqual(utils.ColorizationMode.SYMBOLS, args.parse_colorization_mode('SYMBOLS'))
        self.assertEqual(None, args.parse_colorization_mode('humbug'))

    def test_parse_text_case(self):
        # Test that values are correctly parsed and converted into a text case
        self.assertEqual(utils.TextCase.NONE, args.parse_text_case('none'))
        self.assertEqual(utils.TextCase.UPPER, args.parse_text_case('upper'))
        self.assertEqual(utils.TextCase.UPPER, args.parse_text_case('uc'))
        self.assertEqual(utils.TextCase.LOWER, args.parse_text_case('lower'))
        self.assertEqual(utils.TextCase.LOWER, args.parse_text_case('lc'))
        self.assertEqual(utils.TextCase.SENTENCE, args.parse_text_case('sentence'))
        self.assertEqual(None, args.parse_text_case('hogwash'))

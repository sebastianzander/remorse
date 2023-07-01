import remorse.args as args
import unittest

class ArgsTests(unittest.TestCase):
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

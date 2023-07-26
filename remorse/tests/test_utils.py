import remorse.utils as utils
import unittest

class UtilsTests(unittest.TestCase):
    def test_hexcolor_to_rgb(self):
        expected = (255, 192, 128)
        actual = utils.hexcolor_to_rgb('#ffc080')
        self.assertEqual(expected, actual)
        actual = utils.hexcolor_to_rgb('ffc080')
        self.assertEqual(expected, actual)

        expected = (255, 204, 153)
        actual = utils.hexcolor_to_rgb('#fc9')
        self.assertEqual(expected, actual)

    def test_hexcolor_to_ansi_escape_8bit(self):
        expected = '\x1b[38;5;223m'
        actual = utils.hexcolor_to_ansi_escape_8bit('#ffc080', foreground = True)
        self.assertEqual(expected, actual)

        expected = '\x1b[48;5;231m'
        actual = utils.hexcolor_to_ansi_escape_8bit('#fff', foreground = False)
        self.assertEqual(expected, actual)

    def test_hexcolor_to_ansi_escape_24bit(self):
        expected = '\x1b[38;2;255;192;128m'
        actual = utils.hexcolor_to_ansi_escape_24bit('#ffc080', foreground = True)
        self.assertEqual(expected, actual)

        expected = '\x1b[48;2;0;0;0m'
        actual = utils.hexcolor_to_ansi_escape_24bit('#000', foreground = False)
        self.assertEqual(expected, actual)

    def test_color_to_ansi_escape(self):
        expected = '\x1b[31m'
        actual = utils.color_to_ansi_escape(utils.Color.RED, foreground = True)
        self.assertEqual(expected, actual)

        expected = '\x1b[42m'
        actual = utils.color_to_ansi_escape(utils.Color.GREEN, foreground = False)
        self.assertEqual(expected, actual)

        expected = '\x1b[42m'
        actual = utils.color_to_ansi_escape('2', foreground = False)
        self.assertEqual(expected, actual)

        expected = '\x1b[38;5;123m'
        actual = utils.color_to_ansi_escape(123, foreground = True)
        self.assertEqual(expected, actual)

        expected = '\x1b[38;2;30;215;96m'
        actual = utils.color_to_ansi_escape((30, 215, 96), foreground = True)
        self.assertEqual(expected, actual)

        expected = '\x1b[38;2;215;30;96m'
        actual = utils.color_to_ansi_escape('215, 30, 96', foreground = True)
        self.assertEqual(expected, actual)

        expected = '\x1b[38;2;255;95;95m'
        actual = utils.color_to_ansi_escape('#ff5f5f', foreground = True)
        self.assertEqual(expected, actual)

    def test_clamp(self):
        self.assertEqual(0, utils.clamp(-1, 0, 1))
        self.assertEqual(0, utils.clamp(0, 0, 1))
        self.assertEqual(0.5, utils.clamp(0.5, 0, 1))
        self.assertEqual(1, utils.clamp(1, 0, 1))
        self.assertEqual(1, utils.clamp(2, 0, 1))

    def test_is_close(self):
        self.assertTrue(utils.is_close(1.002, 1.0, 0.01))
        self.assertTrue(utils.is_close(0.998, 1.0, 0.01))
        self.assertTrue(utils.is_close(1.01, 1.0, 0.01))
        self.assertTrue(utils.is_close(0.99, 1.0, 0.01))
        self.assertFalse(utils.is_close(1.02, 1.0, 0.01))
        self.assertFalse(utils.is_close(0.98, 1.0, 0.01))

    def test_wpm_to_spu(self):
        self.assertEqual(0.06, utils.wpm_to_spu(20))
        self.assertEqual(0.12, utils.wpm_to_spu(10))

    def test_spu_to_wpm(self):
        self.assertEqual(10, utils.spu_to_wpm(0.12))
        self.assertEqual(20, utils.spu_to_wpm(0.06))

    def test_preprocess_input_text(self):
        self.assertEqual("SS/", utils.preprocess_input_text("ß\\"))

    def test_preprocess_morse_text(self):
        self.assertEqual("... --- .../--", utils.preprocess_input_morse(" ...   --- ...  // -- "))

    def test_dual_split(self):
        expected0 = "key"
        expected1 = "value"
        actual0, actual1 = utils.dual_split("key:value", ":")
        self.assertEqual(expected0, actual0)
        self.assertEqual(expected1, actual1)

        expected1 = "value: with colon"
        actual0, actual1 = utils.dual_split("key:value: with colon", ":")
        self.assertEqual(expected0, actual0)
        self.assertEqual(expected1, actual1)

        expected1 = None
        actual0, actual1 = utils.dual_split("key", ":")
        self.assertEqual(expected0, actual0)
        self.assertEqual(expected1, actual1)

    def test_nwise(self):
        list = [1, 2, 3, 4, 5]

        expected = [(1,), (2,), (3,), (4,), (5,)]
        actual = [t for t in utils.nwise(list, n = 1)]
        self.assertEqual(expected, actual)

        expected = [(1, 2), (2, 3), (3, 4), (4, 5)]
        actual = [t for t in utils.nwise(list, n = 2)]
        self.assertEqual(expected, actual)

        expected = [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
        actual = [t for t in utils.nwise(list, n = 3)]
        self.assertEqual(expected, actual)

        # overlapped is an alias of nwise
        expected = [(1, 2, 3, 4), (2, 3, 4, 5)]
        actual = [t for t in utils.overlapped(list, group_size = 4)]
        self.assertEqual(expected, actual)

    def test_tuplewise(self):
        list = [1, 2, 3, 4, 5, 6, 7]

        expected = [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]
        actual = [t for t in utils.tuplewise(list, tuple_size = 1)]
        self.assertEqual(expected, actual)

        expected = [(1, 2), (3, 4), (5, 6)]
        actual = [t for t in utils.tuplewise(list, tuple_size = 2)]
        self.assertEqual(expected, actual)

        expected = [(1, 2, 3), (4, 5, 6)]
        actual = [t for t in utils.tuplewise(list, tuple_size = 3)]
        self.assertEqual(expected, actual)

        list = ['A', 'B']

        expected = []
        actual = [t for t in utils.tuplewise(list, tuple_size = 3, strict = True)]
        self.assertEqual(expected, actual)

        expected = [('A', 'B', None)]
        actual = [t for t in utils.tuplewise(list, tuple_size = 3, strict = False)]
        self.assertEqual(expected, actual)

    def test_SimpleMovingAverage(self):
        value = utils.SimpleMovingAverage(window_size = 3)

        self.assertTrue(value.empty())
        self.assertEqual(None, value.sma())

        # Test that the average works as long as there are fewer values than specified in the window size
        value.update(5)
        self.assertEqual(5, value.sma())

        value.update(6)
        self.assertEqual(5.5, value.sma())

        # Test that the oldest value (5) falls out of the moving average
        value.update(11)
        value.update(12)
        self.assertAlmostEqual(9.6666, value.sma(), 3)

        # Test that multiplying with a factor yields the correct moving average
        value.multiply(3)
        self.assertAlmostEqual(29, value.sma(), 3)

        # Test resetting the moving average
        value.reset(20)
        self.assertEqual(20, value.sma())

    def test_StringVerifier(self):
        expected_string = 'abcdefghijk'
        verifier = utils.StringVerifier(expected = expected_string, grace_width = 2)
        entirely_matching = True
        actual = ''

        # Test that if test string matches the expected string entirely even and there is no highlighting even if the
        # test string is shorter
        expected  = 'abcdefg'
        for char in 'abcdefg':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertTrue(entirely_matching)

        verifier.reset()
        entirely_matching = True
        actual = ''

        # Test that if both strings are equal and equally long they are matching entirely and there is no highlighting
        expected  = 'abcdefghijk'
        for char in 'abcdefghijk':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertTrue(entirely_matching)

        verifier.reset()
        entirely_matching = True
        actual = ''

        # Test that if test string is longer than the expected string they are not entirely matching and anything that
        # exceeds the expected string is highlighted green (i.e. \x1b[42m)
        expected  = 'abcdefghijk\x1b[42ml\x1b[0m\x1b[42mm\x1b[0m\x1b[42mn\x1b[0m'
        #            -----------🠇🠇🠇 : 3 additional characters: should result in a green (unexpected) 'lmn' at the end
        for char in 'abcdefghijklmn':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

        verifier.reset()
        entirely_matching = True
        actual = ''

        # Test that if test string is missing a character, the strings are not matching and the expected character is
        # highlighted red (i.e. \x1b[41m) and the actual character displayed normally
        expected  = 'ab\x1b[41mc\x1b[0mdefghijk'
        #            --🠇------- : 'd' instead of 'c': should result in a red (missing) 'c' between 'b' and 'd'
        for char in 'abdefghijk':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

        verifier.reset()
        entirely_matching = True
        actual = ''

        # Test that if test string has an incorrect character, the strings are not matching, the expected character is
        # highlighted red and the unexpected character is highlighted green
        expected  = 'abc\x1b[41md\x1b[0m\x1b[42mp\x1b[0mefghijk'
        #            ---🠇------- : 'p' instead of 'd': should result in a red (missing) 'd' and a green (unexpected) 'p'
        for char in 'abcpefghijk':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

        verifier.reset()
        entirely_matching = True
        actual = ''

        # Test that if test string has an additional character, the strings are not matching, the expected character is
        # highlighted red and the unexpected character is highlighted green; if the expected character follows anyhow it
        # is displayed a second time but normally
        expected  = 'abcd\x1b[41me\x1b[0m\x1b[42mp\x1b[0m\x1b[42me\x1b[0mfgh'
        #            ----🠇---- : 1 additional character: should result in a green (unexpected) 'p'
        for char in 'abcdpefgh':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

        verifier.reset()
        entirely_matching = True
        actual = ''

        # Test that if test string has multiple additional characters, the strings are not matching, the expected
        # character is highlighted red and the unexpected characters are highlighted green; if the expected character
        # follows anyhow it is displayed a second time but normally
        expected  = 'abcd\x1b[41me\x1b[0m\x1b[42mp\x1b[0m\x1b[42mp\x1b[0m\x1b[42mp\x1b[0m\x1b[42me\x1b[0mfgh'
        #            ----🠇🠇🠇---- : 3 additional characters: should result in a green (unexpected) 'ppp'
        for char in 'abcdpppefgh':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

        expected_string = 'Hello, World!'
        verifier.reset(expected = expected_string)
        entirely_matching = True
        actual = ''

        # Test that missing characters are added to the output and highlighted red, and that the strings are not
        # compared beyond what was actually received
        expected  = 'Hell\x1b[41mo\x1b[0m, Wo'
        for char in 'Hell, Wo':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        if expected != actual:
            print('\n\x1b[1;38;5;220;48;5;231m Expected \x1b[0m ', expected)
            print('\x1b[1;38;5;220;48;5;231m  Actual  \x1b[0m ', actual)
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

        # Test that incorrect characters are added to the output and highlighted red, and that the verification fails
        # for all actually received characters beyond the end of the expected string
        entirely_matching = True
        expected += 'rld\x1b[41m!\x1b[0m\x1b[42m.\x1b[0m\x1b[42m.\x1b[0m\x1b[42m.\x1b[0m'
        for char in 'rld...':
            matching, diff_string = verifier.verify(char, additive = True)
            actual += diff_string
            entirely_matching &= matching
        if expected != actual:
            print('\n\x1b[1;38;5;220;48;5;231m Expected \x1b[0m ', expected)
            print('\x1b[1;38;5;220;48;5;231m  Actual  \x1b[0m ', actual)
        self.assertEqual(expected, actual)
        self.assertFalse(entirely_matching)

import remorse.utils as utils
import unittest

class UtilsTests(unittest.TestCase):
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
        self.assertEqual("AEOEUESS", utils.preprocess_input_text("äöüß"))

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

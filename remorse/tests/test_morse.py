import io
import os
import remorse.morse as morse
import tempfile
import time
import unittest

TESTS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MorseTests(unittest.TestCase):
    def test_text_to_morse(self):
        expected = "... --- ..."
        actual = str(morse.text_to_morse("SOS"))
        self.assertEqual(expected, actual)

        # Test that leading and trailing spaces are ignored
        expected = morse.text_to_morse("SOS WE ARE IN TROUBLE")
        actual = morse.text_to_morse(" SOS WE ARE IN TROUBLE")
        self.assertEqual(expected, actual)
        actual = morse.text_to_morse("SOS WE ARE IN TROUBLE ")
        self.assertEqual(expected, actual)

        # Test that consecutive inner spaces are treated as a single space
        actual = morse.text_to_morse("SOS  WE ARE IN    TROUBLE")
        self.assertEqual(expected, actual)

    def test_morse_to_text(self):
        expected = "SOS"
        actual = morse.morse_to_text("... --- ...")
        self.assertEqual(expected, actual)

    def test_MorseString(self):
        self.assertRaises(Exception, morse.MorseString, "a")

        # Test that leading and trailing spaces are ignored
        expected = morse.MorseString("... --- ...")
        actual = morse.MorseString(" ... --- ...  ")
        self.assertEqual(expected, actual)

        # Test that consecutive inner spaces are treated as a single space
        actual = morse.MorseString("...  ---   ...")
        self.assertEqual(expected, actual)

        # Test that leading and trailing word pauses are ignored
        actual = morse.MorseString("/... --- ...")
        self.assertEqual(expected, actual)
        actual = morse.MorseString("... --- .../")
        self.assertEqual(expected, actual)

        # Test that consecutive inner word pauses are treated as a single word pause
        expected = morse.MorseString("... --- .../... --- ...")
        actual = morse.MorseString("... --- ...///... --- ...")
        self.assertEqual(expected, actual)
        actual = morse.MorseString("... --- ... // / ... --- ...")
        self.assertEqual(expected, actual)

    def test_MorsePrinter(self):
        iostream = io.StringIO()
        printer = morse.MorsePrinter(output_device = iostream)

        printer.emit_dit()
        iostream.seek(0)
        self.assertEqual(".", iostream.read())

        printer.emit_intra_character_pause()
        iostream.seek(0)
        self.assertEqual(".", iostream.read())

        printer.emit_dah()
        iostream.seek(0)
        self.assertEqual(".-", iostream.read())

        printer.emit_inter_character_pause()
        printer.emit_dah()
        printer.emit_dah()
        iostream.seek(0)
        self.assertEqual(".- --", iostream.read())

        printer.emit_inter_word_pause()
        printer.emit_dit()
        iostream.seek(0)
        self.assertEqual(".- --/.", iostream.read())

        printer.emit_inter_word_pause()
        printer.emit(morse.MorseString(".-.-.-"))
        iostream.seek(0)
        self.assertEqual(".- --/./.-.-.- ", iostream.read())

    def test_MorseVisualizer(self):
        iostream = io.StringIO()
        visualizer = morse.MorseVisualizer(output_device = iostream)

        visualizer.emit_dit()
        iostream.seek(0)
        self.assertEqual("▄", iostream.read())

        visualizer.emit_intra_character_pause()
        iostream.seek(0)
        self.assertEqual("▄\u2003", iostream.read())

        visualizer.emit_dah()
        iostream.seek(0)
        self.assertEqual("▄\u2003▄▄▄", iostream.read())

        visualizer.emit_inter_character_pause()
        visualizer.emit_dah()
        iostream.seek(0)
        self.assertEqual("▄\u2003▄▄▄\u2003\u2003\u2003▄▄▄", iostream.read())

        visualizer.emit_inter_word_pause()
        visualizer.emit_dit()
        iostream.seek(0)
        self.assertEqual("▄\u2003▄▄▄\u2003\u2003\u2003▄▄▄\u2003\u2003\u2003\u2003\u2003\u2003\u2003▄", iostream.read())

        iostream2 = io.StringIO()
        visualizer2 = morse.MorseVisualizer(output_device = iostream2)
        visualizer2.enable_colored_output()
        visualizer2.set_colorization_mode(morse.ColorizationMode.CHARACTERS)
        visualizer2.set_colors(morse.Color.RED, morse.Color.GREEN)

        visualizer2.pre_emit_character()
        visualizer2.emit_dit()
        visualizer2.emit_intra_character_pause()
        visualizer2.emit_dah()
        visualizer2.emit_inter_character_pause()
        visualizer2.pre_emit_character()
        visualizer2.emit_dit()
        visualizer2.emit_intra_character_pause()
        visualizer2.emit_dah()
        visualizer2.post_emit()
        iostream2.seek(0)
        self.assertEqual("\x1b[31m▄\u2003▄▄▄\u2003\u2003\u2003\x1b[32m▄\u2003▄▄▄\x1b[0m", iostream2.read())

    def test_MorsePlayer(self):
        player = morse.MorsePlayer(speed = 10)

        # The emittance will be muted but this will not affect the duration which we measure
        player.mute()

        # Unmeasured first playback (connection to audio device shall not be measured)
        player.emit_intra_character_pause()

        t0 = time.time()
        player.emit(morse.MorseString("... --- ..."))
        t1 = time.time()

        delta = t1 - t0
        self.assertTrue(3.14 < delta < 3.34)

        player.emit_intra_character_pause()
        player.set_speed("20wpm")

        t0 = time.time()
        player.emit(morse.MorseString("... --- .../.-.-"))
        t1 = time.time()

        delta = t1 - t0
        self.assertTrue(2.6 < delta < 2.8)

    def test_MorseSoundFileReader(self):
        lengths = [105, 90, 90, 10, 210, 90, 110]
        expected = [105, 90, 310, 90, 110]
        morse.MorseSoundFileReader.patch_length_holes(lengths, 20)
        self.assertEqual(expected, lengths)

        lengths = [105, 90, 310, 90, 110]
        expected = [100, 100, 300, 100, 100]
        morse.MorseSoundFileReader.compensate_overhang(lengths, 5)
        self.assertEqual(expected, lengths)

        lengths = [105, 95, 310, 98, 101, 296, 100]
        unit_duration = morse.MorseSoundFileReader.find_unit_duration(lengths)
        self.assertEqual(100, int(unit_duration))

        file_path = os.path.join(TESTS_DIRECTORY, 'audio/sos.mp3')
        reader = morse.MorseSoundFileReader(file_path = file_path, volume_threshold = 0.35, normalize_volume = True,
                                            use_multiprocessing = False, kernel_seconds = 0.001,
                                            min_signal_seconds = 0.01, low_cut_frequency = None,
                                            high_cut_frequency = None, show_plots = False)

        expected = morse.MorseString("... --- ...")
        actual = reader.read()
        self.assertEqual(expected, actual)

    def test_MorseSoundFileWriter(self):
        tmpfile = tempfile.TemporaryFile()

        writer = morse.MorseSoundFileWriter(file = tmpfile, volume = 0.9, frequency = 800, speed = 20,
                                            sample_rate = 8000)
        writer.write(morse.MorseString("... --- ..."))
        writer.close()

        expected = 28844
        actual = tmpfile.tell()
        self.assertEqual(expected, actual)

        tmpfile.seek(0x30)
        expected = 28047
        actual = int.from_bytes(tmpfile.read(2), byteorder = 'little', signed = True)
        self.assertEqual(expected, actual)

        tmpfile.seek(0x3a)
        expected = -28048
        actual = int.from_bytes(tmpfile.read(2), byteorder = 'little', signed = True)
        self.assertEqual(expected, actual)

        tmpfile.seek(0x3ee)
        expected = 0
        actual = int.from_bytes(tmpfile.read(2), byteorder = 'little', signed = True)
        self.assertEqual(expected, actual)

        tmpfile.close()

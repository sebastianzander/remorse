import io
import numpy as np
import os
import remorse.morse as morse
import tempfile
import time
import unittest

TESTS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class SimpleReceiver(morse.StreamReceiver):
    def __init__(self):
        super().__init__()
        self._received = ''

    def receive(self, data: str):
        self._received += data

    def received(self) -> str:
        return self._received

class MorseTests(unittest.TestCase):
    def test_Stream(self):
        stream = morse.Stream()
        simple_receiver = SimpleReceiver()
        stream.subscribe(simple_receiver)

        expected = "Message for streaming"
        for char in expected:
            stream.send(char)

        self.assertEqual(expected, simple_receiver.received())

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
        printer = morse.MorsePrinter(output_device = iostream, strip_escape_sequences = True)

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
        self.assertEqual(".- -- / .", iostream.read())

        printer.emit_inter_word_pause()
        printer.emit(morse.MorseString(".-.-.-"))
        iostream.seek(0)
        self.assertEqual(".- -- / . / .-.-.-", iostream.read())

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
        player = morse.MorsePlayer(frequency = 800, speed = 10, volume = 0.9, sample_rate = 8000)

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

        # Test change of sample rate and implicit reinitalization between emittance
        player.emit(morse.MorseString("..."))
        player.set_sample_rate(22050)
        player.emit_intra_character_pause()
        player.emit(morse.MorseString("---"))
        player.set_sample_rate(44100)
        player.emit_intra_character_pause()
        player.emit(morse.MorseString("..."))

    def test_MorseStringStreamer(self):
        # Test Morse to text conversion
        streamer = morse.MorseStringStreamer(data = "... --- ... / - .. - .- -. .. -.-.", data_is_morse = True)
        simple_receiver = SimpleReceiver()
        streamer.text_stream().subscribe(simple_receiver)
        streamer.read()

        expected = "SOS TITANIC"
        self.assertEqual(expected, simple_receiver.received())

        # Test text to Morse conversion
        streamer = morse.MorseStringStreamer(data = "RUHET IN FRIEDEN", data_is_morse = False)
        simple_receiver = SimpleReceiver()
        streamer.morse_stream().subscribe(simple_receiver)
        streamer.read()

        expected = ".-. ..- .... . -/.. -./..-. .-. .. . -.. . -."
        self.assertEqual(expected, simple_receiver.received())

    def test_MorseSoundStreamer(self):
        signals  = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        morse.MorseSoundStreamer.remove_short_signals(signals, 3)
        self.assertTrue(np.array_equal(expected, signals))

        file_path = os.path.join(TESTS_DIRECTORY, 'audio', 'sos.mp3')
        streamer = morse.MorseSoundStreamer(device = file_path, input = True, output = False,
                                            threshold = 0.35, normalize_volume = True, min_signal_size = '0.01s',
                                            low_cut_frequency = None, high_cut_frequency = None, buffer_size = '0s',
                                            open = True, plot = False)

        expected = morse.MorseString("... --- ...")
        actual = streamer.read()
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

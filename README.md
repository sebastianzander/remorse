# Remorse

Remorse is a command line utility program and library that allows to convert between plain text strings, Morse code
strings and Morse code sound.

## Installation

Remorse is not yet available in a Python package index like PyPI. However, you can download one of the releases from
[here](https://github.com/sebastianzander/remorse/releases) and run `pip install` on the downloaded file, like e.g:

```bash
$ pip install ~/Downloads/remorse-0.2.0.tar.gz
```

Remorse requires the `pyaudio` package which in turn requires the `PortAudio` library. Since it could be that `pip` is
unable to build and install `PortAudio` itself, you may have to
[download PortAudio](http://files.portaudio.com/download.html) for your operating system or
[build PortAudio](http://portaudio.com/docs/v19-doxydocs/compile_linux.html) and then install it manually before you
will be able to install Remorse via `pip`.

## The Program

### Usage Syntax

```bash
$ remorse [format:]<input> --output <format>[:args] [--output <format>[:args]] [options..]
```

### Input and Output Formats

The following formats can be used for input and output. Plain Latin `text`, Morse `code` and `file`s can be used as input
data as is without a format designation prefix. If a file does not exist at the given file path the path will be
interpreted as text instead.

 * `text` / `t` for plain Latin text reading (input) and writing (output)
 * `code` / `c` for Morse code string reading (input) and writing (output) in standard formatting
 * `nicecode` / `n` for Morse code string writing (output) in nice formatting where character width corresponds to duration
 * `sound` / `s` for Morse code live-decoding from an audio input device (input) or playback (output)
 * `file` / `f` for Morse code decoding from a sound file (input) or generation of a sound file (output)

> `sound` needs to be followed by a colon if used as input and you want to choose from a list of available audio input
devices, e.g. `sound:`

> `sound:` may be followed by the name of the audio input device if you know it, e.g. `sound:Microphone`

> `file` needs to be followed by a colon and the file name/path of the file to write to,
e.g. `file:~/sounds/morse.mp3`. If you wish to read from a file you do not need to prefix the path.

### Filtering and Noise Suppression

Remorse supports command line options for enabling filtering of and noise reduction in audio files when trying to decode
Morse signals from them. While **automatic filtering** analyzes the frequency spectrum of the audio data and reduces it
to the most probable and most prominent frequency, **manual filtering** allows you to set up either a low pass, high
pass or band pass filter.

```bash
$ remorse --filtering none | auto | lowpass:lo | highpass:hi | bandpass:lo,hi
```

Low and high pass require an argument specifying the threshold frequency in Hz or kHz, while band pass requires two
arguments: both the low and the high threshold frequency.

**Noise reduction** tries to reduce constant audio signals with a wide range of frequencies and low magnitude.
There are three different levels: low, medium and high.

```bash
$ remorse --noise-reduction none | auto | low | medium | high
```

The automatic mode can be used to let the software decide for itself how hard it will try to find and remove noise.

### Usage Examples

1. Convert a given plain Latin text to a standard Morse code representation:

    ```bash
    $ remorse "Learn Morse today" --output code
    ```

    Console output:

    ```plain
    .-.. . .- .-. -. / -- --- .-. ... . / - --- -.. .- -.--
    ```

2. Convert a given plain text to a visually appealing Morse code representation and to an audible representation, both
in sync character by character. Prefix with `text:` if your text contains actual colons:

    ```bash
    $ remorse "text:Python: Yeah!" --output nicecode,sound
    ```

    Console output:

    ```plain
    ▄ ▄▄▄ ▄▄▄ ▄   ▄▄▄ ▄ ▄▄▄ ▄▄▄   ▄▄▄   ▄ ▄ ▄ ▄   ▄▄▄ ▄▄▄ ▄▄▄   ▄▄▄ ▄   ▄▄▄ ▄▄▄ ▄▄▄ ▄ ▄ ▄          ▄▄▄ ▄ ▄▄▄ ▄▄▄   ▄   ▄ ▄▄▄   ▄ ▄ ▄ ▄   ▄▄▄ ▄ ▄▄▄ ▄ ▄▄▄ ▄▄▄
    ```

3. Reads the Morse code from the given audio file, applies automatic filtering and noise suppression and then converts
it to plain text:

    ```bash
    $ remorse ~/sounds/old_morse.mp3 --output text --filtering auto --noise-reduction low
    ```

4. Reads the Morse code from the given audio file and converts it to a new audio file:

    ```bash
    $ remorse ~/sounds/old_morse.mp3 --output file:~/sounds/new_fresh_morse.mp3 --speed 20wpm --frequency 1khz --sample-rate 44100hz
    ```

5. Reads the Morse code from the input string and saves an audible representation to the specified file using the given
speed, frequency and sample rate:

    ```bash
    $ remorse "... --- ..." --output file:~/sounds/synthetized_morse.wav --speed 20wpm --frequency 600hz --sample-rate 8khz
    ```

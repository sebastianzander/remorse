# Remorse

Remorse is a command line utility program and library that allows to convert between plain text strings, Morse code strings and Morse code sound.

## The Program

### Usage Syntax

```bash
$ remorse <input> --output <format:value> [--output <format:value>] [options..]
```

**Useable Formats**

 * `text` / `t`
 * `code` / `c`
 * `nicecode` / `n` (output only)
 * `sound` / `s` (output only)
 * `file` / `f`

### Usage Examples

1. Convert a given plain text to a standard Morse code representation first and plays an audible representation with the given speed and frequency after that:

    ```bash
    $ remorse "Learn Morse today" --output code --output sound --speed 20wpm --frequency 600hz
    ```

    Console output:

    ```plain
    .-.. . .- .-. -. / -- --- .-. ... . / - --- -.. .- -.--
    ```

2. Convert a given plain text to a visually appealing Morse code representation and to an audible representation, both in sync character by character:

    ```bash
    $ remorse "text:Python: Yeah!" --output nicecode --output sound --simultaneous
    ```

    Console output:

    ```plain
    ▄ ▄▄▄ ▄▄▄ ▄   ▄▄▄ ▄ ▄▄▄ ▄▄▄   ▄▄▄   ▄ ▄ ▄ ▄   ▄▄▄ ▄▄▄ ▄▄▄   ▄▄▄ ▄   ▄▄▄ ▄▄▄ ▄▄▄ ▄ ▄ ▄          ▄▄▄ ▄ ▄▄▄ ▄▄▄   ▄   ▄ ▄▄▄   ▄ ▄ ▄ ▄   ▄▄▄ ▄ ▄▄▄ ▄ ▄▄▄ ▄▄▄
    ```

3. Reads the Morse code from the given audio file and converts it to plain text:

    ```bash
    $ remorse file:~/sounds/old_morse.mp3 --output text --volume-threshold 0.4
    ```

4. Reads the Morse code from the input string and saves an audible representation to the specified file using the given speed, frequency and sample rate:

    ```bash
    $ remorse "code:... --- ..." --output file:~/sounds/synthetized_morse.wav --speed 20wpm --frequency 600hz --sample-rate 8khz
    ```

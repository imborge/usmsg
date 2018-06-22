# usmsg

usmsg (ultrasound message) - hide messages in songs!

## Requirements

- ffmpeg
- numpy
- scipy

## Usage

### Encoding

```bash
python usmsg.py -e -i input_file.wav -o output_file.wav -t "Hello, what's up?"
```

### Decoding

```
python usmsg.py -d -i input_file.wav
```

## Limitations

All characters encoded lasts for one second, if the hidden text is longer than the input audio, the text will be truncated (and vice versa).

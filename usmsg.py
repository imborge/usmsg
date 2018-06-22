import numpy as np
import scipy.io.wavfile as wavfile
import sys
import pprint
import os
import subprocess
from optparse import OptionParser

pp = pprint.PrettyPrinter()

sampleRate = 48000
lo_freq = 22000.0
hi_freq = 24000.0
allowed_characters = [chr(x) for x in range(32, 127)]

def genSamples(sampleRate, frequency, duration):
    return (np.sin(2 * np.pi * np.arange(sampleRate * duration) * frequency / sampleRate)).astype(np.float32)

def char_to_freq(char):
    num_allowed_characters = len(allowed_characters)
    char_pos = allowed_characters.index(char)
    freq_fraction = char_pos / (num_allowed_characters - 1)
    return lo_freq + (freq_fraction * (hi_freq - lo_freq))

def chars_to_freq(chars):
    return [char_to_freq(char) for char in chars]

def chars_to_samples(chars):
    frequencies = chars_to_freq(chars)
    samples = [genSamples(sampleRate, frequency, 1.0) for frequency in frequencies]
    return np.concatenate(tuple(samples))

def save_samples(filename, samples):
    wavfile.write(filename, sampleRate, samples)

def load_samples(filename):
    return wavfile.read(filename)

def encode(input_audio_file, text, output_filename):
    samples = chars_to_samples(text)
    tmp_filename = "___tmp_encoding1.wav"
    tmp_left_filename = "___tmp_left.wav"   # left audio channel
    tmp_right_filename = "___tmp_right.wav" # right audio channel
    save_samples(tmp_filename, samples)
    
    # extract left and right from input audio file
    subprocess.run(
        ["ffmpeg", "-i", input_audio_file, "-ar", "48000", "-acodec", "pcm_f32le", 
         "-map_channel", "0.0.0", tmp_left_filename, "-map_channel", "0.0.1", tmp_right_filename],
        input="y\ny\ny\ny\ny\n", encoding="ascii", # hacky solution to enter "y" to replace files
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # put our generated audio on the left channel, and input audio on the right
    subprocess.run(
        ["ffmpeg", "-i", tmp_filename, "-i", tmp_right_filename, "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]", 
         "-map", "[aout]", "-acodec", "pcm_f32le", output_filename],
        input="y\ny\ny\ny\ny\n", encoding="ascii", # hacky solution to enter "y" to replace files
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        os.remove(tmp_filename)
        os.remove(tmp_left_filename)
        os.remove(tmp_right_filename)
    except OSError:
        pass
    
def get_freq_peaks(sample_rate, data):
    try:
        fft = np.fft.fft(data)
        n = data.size
        freqs = np.fft.fftfreq(n, d=1)
        idx = np.argmax(np.abs(fft))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * sample_rate)
        return freq_in_hertz
    except:
        return 90000

def find_closest_freq(needle, haystack):
    closest = None
    for freq in haystack:
        if closest == None or abs(needle - freq) < abs(needle - closest):
            closest = freq
    return closest

def decode(filename, table, inverse_table):
    # split left and right channels
    tmp_left_filename = "___tmp_left.wav"
    tmp_right_filename = "___tmp_right.wav"
    subprocess.run(
        ["ffmpeg", "-i", filename,
         "-map_channel", "0.0.0", tmp_left_filename, "-map_channel", "0.0.1", tmp_right_filename],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    rate, data = load_samples(tmp_left_filename)
    duration = int(data.size / rate)
    freqs = [get_freq_peaks(rate, data[rate*i:rate*i+rate]) for i in range(0, duration)]
    chars = []
    for freq in freqs:
        closest_freq = find_closest_freq(freq, inverse_table)
        chars.append(inverse_table[closest_freq])
    try:
        os.remove(tmp_left_filename)
        os.remove(tmp_right_filename)
    except OSError:
        pass

    print()
    print("Message:")
    print("".join(chars))

def build_tables(allowed_characters, min_freq, max_freq):
    table = {}
    inverse_table = {}
    for char in allowed_characters:
        freq = char_to_freq(char)
        table[char] = freq
        inverse_table[freq] = char
    return table, inverse_table
        

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input_audio",
                      action="store", type="string", dest="input_audio")
    parser.add_option("-e", "--encode", action="store_true", dest="encode")
    parser.add_option("-d", "--decode", action="store_false", dest="encode")
    parser.add_option("-t", "--text", metavar="TEXT", dest="text")
    parser.add_option("-o", "--output_audio",
                      action="store", type="string", dest="output_audio")

    options, args = parser.parse_args()
    
    if options.encode and ((not options.text) or (not options.input_audio) or (not options.output_audio)):
        parser.error("Need to specify text, input audio, and output audio when encoding")
    
    if not options.encode and not options.input_audio:
        parser.error("Need to specify input audio when decoding")

    table, inverse_table = build_tables(allowed_characters, lo_freq, hi_freq)

    if options.encode:
        encode(input_audio_file=options.input_audio,
               text=options.text,
               output_filename=options.output_audio)
    else:
        decode(filename=options.input_audio, 
               table=table, 
               inverse_table=inverse_table)

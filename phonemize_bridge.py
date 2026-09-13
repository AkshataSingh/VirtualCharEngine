import sys
from piper import PiperVoice

def main():
    text = sys.stdin.read()
    voice = PiperVoice.load("en_US-lessac-medium.onnx")
    phonemes = voice.phonemize(text)
    ids = voice.phonemes_to_ids(phonemes[0])
    print(",".join(str(i) for i in ids))

if __name__ == "__main__":
    main()
import subprocess
import time
import wave

TEXT = "Hello! I'm here to assist you with any questions or concerns you might have."
MODEL ="en_US-lessac-medium-pruned.onnx"
OUTPUT_FILE = "pruned_output.wav"

start = time.perf_counter()
subprocess.run(
    ["piper", "--model", MODEL, "--output_file", OUTPUT_FILE],
    input=TEXT,
    text=True,
    check=True,
)
end = time.perf_counter()

synthesis_time = end - start

with wave.open(OUTPUT_FILE, "rb") as wf:
    frames = wf.getnframes()
    rate = wf.getframerate()
    audio_duration = frames / float(rate)

rtf = synthesis_time / audio_duration

print("\n--- pruned TTS Latency ---")
print(f"Synthesis time: {synthesis_time:.3f} s")
print(f"Audio duration: {audio_duration:.3f} s")
print(f"Real-Time Factor (RTF): {rtf:.3f}  (lower is better, <1.0 = faster than real-time)")
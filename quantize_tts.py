import shutil
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_IN = "en_US-lessac-medium.onnx"
MODEL_OUT = "en_US-lessac-medium-int8.onnx"

quantize_dynamic(
    model_input=MODEL_IN,
    model_output=MODEL_OUT,
    weight_type=QuantType.QUInt8,
)

# Piper needs a matching .onnx.json config file next to the model —
# it's just metadata (phonemes, speaker info), not weights, so we copy it as-is
shutil.copy("en_US-lessac-medium.onnx.json", "en_US-lessac-medium-int8.onnx.json")

print(f"Quantized model saved to {MODEL_OUT}")
import onnx
import numpy as np
from onnx import numpy_helper

MODEL_IN = "en_US-lessac-medium.onnx"
MODEL_OUT = "en_US-lessac-medium-pruned.onnx"
SPARSITY = 0.20

model = onnx.load(MODEL_IN)

all_weights = []
for initializer in model.graph.initializer:
    arr = numpy_helper.to_array(initializer)
    if arr.dtype in (np.float32, np.float64) and arr.size > 1:
        all_weights.append(np.abs(arr).flatten())

all_weights = np.concatenate(all_weights)
threshold = np.quantile(all_weights, SPARSITY)
print(f"Global magnitude threshold for {SPARSITY*100:.0f}% sparsity: {threshold:.6f}")

new_initializers = []
total_params = 0
total_zeroed = 0
for initializer in model.graph.initializer:
    arr = numpy_helper.to_array(initializer)
    if arr.dtype in (np.float32, np.float64) and arr.size > 1:
        mask = np.abs(arr) < threshold
        total_params += arr.size
        total_zeroed += mask.sum()
        arr = arr.copy()
        arr[mask] = 0
        new_initializers.append(numpy_helper.from_array(arr, name=initializer.name))
    else:
        new_initializers.append(initializer)

del model.graph.initializer[:]
model.graph.initializer.extend(new_initializers)

onnx.save(model, MODEL_OUT)
print(f"Actual achieved sparsity: {total_zeroed/total_params*100:.1f}%")
print(f"Saved pruned model to {MODEL_OUT}")
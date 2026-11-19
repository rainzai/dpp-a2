import os

import numpy as np
import onnx
import onnxruntime as ort

os.makedirs("../data", exist_ok=True)
model = onnx.load("../models/bert-base-uncased.onnx")

nodes_to_extract = {
    "Q": "/bert/encoder/layer.0/attention/self/query/Add_output_0",
    "K": "/bert/encoder/layer.0/attention/self/key/Add_output_0",
    "V": "/bert/encoder/layer.0/attention/self/value/Add_output_0",
}

# Add them to outputs
for name in nodes_to_extract.values():
    info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(info)

model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "../models/model_extract.onnx")

# Run Inference
session = ort.InferenceSession("../models/model_extract.onnx")

# randint(0, vocab_size), vocab size can usually be found at the start
input_feed = {
    "input_ids": np.random.randint(0, 30522, (1, 128), dtype=np.int64),
    "attention_mask": np.ones((1, 128), dtype=np.int64),
    "token_type_ids": np.zeros((1, 128), dtype=np.int64),
}

outputs = session.run(None, input_feed)
output_names = [o.name for o in session.get_outputs()]

# Save
for label, node_name in nodes_to_extract.items():
    idx = output_names.index(node_name)
    tensor = np.array(outputs[idx], dtype=np.float32)

    np.save(f"../data/{label}.npy", tensor)
    tensor.tofile(f"../data/{label}.bin")
    print(f"Saved {label} (shape: {tensor.shape})")

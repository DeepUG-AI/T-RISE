from onnxruntime.quantization import quantize_dynamic, QuantType

# Input and output model paths
model_fp32 = "model_cpu.onnx"
model_int8 = "model_cpu_int8.onnx"

# Quantize the FP32 model to INT8
quantize_dynamic(
    model_input=model_fp32,  # Path to the FP32 model
    model_output=model_int8,  # Path to save the INT8 model
    weight_type=QuantType.QUInt8  # Quantization type
)

print(f"Model has been quantized to INT8 and saved as {model_int8}.")

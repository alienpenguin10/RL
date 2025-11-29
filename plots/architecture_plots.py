import torch
from agents.networks import PolicyNetwork, ValueNetwork

# 1. Instantiate your model
policy_model = PolicyNetwork()
value_model = ValueNetwork()
value_model.eval() # Set to evaluation mode

# 2. Create a dummy input matching your shape (Batch Size, Channels, Height, Width)
# You specified 96x96 RGB images, so:
dummy_input = torch.randn(1, 3, 96, 96)

# 3. Export to ONNX
torch.onnx.export(
    value_model, 
    dummy_input, 
    "value_network.onnx", 
    input_names=['image_input'], 
    output_names=['value'],
    opset_version=11
)

print("Model saved to policy_network.onnx")
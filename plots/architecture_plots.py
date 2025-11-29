import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from agents.networks import PolicyNetwork, ValueNetwork, ConvNet

# 1. Instantiate your model
policy_model = PolicyNetwork()
# value_model = ValueNetwork()
# conv_model = ConvNet()
policy_model.eval() # Set to evaluation mode

# 2. Create a dummy input matching your shape (Batch Size, Channels, Height, Width)
# You specified 96x96 RGB images, so:
dummy_input = torch.randn(1, 3, 96, 96)

# 3. Export to ONNX
torch.onnx.export(
    policy_model,
    dummy_input,
    "policy_model.onnx",
    input_names=['image_input'],
    output_names=['value'],
    opset_version=18
)

print("Model saved to policy_model.onnx")
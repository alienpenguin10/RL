import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from agents.networks import ConvNet, PolicyNetwork, ValueNetwork

policy_model = PolicyNetwork()
# value_model = ValueNetwork()
# conv_model = ConvNet()
policy_model.eval()
dummy_input = torch.randn(1, 3, 96, 96)

torch.onnx.export(
    policy_model,
    dummy_input,
    "policy_model.onnx",
    input_names=["image_input"],
    output_names=["value"],
    opset_version=18,
)

print("Model saved to policy_model.onnx")

import torch
import torch.nn as nn
from sam3.model_builder import build_sam3_image_model

# Test model loading and forward pass
model = build_sam3_image_model()
model = nn.DataParallel(model, device_ids=[0, 1])
model.eval()

# Create dummy input
batch_size = 2
image = torch.randn(batch_size, 3, 1008, 1008)
points = torch.randn(batch_size, 1, 2)
labels = torch.ones(batch_size, 1, dtype=torch.int64)

print('Model loaded successfully')
print('Testing forward pass...')

with torch.no_grad():
    try:
        output = model(image, points, labels)
        print('Forward pass successful!')
        print(f'Output keys: {list(output.keys())}')
    except Exception as e:
        print(f'Error during forward pass: {e}')
        import traceback
        traceback.print_exc()
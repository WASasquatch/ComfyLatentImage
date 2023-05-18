# ComfyLatentImage

## Usage

```python
from PIL import Image
import safetensors.torch
import piexif
import zipfile
from io import BytesIO

from ComfyImage import ComfyLatentImage

# Input Data
safetensor = safetensors.torch.load_file("/content/ComfyUI_00044_.latent")
tensor = safetensor["latent_tensor"]
image = Image.open("/content/ComfyUI2_00001_.jpg")
print("Tensor Shape:", tensor.shape)

image_path = 'image.latent.webp'
ComfyLatentImage.saveComfyLatent(tensor, image, image_path, 512)
comfylatent = Image.open(image_path)

# Example Load usage
extracted_tensor = ComfyLatentImage.loadComfyLatent(comfylatent)
print("Extracted Tensor Shape:", extracted_tensor['latent_tensor'].shape)
```

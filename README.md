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

filename = 'image.latent.png'
ComfyLatentImage.saveComfyLatent(tensor, image, filename, 512, 'png')
comfylatent = Image.open(filename)

# Example Load usage
extracted_tensor = ComfyLatentImage.loadComfyLatent(comfylatent)
print("Extracted Tensor Shape:", extracted_tensor['latent_tensor'].shape)
```

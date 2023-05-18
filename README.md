# ComfyLatentImage

## Usage

```python
from PIL import Image
import safetensors.torch
import piexif
import zipfile
from io import BytesIO

from ComfyImage import ComfyLatentImageIO

# Input Data
safetensor = safetensors.torch.load_file("/content/ComfyUI_00044_.latent")
tensor = safetensor["latent_tensor"]
image = Image.open("/content/ComfyUI2_00001_.jpg")
print("Tensor Shape:", tensor.shape)

# Output path and filename
latent_image_file = 'image.latent.png'

# Save image with latent embedded
comfyio = ComfyLatentImageIO(mdim=1024, format='png')
comfyio.save(tensor, image, latent_image_file)

# Load embedded latent image
comfylatent = Image.open(latent_image_file)

# Extract latent tensor
extracted_tensor = comfyio.load(comfylatent)
print("Extracted Tensor Shape:", extracted_tensor['latent_tensor'].shape)

```

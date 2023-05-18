# ComfyLatentImageIO

The idea behind ComfyLatentImageIO is a way to store a latent tensor that has been encoded by a VAE for use in stable diffusion workflows, while being a regular image format that can be easily distinguisable on a computer, and shared online, and viewed, without any extra tools to process a tensor on image browsers and users CPUs. 

## Pros
- Easily manageable with a image of what they represent
- Easily shareable in the community and retaining the integrity of what they represent
- Image can still store workflow/prompt data as well
- Image can be viewed on cross-platofrm easily without preparation
- Can be used in web platforms without special handling
- Can be used directly in other DCC apps, especially when PNG

## Cons
- Slightly larger than a raw safetensors.latent image itself (when webp)
- Only requires ComfyUI to handle new format
- ???

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

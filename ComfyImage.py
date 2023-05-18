from PIL import Image, PngImagePlugin
import safetensors.torch
import piexif
import zipfile
import os
from io import BytesIO


class ComfyLatentImageIO():
    def __init__(self, mdim=1280, format='webp'):
        self.mdim = mdim
        self.format = format

    def save(self, tensor, image, image_path, mdim=1280, format='webp'):
        tensor = {"latent_tensor": tensor}
        tensor_bytes = safetensors.torch.save(tensor)
        compressed_data = BytesIO()
        with zipfile.ZipFile(compressed_data, mode='w') as archive:
            archive.writestr("latent_tensor", tensor_bytes)
        image_with_metadata = image.copy()
        width, height = image_with_metadata.size
        if width > mdim or height > mdim:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width = mdim
                new_height = int(mdim / aspect_ratio)
            else:
                new_width = int(mdim * aspect_ratio)
                new_height = mdim
            image_with_metadata = image_with_metadata.resize((new_width, new_height), Image.ANTIALIAS)
        exif_data = piexif.load(image_with_metadata.info["exif"])
        exif_data["Exif"][piexif.ExifIFD.UserComment] = compressed_data.getvalue()
        exif_bytes = piexif.dump(exif_data)
        
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext == '.webp':
            image_with_metadata.save(image_path, format='webp', exif=exif_bytes)
        elif file_ext == '.png':
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("UserComment", compressed_data.getvalue())
            image_with_metadata.save(image_path, format='png', exif=exif_bytes, pnginfo=png_info, optimized=True)
        else:
            raise ValueError("Invalid file extension. Only '.webp' and '.png' extensions are supported.")

    def load(self, image):
        exif_data = piexif.load(image.info["exif"])
        if piexif.ExifIFD.UserComment in exif_data["Exif"]:
            compressed_data = exif_data["Exif"][piexif.ExifIFD.UserComment]
            compressed_data_io = BytesIO(compressed_data)
            with zipfile.ZipFile(compressed_data_io, mode='r') as archive:
                tensor_bytes = archive.read("latent_tensor")
            tensor = safetensors.torch.load(tensor_bytes)
            return tensor
        return None

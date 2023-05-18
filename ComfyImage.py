from PIL import Image
import safetensors.torch
import piexif
import zipfile
from io import BytesIO


class ComfyLatentImage():

    @staticmethod
    def saveComfyLatent(tensor, image, image_path, mdim=1280):
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
        image_with_metadata.save(image_path, format='webp', exif=exif_bytes)

    @staticmethod
    def loadComfyLatent(image):
        exif_data = piexif.load(image.info["exif"])
        if piexif.ExifIFD.UserComment in exif_data["Exif"]:
            compressed_data = exif_data["Exif"][piexif.ExifIFD.UserComment]
            compressed_data_io = BytesIO(compressed_data)
            with zipfile.ZipFile(compressed_data_io, mode='r') as archive:
                tensor_bytes = archive.read("latent_tensor")
            tensor = safetensors.torch.load(tensor_bytes)
            return tensor
        return None

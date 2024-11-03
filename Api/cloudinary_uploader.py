import os
import dotenv
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader

load_dotenv()


class cloudinary_uploader:
    def __init__(self):
        pass

    def upload_to_cloudinary(self, images, pre=''):
        image_urls = []
        for file in images:
            # file_name = f'HelloWorldBackend/latex_image+{file:03}.png'
            image_url = cloudinary.uploader.upload(file, public_id=file)  # not all files have a name
            image_urls.append(image_url)
        # Put files in folder or give them the same tag
        return image_urls

    def clear_images(self, image_urls, audio_urls):
        for image in image_urls:
            cloudinary.uploader.destroy(image)
        for audio in audio_urls:
            cloudinary.uploader.destroy(audio)
# https://res.cloudinary.com/shay/video/upload/w_500,h_400/w_500,h_400,l_latex_image_009_ksqfdm,fl_splice,du_3/so_0,fl_layer_apply/l_latex_image_009_ksqfdm,so_3/dog_vhvlyj.mp4
# l_latex_image_009_ksqfdm the second time changes the picture to be that
# Need to upload photos and sound, then build an api request that combines everything
# Issue with my path

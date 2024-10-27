import os
import uuid

import boto3


class aws_uploader:
    def __init__(self):
        boto3.client('s3')
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'manors-videos-bucket'

    def upload(self, file_path='final_movie.mp4', object_name='video.mp4'):
        file_path = file_path
        object_name = str(uuid.uuid1()) + object_name

        # Upload the file
        self.s3_client.upload_file(file_path, self.bucket_name, object_name)
        url = f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
        os.remove(file_path)
        return url



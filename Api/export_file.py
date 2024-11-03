import os
from uuid import uuid1

from google.cloud import storage

bn = 'helloworldvideostorage'
basic_path = '/Users/shay/PycharmProjects/HelloWorldBackend/'
secrets_location = '/Users/shay/PycharmProjects/HelloWorldBackend/Api/helloworldhackathonyoutube-403e869baf15.json'


def upload_to_bucket(path_to_file='final_movie.mp4', bucket_name=bn):
    """ Upload data to a bucket"""

    storage_client = storage.Client.from_service_account_json(secrets_location)

    # print(buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path_to_file)
    blob.upload_from_filename(basic_path + path_to_file)

    # returns a public url
    print(blob.public_url)
    return blob.public_url


def get_in_bucket(file_name, bucket_name=bn):
    storage_client = storage.Client.from_service_account_json(secrets_location)

    bucket = storage_client.get_bucket(bucket_name)
    return bucket.get_blob(basic_path + file_name)

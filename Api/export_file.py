from uuid import uuid1

from google.cloud import storage


# pip install --upgrade google-cloud-storage.
def upload_to_bucket(path_to_file='final_movie.jpeg', bucket_name='helloworldvideostorage'):
    """ Upload data to a bucket"""

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'helloworldhackathonyoutube-403e869baf15.json')

    # print(buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path_to_file)
    blob.upload_from_filename(path_to_file)

    # returns a public url
    print(blob.public_url)
    return blob.public_url


upload_to_bucket('HelloWorldBackend/background.jpeg')

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from flask.cli import load_dotenv
import urllib.parse

# Default bucket name
DEFAULT_BUCKET_NAME = 'manors-videos-bucket'

load_dotenv()

def upload_to_bucket(path_to_file='final_movie.mp4', bucket_name=DEFAULT_BUCKET_NAME, object_name=None, public=True):
    """Upload a file to an S3 bucket

    :param path_to_file: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified, uses the basename of path_to_file
    :param public: Whether the uploaded file should be publicly accessible
    :return: URL of the uploaded file or None if failed
    """
    # If S3 object_name was not specified, use file basename
    if object_name is None:
        object_name = os.path.basename(path_to_file)

    # Encode the object_name to ensure special characters are properly handled
    # encoded_object_name = urllib.parse.quote(object_name, safe='')
    encoded_object_name = object_name

    # Initialize S3 client using environment variables
    s3_client = boto3.client('s3')

    try:
        # Upload the file with the encoded object name
        s3_client.upload_file(path_to_file, bucket_name, encoded_object_name)
        # obj = s3_client.get_object(Bucket=bucket_name, Key=encoded_object_name)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=3600  # URL valid for 1 hour
        )

    except FileNotFoundError:
        print(f"The file {path_to_file} was not found.")
        return None
    except NoCredentialsError:
        print("AWS credentials not available.")
        return None
    except ClientError as e:
        print(f"Failed to upload {path_to_file} to {bucket_name}/{encoded_object_name}: {e}")
        return None
    #
    # # Construct the public URL
    # region = s3_client.get_bucket_location(Bucket=bucket_name)['LocationConstraint']
    # if region is None:
    #     region = 'us-east-1'
    #
    # if region == 'us-east-1':
    #     url = f"https://{bucket_name}.s3.amazonaws.com/{encoded_object_name}"
    # else:
    #     url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{encoded_object_name}"

    print(presigned_url)
    return presigned_url



def get_in_bucket(file_name, bucket_name=DEFAULT_BUCKET_NAME):
    """Retrieve an object from an S3 bucket

    :param file_name: S3 object name
    :param bucket_name: Bucket name
    :return: Object content as bytes or None if failed
    """
    # Initialize S3 client using environment variables
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        return response['Body'].read()
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            print(f"The object {file_name} does not exist in bucket {bucket_name}.")
        else:
            print(f"Failed to retrieve {file_name} from {bucket_name}: {e}")
        return None


def file_exists_in_bucket(filename) -> str:
    s3_client = boto3.client('s3')

    try:
        # Attempt to retrieve the metadata of the object
        s3_client.head_object(Bucket=DEFAULT_BUCKET_NAME, Key=filename)
        return f"https://{DEFAULT_BUCKET_NAME}.s3.us-east-2.amazonaws.com/{filename}"
    except:
        return ""

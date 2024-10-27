import os
import time

import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.errors
import googleapiclient.discovery
import googleapiclient.http

class YouTubeUploader:
    def __init__(self, client_secrets_file, scopes=None):
        if scopes is None:
            scopes = ["https://www.googleapis.com/auth/youtube.upload"]
        self.client_secrets_file = client_secrets_file
        self.scopes = scopes
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.credentials = None
        self.youtube = None

    def authenticate(self):
        # Get credentials and create an API client
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            self.client_secrets_file, self.scopes)
        self.credentials = flow.run_local_server(port=0)
        self.youtube = googleapiclient.discovery.build(
            self.api_service_name, self.api_version, credentials=self.credentials)

    def upload_video(self, file_path, title, description, category_id="22", tags=None, privacy_status="public"):
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
            }
        }

        # Call the API's videos.insert method to create and upload the video.
        media_body = googleapiclient.http.MediaFileUpload(file_path, chunksize=-1, resumable=True)

        request = self.youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media_body
        )

        response = None
        error = None
        retry = 0
        max_retries = 10
        while response is None:
            try:
                print("Uploading video...")
                status, response = request.next_chunk()
                if response is not None:
                    print(f"Video uploaded successfully. Video ID: {response['id']}")
                    return response
            except googleapiclient.errors.HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
                else:
                    raise
            except Exception as e:
                error = f"An error occurred: {e}"

            if error:
                print(error)
                retry += 1
                if retry > max_retries:
                    print("No longer attempting to retry.")
                    break

                sleep_seconds = (2 ** retry)
                print(f"Sleeping {sleep_seconds} seconds and retrying...")
                time.sleep(sleep_seconds)
        return None

def main(path, problem):
    CLIENT_SECRETS_FILE = "/Users/shay/PycharmProjects/HelloWorld/Secrets/client_secrets.json"  # Path to your client_secret.json file
    uploader = YouTubeUploader(client_secrets_file=CLIENT_SECRETS_FILE)

    # Authenticate the user
    uploader.authenticate()

    # Video details
    file_path = path
    title = f'How to solve {problem}'
    description = f'A solution for how to solve the #math problem {problem}'
    tags = ["math", "science", "calculus"]  # Optional
    category_id = "22"  # See YouTube API documentation for category IDs
    privacy_status = "public"  # Options: "public", "private", "unlisted"

    # Upload the video and get the video URL
    video_url = uploader.upload_video(
        file_path=file_path,
        title=title,
        description=description,
        tags=tags,
        category_id=category_id,
        privacy_status=privacy_status
    )

    if video_url:
        print(f"Video URL: {video_url}")
        return video_url
    else:
        print("Failed to upload video.")

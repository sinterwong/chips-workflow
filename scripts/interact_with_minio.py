"""
This file is used to upload the images to minio.
"""

import minio
import os
import time
from imutils import paths


class MinioClient:
    def __init__(self, endpoint, access_key, secret_key, bucket_name):
        # Create client with access key and secret key
        self.client = minio.Minio(endpoint,
                                  access_key=access_key,
                                  secret_key=secret_key,
                                  secure=False)

        # print(self.client.list_buckets())
        # exit()
        # make a bucket with the make_bucket API call.
        self._create_bucket(bucket_name)

    def _create_bucket(self, bucket_name):
        if bucket_name in [bucket.name for bucket in self.client.list_buckets()]:
            print("The bucket already exists.")
            return
        try:
            self.client.make_bucket(bucket_name)
        except minio.error.MinioException as err:
            print(err)

    def upload(self, bucket_name, object_name, file_path):
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
        except minio.error.S3Error as err:
            print(err)


bucket_name = 'facerecognitionimagesbackup'

# Create client with access key and secret key
client = MinioClient('localhost:20503',
                     access_key='zkfd',
                     secret_key='ZKFD123.com', bucket_name=bucket_name)

folder = '/home/wangxt/workspace/projects/pyonnx-example/temp'


def task():
    # Check if there is a new file in the folder
    # If there is a new file, upload it to minio
    # If upload success, delete the local file
    # Check the folder whether there is new files
    images = paths.list_images(folder)

    for im in images:
        client.upload(bucket_name,
                      os.path.basename(im), im)
        # Delete the local file
        os.remove(im)


if __name__ == '__main__':
    while True:
        task()
        # Time interval is 5 seconds
        time.sleep(5)

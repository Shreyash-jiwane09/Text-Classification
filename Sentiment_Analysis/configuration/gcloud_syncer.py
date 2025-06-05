import os
import subprocess
import shlex
import platform

class GCloudSync:

    def sync_folder_to_gcloud(self, bucket_name, filepath, filename):
        full_path = os.path.abspath(os.path.join(filepath, filename))

        # Use double quotes for paths with spaces or Windows-style backslashes
        if platform.system() == "Windows":
            full_path = f'"{full_path}"'  # Wrap path in double quotes

        command = f"gsutil cp {full_path} gs://{bucket_name}/"
        os.system(command)

    def sync_folder_from_gcloud(self, bucket_name, folder_name, destination):
        destination_path = os.path.abspath(destination)
        if platform.system() == "Windows":
            destination_path = f'"{destination_path}"'

        command = f"gsutil cp gs://{bucket_name}/{folder_name} {destination_path}/{folder_name}"
        os.system(command)

import os

class GCloudSync:

    def sync_folder_to_gcloud(self, bucket_name, filepath, filename):
        command = f"gsutil cp {filepath}/{filename} gs://{bucket_name}/"
        os.system(command)

    def sync_folder_from_gcloud(self, bucket_name, folder_name, destination):
        command = f"gsutil cp gs://{bucket_name}/{folder_name} {destination}/{folder_name}"
        os.system(command)

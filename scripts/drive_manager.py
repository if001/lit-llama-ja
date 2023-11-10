import io
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import service_account


class DriveManager():
    def __init__(self, cred_file_path, verbose = False):
        self._service = None
        self.connect(cred_file_path)
        self._verbose = verbose

    def connect(self, cred_file_path):
        # SCOPES = ['https://www.googleapis.com/auth/drive.file']
        SCOPES = ['https://www.googleapis.com/auth/drive']

        sa_creds = service_account.Credentials.from_service_account_file(cred_file_path)
        scoped_creds = sa_creds.with_scopes(SCOPES)
        self._service = build('drive', 'v3', credentials=scoped_creds)

    def get_files(self):
        parent_id = '1pFMNjJKIdX2C3FfPv1XaW_JENNSnTfMu'
        parent_id = 'root'
        # folder_name = 'paperspace'
        folder_name = 'root'
        query = f"mimeType='application/vnd.google-apps.folder' and  name='{folder_name}' and '{parent_id}' in parents and trashed=false"
        query = f"mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
        query = f"parents in '{parent_id}' and trashed = false"
        response = self._service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        # response = self._service.files().list(upportsAllDrives=True, includeItemsFromAllDrives=True, q=query, fields="nextPageToken, files(id, name)").execute()
    
        files = response.get('files', [])
        for v in files:
            print(v)

    def share(self, file_id):
        # file_id = '1wCX9HibsoZR29L2387APbkXJkZdjc0ss'
        response = self._service.files().get(fileId=file_id, fields='permissions').execute()
        # print('p', response['permissions'])
        body = {
            'type': 'user',
            'role': 'writer',
            'emailAddress': 'otomijuf.003@gmail.com', 
        }
        response = self._service.permissions().create(fileId=file_id,
                                                       body=body,                                                       
                                                       ).execute()        
        print('add role', response)

    def get_folder_id(self, folder_path):
        folder_names = folder_path.strip('/').split('/')
        parent_id = 'root'
        for folder_name in folder_names:
            query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and '{parent_id}' in parents and trashed=false"
            response = self._service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            files = response.get('files', [])
            if not files:
                # print('No folder found with the name:', folder_name)
                break
            elif len(files) > 1:
                 #print('Multiple folders found with the name:', folder_name)
                break
            else:
                folder = files[0]
                # print('Found folder:', folder_name, 'ID:', folder.get('id'))
                parent_id = folder.get('id')  # Set the parent_id for the next iteration
        
        if parent_id != 'root' and files and len(files) == 1:
            return parent_id                    
        print('Could not find a unique path to folder')
        return ''

    def upload(self, file_path, folder_id = None):
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        file_metadata['parents'] = folder_id        
        media = MediaFileUpload(file_path, mimetype='text/plain')
        file = self._service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = file.get('id')
        print('upload success: %s' % file.get('id'))
        self.share(file_id)

    def download_from_dir(self, save_path, storage_dir_id):
        results = self._service.files().list(
            q=f"'{storage_dir_id}' in parents",
            pageSize=100,
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])

        for item in items:
            file_id = item['id']
            file_name = item['name']
            request = self._service.files().get_media(fileId=file_id)
            file_path = os.path.join(save_path, file_name)
            
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Download {file_name} ({int(status.progress() * 100)}%)")    

    def download(self, file_id, save_path):
        request = self._service.files().get_media(fileId=file_id)
                
        with io.FileIO(save_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {save_path} ({int(status.progress() * 100)}%)")

    def remove(self, file_id):
        response = self._service.files().delete(fileId=file_id).execute()
        print('delete...', response)

def main(    
    cred_pth: str = "../../../google_drive_token/peparspace_gdrive_credential.json",
    base_folder_id: str = "1pFMNjJKIdX2C3FfPv1XaW_JENNSnTfMu",    
    upload_file: str = "",
    download_file_id: str = "",
    save_file_path: str = "",
    remove_file_id: str = "",
    get_files: bool = False
    ):
    mng = DriveManager(cred_pth, verbose=True)
    
    if(get_files):
        mng.get_files()

    if (upload_file != ""):
        mng.upload(upload_file, base_folder_id)
    if (download_file_id != ""):
        mng.download(download_file_id, save_file_path)
    if (remove_file_id != ""):
        mng.remove(remove_file_id)
        mng.get_files()


if __name__ == '__main__':
    from jsonargparse.cli import CLI
    CLI(main)
import requests
import os
import zipfile
import pandas as pd

def check_dataset():
    url = "https://springernature.figshare.com/ndownloader/files/10797959"
    print(f"Downloading {url}...")
    
    response = requests.get(url, allow_redirects=True)
    
    # Save it temporarily
    file_name = "iiitd_dataset.zip" # Figshare often zips them, or it might be a CSV
    
    # Check headers for content type
    content_disp = response.headers.get('content-disposition', '')
    if 'filename=' in content_disp:
        file_name = content_disp.split('filename=')[1].strip('"')
    print(f"Saving as {file_name}")
    
    with open(file_name, 'wb') as f:
        f.write(response.content)
        
    print(f"Downloaded size: {os.path.getsize(file_name) / (1024*1024):.2f} MB")
    
    if file_name.endswith('.zip'):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall("iiitd_data")
            print("Extracted files:", zip_ref.namelist())
    elif file_name.endswith(('.csv', '.txt')):
        df = pd.read_csv(file_name, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())
    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_name, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())

if __name__ == "__main__":
    check_dataset()

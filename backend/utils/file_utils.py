import os
import shutil

UPLOAD_DIR="uploads"

os.makedirs(UPLOAD_DIR,exist_ok=True)

def save_file(upload_file):
    path=os.path.join(UPLOAD_DIR, upload_file.filename)

    with open(path,"wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return path
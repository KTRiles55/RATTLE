import os
from werkzeug.utils import secure_filename

def fetch_file(file_capture, upload_folder):
    file_path = ""

    # Add or overwrite file in project file directory
    if file_capture:
        filename = secure_filename(file_capture.filename)
        file_path = os.path.join(upload_folder, filename)
        file_capture.save(file_path)
        return file_path

    else:
        return None

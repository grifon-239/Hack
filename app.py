from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_dropzone import Dropzone
import os, shutil
import time


app = Flask(__name__)
dropzone = Dropzone(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAPS_FOLDER'] = 'maps'
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_MAX_FILES'] = 100
app.config['DROPZONE_PARALLEL_UPLOADS'] = 100

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    maps = os.listdir(app.config['MAPS_FOLDER'])
    return render_template('index.html', files=files, maps=maps)



@app.route('/upload', methods=['POST'])
def upload():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except:
            pass

    for key, file in request.files.items():
        if key.startswith('file'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    return '', 204


@app.route('/get_file_path', methods=['POST'])
def get_file_path():
    selected_file = request.form.get('selected_file')
    map_path = os.path.join(app.config['MAPS_FOLDER'], selected_file)
    return f"Path to the selected map: {map_path}"


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import trainedmodel as tm

UPLOAD_FOLDER = 'C:/Users/earth/Downloads/Workspace/CVFlask/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# home page
@app.route('/')
def upload_files():
   return render_template('upload.html')

# upload file
@app.route('/preview', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      return render_template('preview.html', id=f.filename)

# view model predictions on file
@app.route('/predict', methods = ['GET', 'POST'])
def preview_file():
    test = Path('./uploads')
    image = tm.loadimage(test)
    output = tm.predictimage(image)
    return render_template('predict.html', output=output)
    

if __name__ == '__main__':
   app.run(debug = True)




















# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# app = Flask(__name__)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('download_file', name=filename))
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''


# # download file 
# from flask import send_from_directory

# @app.route('/uploads/<name>')
# def download_file(name):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], name)
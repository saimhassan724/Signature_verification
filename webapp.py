from flask import Flask, render_template, flash, redirect, request
from flask import send_from_directory
import tensorflow as tf
import numpy as np

from keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='template')
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

UPLOAD_FOLDER = '.\static\image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)


def Predict(dirpath=".\static\image", valid_extensions=('jpg', 'jpeg', 'png')):
    """
    Get the latest image file in the given directory
    """

    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
                   f.rsplit('.', 1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime)


first_image = Predict()
print("First Image: ", first_image)


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input()

    return render_template("upload.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    saved_model = load_model('static/model/vgg16_1.h5')
    path = Predict()
    img1 = tf.keras.utils.load_img(path, target_size=(224, 224))
    img1 = np.asarray(img1)

    img1 = np.expand_dims(img1, axis=0)
    output1 = saved_model.predict(img1)
    print(output1)
    if output1[0][0] == 1:
        output = "Forged Signature"
    else:
        output = "Original Signature"
    print(output)
    return render_template('upload.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)

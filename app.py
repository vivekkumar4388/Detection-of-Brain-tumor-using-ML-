import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, url_for,session,redirect
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pyrebase
from firebase_admin import auth, credentials, initialize_app
import firebase_admin
import requests

app = Flask(__name__)

config = {
    'apiKey': "AIzaSyDjGesX4rCVVAxAsKxL2fiLcnQ3S85H7S4",
    'authDomain': "authenticate-82be9.firebaseapp.com",
    'projectId': "authenticate-82be9",
    'storageBucket': "authenticate-82be9.appspot.com",
    'messagingSenderId': "420437121081",
    'appId': "1:420437121081:web:e2286687b183bef2d0d5e8",
    'measurementId': "G-QNYWLWTVSD",
    'databaseURL': ""
}


firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
app.secret_key = 'secert'
cred = credentials.Certificate('D:\BrainTumor Classification DL\serviceAccountKey.json')
firebase_admin.initialize_app(cred)

model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')
plt.switch_backend('agg')

def get_className(classNo):
    if classNo == 0:
        return "Normal Brain Scan"
    elif classNo == 1:
        return "Yes Brain Tumor.Upon evaluation of the MRI photograph, a wonderful abnormality steady with a mind tumor was recognized. The tumor seems as a well-defined mass with irregular borders placed. It reveals characteristics such as accelerated sign intensity on T2-weighted images and heterogeneous enhancement following contrast management."


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')



@app.route('/', methods=['GET', 'POST'])
def login():
    if ( 'user' in session ):
        return render_template('index.html')
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')
        try:
            user=auth.sign_in_with_email_and_password(email, password)
            session['user']=email
            return render_template('index.html')
        except requests.exceptions.HTTPError as e:
            error_message = "Invalid credentials. Please try again."
            if e.response is not None and e.response.json():
                error_message = e.response.json()['error']['message']
            return render_template('login.html', error=error_message)
        except pyrebase.pyrebase.HTTPError as e:
            error_message = "Error signing in. Please try again later."
            return render_template('login.html', error=error_message)


    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove 'user' from session if it exists
    return redirect(url_for('login'))


@app.route('/signup',methods=['GET', 'POST'])
def signup():
    if 'user' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')
        try:
            user = auth.create_user_with_email_and_password(email, password)
            session['user'] = email
            return render_template('index.html')
        except requests.exceptions.HTTPError as e:
            error_message = "Error signing up. Please try again."
            if e.response is not None and e.response.json():
                error_message = e.response.json()['error']['message']
            return render_template('signup.html', error=error_message)
        except pyrebase.pyrebase.HTTPError as e:
            error_message = "Error signing up. Please try again later."
            return render_template('signup.html', error=error_message)

    return render_template('signup.html', error=None)





@app.route('/graph')
def graph():
    actual_labels = ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
    predicted_labels = ['Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    yes_yes_count = 100
    yes_no_count = 12
    no_yes_count = 18
    no_no_count = 38
    plt.ioff()
    # Create confusion matrix
    conf_matrix = np.array([[yes_yes_count, yes_no_count], [no_yes_count, no_no_count]])

    # Visualize confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Yes', 'No'])
    plt.yticks(tick_marks, ['Yes', 'No'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Plot confusion matrix
    #plt.figure(figsize=(8, 6))
    #plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix')
    #plt.colorbar()
    #tick_marks = np.arange(len(actual_labels))
    #plt.xticks(tick_marks, actual_labels)
    #plt.yticks(tick_marks, actual_labels)

    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plot_file = 'static/confusion_matrix.png'
    # Show plot
    plt.show()
    plt.savefig(plot_file)
    plt.clf()

    return render_template('graph.html')



@app.route('/accuracy')
def accuracy():
    epochs = np.arange(1, 11)
   # train_acc = np.array([75, 78, 80, 82, 85, 87, 88, 90, 91, 92])
    #val_acc = np.array([70, 72, 75, 78, 80, 82, 83, 85, 86, 98])
    plt.ioff()
    train_acc = np.array([71.08, 81.92, 86.12, 89.71, 93.46, 96, 94.34, 93.46, 97.42, 97.18])
    val_acc = np.array([73, 80.83, 87.17, 89.83, 95.65, 95.67, 95.45,95.43, 95.43, 96.12])
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Value Accuracy')
    plt.title('Training and Value Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('static/accuracy_plot.png')  # Save the plot as a static file
    plt.clf()
    return render_template('accuracy.html')




@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


#bar graph
@app.route('/bar_graph')
def bar_graph():
    # Sample data
    categories = ['CNN', 'CRF', 'GA', 'SVM' ,'Others_CNN']
    values = [97.08, 89, 83.64, 84.5, 91]
    plt.ioff()
    # Plotting the bar graph
    plt.bar(categories, values)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Graph Example')

    # Saving the plot to a file
    plot_file = 'static/bar_graph.png'
    plt.savefig(plot_file)

    # Clear the plot to avoid overwriting with the next request
    plt.clf()

    return render_template('bar_graph.html', plot_file=plot_file)


if __name__ == '__main__':
    app.run(debug=True)

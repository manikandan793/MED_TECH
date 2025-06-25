from flask import Flask, render_template, request, send_file
import os
import keras
from PIL import Image
import numpy as np
import cv2
from PyPDF2 import PdfReader
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader
from fpdf import FPDF

# Initialize Flask app
app = Flask(__name__)

# Define function to return the class of the ear condition
def return_class(img_path):
    try:
        # Load the pre-trained model for ear conditions
        model = keras.models.load_model('main_model.h5')
        print(f"Image path: {img_path}")

        # Open and preprocess the image
        test_img = Image.open(img_path)
        test_img_array = np.asarray(test_img)
        test_img_resize = cv2.resize(test_img_array, (256, 256))
        test_img_reshape = np.reshape(test_img_resize, (1, 256, 256, 3))

        # Predict the class
        predictions = model.predict(test_img_reshape)
        classes = ["Chronic otitis media", "Ear wax plug", "Myringlorisis", "Normal ear drum"]
        predicted_class = classes[np.argmax(predictions[0])]

        return predicted_class
    except Exception as e:
        print(f"Error in return_class: {e}")
        return "Error in prediction"


def return_eye_class(img_path):
    try:
        # Load the pre-trained model for eye diseases
        eye_model = keras.models.load_model('model_eye.h5')
        print(f"Eye image path: {img_path}")

        # Open and preprocess the image
        eye_img = Image.open(img_path)
        eye_img_array = np.asarray(eye_img)
        print(f"Original image shape: {eye_img_array.shape}")
        eye_img_resize = cv2.resize(eye_img_array, (224, 224))
        print(f"Resized image shape: {eye_img_resize.shape}")
        eye_img_reshape = np.reshape(eye_img_resize, (1, 224, 224, 3))
        print(f"Reshaped image shape: {eye_img_reshape.shape}")

        # Predict the class
        eye_predictions = eye_model.predict(eye_img_reshape)
        print(f"Predictions: {eye_predictions}")
        eye_classes = ["Cataract", "diabetic_retinopathy", "Glaucoma", "Normal eye"]
        eye_predicted_class = eye_classes[np.argmax(eye_predictions[0])]

        return eye_predicted_class
    except Exception as e:
        print(f"Error in return_eye_class: {e}")
        return f"Error in prediction: {e}"



@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Ensure 'uploads' and 'summaries' directories exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        if not os.path.exists('summaries'):
            os.makedirs('summaries')

        if 'pdf' not in request.files:
            return 'No file uploaded', 400
        
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return 'No selected file', 400

        # Save uploaded file
        save_path = os.path.join('uploads', pdf_file.filename)
        pdf_file.save(save_path)

        # Extract text from PDF
        report_text = extract_text_from_pdf(save_path)

        if report_text:
            # Summarize the report
            summary = summarize_report(report_text, max_length=500, min_length=200)

            # Save the summarized report as a PDF
            output_filename = os.path.join('summaries', f"summary_{pdf_file.filename}")
            save_summary_to_pdf(summary, output_filename)

            # Send the summarized PDF as a downloadable file
            return send_file(output_filename, as_attachment=True)
        else:
            return 'Could not extract text from the PDF', 400
    except Exception as e:
        return f"An error occurred: {e}", 500

# Helper functions
def summarize_report(report, max_length=500, min_length=200):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    input_text = "summarize: " + report
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def save_summary_to_pdf(summary, output_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'I', 12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(output_filename)

# Route to render the summarizer page (using sum.html)
@app.route('/summarizer')
def summarizer():
    return render_template('sum.html')

# Define the main route
@app.route('/')
def main():
    return render_template('main.html')

# Define the home route
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/eyeabout')
def eyeabout():
    return render_template('eyeabout.html')

# Define the about route
@app.route('/about')
def about():
    return render_template('about.html')

# Define the detection page route
@app.route('/dect')
def dect():
    return render_template('dect.html')

@app.route('/dectx')
def dectx():
    return render_template('dectx.html')

@app.route('/xrayabout')
def xrayabout():
    return render_template('xrayabout.html')

@app.route('/lungabout')
def lungabout():
    return render_template('lungabout.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/indexeye')
def indexeye():
    return render_template('indexeye.html')

# Define the result route for ear conditions
@app.route('/result', methods=['POST'])
def result():
    try:
        if request.method == 'POST':
            input_image = request.files['input_image']
            print(f"Uploaded file: {input_image.filename}")
            save_path = os.path.join('static', input_image.filename)
            input_image.save(save_path)
            output = return_class(save_path)
            return render_template("index.html", img_path=input_image.filename, output=output)
    except Exception as e:
        print(f"Error in result function: {e}")
        return "An error occurred during the prediction process."

# Define the result route for eye diseases
@app.route('/result_eye', methods=['POST'])
def result_eye():
    try:
        if request.method == 'POST':
            input_image = request.files['input_image']
            print(f"Uploaded file: {input_image.filename}")
            save_path = os.path.join('static', input_image.filename)
            input_image.save(save_path)
            output = return_eye_class(save_path)
            return render_template("indexeye.html", img_path=input_image.filename, output=output)
    except Exception as e:
        print(f"Error in result_eye function: {e}")
        return f"An error occurred during the prediction process: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

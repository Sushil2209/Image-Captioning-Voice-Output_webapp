from flask import Flask, render_template, request, redirect
import os
from PIL import Image
from dotenv import load_dotenv
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# Load Hugging Face token from .env
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", token=token)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", token=token)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize Flask app
app = Flask(__name__)

# Folder configs
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audios'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# 🏠 Homepage
@app.route('/')
def index():
    return render_template('index.html')

# 📤 Upload handler
@app.route('/upload', methods=['POST'])
def upload():
    image = request.files.get('image')
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Process image
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Language selection (default to English)
        language = request.form.get('language', 'en')

        # Convert caption to audio
        tts = gTTS(text=caption, lang=language)
        filename = image.filename.rsplit('.', 1)[0] + ".mp3"
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        tts.save(audio_path)

        return render_template('index.html',
                               caption=caption,
                               image_filename=image.filename,
                               audio_filename=filename)

    return 'No image uploaded'

# ▶️ Run app
if __name__ == '__main__':
    app.run(debug=True)

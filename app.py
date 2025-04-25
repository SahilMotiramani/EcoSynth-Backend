from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import uuid
# import pyttsx3  # Removed: not compatible with Render
import edge_tts  # Free Microsoft Edge TTS
import asyncio
from TTS.api import TTS
import torch

from translator import MultiLanguageTranslator
from werkzeug.utils import secure_filename
from denoise import remove_noise
import time
import requests
import numpy as np
import cv2
import tempfile
import subprocess
from transformers import pipeline
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
import logging
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Configure folders
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("temp_files", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)

try:
    script_generator = pipeline("text-generation", model="gpt-4o-2024-08-06")
    logger.info("Script generator model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load script generator model: {e}")
    script_generator = None

# Mock lip sync function (in production, replace with actual lip sync model)
def generate_lip_sync_video(script, audio_path, output_path):
    """
    Generate a lip-synced video using the provided script and audio.
    This is a simplified mock implementation.
    In a real implementation, you would use a model like Wav2Lip.
    """
    logger.info("Starting lip sync generation")
    
    try:
        # Create video variables
        height, width = 480, 640
        fps = 30
        duration = 10  # seconds

        # Create a temporary video file
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        # Create a video with text (simulating the lip-synced output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        words = script.split()
        total_frames = fps * duration
        words_per_frame = max(1, len(words) // total_frames)

        for i in range(total_frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            start_idx = min(i * words_per_frame, len(words) - 1)
            end_idx = min(start_idx + 10, len(words))
            text = ' '.join(words[start_idx:end_idx])

            cv2.putText(img, text, (50, height // 2), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (width // 2, height // 4), 60, (0, 0, 255), -1)
            cv2.circle(img, (width // 2 - 20, height // 4 - 15), 10, (255, 255, 255), -1)
            cv2.circle(img, (width // 2 + 20, height // 4 - 15), 10, (255, 255, 255), -1)
            mouth_height = 10 + (5 * (i % 10)) // 10
            cv2.ellipse(img, (width // 2, height // 4 + 20), (30, mouth_height), 0, 0, 180, (255, 255, 255), -1)

            out.write(img)

        out.release()

        # Use context managers to safely open and close video/audio
        with VideoFileClip(temp_video) as video_clip, AudioFileClip(audio_path) as audio_clip:
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration

            if video_duration > audio_duration:
                video_clip = video_clip.subclip(0, audio_duration)

            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Safe to delete now
        os.remove(temp_video)

        logger.info(f"Lip sync video generated successfully at {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error in lip sync generation: {e}")
        return False

API_KEY = "AIzaSyBZ8y3HtDiPYDDM8MDznGgnF7-6LZWk0zk"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

AUDIO_OUTPUT_FOLDER = 'static/audio_output'
PROCESSED_FOLDER = 'processed'
UPLOAD_FOLDER = 'uploads'
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['AUDIO_OUTPUT_FOLDER'] = AUDIO_OUTPUT_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize TTS model for voice cloning (load only once)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tts_clone = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts").to(device)
except Exception as e:
    print(f"Warning: Could not load voice cloning model. Error: {e}")
    tts_clone = None

# Initialize translator
translator = MultiLanguageTranslator()
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

# Character Voice Database
CHARACTER_VOICES = {
    "child": {
        "edge_voice": "en-US-AnaNeural",
        "description": "Happy Child (6-8 years)",
    },
    "teen_boy": {
        "edge_voice": "en-US-GuyNeural",
        "description": "Teenage Boy (14-16)",
    },
    "young_woman": {
        "edge_voice": "en-US-AriaNeural",
        "description": "Young Woman (20s)",
    },
    "businessman": {
        "edge_voice": "en-US-DavisNeural",
        "description": "Professional Businessman",
    },
    "grandma": {
        "edge_voice": "en-US-JennyNeural",
        "description": "Kind Grandmother",
    },
    "grandpa": {
        "edge_voice": "en-US-JasonNeural",
        "description": "Wise Grandfather",
    },
    "robot": {
        "edge_voice": "en-US-TonyNeural",
        "description": "Futuristic Robot",
    },
    "storyteller": {
        "edge_voice": "en-US-BrianNeural",
        "description": "Dramatic Storyteller",
    },
    "announcer": {
        "edge_voice": "en-US-EricNeural",
        "description": "Sports Announcer",
    },
    "whisper": {
        "edge_voice": "en-US-JaneNeural",
        "description": "Mysterious Whisper",
    }
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

class FreeTTS:
    @staticmethod
    def gtts_convert(text, output_file, character=None):
        """Google TTS fallback when pyttsx3 is not available"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_file)
            return True
        except Exception as e:
            print(f"gTTS error: {e}")
            return False

    @staticmethod
    async def edge_tts_convert(text, output_file, character):
        """Online TTS with Edge's neural voices"""
        try:
            voice = CHARACTER_VOICES[character]["edge_voice"]
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            return True
        except Exception as e:
            print(f"EdgeTTS error: {e}")
            return False
        
@app.route('/api/generate-script', methods=['POST'])
def generate_script():
    try:
        data = request.json
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Use Gemini AI to generate text
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')  # Gemini Pro model
        response = model.generate_content(prompt)

        script = response.text if response and response.text else "Failed to generate script."

        return jsonify({'script': script})
    except Exception as e:
        logger.error(f"Error generating script: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-speech1', methods=['POST'])
def text_to_speech1():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate a unique ID for this file
        file_id = str(uuid.uuid4())
        audio_path = f"temp_files/{file_id}.mp3"
        
        # Generate speech from text
        tts = gTTS(text=text, lang='en')
        tts.save(audio_path)
        
        return jsonify({'audio_path': audio_path, 'file_id': file_id})
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    try:
        data = request.json
        script = data.get('script', '')
        audio_path = data.get('audio_path', '')
        
        if not script or not audio_path:
            return jsonify({'error': 'Script and audio path are required'}), 400
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        # Generate a unique ID for the output video
        video_id = str(uuid.uuid4())
        output_path = f"output_videos/{video_id}.mp4"
        
        # Generate lip sync video
        success = generate_lip_sync_video(script, audio_path, output_path)
        
        if not success:
            return jsonify({'error': 'Failed to generate video'}), 500
        
        return jsonify({'video_path': output_path, 'video_id': video_id})
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-audio/<file_id>', methods=['GET'])
def get_audio(file_id):
    try:
        audio_path = f"temp_files/{file_id}.mp3"
        
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        return send_file(audio_path, mimetype='audio/mpeg')
    except Exception as e:
        logger.error(f"Error retrieving audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-video/<video_id>', methods=['GET'])
def get_video(video_id):
    try:
        video_path = f"output_videos/{video_id}.mp4"
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        logger.error(f"Error retrieving video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/text-to-speech', methods=['POST'])
async def text_to_speech():
    data = request.json
    text = data.get('text')
    character = data.get('character', 'young_woman')
    engine = data.get('engine', 'edge')  # 'edge' or 'gtts' (replacing pyttsx3)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        output_filename = f"tts_{uuid.uuid4()}.mp3"
        output_path = os.path.join(app.config['AUDIO_OUTPUT_FOLDER'], output_filename)
        
        if engine == 'gtts':
            success = FreeTTS.gtts_convert(text, output_path, character)
        else:
            success = await FreeTTS.edge_tts_convert(text, output_path, character)
        
        if not success:
            return jsonify({'error': 'TTS conversion failed'}), 500
            
        return jsonify({
            'audio_url': f"/audio_output/{output_filename}",
            'message': 'Text converted to speech successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clone-voice', methods=['POST'])
async def clone_voice():
    if 'file' not in request.files or not request.files['file']:
        return jsonify({'error': 'No audio file provided'}), 400
    
    text = request.form.get('text', 'Hello, this is my cloned voice')
    name = request.form.get('name', 'clone')
    
    if not tts_clone:
        return jsonify({'error': 'Voice cloning is not available on this server'}), 501
    
    try:
        # Save uploaded voice sample
        file = request.files['file']
        filename = f"sample_{uuid.uuid4()}.wav"
        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(sample_path)
        
        # Generate output filename
        output_filename = f"clone_{name}_{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['AUDIO_OUTPUT_FOLDER'], output_filename)
        
        # Perform voice cloning
        tts_clone.tts_to_file(
            text=text,
            speaker_wav=sample_path,
            language="en",
            file_path=output_path
        )
        
        return jsonify({
            'original_sample': f"/uploads/{filename}",
            'cloned_audio': f"/audio_output/{output_filename}",
            'message': 'Voice cloned successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process audio
        output_path = os.path.join(PROCESSED_FOLDER, "cleaned_" + filename)
        remove_noise(filepath, output_path)

        return jsonify({"processed_file": f"/download/cleaned_{filename}"})

    return jsonify({"error": "Invalid file format"}), 400

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

@app.route('/translate', methods=['POST'])
def translate_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Get parameters
        source_lang = request.form.get('source_lang', 'en')
        target_lang = request.form.get('target_lang', 'hi')
        voice_type = int(request.form.get('voice_type', 1))
        
        # Save file
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate audio output filename
        output_filename = f"translated_{uuid.uuid4()}.mp3"
        output_path = os.path.join(app.config['AUDIO_OUTPUT_FOLDER'], output_filename)
        
        # Add validation
        if not translator:
            return jsonify({'error': 'Translator not initialized'}), 500
            
        # Debug logging
        logger.info(f"Starting translation: {source_lang}->{target_lang}")
        
        result = translator.full_translation_pipeline(
            filepath,
            source_lang,
            target_lang,
            voice_option=voice_type,
            output_audio_path=output_path
        )
        
        if not result or 'error' in result:
            error_msg = result.get('error', 'Unknown translation error')
            logger.error(f"Translation failed: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        if 'audio_url' not in result:
            result['audio_url'] = f"/audio_output/{output_filename}"
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Translation failed: {str(e)}"}), 500
    
@app.route('/character-voices', methods=['GET'])
def get_character_voices():
    """Return available character voices"""
    simplified_voices = {k: v['description'] for k, v in CHARACTER_VOICES.items()}
    return jsonify(simplified_voices)

@app.route('/languages', methods=['GET'])
def get_languages():
    """Get list of supported languages"""
    languages = translator.get_supported_languages()
    return jsonify(languages)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/audio_output/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_OUTPUT_FOLDER'], filename)

# Add a root route to indicate server is running
@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'message': 'AI Voice Assistant API is running',
        'endpoints': {
            'health_check': '/api/health',
            'text_to_speech': '/text-to-speech',
            'generate_script': '/api/generate-script',
            'character_voices': '/character-voices',
            'languages': '/languages'
        }
    })

if __name__ == '__main__':
    # Use environment variable PORT for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

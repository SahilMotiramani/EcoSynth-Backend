import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import pygame
import time
import pyttsx3
import platform

class MultiLanguageTranslator:
    # Supported languages with their codes and display names
    SUPPORTED_LANGUAGES = {
        'hi': 'Hindi',
        'mr': 'Marathi',
        'pa': 'Punjabi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'bn': 'Bengali',
        'gu': 'Gujarati',
        'en': 'English',
        'zh-cn': 'Chinese (Simplified)',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'ru': 'Russian',
        'ar': 'Arabic'
    }
    
    # Voice types for TTS
    VOICE_TYPES = {
        1: "Adult Male",
        2: "Adult Female",
        3: "Elderly Voice (Slower)",
        4: "Child Voice (Faster)", 
        5: "Default Voice"
    }
    
    # Indian languages
    INDIAN_LANGUAGES = ['hi', 'mr', 'pa', 'ta', 'te', 'kn', 'ml', 'bn', 'gu']

    def __init__(self):
        pygame.mixer.init()
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.temp_file = "temp_translation.mp3"
        
        # Initialize pyttsx3 engine for voice options
        try:
            self.engine = pyttsx3.init()
            self.system_type = platform.system()  # 'Windows', 'Darwin' (Mac), 'Linux'
            print(f"System detected: {self.system_type}")
            
            # Check available voices
            voices = self.engine.getProperty('voices')
            self.available_voices = len(voices)
            print(f"Available system voices: {self.available_voices}")
            
            if self.available_voices > 0:
                for i, voice in enumerate(voices):
                    print(f"Voice {i+1}: {voice.name} ({voice.id})")
        except Exception as e:
            print(f"Warning: pyttsx3 initialization issue: {e}")
            print("Some voice features may be limited.")
            self.engine = None
            self.system_type = platform.system()
            self.available_voices = 0
    
    def get_supported_languages(self):
        """Return supported languages in a format suitable for API responses"""
        return [{'code': code, 'name': name} for code, name in self.SUPPORTED_LANGUAGES.items()]
    
    def audio_to_text(self, audio_path, source_lang='en'):
        """Convert audio file to text in the specified language"""
        try:
            with sr.AudioFile(audio_path) as source:
                print(f"Processing {self.SUPPORTED_LANGUAGES.get(source_lang, source_lang)} audio file...")
                audio = self.recognizer.record(source)
                
                try:
                    # Use the appropriate language for recognition
                    recognized_text = self.recognizer.recognize_google(audio, language=source_lang)
                    print(f"Recognized {self.SUPPORTED_LANGUAGES.get(source_lang, source_lang)} text: {recognized_text}")
                    return recognized_text
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    return None
        except Exception as e:
            print(f"Audio file error: {e}")
            return None
    
    def translate_text(self, text, source_lang, target_lang):
        """More robust translation with error handling"""
        try:
            # Ensure text is valid
            if not text or not isinstance(text, str):
                print("Invalid text for translation")
                return None
            
            # Handle language codes
            source_lang = source_lang if source_lang in self.SUPPORTED_LANGUAGES else 'auto'
            target_lang = target_lang if target_lang in self.SUPPORTED_LANGUAGES else 'en'
        
        # Attempt translation with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    translation = self.translator.translate(text, src=source_lang, dest=target_lang)
                    if hasattr(translation, 'text'):
                        print(f"Translation successful: {translation.text}")
                        return translation.text
                    time.sleep(1)  # Wait before retry
                except AttributeError as e:
                    print(f"Translation parse error (attempt {attempt+1}): {e}")
                    time.sleep(2)
                except Exception as e:
                    print(f"Translation error (attempt {attempt+1}): {e}")
                    time.sleep(1)

            print("Max translation attempts reached")
            return None
        
        except Exception as e:
            print(f"Unexpected translation error: {e}")
            return None
    def speak_indian_language(self, text, lang_code, voice_option, output_path="static/audio-output"):
        """Special handling for Indian languages"""
        try:
            # Create standard gTTS object
            tts = gTTS(text=text, lang=lang_code)
            
            # Save to either specified output path or temp file
            save_path = output_path if output_path else self.temp_file
            tts.save(save_path)
            
            # Apply voice option modifications
            if voice_option == 2:  # Slower speech
                print("Using slower speech rate")
                pygame.mixer.music.load(save_path)
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play()
                
                # Slow down playback using tick rate
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(8)  # Lower value = slower playback
                        
            elif voice_option == 3:  # Faster speech
                print("Using faster speech rate")
                pygame.mixer.music.load(save_path)
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play()
                
                # Standard playback but with faster tick rate
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(15)  # Higher value = faster playback
                    
            else:  # Default speech
                pygame.mixer.music.load(save_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
            time.sleep(0.5)
            if not output_path and os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                
        except Exception as e:
            print(f"Indian language speech synthesis error: {e}")
            return False
        return True
    
    def speak_with_pyttsx3(self, text, voice_option):
        """Speak using pyttsx3 with appropriate voice settings"""
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')
            current_rate = self.engine.getProperty('rate')
            
            # Handle voice selection based on system and options
            if self.system_type == 'Windows' and len(voices) >= 2:
                if voice_option == 1:  # Voice 1 (typically male)
                    self.engine.setProperty('voice', voices[0].id)
                    self.engine.setProperty('rate', current_rate)  # Default rate
                elif voice_option == 2 and len(voices) >= 2:  # Voice 2 (typically female)
                    self.engine.setProperty('voice', voices[1].id)
                    self.engine.setProperty('rate', current_rate)  # Default rate
                elif voice_option == 3:  # Slower speech
                    # Keep current voice but slow down
                    self.engine.setProperty('rate', current_rate * 0.7)  # 70% of normal speed
                elif voice_option == 4:  # Faster speech
                    # Keep current voice but speed up
                    self.engine.setProperty('rate', current_rate * 1.3)  # 130% of normal speed
                else:  # Default speech rate
                    self.engine.setProperty('rate', current_rate)
            else:
                # For non-Windows or systems with only one voice
                if voice_option == 2:  # Slower speech
                    self.engine.setProperty('rate', current_rate * 0.7)
                elif voice_option == 3:  # Faster speech
                    self.engine.setProperty('rate', current_rate * 1.3)
                else:
                    self.engine.setProperty('rate', current_rate)  # Default rate
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
            # Reset rate to default after speaking
            self.engine.setProperty('rate', current_rate)
            
        except Exception as e:
            print(f"pyttsx3 speech error: {e}")
            return False
            
        return True
    
    def speak_translation_with_voice(self, text, lang_code, voice_option, output_path=None):
        """Speak translated text aloud with selected voice type and save to file if output_path specified"""
        # Special handling for Indian languages
        if lang_code in self.INDIAN_LANGUAGES:
            return self.speak_indian_language(text, lang_code, voice_option, output_path)
            
        # For languages well-supported by pyttsx3
        if lang_code in ['en', 'es', 'fr', 'de'] and self.engine:
            # If we need to save to file, use gTTS instead
            if output_path:
                try:
                    tts = gTTS(text=text, lang=lang_code)
                    tts.save(output_path)
                    return True
                except Exception as e:
                    print(f"gTTS save to file error: {e}")
                    return False
            # Otherwise use pyttsx3 for immediate playback
            else:
                return self.speak_with_pyttsx3(text, voice_option)
        
        # For other languages or if pyttsx3 fails, use gTTS
        try:
            tts = gTTS(text=text, lang=lang_code)
            
            # Save to file if output path specified
            if output_path:
                tts.save(output_path)
                return True
            # Otherwise play immediately
            else:
                save_path = self.temp_file
                tts.save(save_path)
                
                # Apply voice option modifications using pygame playback rates
                if voice_option in [3, 4]:  # Slower speech for Elderly option
                    print("Using slower speech rate")
                    pygame.mixer.music.load(save_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(8)  # Slower playback
                elif voice_option == 5:  # Faster speech for Child option
                    print("Using faster speech rate")
                    pygame.mixer.music.load(save_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(15)  # Faster playback
                else:  # Default speech rate
                    pygame.mixer.music.load(save_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                
                time.sleep(0.5)
                if os.path.exists(save_path):
                    os.remove(save_path)
                return True
                
        except Exception as e:
            print(f"Speech synthesis error: {e}")
            return False
    
    def full_translation_pipeline(self, audio_path, source_lang, target_lang, voice_option=1, output_audio_path=None):
        """More robust pipeline with better error handling"""
        try:
            # Validate languages
            if source_lang not in self.SUPPORTED_LANGUAGES:
                return {'error': f'Unsupported source language: {source_lang}'}
            if target_lang not in self.SUPPORTED_LANGUAGES:
                return {'error': f'Unsupported target language: {target_lang}'}
        
        # Step 1: Audio to Text
            source_text = self.audio_to_text(audio_path, source_lang)
            if not source_text:
                return {'error': 'Speech recognition failed', 'step': 'audio_to_text'}

            # Step 2: Text Translation
            translated_text = self.translate_text(source_text, source_lang, target_lang)
            if not translated_text:
                return {'error': 'Text translation failed', 'step': 'translation'}
        
        # Step 3: Speech Synthesis
            output_path = output_audio_path or "temp_translation.mp3"
            success = self.speak_translation_with_voice(
                translated_text, 
                target_lang, 
                voice_option,
                output_path
            )
        
            if not success:
                return {'error': 'Speech synthesis failed', 'step': 'tts'}
        
            return {
                'success': True,
                'original_text': source_text,
                'translated_text': translated_text,
                'audio_path': output_path if output_audio_path else None,
                'languages': {
                    'source': self.SUPPORTED_LANGUAGES[source_lang],
                    'target': self.SUPPORTED_LANGUAGES[target_lang]
                }
            }
        
        except Exception as e:
            return {'error': f'Pipeline error: {str(e)}', 'step': 'unknown'}
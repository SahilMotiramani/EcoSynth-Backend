from google.cloud import texttospeech
import os

class GoogleTTS:
    def __init__(self):
        # Initialize the client (will use GOOGLE_APPLICATION_CREDENTIALS from env)
        self.client = texttospeech.TextToSpeechClient()

    def convert_text_to_speech(self, text, language_code="en-US", voice_name=None, output_file="output.mp3"):
        """Convert text to speech using Google Cloud TTS API"""
        try:
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name if voice_name else None,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Write the response to the output file
            with open(output_file, "wb") as out:
                out.write(response.audio_content)
            
            return True

        except Exception as e:
            print(f"Error in Google TTS: {str(e)}")
            return False
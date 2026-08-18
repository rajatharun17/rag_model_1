import os
import requests
from typing import Dict, Any
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

class ElevenLabsSTTEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is missing from .env")
            
        # Initialize official client
        self.client = ElevenLabs(api_key=self.api_key)
    def transcribe(self, audio_file_path: str, language_code: str = "eng") -> dict:
        if not os.path.exists(audio_file_path):
            return {"transcript": "", "success": False, "error": "Audio file missing."}

        try:
            with open(audio_file_path, "rb") as audio_file:
                # Official Scribe v2 call
                transcription = self.client.speech_to_text.convert(
                    file=audio_file,
                    model_id="scribe_v2",
                    language_code=language_code
                )
                
            # Extract transcript text
            text = getattr(transcription, "text", str(transcription))
            return {"transcript": text, "success": True}

        except Exception as e:
            return {"transcript": "", "success": False, "error": str(e)}

    def _transcribe_via_rest(self, audio_file_path: str, language_code: str) -> Dict[str, Any]:
        """Direct REST POST request to ElevenLabs Scribe endpoint."""
        endpoint = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": self.api_key}

        try:
            with open(audio_file_path, "rb") as audio_file:
                files = {"file": (os.path.basename(audio_file_path), audio_file, "audio/wav")}
                data = {
                    "model_id": "scribe_v2",
                    "language_code": language_code
                }
                
                response = requests.post(endpoint, headers=headers, files=files, data=data)

            if response.status_code == 200:
                res_json = response.json()
                # Extract transcript text from response payload
                transcript = res_json.get("text", res_json.get("transcript", ""))
                return {"transcript": transcript, "success": True}
            else:
                return {
                    "transcript": "",
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {"transcript": "", "success": False, "error": str(e)}
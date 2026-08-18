# Change the import statement at the top of main.py
from src.stt_engine import ElevenLabsSTTEngine

# Update Section 5 inside main() function:
def main():
    # ... [Sections 1-4 remain unchanged] ...

    print("\n--- 5. Voice Input Run (ElevenLabs Scribe STT) ---")
    stt = ElevenLabsSTTEngine()
    sample_audio = "data/sample.wav"
    
    if os.path.exists(sample_audio):
        # 'ta' for Tamil, 'en' for English
        stt_result = stt.transcribe(sample_audio, language_code="ta")
        if stt_result["success"]:
            print(f"Transcribed Text: {stt_result['transcript']}")
            res = harness.run_pipeline(stt_result['transcript'])
            print("Pipeline Output:", json.dumps(res, indent=2))
        else:
            print(f"STT Error: {stt_result['error']}")
    else:
        print(f"No voice file found at '{sample_audio}'. Simulating voice query...")
        res = harness.run_pipeline("What is the capital of Tamil Nadu?")
        print("Pipeline Output:", json.dumps(res, indent=2))


import os
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_size="small", device="cuda", compute_type="int8"):
        # Define the local path for the model
        self.model_path = os.path.join(os.getcwd(), "models", model_size)
        
        print(f"Loading model '{model_size}' into {device} memory...")
        # The model is loaded once here during initialization
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=self.model_path
        )

    def transcribe(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return

        segments, info = self.model.transcribe(file_path, beam_size=5)
        
        print(f"\nProcessing: {os.path.basename(file_path)}")
        print(f"Detected language: {info.language} ({info.language_probability:.2f})")

        results = []
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            results.append(segment.text)
        
        return " ".join(results)

if __name__ == "__main__":
    # 1. Initialize the transcriber (Downloads/Loads model once)
    ai_transcriber = Transcriber()

    # 2. Reuse the same loaded model for multiple files
    audio_files = ["sample.mp3", "sample.mp3"] 
    
    for file in audio_files:
        full_path = os.path.join(os.getcwd(), file)
        ai_transcriber.transcribe(full_path)
        
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import uvicorn

# --- Model Logic ---
class Transcriber:
    def __init__(self, model_size="small", device="cuda", compute_type="int8"):
        self.model_path = os.path.join(os.getcwd(), "models", model_size)
        print(f"Loading model '{model_size}'...")
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=self.model_path
        )

    def transcribe(self, file_path: str):
        segments, info = self.model.transcribe(file_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return {"language": info.language, "text": text.strip()}

# --- FastAPI Setup ---
# Initialize a global variable for the transcriber
ai_transcriber = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global ai_transcriber
    ai_transcriber = Transcriber()
    yield
    # Clean up (if needed) on shutdown
    del ai_transcriber

app = FastAPI(lifespan=lifespan)

@app.post("/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    # 1. Basic validation
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio format.")

    # 2. Save temporary file
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Transcribe using the reusable model
        result = ai_transcriber.transcribe(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 4. Clean up: Remove temp file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
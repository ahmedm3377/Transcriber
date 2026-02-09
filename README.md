## Try Whisper

FastAPI service and a small CLI example for transcribing audio with `faster-whisper`.

### Features
- Reuses a single Whisper model instance for multiple requests.
- Simple `/transcribe` endpoint for audio files.
- Optional local model cache under `models/`.

### Project Structure
- `app.py` — FastAPI server with `/transcribe`.
- `main.py` — Minimal CLI usage example.
- `models/` — Model cache directory.

### Requirements
- Python >= 3.12
- CUDA-capable GPU recommended (default device is `cuda`).

### Install
If you use `uv`:
```bash
uv sync
```

If you use `pip`:
```bash
pip install -r requirements.txt
```


### Run the API Server
```bash
python app.py
```
The server starts at `http://0.0.0.0:8000`.

### Use the API
```bash
curl -X POST "http://localhost:8000/transcribe" \
	-H "accept: application/json" \
	-H "Content-Type: multipart/form-data" \
	-F "file=@/path/to/audio.mp3"
```

Response example:
```json
{
	"language": "en",
	"text": "Hello world"
}
```

### CLI Example
Update the file list in `main.py` and run:
```bash
python main.py
```

### Configuration
The default model configuration in `app.py`:
- `model_size`: `small`
- `device`: `cuda`
- `compute_type`: `int8`

To run on CPU, change the `device` to `cpu` (and optionally `compute_type` to `int8` or `float32`).

### Notes
- The first run downloads the model into `models/small/`.
- The API only accepts `audio/*` content types.


import sys
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import whisperx
import torch
import numpy as np

app = FastAPI()

# API Key Setup
API_KEY = "boy_i_sure_hope_you_supplied_one_from_the_command_line_cause_this_string_is_going_in_the_repo"

print("loading model, yeehaw")
# Load model and transcribe audio
if torch.cuda.is_available():
    # https://huggingface.co/openai/whisper-large-v2
    model = "large-v2"
    audio_model = whisperx.load_model(model, device="cuda", language="en")
else:
    print("YOU GOT NO GPU ACTIVE YO")
    # https://huggingface.co/openai/whisper-small.en
    model = "small.en"
    audio_model = whisperx.load_model(model, device="cpu", language="en", compute_type="float32")
print("🚀")

def get_api_key(request: Request):
    # we in dev baybee
    return API_KEY
    # try:
    #     return request.headers["Authorization"]
    # except KeyError:
    #     raise HTTPException(status_code=401, detail="API Key missing")

# Middleware for API Key Check
@app.middleware("http")
async def api_key_check(request: Request, call_next):
    api_key = get_api_key(request)
    if api_key != API_KEY:
        return JSONResponse(status_code=401, detail="Invalid API Key")
    return await call_next(request)

# Health Check Endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello from the transcription API"}

# Transcribe Endpoint
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="File format not supported. Please upload a WAV file.")
    
    try:
        # Read audio file into memory
        audio_content = await file.read()
        audio_array = np.frombuffer(audio_content, np.int16).flatten().astype(np.float32) / 32768.0
        
        # Transcribe the audio
        result = audio_model.transcribe(audio_array, batch_size=16)
        
        return {"transcription": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]  # Dynamic API key from command line argument
    uvicorn.run(app, host="0.0.0.0", port=8756)

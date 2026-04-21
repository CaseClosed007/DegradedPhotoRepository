# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import subprocess

from clean_my_drive import generate_scan_report
from src.enhance import enhance_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanRequest(BaseModel):
    folder_path: str

class EnhanceRequest(BaseModel):
    image_path: str
    reason: str

# --- THE MISSING ENDPOINT ---
@app.get("/api/browse")
def browse_for_folder():
    """Opens a native macOS folder picker using AppleScript."""
    try:
        apple_script = 'POSIX path of (choose folder with prompt "Select Gallery Folder to Scan")'
        result = subprocess.run(
            ["osascript", "-e", apple_script],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            folder_path = result.stdout.strip()
            return {"folder_path": folder_path}
        else:
            return {"folder_path": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ----------------------------

@app.post("/api/scan")
def run_scan(request: ScanRequest):
    """Receives a folder path, runs the AI, and returns the results."""
    target_dir = request.folder_path
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {target_dir}")
    try:
        results = generate_scan_report(
            target_dir=target_dir,
            model_path="models/weights/best_mobilenetv2.keras",
            blur_threshold=100.0
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/degraded-photos")
def get_photos():
    """Reads the JSON manifest and sends it to the frontend."""
    manifest_path = 'scan_results.json'
    if not os.path.exists(manifest_path):
        return [] 
    with open(manifest_path, 'r') as f:
        return json.load(f)

@app.post("/api/enhance")
def enhance_photo(request: EnhanceRequest):
    """Enhances a degraded photo and returns the path to the saved result."""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image not found on disk")
    success, enhanced_path = enhance_image(request.image_path, request.reason)
    if not success:
        raise HTTPException(status_code=500, detail="Enhancement failed — image may be unreadable")
    return {"enhanced_path": enhanced_path}

@app.get("/api/image")
def get_image(path: str):
    """Serves the actual image bytes securely to the browser."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found on disk")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
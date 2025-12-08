from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import json
import uvicorn
import time
import requests

from app.config import (
    SUPABASE_URL, SUPABASE_KEY, STUDENT_BUCKET,
    USE_GPU, DET_SIZE, SIMILARITY_THRESHOLD, DET_CONF_THRESHOLD
)

from app.services.supabase_service import SupabaseService
from app.services.face_service import FaceService

app = FastAPI(title="Face Attendance API")

supa = SupabaseService(SUPABASE_URL, SUPABASE_KEY)

face_svc = FaceService(
    supa,
    student_bucket=STUDENT_BUCKET,
    use_gpu=USE_GPU,
    det_size=(DET_SIZE, DET_SIZE),
    sim_threshold=SIMILARITY_THRESHOLD,
    det_conf_threshold=DET_CONF_THRESHOLD
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recognize_upload")
def recognize_upload(
    session_id: str = Form(...),
    enrolled: str = Form(...),
    image_url: str = Form(...),
    image_name: Optional[str] = Form(None)
):

    # Parse enrolled list
    try:
        enrolled_list = json.loads(enrolled)
        if not isinstance(enrolled_list, list):
            raise ValueError("enrolled must be a list or comma-separated string")
    except Exception:
        enrolled_list = [x.strip() for x in enrolled.split(",") if x.strip()]

    # Normalize to lowercase strings (safely handle non-string items)
    normalized = []
    for item in enrolled_list:
        try:
            s = str(item).strip().lower()
            if s:
                normalized.append(s)
        except Exception:
            continue
    enrolled_list = normalized

    # Fetch classroom image from URL
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        frame_bytes = response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(400, f"Cannot fetch image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Error reading image from URL: {str(e)}")

    # Build embeddings for only these students
    try:
        names_arr, embs_arr = face_svc.build_embeddings_for_students(enrolled_list)
    except Exception as e:
        raise HTTPException(500, str(e))

    # Perform recognition
    try:
        recognized = face_svc.recognize_frame(frame_bytes, names_arr, embs_arr)
    except Exception as e:
        raise HTTPException(500, str(e))

    # Build attendance response
    attendance = {}
    for r in enrolled_list:
        attendance[r] = "present" if r in recognized else "absent"

    # Print to terminal
    print("\n--- Attendance (Session:", session_id, ") ---")
    for k,v in attendance.items():
        print(k, ":", v)
    print("--------------------------------------------\n")

    # Generate image name if not provided
    if not image_name:
        image_name = f"CR000_{int(time.time())}"

    return {
        "session_id": session_id,
        "image_name": image_name,
        "attendance": attendance,
        "recognized_summary": recognized,
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
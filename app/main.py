from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import json
import uvicorn
import time
import requests

from app.config import (
    SUPABASE_URL, SUPABASE_KEY, STUDENT_BUCKET,
    USE_GPU, DET_SIZE, SIMILARITY_THRESHOLD, DET_CONF_THRESHOLD, SUPABASE_SERVICE_ROLE_KEY
)

from app.services.supabase_service import SupabaseService
from app.services.face_service import FaceService

app = FastAPI(title="Face Attendance API")

supa = SupabaseService(SUPABASE_URL, SUPABASE_KEY, service_role_key=SUPABASE_SERVICE_ROLE_KEY)

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

    try:
        enrolled_list = json.loads(enrolled)
        if not isinstance(enrolled_list, list):
            raise ValueError()
    except Exception:
        enrolled_list = [x.strip() for x in enrolled.split(",") if x.strip()]

    enrolled_list = [str(x).strip().lower() for x in enrolled_list]

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        frame_bytes = response.content
    except Exception as e:
        raise HTTPException(400, f"Cannot fetch image: {str(e)}")

    # Build embeddings (be defensive about return shape)
    try:
        embeds_result = face_svc.build_embeddings_for_students(enrolled_list)
        # Normalize build_embeddings_for_students return
        if isinstance(embeds_result, (tuple, list)) and len(embeds_result) == 2:
            names_arr, embs_arr = embeds_result
        elif isinstance(embeds_result, dict):
            # dict of name -> embedding
            names_arr = list(embeds_result.keys())
            embs_arr = [embeds_result[n] for n in names_arr]
        elif isinstance(embeds_result, list):
            # perhaps it's a list of embeddings aligned with enrolled_list
            names_arr = enrolled_list
            embs_arr = embeds_result
        else:
            raise ValueError(f"Unexpected return from build_embeddings_for_students: {type(embeds_result)}")
    except Exception as e:
        # give a clear message and log for debugging
        raise HTTPException(500, f"Error building embeddings: {str(e)}")

    # NOW RETURNS similarity scores (defensive handling for multiple possible return types)
    try:
        recog_result = face_svc.recognize_frame(frame_bytes, names_arr, embs_arr)

        total_present = 0  # default in case something unexpected happens later

        # Case A: tuple/list
        if isinstance(recog_result, (tuple, list)):
            if len(recog_result) == 3:
                # recognized, similarity_map, total_present
                recognized, similarity_map, total_present = recog_result
            elif len(recog_result) == 2:
                # recognized, similarity_map
                recognized, similarity_map = recog_result
                total_present = len(recognized)
            else:
                raise ValueError(f"Unexpected tuple/list length from recognize_frame: {len(recog_result)}")

        # Case B: returned only a dict -> treat it as similarity_map
        elif isinstance(recog_result, dict):
            similarity_map = recog_result
            sim_threshold = getattr(face_svc, "sim_threshold", SIMILARITY_THRESHOLD)
            recognized = [name for name, sim in similarity_map.items() if (sim or 0.0) >= sim_threshold]
            total_present = len(recognized)

        # Case C: returned only a list -> treat as recognized names
        elif isinstance(recog_result, list):
            recognized = recog_result
            similarity_map = {name: (1.0 if name in recognized else 0.0) for name in enrolled_list}
            total_present = len(recognized)

        else:
            raise ValueError(f"Unexpected return from recognize_frame: {type(recog_result)}")
    except Exception as e:
        raise HTTPException(500, f"{str(e)}")

    # Build attendance with similarity %
    attendance = {}
    for r in enrolled_list:
        status = "present" if r in recognized else "absent"
        # current similarity_map values are cosine scores like 0.8; keep same behavior as before
        similarity = round(float(similarity_map.get(r, 0.0)), 2)
        attendance[r] = {
            "status": status,
            "similarity_percent": similarity
        }

    print("\n--- Attendance (Session:", session_id, ") ---")
    for k, v in attendance.items():
        print(k, ":", v)
    print("--------------------------------------------\n")

    if not image_name:
        image_name = f"CR000_{int(time.time())}"

    return {
        "session_id": session_id,
        "image_name": image_name,
        "attendance": attendance,
        "recognized_summary": recognized,
        "total_present": total_present
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

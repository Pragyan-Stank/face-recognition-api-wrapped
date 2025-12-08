# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
STUDENT_BUCKET = os.getenv("STUDENT_IMAGES_BUCKET", "student-images")
FRAMES_BUCKET = os.getenv("CLASSROOM_FRAMES_BUCKET", "classroom-frames")

USE_GPU = os.getenv("USE_GPU", "false").lower() in ("1", "true", "yes")
DET_SIZE = int(os.getenv("DETECTION_SIZE", "1024"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
DET_CONF_THRESHOLD = float(os.getenv("DET_CONF_THRESHOLD", "0.25"))

# sanity check
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY in .env")

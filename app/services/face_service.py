# app/services/face_service.py
from insightface.app import FaceAnalysis
import numpy as np
from typing import Tuple, Dict, List, Optional
from app.utils.image_utils import bytes_to_bgr_image
from app.services.supabase_service import SupabaseService
import cv2
import time
from collections import defaultdict
import re

class FaceService:
    def __init__(
        self,
        supa: SupabaseService,
        student_bucket: str,
        use_gpu: bool = False,
        det_size: Tuple[int,int] = (1024,1024),
        sim_threshold: float = 0.55,
        det_conf_threshold: float = 0.25,
    ):
        self.supa = supa
        self.student_bucket = student_bucket
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.sim_threshold = sim_threshold
        self.det_conf_threshold = det_conf_threshold
        self.fa = None

    def init_face_app(self):
        """Initialize InsightFace once."""
        if self.fa is None:
            ctx_id = 0 if self.use_gpu else -1
            print(f"[FaceService] Initializing FaceAnalysis (ctx_id={ctx_id}, det_size={self.det_size})")
            self.fa = FaceAnalysis(allowed_modules=['detection','recognition'])
            self.fa.prepare(ctx_id=ctx_id, det_size=self.det_size)
            print("[FaceService] FaceAnalysis ready.")

    def build_embeddings_for_students(self, roll_ids: List[str]):
        """
        Build face embeddings ONLY for the requested roll numbers.
        Returns:
            student_names: List[str] of roll_ids
            student_embs: np.ndarray embeddings
        """
        self.init_face_app()
        all_file_paths = self.supa.list_all_files_recursive(self.student_bucket)

        # Map roll_id → images in bucket
        roll_to_images = defaultdict(list)
        for p in all_file_paths:
            if "/" in p:
                folder = p.split("/")[0].lower().strip()
                if folder in roll_ids:
                    roll_to_images[folder].append(p)

        names = []
        embs = []

        for rid in roll_ids:
            imgs = roll_to_images.get(rid, [])
            if not imgs:
                print(f"[WARN] No images found for {rid}")
                continue

            for p in imgs:
                img_bytes = self.supa.download_bytes(self.student_bucket, p)
                if not img_bytes:
                    print(f"[WARN] Could not download {p}")
                    continue

                img_bgr = bytes_to_bgr_image(img_bytes)
                if img_bgr is None:
                    print(f"[WARN] Cannot decode {p}")
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                faces = self.fa.get(img_rgb)
                if not faces:
                    print(f"[WARN] No face detected in {p}")
                    continue

                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = np.asarray(face.embedding, dtype=np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-10)

                names.append(rid)
                embs.append(emb)

        if not embs:
            raise RuntimeError("No embeddings created for requested roll numbers.")

        names_arr = np.array(names)
        embs_arr = np.vstack(embs).astype(np.float32)

        print(f"[FaceService] Built {len(embs_arr)} embeddings for students: {roll_ids}")

        return names_arr, embs_arr

    def recognize_frame(self, frame_bytes, names_arr, embs_arr):
        """
        Recognize faces in classroom image using given embeddings.
        """
        img_bgr = bytes_to_bgr_image(frame_bytes)
        if img_bgr is None:
            raise RuntimeError("Failed to decode classroom image.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = self.fa.get(img_rgb)
        faces = [f for f in faces if getattr(f,"det_score",1.0) >= self.det_conf_threshold]

        recognized = {}

        for f in faces:
            emb = np.asarray(f.embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)

            sims = np.dot(embs_arr, emb)
            idx = int(np.argmax(sims))
            score = float(sims[idx])

            if score >= self.sim_threshold:
                name = names_arr[idx].lower()
                if name not in recognized or recognized[name] < score:
                    recognized[name] = score

        return recognized

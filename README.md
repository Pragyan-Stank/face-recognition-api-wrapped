# Face Recognition based Attendance System

### Overview

Face Attendance API is a production-oriented computer vision service that automates classroom attendance using face recognition. The system retrieves enrolled student images from Supabase, builds facial embeddings using InsightFace, and compares them against a live classroom image to determine presence based on cosine similarity. It exposes a REST API that returns per-student attendance status along with similarity confidence scores.

This project demonstrates an end-to-end face recognition pipeline — cloud image storage, model inference, similarity matching, and structured API responses — designed for real-world deployment and scalability.

### Key Features

- Automated attendance detection from a single classroom image
- Face detection and recognition using InsightFace
- Embedding-based matching with cosine similarity
- Robust image retrieval from Supabase (public URLs, signed URLs, fallback mechanisms)
- Configurable thresholds for similarity and detection confidence
- GPU-accelerated inference support (optional)
- Clean REST API built using FastAPI

### How It Works

- Student reference images are stored in Supabase (organized by roll ID).
- The API dynamically builds normalized facial embeddings for enrolled students.
- A classroom image is fetched via URL and processed using InsightFace.
- Detected faces are matched against enrolled embeddings using cosine similarity.
- Attendance is returned with present / absent status and similarity scores.

#### Example Output
> bt23eci024_pragyan → present (0.72)

> bt23eci044_kaushik → absent (0.18)


### Tech Stack

Python

FastAPI (REST API)

InsightFace (Face detection & recognition)

OpenCV & NumPy (image processing)

Supabase (cloud storage & signed URLs)

Uvicorn (ASGI server)


### Use Cases

1. Classroom attendance automation

2. Smart classrooms & exam proctoring

3. Access control and identity verification

4. Edge-deployable face recognition systems

### Design & Engineering Highlights

- Embedding normalization and cosine similarity ensure stable identity matching
- Defensive API design handles varied input formats and inference edge cases
- Storage-agnostic architecture allows model or backend replacement
- Designed with production considerations: scalability, extensibility, and security



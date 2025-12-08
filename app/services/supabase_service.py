# app/services/supabase_service.py
from supabase import create_client
from typing import List, Dict, Optional
import requests
from urllib.parse import urljoin
import time

class SupabaseService:
    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)
        self.url = url
        self.key = key

    def list_all_files_recursive(self, bucket: str, prefix: str = ""):
        """
        Recursively list all files under a prefix in a bucket.
        Returns a list of file paths including subfolders.
        """
        results = []
        stack = [prefix]

        while stack:
            current_prefix = stack.pop()

            try:
                items = self.client.storage.from_(bucket).list(current_prefix, {"limit": 1000})
            except Exception as e:
                print("list error:", e)
                continue

            if not items:
                continue

            for item in items:
                name = item.get("name") or item.get("path") or ""
                if not name:
                    continue

                full_path = f"{current_prefix}/{name}" if current_prefix else name

                if item.get("metadata") is None:  
                    # this means it's a FOLDER
                    stack.append(full_path)
                else:
                    # this is a FILE
                    results.append(full_path)

        return results

    def get_public_url(self, bucket: str, path: str) -> Optional[str]:
        try:
            r = self.client.storage.from_(bucket).get_public_url(path)
            # different versions use different keys; check common ones
            if isinstance(r, dict):
                return r.get("publicURL") or r.get("publicUrl") or r.get("public_url")
            # If str returned:
            if isinstance(r, str):
                return r
        except Exception:
            pass
        return None

    def create_signed_url(self, bucket: str, path: str, expires_in: int = 3600) -> Optional[str]:
        try:
            r = self.client.storage.from_(bucket).create_signed_url(path, expires_in)
            if isinstance(r, dict):
                return r.get("signedURL") or r.get("signedUrl") or r.get("signed_url")
            if isinstance(r, str):
                return r
        except Exception:
            pass
        return None

    def download_bytes(self, bucket: str, path: str) -> Optional[bytes]:
        """
        Preferred: try to get public/signed URL and fetch via requests (works consistently).
        Fallback: try client's download method if available.
        """
        # Try public URL
        url = self.get_public_url(bucket, path)
        if url:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return r.content
            except Exception:
                pass

        # Try signed URL
        url = self.create_signed_url(bucket, path, expires_in=120)
        if url:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return r.content
            except Exception:
                pass

        # Try direct download via client (some supabase-py versions support .download)
        try:
            resp = self.client.storage.from_(bucket).download(path)
            # resp might be bytes, or a requests.Response
            if isinstance(resp, (bytes, bytearray)):
                return bytes(resp)
            # if it's a dict with 'content' key
            if isinstance(resp, dict) and 'content' in resp:
                return resp['content']
            # if resp has .content
            if hasattr(resp, "content"):
                return resp.content
        except Exception:
            pass

        return None

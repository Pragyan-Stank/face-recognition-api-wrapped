# app/services/supabase_service.py
from supabase import create_client
from typing import List, Dict, Optional, Any
import requests
import time

class SupabaseService:
    def __init__(self, url: str, key: str, service_role_key: Optional[str] = None):
        """
        key: typically anon/public key
        service_role_key: the secret service_role key (use server-side only) — recommended for signed URLs
        """
        self.client = create_client(url, key)
        self.url = url
        self.key = key
        # fallback to anon key if service_role_key not provided
        self.service_role_key = service_role_key or key

    def _extract_url_from_response(self, resp: Any, keys: List[str]):
        """
        Normalize various response shapes returned by different supabase-py versions.
        Tries to extract url-like values from dicts, tuples, strings, or Response-like objects.
        """
        if resp is None:
            return None

        # direct string
        if isinstance(resp, str):
            return resp

        # dict-like
        if isinstance(resp, dict):
            for k in keys:
                if k in resp and resp[k]:
                    return resp[k]
            # some shapes nest under 'data'
            if 'data' in resp and isinstance(resp['data'], dict):
                for k in keys:
                    if k in resp['data'] and resp['data'][k]:
                        return resp['data'][k]
            return None

        # tuple/list (status, dict) or (dict, )
        if isinstance(resp, (list, tuple)):
            for item in resp:
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    for k in keys:
                        if k in item and item[k]:
                            return item[k]
                    if 'data' in item and isinstance(item['data'], dict):
                        for k in keys:
                            if k in item['data'] and item['data'][k]:
                                return item['data'][k]

        # response-like objects (try to parse .json())
        if hasattr(resp, "json"):
            try:
                j = resp.json()
                return self._extract_url_from_response(j, keys)
            except Exception:
                pass

        return None

    def list_all_files_recursive(self, bucket: str, prefix: str = ""):
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

                # older supabase returns folders as items with no metadata
                if item.get("metadata") is None:
                    stack.append(full_path)
                else:
                    results.append(full_path)

        return results

    def get_public_url(self, bucket: str, path: str) -> Optional[str]:
        """
        Try to get the public URL for an object (works if object/bucket is public).
        """
        try:
            path = path.lstrip("/")
            resp = self.client.storage.from_(bucket).get_public_url(path)
            url = self._extract_url_from_response(resp, ["publicURL", "publicUrl", "public_url", "url"])
            return url
        except Exception as e:
            # debug print to help during dev — remove if noisy
            print("get_public_url error:", e)
            return None

    def create_signed_url(self, bucket: str, path: str, expires_in: int = 3600) -> Optional[str]:
        """
        Create a signed URL. Uses the service role key client if it's different from anon key.
        """
        try:
            path = path.lstrip("/")

            # If service_role_key differs from initial key, use a temporary client with service role key
            if self.service_role_key and self.service_role_key != self.key:
                temp_client = create_client(self.url, self.service_role_key)
                resp = temp_client.storage.from_(bucket).create_signed_url(path, expires_in)
            else:
                resp = self.client.storage.from_(bucket).create_signed_url(path, expires_in)

            url = self._extract_url_from_response(resp, ["signedURL", "signedUrl", "signed_url", "signedurl"])
            return url
        except Exception as e:
            print("create_signed_url error:", e)
            return None

    def download_bytes(self, bucket: str, path: str) -> Optional[bytes]:
        """
        Robust download:
         1) Try public URL
         2) Try signed URL (service role)
         3) Try direct client.download fallback
        """
        path = path.lstrip("/")

        # 1) public
        url = self.get_public_url(bucket, path)
        if url:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return r.content
            except Exception as e:
                print("public URL fetch failed:", e)

        # 2) signed
        url = self.create_signed_url(bucket, path, expires_in=120)
        if url:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return r.content
            except Exception as e:
                print("signed URL fetch failed:", e)

        # 3) client.download fallback
        try:
            resp = self.client.storage.from_(bucket).download(path)
            if isinstance(resp, (bytes, bytearray)):
                return bytes(resp)
            if isinstance(resp, dict) and 'content' in resp:
                return resp['content']
            if hasattr(resp, "content"):
                return resp.content
        except Exception as e:
            print("client.download failed:", e)

        return None

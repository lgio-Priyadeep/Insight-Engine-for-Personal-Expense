import hashlib

def stable_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]

"""Verify files against the versioned source manifest."""
import hashlib,json
from pathlib import Path
root=Path(__file__).parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for name,digest in manifest['artifacts'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('Versioned source hashes verified.')

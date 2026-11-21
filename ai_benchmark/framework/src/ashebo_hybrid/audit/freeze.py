"""Hash snapshots detect later changes; they are not independent blind timestamps."""
import datetime
import hashlib
import json
from pathlib import Path

def _digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze_artifacts(files,protocol,output):
    target=Path(output).resolve()
    if target.exists():raise FileExistsError(target)
    entries={}
    for name,path in files.items():
        path=Path(path).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        stored=str(path.relative_to(target.parent)) if path.is_relative_to(target.parent) else str(path)
        entries[name]=dict(path=stored,sha256=_digest(path))
    encoded=json.dumps(protocol,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=protocol,protocol_sha256=hashlib.sha256(encoded).hexdigest(),files=entries)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(record,indent=2)+"\n");return record

def verify_freeze(record_path):
    path=Path(record_path).resolve();record=json.loads(path.read_text());changed=[]
    for name,item in record["files"].items():
        current=path.parent/item["path"]
        if not current.is_file() or _digest(current)!=item["sha256"]:changed.append(name)
    encoded=json.dumps(record["protocol"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest()!=record["protocol_sha256"]:changed.append("protocol")
    return dict(verified=not changed,changed=changed)

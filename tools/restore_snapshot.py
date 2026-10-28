"""Restore one original project snapshot from the deduplicated release archive."""
import argparse,hashlib,json,os,tarfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('archive',type=Path)
p.add_argument('--snapshot',required=True,choices=['laptop','windy-older','windy-current'])
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error('Output must not already exist')
a.output.mkdir(parents=True)
with tarfile.open(a.archive,'r|gz') as tf:
 first=next(iter(tf))
 manifest=json.load(tf.extractfile(first))[a.snapshot]
 wanted={}
 for rel,meta in manifest.items():
  if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Unsafe archive path')
  if 'symlink' in meta:
   dest=a.output/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(meta['symlink']);continue
  wanted.setdefault(meta['sha256'],[]).append((rel,meta))
 for member in tf:
  digest=member.name.removeprefix('objects/')
  if digest not in wanted:continue
  data=tf.extractfile(member).read()
  if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Checksum mismatch: '+member.name)
  for rel,meta in wanted.pop(digest):
   dest=a.output/rel
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
   dest.chmod(meta['mode']);os.utime(dest,ns=(meta['mtime_ns'],meta['mtime_ns']))
 if wanted:raise ValueError('Missing archive objects')
print(f'Restored and verified {len(manifest)} files to {a.output}')

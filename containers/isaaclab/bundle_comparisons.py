"""Package the working source (including uncommitted changes), never runs/secrets.

The immutable SIF is hashed, not duplicated. Transfer it alongside this output.
No environment rebuild, Docker conversion or image publishing is performed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ROOTS = {'legged_gym', 'rsl_rl', 'resources', 'scripts', 'containers', 'tests', 'docs'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_paths(root):
    names = subprocess.check_output(['git','-C',str(root),'ls-files','-co','--exclude-standard','-z'])
    for name in sorted(set(names.decode().split('\0')) - {''}):
        rel = Path(name)
        if rel.parts[0] not in ROOTS and name not in ('pyproject.toml','README.md','LICENSE','.gitignore'):
            continue
        if (any(p in ('__pycache__','logs','runs','.git','.env') for p in rel.parts)
                or rel.suffix in ('.sif','.pt','.pth','.npz','.tar','.gz','.pyc')
                or name.endswith(('.pem','.key'))):
            continue
        path = root/rel
        if path.is_symlink():
            raise ValueError('Source bundle requires regular files; audit symlink: '+name)
        if path.is_file():
            yield rel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; never overwritten')
    args = p.parse_args()
    image = args.image.resolve(strict=True)
    if image.suffix != '.sif' or not image.is_file():
        p.error('--image must name an existing Apptainer .sif')
    files = list(source_paths(ROOT))
    git = lambda *cmd: subprocess.check_output(['git','-C',str(ROOT),*cmd],text=True).strip()
    provenance = dict(version=1, git_sha=git('rev-parse','HEAD'),
                      git_status=git('status','--short'), branch=git('branch','--show-current'),
                      source_files={str(f):sha256(ROOT/f) for f in files})
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output/'hardpact-comparisons-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for file in files:
            tar.add(ROOT/file, arcname=str(Path('hardpact-comparisons')/file), recursive=False)
        data = json.dumps(provenance,indent=2).encode()
        info = tarfile.TarInfo('hardpact-comparisons/CONTAINER_SOURCE.json')
        info.size = len(data); info.mode = 0o644
        tar.addfile(info,io.BytesIO(data))
    manifest = dict(version=1, source=archive.name, source_sha256=sha256(archive),
                    image=image.name, image_sha256=sha256(image), image_bytes=image.stat().st_size,
                    git_sha=provenance['git_sha'], git_status=provenance['git_status'],
                    image_copied=False, checkpoints_included=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'SHA256SUMS').write_text(f"{manifest['source_sha256']}  {archive.name}\n"
                                       f"{manifest['image_sha256']}  {image.name}\n")
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()

"""Fetch exact author sources; no third-party code is silently reimplemented."""
import json
from pathlib import Path
import subprocess


def main():
    here = Path(__file__).resolve().parent
    for item in json.loads((here / 'vendor_revisions.json').read_text()):
        target = here / 'vendor' / item['folder']
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            subprocess.run(['git', 'clone', '--no-checkout', item['url'], str(target)], check=True)
        else:
            actual_url = subprocess.check_output(['git', '-C', str(target), 'remote', 'get-url', 'origin'], text=True).strip()
            if actual_url != item['url']:
                raise RuntimeError('Existing vendor remote differs from pinned source')
            if subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
                raise RuntimeError('Preserve local vendor changes before fetching')
        subprocess.run(['git', '-C', str(target), 'fetch', 'origin', item['commit']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', item['commit']], check=True)


if __name__ == '__main__':
    main()

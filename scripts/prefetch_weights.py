"""Pre-download pinned checkpoints. Network only -- imports no torch, so it is
safe to run while a model is resident in another process.

The link to huggingface.co from here drops multi-GB streams: small files always
land, every safetensors shard dies partway with ReadTimeout or ECONNRESET. So
each repo is retried, and HF_HUB_DOWNLOAD_TIMEOUT is raised well above the 10s
default. huggingface_hub resumes from the .incomplete blob, so a retry picks up
where the last attempt stopped rather than restarting the shard.
"""
import os, re, sys, time
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from pathlib import Path
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
src = (REPO / "src/judges/local_runner.py").read_text()
block = src.split("MODEL_REVISIONS", 1)[1].split("}", 1)[0]
REV = dict(re.findall(r'"([^"]+)":\s*"([0-9a-f]{40})"', block))

ALLOW = ["*.json", "*.txt", "*.model", "*.safetensors"]
IGNORE = ["consolidated*", "original/*", "*.pth", "*.bin", "*.gguf"]
ATTEMPTS = 40

for name in sys.argv[1:]:
    rev = REV[name]          # KeyError here is the point: no unpinned fetches
    print(f"=== {name} @ {rev[:12]}", flush=True)
    for i in range(1, ATTEMPTS + 1):
        try:
            p = snapshot_download(name, revision=rev, allow_patterns=ALLOW,
                                  ignore_patterns=IGNORE, max_workers=1)
        except Exception as exc:
            got = sum(f.stat().st_size for f in
                      (Path.home()/".cache/huggingface/hub").rglob("*.incomplete"))
            print(f"--- attempt {i}/{ATTEMPTS} failed ({type(exc).__name__}); "
                  f"{got/2**30:.2f} GiB held in .incomplete; retrying",
                  flush=True)
            time.sleep(min(5 * i, 60))
            continue
        tot = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
        print(f"=== done {name}: {tot/2**30:.1f} GiB in {i} attempt(s)", flush=True)
        break
    else:
        print(f"=== GAVE UP on {name} after {ATTEMPTS} attempts", flush=True)
        sys.exit(1)

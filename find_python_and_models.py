#!/usr/bin/env python
"""Find conda python and check pretrained models"""
import os, subprocess, glob

os.chdir("/root/autodl-tmp")

# Try to find conda python
candidates = [
    "/opt/conda/bin/python",
    "/opt/conda/envs/alignn/bin/python",
    os.path.expanduser("~/miniconda3/bin/python"),
    os.path.expanduser("~/miniconda3/envs/alignn/bin/python"),
]
for p in candidates:
    if os.path.exists(p):
        print(f"Found python at: {p}")
        break
else:
    # Glob for any python
    results = glob.glob("/opt/conda/bin/python*")
    print("Glob /opt/conda/bin/python*:", results)
    results += glob.glob("/root/miniconda3/bin/python*")
    print("Glob /root/miniconda3/bin/python*:", results)

# Try alignn
try:
    from alignn.pretrained import get_pretrained_models
    m = get_pretrained_models()
    ks = sorted(m.keys())
    open("/root/autodl-tmp/models_list.txt", "w").write("\n".join(ks))
    print(f"\nFound {len(ks)} pretrained models:")
    for k in ks:
        print(" ", k)
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

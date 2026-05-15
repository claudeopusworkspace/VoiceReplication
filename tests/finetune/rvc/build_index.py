"""Build the FAISS retrieval index for a trained RVC voice.

The upstream `tools/infer/train-index-v2.py` has a hardcoded `./logs/anz/...`
path and writes its outputs to `tools/infer/` — meant as a template, not a
reusable script. The WebUI has the parameterized version inline; this is a
clean CLI replica.

Usage: build_index.py <exp_name>
Writes:
  logs/<exp_name>/added_IVF<n_ivf>_Flat_nprobe_<np>_<exp_name>_v2.index
  logs/<exp_name>/total_fea.npy
"""
import os
import sys
import traceback
from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

EXP = sys.argv[1] if len(sys.argv) > 1 else "diana_rvc"
REPO_ROOT = "/workspace/VoiceReplication/specialized/rvc"
EXP_DIR = f"{REPO_ROOT}/logs/{EXP}"
FEATURE_DIR = f"{EXP_DIR}/3_feature768"
n_cpu = cpu_count()

os.chdir(REPO_ROOT)

print(f"Loading features from {FEATURE_DIR}")
npys = []
for name in sorted(os.listdir(FEATURE_DIR)):
    if not name.endswith(".npy"):
        continue
    phone = np.load(f"{FEATURE_DIR}/{name}")
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]
print(f"big_npy shape: {big_npy.shape}")

if big_npy.shape[0] > 2e5:
    print(f"kmeans-reducing {big_npy.shape[0]} → 10000 centers")
    try:
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * n_cpu,
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )
    except Exception:
        traceback.print_exc()

np.save(f"{EXP_DIR}/total_fea.npy", big_npy)

n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
print(f"FAISS IVF n_ivf={n_ivf}")
index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(index, f"{EXP_DIR}/trained_IVF{n_ivf}_Flat_nprobe_1_{EXP}_v2.index")

print("Adding vectors to index...")
batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])
out_path = f"{EXP_DIR}/added_IVF{n_ivf}_Flat_nprobe_1_{EXP}_v2.index"
faiss.write_index(index, out_path)
print(f"Wrote {out_path}")

"""Fish Speech temperature sweep — wider spread.

6 temperatures spanning the schema's full range (0.1 to 1.0), holding top_p and
repetition_penalty constant at the A_lowtemp values (0.85 / 1.2) so the
temp=0.5 cell reproduces A_lowtemp exactly. Same ref (05_excited_ocean) and
same seed (42) across variants.

Output: tests/outputs/harness/fish-speech_sweep2/T<temp>__<sentence_id>.wav
"""
import json
import os
import queue
import sys
import time
from pathlib import Path

FISH_DIR = Path("/workspace/VoiceReplication/generators/fish-speech")
ROOT = Path("/workspace/VoiceReplication")
os.chdir(FISH_DIR)
sys.path.insert(0, str(FISH_DIR))
sys.path.insert(0, str(ROOT / "tests" / "harness"))
os.environ["EINX_FILTER_TRACEBACK"] = "false"

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from fish_speech.inference_engine import TTSInferenceEngine  # noqa: E402
from fish_speech.models.dac.inference import load_model as load_decoder_model  # noqa: E402
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue  # noqa: E402
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest  # noqa: E402

LLAMA_CKPT = FISH_DIR / "checkpoints" / "s2-pro"
DECODER_CKPT = FISH_DIR / "checkpoints" / "s2-pro" / "codec.pth"

REF_ID = "05_excited_ocean"
REF_WAV = ROOT / "tests" / "harness" / "refs" / f"{REF_ID}.wav"
REF_TXT = ROOT / "tests" / "harness" / "refs" / f"{REF_ID}.txt"
SENTENCES = json.loads((ROOT / "tests" / "harness" / "sentences.json").read_text())

TEMPS = [0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
TOP_P = 0.85
REP_PEN = 1.2
SEED = 42

OUT_DIR = ROOT / "tests" / "outputs" / "harness" / "fish-speech_sweep2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    assert torch.cuda.is_available(), "need CUDA"
    print(f"ref={REF_ID}  top_p={TOP_P}  rep_pen={REP_PEN}  seed={SEED}", flush=True)
    print(f"temps={TEMPS}  sentences={list(SENTENCES)}", flush=True)

    t0 = time.time()
    llama_queue: queue.Queue = launch_thread_safe_queue(
        checkpoint_path=LLAMA_CKPT, device="cuda", precision=torch.bfloat16, compile=False
    )
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq", checkpoint_path=DECODER_CKPT, device="cuda"
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue, decoder_model=decoder_model,
        precision=torch.bfloat16, compile=False,
    )
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    ref_bytes = REF_WAV.read_bytes()
    ref_text = REF_TXT.read_text().strip()
    references = [ServeReferenceAudio(audio=ref_bytes, text=ref_text)]

    for temp in TEMPS:
        for sid, text in SENTENCES.items():
            tag = f"T{int(round(temp*100)):03d}"  # e.g. T010, T085
            out = OUT_DIR / f"{tag}__{sid}.wav"
            req = ServeTTSRequest(
                text=text,
                references=references,
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                seed=SEED,
                temperature=temp,
                top_p=TOP_P,
                repetition_penalty=REP_PEN,
                format="wav",
                streaming=False,
            )
            t0 = time.time()
            audio, sr, err = None, None, None
            for r in engine.inference(req):
                if r.code == "error":
                    err = str(r.error); break
                if r.code == "final":
                    sr, audio = r.audio; break
            gen_s = time.time() - t0
            if err or audio is None:
                print(f"  FAIL {tag}/{sid}: {err}", flush=True)
                continue
            sf.write(str(out), audio, sr)
            print(f"  {tag}/{sid}: {audio.shape[-1]/sr:.2f}s in {gen_s:.2f}s → {out.name}", flush=True)


if __name__ == "__main__":
    main()

# Smoke tests

One script per model, named `smoke_<model>.py` (the `smoke_` prefix avoids
Python module-name collisions when the file name matches the package being
imported — e.g. `chatterbox.py` would shadow the installed `chatterbox`
package). Each:
1. Loads the model's bundled or default pretrained weights.
2. Generates audio from a fixed sentence (see `SMOKE_TEXT` below).
3. Saves to `tests/outputs/smoke/<model>.wav`.
4. Prints generation time + output sample-rate + duration.
5. Exits 0 on success, non-zero on failure.

The goal is **"does the model run at all on this hardware?"** — not quality.
Quality comparison comes later, against the character reference voice.

## Run pattern

Each model has its own venv, so smoke tests are run via the venv's python:

```bash
cd /workspace/VoiceReplication
generators/chatterbox/.venv/bin/python tests/smoke/chatterbox.py
```

## Smoke text

```
Hello! This is a test of voice synthesis. The output should sound clear and natural.
```

Chosen for: short enough to fit in any model's single-shot limit, mix of
exclamation + statement, no weird punctuation, contains common phonemes.

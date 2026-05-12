# reference_voice/

Drop the character's voice data here. The actual audio files are gitignored —
this README is just a placeholder so the directory survives in git.

Suggested layout once Woj brings the data:

```
reference_voice/
├── clips/              # Individual clean clips (for zero-shot models)
│   ├── clip_01.wav
│   └── ...
├── full/               # Longer continuous audio (for fine-tuning candidates)
│   └── character_full.wav
└── transcripts.txt     # Aligned transcripts if available (StyleTTS2 / GPT-SoVITS fine-tune needs these)
```

We can revisit the layout once we see what you've gathered.

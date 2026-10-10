# Optional target-speaker adaptation protocol

This is an **optional separate experiment** for adapting a model to one consenting speaker. You do not need to record your own voice to train general Indian-English accent adaptation; use the public multi-speaker workflow in [`ACCENT_ADAPTATION.md`](ACCENT_ADAPTATION.md) for that. Runtime voice cloning can use a short consented reference clip without training on the target speaker. This optional protocol is for a later personalized-weight experiment only.

## Before recording

- Record only your own voice, or a speaker who has explicitly agreed to recording, model training, and synthetic speech generation.
- Use a quiet room and the same microphone, room, and distance (about 15–20 cm) for all sessions. Disable noise suppression, reverb, and automatic voice effects if possible.
- Save lossless mono WAV at 44.1 or 48 kHz, 16-bit PCM. Do not denoise or normalize clips after recording; the preparation pipeline will inspect them and XTTS will resample as needed.
- Speak in the target voice and accent: natural Indian English. Do not imitate a different accent. Use a comfortable, consistent loudness and vary natural intonation rather than performing exaggerated pitch changes.
- Record in separate sessions on at least three occasions. This makes a held-out session evaluation possible and reduces leakage between training and test clips.

## Dataset target

Aim for **30–45 minutes of usable English speech**, split over 3–5 sessions, plus accurate verbatim transcripts. A minimum viable pilot is 15 minutes, but expect weaker adaptation and report it as a pilot. Record each prompt as a separate file; keep clips around 4–12 seconds. Avoid long silences, clipping, overlapping speech, music, and edits that cut off consonants.

Record the complete English set first. Optional Hindi or code-switched recordings can be stored separately for later experiments, but should not be mixed into the initial English adaptation run. That keeps the first experiment interpretable and gives the model sufficient examples of the English sounds and accent we want it to generate.

## Session prompt pack

Read each line once in a conversational voice. If you stumble, repeat that clip; do not concatenate takes. Keep the same wording in the transcript.

### Session A — everyday statements

1. Good morning, I will call you after I reach the office.
2. Please send me the address when you have a moment.
3. I usually take the train, but today I came by bus.
4. We can finish this now and review it again tomorrow.
5. The weather is warm, and there is a little breeze outside.
6. I have kept the documents on the table near the window.
7. Let us begin with the first point and take our time.
8. I was busy in the afternoon, so I could not reply earlier.
9. The shop closes at eight, but the pharmacy stays open later.
10. I think this version sounds clearer than the previous one.
11. Please check the numbers once before we share the report.
12. We are meeting our friends for lunch this Sunday.

### Session B — questions, names, and numbers

1. Could you tell me which platform the train leaves from?
2. Are you free this evening, or would tomorrow work better?
3. How many people are joining us for the presentation?
4. My appointment is at 4:30 on the twenty-second of June.
5. The total comes to one thousand two hundred and fifty rupees.
6. Is the delivery expected between ten and eleven in the morning?
7. Would you like tea, coffee, or a glass of water?
8. I did not say Friday; I said Thursday.
9. Please call Ananya, Rohan, and Mr. Mehta before noon.
10. What would you change if we had one more week?
11. Did you receive the message I sent a few minutes ago?
12. How far is the station from the hotel?

### Session C — longer natural passages

1. On Saturday morning, I walked to the market to buy vegetables. It was already busy, but I found everything I needed and came home before lunch.
2. When we explain a new idea, it helps to give one simple example first. Then people can ask questions and connect the example to their own experience.
3. I enjoy listening to music while I work, although I turn it off when I need to concentrate on something important.
4. The meeting went well because everyone had time to share their thoughts. We wrote down the main decisions and agreed on what to do next.
5. If the weather stays clear, we can leave early, stop for breakfast on the way, and reach the city before the traffic gets heavy.
6. I want the translated speech to sound like the way I normally speak English: clear, relaxed, and easy to understand.

### Session D — only if more coverage is needed

Record 10–15 minutes of unscripted English about familiar topics (a daily routine, a favourite place, a recent project, or directions to a familiar location). Avoid private personal details. Afterwards, transcribe exactly what was said and check names and numbers manually.

## File and manifest format

Store data outside Git, for example under `data/private_speaker/`. Use stable filenames such as `session_a_001.wav`. Create a CSV with these columns:

```csv
audio_path,text,speaker_id,session_id,consent
/absolute/path/session_a_001.wav,"Good morning, I will call you after I reach the office.",target_speaker,session_a,yes
```

Use the same `speaker_id` for all clips and a different `session_id` for every separate recording session. The `consent` field must be `yes`. Keep the audio and manifest private and out of Git.

When ready, prepare the session-disjoint training and evaluation lists:

```bash
python scripts/prepare_speaker_adaptation.py /absolute/path/manifest.csv --output data/private_speaker/xtts
```

The script reports recording duration, format, clipping, and the session split. Review the report and listen to every clip before fine-tuning. Do not train if the manifest has transcript mistakes or if the evaluation session was also used to tune settings.

## Evaluation recordings

The held-out session is for measurement, not training. Use 20–30 sentences absent from training prompts, including statements, questions, numbers, and common Indian names. Generate exactly the same text with the base and adapted checkpoints. Compare ASR word error rate, speaker embedding similarity, pitch statistics/contour, and blinded human ratings for voice match, accent/prosody, and naturalness. Treat pitch and embedding metrics as supporting evidence; human ratings remain necessary.

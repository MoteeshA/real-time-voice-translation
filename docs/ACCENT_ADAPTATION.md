# Public-data Indian-English accent adaptation

This is the primary research path for improving **accent**, not a requirement to train on the user's voice. At inference, XTTS still receives a short, consented reference clip to condition the speaker identity. Public multi-speaker speech teaches the model general Indian-English pronunciation and prosody; it cannot teach the specific identity of the person in the reference.

## Dataset requirements

Use a licensed, transcribed, multi-speaker **English speech** corpus with reliable evidence that the clips represent Indian English. Required fields are audio path, transcript, speaker ID, and preferably source/region or self-reported accent metadata. Do not infer accent from a name or appearance. Filter corrupted, noisy, clipped, overlapping, and transcript-mismatched clips. Keep the dataset license, release version, citation, download date, and filtering rules with the experiment.

**First dataset candidate: GLOBE.** It is an English TTS corpus with transcripts, speaker IDs, and accent metadata; it includes an `India and South Asia (India, Pakistan, Sri Lanka)` accent label. The public dataset card identifies it as CC0-1.0 and offers a V2 release; its card recommends V2 to avoid a volume issue in V3. This regional label does not isolate India, so check how many usable speakers and hours remain after filtering and describe the grouping accurately in the paper. [Dataset card and release details](https://huggingface.co/datasets/MushanW/GLOBE) · [GLOBE authors' project page](https://globecorpus.github.io/).

Mozilla Common Voice also publishes English and Hindi collections under CC0, but the English release must be inspected for useful Indian-English speaker metadata and enough clean target-accent speech before it is selected. [Official dataset page](https://commonvoice.mozilla.org/en/datasets). IndicVoices-R is valuable Indian-language TTS data across 22 Indian languages, but it is not itself an Indian-English corpus. [Official dataset repository](https://github.com/AI4Bharat/IndicVoices-R).

The existing VCTK subset is useful as a general English fine-tuning/debug baseline, but it is not the target Indian-English accent dataset. Do not report VCTK adaptation as evidence of Indian-accent improvement.

## Experiment

1. Keep the unmodified XTTS checkpoint as the baseline.
2. Prepare a speaker-disjoint multi-speaker train/evaluation split from the verified Indian-English corpus with `scripts/prepare_xtts_dataset.py`.
3. Fine-tune only the documented XTTS component in a small pilot; retain the exact baseline checkpoint and settings. Monitor overfitting, intelligibility, and speaker similarity. Fine-tuning is a hypothesis to test, not a guaranteed accent fix.
4. Select test speakers excluded from all training/tuning. At inference, provide the same short reference clip and same English test text to baseline and adapted models.
5. Evaluate accent/naturalness with blinded Indian-English listener ratings, intelligibility with ASR plus human checks, speaker identity with embedding and listener ratings, and latency. Report confidence intervals and per-speaker results.
6. Only after this offline comparison succeeds, connect the adapted speech model to Hindi ASR/translation. Hold the translated English text fixed when comparing TTS so translation errors are not mistaken for accent errors.

## What the result can and cannot claim

If listeners prefer the adapted model on held-out speakers, the evidence supports improved general Indian-English delivery under the tested conditions. It does not prove a perfect match for every Indian region, speaker, or sentence. The short runtime reference remains necessary for speaker identity. Recording the target user is optional and belongs in a separate speaker-adaptation experiment, not in the public accent-training set.

## Compute and license

XTTS-v2 is under the non-commercial Coqui Public Model License. The Apple M2/8 GB machine successfully loaded XTTS and completed the first training batch, but CPU fine-tuning of its 518M-parameter GPT is not practical here: the optimizer caused a 7.5 GB physical footprint (9 GB peak) and heavy swap use before the pilot could advance to its first logged interval. No adapted checkpoint was produced. The repeatable launcher is `scripts/train_globe_xtts_pilot.py`; run it on a CUDA GPU with enough memory, then preserve the same speaker-disjoint split and settings. Never upload private recordings or checkpoints to a public share.

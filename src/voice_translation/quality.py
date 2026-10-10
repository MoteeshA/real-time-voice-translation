"""Local, bounded speaker/pitch comparison; no exact-match certification."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf
import librosa
from .synthesizer import TTS_LOCK

SPEAKER_MODEL='microsoft/wavlm-base-plus-sv'
SPEAKER_REVISION='feb593a6c23c1cc3d9510425c29b0a14d2b07b1e'
ANALYSIS_SECONDS=15


def load_comparison_audio(path):
    if not path or not Path(path).is_file():
        raise ValueError('Provide both a reference and generated recording.')
    info=sf.info(path)
    if info.duration<1.5:
        raise ValueError('Use at least 1.5 seconds per clip for this preliminary comparison.')
    y,sr=sf.read(path,frames=min(info.frames,int(ANALYSIS_SECONDS*info.samplerate)),dtype='float32')
    y=y.mean(axis=1) if y.ndim>1 else y
    if not y.size or not np.isfinite(y).all() or np.max(np.abs(y))<.001:
        raise ValueError('The comparison recording is silent or contains invalid samples.')
    x=librosa.resample(y,orig_sr=sr,target_sr=16000)
    return x,{'duration_seconds':info.duration,'analysed_seconds':len(y)/sr,
              'sample_rate':sr,'hard_clip_fraction':float(np.mean(np.abs(y)>=.9999)),
              'rms_dbfs':float(20*np.log10(max(np.sqrt(np.mean(y*y)),1e-12)))}


@lru_cache(maxsize=1)
def _speaker_verifier():
    from transformers import Wav2Vec2FeatureExtractor,WavLMForXVector
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(SPEAKER_MODEL,revision=SPEAKER_REVISION)
    model=WavLMForXVector.from_pretrained(SPEAKER_MODEL,revision=SPEAKER_REVISION).eval()
    return extractor,model


def pitch_summary(audio):
    f0,voiced,_=librosa.pyin(audio,fmin=65,fmax=500,sr=16000,frame_length=1024,hop_length=160)
    values=f0[voiced & np.isfinite(f0)]
    return {'median_hz':float(np.median(values)) if len(values) else None,
            'spread_semitones':float(np.std(12*np.log2(values))) if len(values) else None}


def compare_voice(reference_path,generated_path):
    # Validate before downloading or loading the independent verifier.
    reference,ref_info=load_comparison_audio(reference_path)
    generated,gen_info=load_comparison_audio(generated_path)
    import torch
    with TTS_LOCK:
        extractor,model=_speaker_verifier()
        embeddings=[]
        for audio in (reference,generated):
            inputs=extractor(audio,sampling_rate=16000,return_tensors='pt')
            with torch.inference_mode():
                embeddings.append(torch.nn.functional.normalize(model(**inputs).embeddings,dim=-1))
        score=float(torch.sum(embeddings[0]*embeddings[1]).item())
        if not np.isfinite(score):
            raise RuntimeError('The speaker verifier returned an invalid score.')
        score=float(np.clip(score,-1.,1.))
        ref_info['pitch']=pitch_summary(reference);gen_info['pitch']=pitch_summary(generated)
    return {'speaker_cosine':score,'reference':ref_info,'generated':gen_info,
            'model':SPEAKER_MODEL,'revision':SPEAKER_REVISION,
            'reference_sha256':hashlib.file_digest(Path(reference_path).open('rb'),'sha256').hexdigest(),
            'generated_sha256':hashlib.file_digest(Path(generated_path).open('rb'),'sha256').hexdigest(),
            'limitations':['Cosine is not a percentage or an exact-match verdict.',
                           'Pitch and timing require matching spoken text for a fair comparison.',
                           'Accent, emotion and naturalness are not certified by this check.',
                           'Only the first 15 seconds are analysed; short clips and cross-language comparisons are less reliable.'],
            'recordings_uploaded':False}


def format_comparison(report):
    def number(value):return 'Not detected' if value is None else f'{value:.1f}'
    r,g=report['reference'],report['generated']
    return (f"**Independent speaker cosine: {report['speaker_cosine']:.3f}** (not a percentage or exact-match verdict).\n\n"
            '| Measurement | Reference | Generated |\n|---|---:|---:|\n'
            f"| Clip duration (seconds) | {r['duration_seconds']:.2f} | {g['duration_seconds']:.2f} |\n"
            f"| Median voiced pitch (Hz) | {number(r['pitch']['median_hz'])} | {number(g['pitch']['median_hz'])} |\n"
            f"| Pitch spread (semitones) | {number(r['pitch']['spread_semitones'])} | {number(g['pitch']['spread_semitones'])} |\n"
            f"| Hard-clipped samples | {100*r['hard_clip_fraction']:.3f}% | {100*g['hard_clip_fraction']:.3f}% |\n\n"
            'Processing is local. Only the first 15 seconds are analysed. Compare the same sentence for pitch and timing; listen separately for accent, expressive delivery and an AI-like feel.')

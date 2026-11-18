from __future__ import annotations
import os
import io
import copy
import traceback
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import requests
import joblib
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import models, transforms
from transformers import (
    AutoTokenizer, AutoModel,
    Wav2Vec2Processor, Wav2Vec2Model
)
import subprocess
import tempfile
import os
import io
import numpy as np
import soundfile as sf
from torchaudio.transforms import MFCC
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from pedagogical_strategy import select_strategy
except ImportError:
    select_strategy = None


# ----------------------------- Paths/config -----------------------------

ROOT = Path(os.getenv("MELD_PROJECT_ROOT", r"D:\MELD_Project\repo_style_final"))

ATTENTION_CKPT = ROOT / "results" / "github_clean_new" / "checkpoint.pt"
RELIABILITY_CKPT = ROOT / "results" / "reliability_aware_new" / "reliability_checkpoint_new.pt"
TEMPORAL_CKPT = ROOT / "results" / "temporal_svm_tuned_github" / "temporal_checkpoint.pt"
SVM_PATH = ROOT / "results" / "temporal_class_corrected" / "svm_temporal.joblib"

# Used to reproduce the temporal feature scaling selected by your training script.
BASE_EMBEDDINGS = ROOT / "results" / "reliability_aware_balanced_fix" / "reliability_embeddings.npz"
TEMPORAL_EMBEDDINGS = ROOT / "results" / "temporal_svm_tuned_github" / "temporal_embeddings.npz"

TEXT_MODEL_ID = "FacebookAI/roberta-base"
AUDIO_MODEL_ID = "facebook/wav2vec2-base-960h"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")

EMOTIONS = ("anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear")
CLASS_BIAS = np.array([-0.08, 0.0, 0.08, 0.03, 0.0, 0.0, 0.0], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet50/LSTM feature extractor settings from Video_vector(3).py
NUM_VIDEO_FRAMES = 10
VIDEO_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ----------------------------- Model classes ----------------------------

class EmotionAnalysisModel(nn.Module):
    """Same architecture as Multihead_attention_NN_GitHub_clean(6).py."""
    def __init__(self):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.rand(512))
        self.alpha2 = nn.Parameter(torch.rand(512))
        self.alpha3 = nn.Parameter(torch.rand(512))
        self.bilstm_voice = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.bilstm_text = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.bilstm_video = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.multihead_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.project_V3 = nn.Linear(25, 512)
        self.project_V4 = nn.Linear(25, 512)
        self.project_V2 = nn.Linear(768, 512)

    def forward(self, V1, V2, V3, V4, A2, extract=False):
        voice = V1 * self.alpha1 + self.project_V3(V3) * self.alpha2 + self.project_V4(V4) * self.alpha3
        F_voice, _ = self.bilstm_voice(voice.unsqueeze(1))
        F_text, _ = self.bilstm_text(self.project_V2(V2).unsqueeze(1))
        F_video, _ = self.bilstm_video(A2.unsqueeze(1))
        tokens = torch.cat((F_voice, F_text, F_video), dim=1)
        attended, _ = self.multihead_attention(tokens, tokens, tokens)
        fused = attended[:, -1, :]
        return fused if extract else self.fc(fused)


class ReliabilityHead(nn.Module):
    """Same architecture as Reliability_Aware_Fusion(4).py."""
    def __init__(self, pretrained_fc):
        super().__init__()
        self.scorers = nn.ModuleList([nn.Linear(512, 1) for _ in range(3)])
        for layer in self.scorers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.fc = copy.deepcopy(pretrained_fc)
        self.fc.requires_grad_(True)
        self.pool_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens, attention):
        scores = torch.cat([layer(tokens[:, i, :]) for i, layer in enumerate(self.scorers)], dim=1)
        weights = torch.softmax(scores, dim=1)
        weighted = tokens * (3.0 * weights.unsqueeze(-1))
        attended, _ = attention(weighted, weighted, weighted, need_weights=False)
        original_fused = attended[:, -1, :]
        context = (attended * weights.unsqueeze(-1)).sum(dim=1)
        fused = original_fused + torch.tanh(self.pool_strength) * context
        return self.fc(fused), fused, weights


class TemporalGRU(nn.Module):
    """Same architecture as Temporal_Emotion_Model_final(6).py."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.strength = nn.Parameter(torch.tensor(0.1))
        self.fc = nn.Linear(512, 7)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        result, _ = self.gru(packed)
        context, _ = pad_packed_sequence(result, batch_first=True, total_length=x.shape[1])
        fused = x + self.strength * context
        return self.fc(fused), fused


class VideoLSTM(nn.Module):
    """Exact feature-extractor LSTM shape from Video_vector(3).py."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, frame_vectors):
        output, _ = self.lstm(frame_vectors)
        return self.fc(output[:, -1, :])


# ----------------------------- Load checkpoints -------------------------

def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required model file not found: {path}")


def load_all_models():
    for p in (ATTENTION_CKPT, RELIABILITY_CKPT, TEMPORAL_CKPT, SVM_PATH,
              BASE_EMBEDDINGS, TEMPORAL_EMBEDDINGS):
        require_file(p)

    # The video feature script seeds the randomly initialized LSTM at 42.
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    attention_model = EmotionAnalysisModel().to(DEVICE)
    attention_state = torch.load(ATTENTION_CKPT, map_location=DEVICE, weights_only=False)
    attention_model.load_state_dict(attention_state["model"], strict=True)
    attention_model.eval()

    reliability = ReliabilityHead(attention_model.fc).to(DEVICE)
    reliability_state = torch.load(RELIABILITY_CKPT, map_location=DEVICE, weights_only=False)
    reliability.load_state_dict(reliability_state["model"], strict=True)
    reliability.eval()

    temporal = TemporalGRU().to(DEVICE)
    temporal_state = torch.load(TEMPORAL_CKPT, map_location=DEVICE, weights_only=False)
    if temporal_state.get("protocol") != "github":
        raise RuntimeError("Temporal checkpoint protocol is not 'github'.")
    temporal.load_state_dict(temporal_state["model"], strict=True)
    temporal.eval()

    svm = joblib.load(SVM_PATH)
    if int(getattr(svm, "n_features_in_", -1)) != 1024:
        raise RuntimeError("Saved SVM must accept exactly 1024 features.")
    if not hasattr(svm, "predict_proba"):
        raise RuntimeError("SVM checkpoint has no predict_proba; class confidence code cannot run.")

    # Recreate gain=.25 feature scaling using the training script's fit indices.
    with np.load(BASE_EMBEDDINGS, allow_pickle=False) as z:
        base_all = z["embeddings"].astype(np.float32)
        y = z["labels"].astype(np.int64)
        ids = z["sample_ids"].astype(str)
    with np.load(TEMPORAL_EMBEDDINGS, allow_pickle=False) as z:
        temporal_all = z["embeddings"].astype(np.float32)
        temporal_ids = z["sample_ids"].astype(str)

    if base_all.shape != temporal_all.shape or base_all.shape[1] != 512:
        raise RuntimeError(f"Embedding shape mismatch: {base_all.shape} vs {temporal_all.shape}")
    if not np.array_equal(ids, temporal_ids):
        raise RuntimeError("Base and temporal embedding sample IDs are not aligned.")

    # Reproduce github protocol split exactly (seed 42, test_size=.2,
    # then validation=.15 stratified on the remaining rows).
    from sklearn.model_selection import train_test_split
    all_rows = np.arange(len(y))
    train_rows, _test_rows = train_test_split(all_rows, test_size=.2, random_state=42)
    fit_rows, _val_rows = train_test_split(
        train_rows, test_size=.15, stratify=y[train_rows], random_state=42
    )
    delta = temporal_all - base_all
    base_scale = float(np.sqrt(np.mean(np.var(base_all[fit_rows], axis=0, dtype=np.float64))))
    delta_scale = float(np.sqrt(np.mean(np.var(delta[fit_rows], axis=0, dtype=np.float64))))
    temporal_scale = min(base_scale / max(delta_scale, 1e-8), 100.0)

    return attention_model, reliability, temporal, svm, temporal_scale


# ----------------------------- Feature extraction -----------------------

@torch.inference_mode()
def extract_text(text: str, tokenizer, model) -> np.ndarray:
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    vec = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    result = vec[0].float().cpu().numpy()
    if result.shape != (768,):
        raise RuntimeError(f"Expected RoBERTa V2=768, got {result.shape}")
    return result


def decode_audio(data: bytes, filename="audio.m4a"):
    if not data:
        raise ValueError("Uploaded audio is empty")

    temp_path = None

    try:
        suffix = Path(filename).suffix or ".m4a"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:
            temp.write(data)
            temp_path = temp.name

        result = subprocess.run(
            [
                "ffmpeg",
                "-v", "error",
                "-i", temp_path,
                "-f", "wav",
                "-ac", "1",
                "-ar", "16000",
                "pipe:1"
            ],
            capture_output=True,
            check=True,
            timeout=120
        )

        waveform, sample_rate = sf.read(
            io.BytesIO(result.stdout),
            dtype="float32"
        )

        if sample_rate != 16000:
            raise ValueError(f"Unexpected sample rate: {sample_rate}")

        if waveform.ndim != 1:
            raise ValueError("Audio is not mono")

        if waveform.size == 0:
            raise ValueError("Decoded audio is empty")

        print(
            f"Audio decoded successfully: "
            f"{len(waveform)} samples, {sample_rate} Hz"
        )

        return waveform

    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"FFmpeg audio decoding failed: "
            f"{e.stderr.decode(errors='replace')}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@torch.inference_mode()
def extract_audio(y, processor, wav2vec, mfcc_transform):
    # Match Audio_vector_github(4).py: V1 = mean hidden-state then first 512 dims.
    encoded = processor(y, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    hidden = wav2vec(encoded).last_hidden_state
    v1 = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()[:512]

    mfcc = mfcc_transform(torch.from_numpy(y).unsqueeze(0))
    v3 = mfcc.mean(dim=2).squeeze(0).cpu().numpy().astype(np.float32)[:25]

    y22 = librosa.resample(y, orig_sr=16000, target_sr=22050)
    sr = 22050
    centroid = float(librosa.feature.spectral_centroid(y=y22, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=y22).mean())
    rms = float(librosa.feature.rms(y=y22).mean())
    tempo, _ = librosa.beat.beat_track(y=y22, sr=sr)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    pitches, _ = librosa.piptrack(y=y22, sr=sr)
    positive = pitches[pitches > 0]
    pitch = float(positive.mean()) if positive.size else 0.0
    v4 = np.tile(np.array([centroid, zcr, rms, tempo_value, pitch], dtype=np.float32), 5)[:25]

    for name, vec, dim in (("V1", v1, 512), ("V3", v3, 25), ("V4", v4, 25)):
        if vec.shape != (dim,) or not np.isfinite(vec).all():
            raise APIError(422, f"Audio feature extraction produced invalid {name}.")
    return v1, v3, v4


def decode_video(data: bytes, suffix: str = ".mp4"):
    # OpenCV VideoCapture requires a path; write a temporary file and remove it.
    import tempfile
    suffix = suffix if suffix in {".mp4", ".mov", ".avi", ".mkv"} else ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise APIError(400, "Cannot open video file.")
        frames = []
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count > 0:
                indices = np.linspace(0, count - 1, NUM_VIDEO_FRAMES, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frames.append(frame)
            else:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if frames:
                    indices = np.linspace(0, len(frames)-1, NUM_VIDEO_FRAMES, dtype=int)
                    frames = [frames[i] for i in indices]
        finally:
            cap.release()
        if not frames:
            raise APIError(400, "No decodable video frames.")
        while len(frames) < NUM_VIDEO_FRAMES:
            frames.append(frames[-1])
        return frames[:NUM_VIDEO_FRAMES]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@torch.inference_mode()
def extract_video(data: bytes, backbone, video_lstm, suffix: str = ".mp4"):
    frames = decode_video(data, suffix)
    batch = torch.stack([VIDEO_PREPROCESS(frame) for frame in frames]).to(DEVICE)
    frame_vectors = backbone(batch).flatten(1)  # [10, 2048]
    vec = video_lstm(frame_vectors.unsqueeze(0))[0].float().cpu().numpy()
    if vec.shape != (512,) or not np.isfinite(vec).all():
        raise APIError(422, "Video feature extraction produced invalid A2.")
    return vec


# ----------------------------- Model inference --------------------------

@torch.inference_mode()
def infer_emotion(text, audio_bytes, video_bytes, history, tokenizer, text_model,
                  audio_processor, wav2vec, mfcc_transform, video_backbone, video_lstm,
                  attention_model, reliability_model, temporal_model, svm, temporal_scale,
                  audio_filename="upload.m4a",video_filename= "upload.mp4"):
    V2 = extract_text(text, tokenizer, text_model)

    if audio_bytes:
        waveform = decode_audio(audio_bytes, audio_filename)

        V1, V3, V4 = extract_audio(
            waveform,
            audio_processor,
            wav2vec,
            mfcc_transform )
    else:
        V1 = np.zeros(512, dtype=np.float32)
        V3 = np.zeros(25, dtype=np.float32)
        V4 = np.zeros(25, dtype=np.float32)

    if video_bytes:
        A2 = extract_video(video_bytes, video_backbone, video_lstm, Path(video_filename).suffix.lower())
    else:
        A2 = np.zeros(512, dtype=np.float32)

    inputs = [
        torch.from_numpy(V1).unsqueeze(0).to(DEVICE),
        torch.from_numpy(V2).unsqueeze(0).to(DEVICE),
        torch.from_numpy(V3).unsqueeze(0).to(DEVICE),
        torch.from_numpy(V4).unsqueeze(0).to(DEVICE),
        torch.from_numpy(A2).unsqueeze(0).to(DEVICE),
    ]

    # Multihead model generates the modality tokens, preserving original token order:
    # audio, text, video. The trained attention module is then used inside reliability fusion.
    v1, v2, v3, v4, a2 = inputs
    voice = v1 * attention_model.alpha1 + attention_model.project_V3(v3) * attention_model.alpha2 + attention_model.project_V4(v4) * attention_model.alpha3
    audio_token, _ = attention_model.bilstm_voice(voice.unsqueeze(1))
    text_token, _ = attention_model.bilstm_text(attention_model.project_V2(v2).unsqueeze(1))
    video_token, _ = attention_model.bilstm_video(a2.unsqueeze(1))
    tokens = torch.cat([audio_token, text_token, video_token], dim=1)

    # Explicitly run trained Multihead Attention; ReliabilityHead also uses it internally,
    # exactly as in Reliability_Aware_Fusion.py.
    attended_tokens, _ = attention_model.multihead_attention(tokens, tokens, tokens, need_weights=False)
    _, reliability_fused, reliability_weights = reliability_model(tokens, attention_model.multihead_attention)

    current_base = reliability_fused[0].cpu().numpy().astype(np.float32)

    # Build the temporal sequence from the student's preceding session vectors.
    # Store reliability embeddings (not temporal embeddings) as GRU inputs.
    sequence = [np.asarray(v, dtype=np.float32) for v in history[-30:]] + [current_base]
    seq_tensor = torch.from_numpy(np.stack(sequence)).unsqueeze(0).to(DEVICE)
    length = torch.tensor([len(sequence)], dtype=torch.long)
    _, contextual = temporal_model(seq_tensor, length)
    current_context = contextual[0, -1].cpu().numpy().astype(np.float32)

    temporal_delta = current_context - current_base
    final_vector = np.concatenate([
        current_base,
        0.25 * temporal_scale * temporal_delta
    ]).astype(np.float32)

    if final_vector.shape != (1024,) or not np.isfinite(final_vector).all():
        raise APIError(422, "Final SVM vector must be 1024 finite values.")

    # Match final temporal model's probability + log-prior correction.
    probabilities = svm.predict_proba(final_vector.reshape(1, -1))[0]
    svm_classes = np.asarray(svm.classes_, dtype=int)
    scores = np.log(np.maximum(probabilities, 1e-12)) + CLASS_BIAS[svm_classes]
    pred_idx = int(np.argmax(scores))
    emotion = EMOTIONS[int(svm_classes[pred_idx])]

    shifted = scores - scores.max()
    adjusted = np.exp(shifted)
    adjusted /= adjusted.sum()
    confidence = float(adjusted[pred_idx])
    sorted_scores = np.sort(adjusted)
    margin = float(sorted_scores[-1] - sorted_scores[-2])
    needs_confirmation = confidence < 0.60 or margin < 0.15

    return {
        "emotion": emotion,
        "confidence": confidence,
        "needs_confirmation": needs_confirmation,
        "reliability_weights": reliability_weights[0].cpu().numpy().tolist(),
        "reliability_embedding": current_base,
        "warning": None if audio_bytes and video_bytes else (
            "One or more modalities were missing and zero-filled. "
            "This inference mode was not validated by the all-modality MELD training pipeline."
        ),
    }


# ----------------------------- Strategy + Ollama ------------------------

def choose_strategy(emotion, needs_confirmation, temporal_state):
    if select_strategy is not None:
        return select_strategy(
            emotion=emotion,
            needs_confirmation_heuristic=needs_confirmation,
            temporal_state=temporal_state
        )
    return {
        "strategy": "Gentle check-in",
        "instructions": "Ask what the learner finds difficult; do not assert an emotion.",
        "reason": "strategy_module_unavailable"
    }



def generate_answer(question, emotion, confidence, needs_confirmation, temporal_state):
    strategy = choose_strategy(emotion, needs_confirmation, temporal_state)

    emotion_signal = (
        emotion
        if not needs_confirmation
        else "uncertain; do not assume"
    )

    
    system_prompt = """
You are EmpathIQ, an adaptive educational assistant.

Your primary responsibility is to answer the student's actual question
clearly and accurately.

STRICT RESPONSE RULES:
1. ALWAYS answer the student's question directly first.
2. Never begin with a greeting, emotional check-in, or question asking
   what the student finds confusing.
3. Do not ask the student to explain their question again if it is clear.
4. Explain the answer in simple, student-friendly language.
5. Use examples when they help understanding.
6. Adapt the explanation's tone, complexity, and pacing using the
   internal emotion and temporal-state information.
7. Never reveal emotion labels, confidence scores, model predictions,
   or internal temporal-state information to the student.
8. Do not assume the student is confused, sad, frustrated, or happy.
9. Ask a follow-up question only when the original question is genuinely
   ambiguous or when a question would help deepen understanding.
10. Avoid generic motivational messages and unnecessary check-ins.

ADAPTIVE TEACHING:
- Confusion: explain in smaller steps and clarify difficult concepts.
- Frustration: be calm, concise, and focus on the immediate problem.
- Sadness or fear: use patient, supportive language without assuming
  the student feels that way.
- Joy or surprise: encourage exploration and deeper understanding.
- Neutral or uncertain state: provide a direct academic explanation.
- Persistent learning difficulty: try a different explanation or example.

The educational question always takes priority over emotional adaptation.

Return only the response intended for the student.
RESPONSE LENGTH:
- Keep answers concise, normally 100-500 words.
- Use numbered steps only when necessary.
- Use short paragraphs.
- Avoid repeating the same explanation.
- Do not end with a generic question such as
  "What part do you find difficult?"
- End with the explanation unless a follow-up is genuinely necessary.
"""

    user_prompt = f"""
Student's question:
{question}

Internal teaching context:
Emotion signal: {emotion_signal}
Temporal learning state: {temporal_state}
Teaching strategy: {strategy.get('strategy', 'direct explanation')}
Teaching instructions: {strategy.get('instructions', '')}

Answer the student's question directly.
Use the internal context only to adapt your teaching style.
Do not mention the internal context or emotion predictions.
"""


    # Keep your existing Ollama request and response code below this point.

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "options": {"temperature": 0.4}
            },
            timeout=180
        )

        r.raise_for_status()

        answer = r.json().get("message", {}).get("content", "").strip()

        if not answer:
            raise APIError(502, "Ollama returned an empty answer.")

    except requests.exceptions.ConnectionError:
        raise APIError(503, "Ollama is unavailable. Start Ollama first.")

    except requests.exceptions.Timeout:
        raise APIError(504, "Ollama timed out.")

    except requests.exceptions.RequestException as exc:
        raise APIError(502, f"Ollama request failed: {exc}")
    return answer, strategy


# ----------------------------- Flask setup ------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150 MB upload limit

class APIError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(str(detail))
        self.status_code = int(status_code)
        self.detail = str(detail)

@app.errorhandler(APIError)
def handle_api_error(exc):
    return jsonify({"detail": exc.detail}), exc.status_code

@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    traceback.print_exc()
    return jsonify({"detail": f"{type(exc).__name__}: {exc}"}), 500

@app.after_request
def add_cors_headers(response):
    # Supports a frontend opened via VS Code Live Server.
    origin = request.headers.get("Origin")
    if origin and (origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

models_loaded = False
load_error = None
session_history: dict[str, list[np.ndarray]] = {}

def initialize_models():
    global models_loaded, load_error
    global attention_model, reliability_model, temporal_model, svm_model, temporal_scale
    global tokenizer, text_model, audio_processor, wav2vec, mfcc_transform
    global video_backbone, video_lstm

    try:
        (attention_model, reliability_model, temporal_model,
         svm_model, temporal_scale) = load_all_models()

        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        text_model = AutoModel.from_pretrained(TEXT_MODEL_ID, add_pooling_layer=False).to(DEVICE).eval()

        audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_ID)
        wav2vec = Wav2Vec2Model.from_pretrained(AUDIO_MODEL_ID,ignore_mismatched_sizes=False).to(DEVICE).eval()
        mfcc_transform = MFCC(sample_rate=16000, n_mfcc=25)

        video_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        video_backbone = nn.Sequential(*list(video_backbone.children())[:-1]).to(DEVICE).eval()

        # Video_vector uses a seed-42 initialized LSTM feature extractor.
        torch.manual_seed(42)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(42)
        video_lstm = VideoLSTM().to(DEVICE).eval()

        models_loaded = True
        load_error = None
        print(f"MELD Flask backend ready on {DEVICE}", flush=True)
    except Exception as exc:
        models_loaded = False
        load_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print("MELD backend startup failed:", load_error, flush=True)

# Load models once when Flask process starts.
initialize_models()

def truthy_form(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "MELD Adaptive Learning Backend (Flask)",
        "status": "ready" if models_loaded else "model_load_failed",
        "device": str(DEVICE),
        "ollama_model": OLLAMA_MODEL,
        "models_loaded": models_loaded,
        "load_error": load_error,
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.ok
    except requests.RequestException:
        ollama_ok = False
    return jsonify({
        "backend": "ok",
        "models_loaded": models_loaded,
        "model_load_error": load_error,
        "ollama_available": ollama_ok,
        "device": str(DEVICE),
    })


@app.route("/respond", methods=["POST", "OPTIONS"])
def respond():
    if request.method == "OPTIONS":
        return ("", 204)

    if not models_loaded:
        raise APIError(503, f"Model startup failed: {load_error}")

    # Text is the academic question; audio/video are optional
    # emotion-recognition inputs.
    question = str(request.form.get("question", "")).strip()

    if not question:
        raise APIError(
            400,
            "Please enter your academic question. Audio/video can be optional."
        )

    if len(question) > 5000:
        raise APIError(400, "Question exceeds 5000 characters.")

    session_id = str(
        request.form.get("session_id", "default")
    ).strip() or "default"

    if len(session_id) > 100:
        raise APIError(400, "Session ID exceeds 100 characters.")

    audio_file = request.files.get("audio")
    video_file = request.files.get("video")

    audio_bytes = None
    video_bytes = None

    audio_filename = "upload.m4a"
    video_filename = "upload.mp4"

    # ---------------- Audio upload ----------------
    if audio_file and audio_file.filename:
        if not truthy_form(request.form.get("audio_consent")):
            raise APIError(
                400,
                "Audio consent is required before processing."
            )

        audio_filename = audio_file.filename

        suffix = Path(audio_filename).suffix.lower()

        if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
            raise APIError(
                400,
                "Audio type must be WAV, MP3, M4A, OGG, or FLAC."
            )

        audio_bytes = audio_file.read()

        if not audio_bytes:
            raise APIError(400, "Audio upload is empty.")

    # ---------------- Video upload ----------------
    if video_file and video_file.filename:
        if not truthy_form(request.form.get("video_consent")):
            raise APIError(
                400,
                "Video consent is required before processing."
            )

        video_filename = video_file.filename

        suffix = Path(video_filename).suffix.lower()

        if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
            raise APIError(
                400,
                "Video type must be MP4, MOV, AVI, or MKV."
            )

        video_bytes = video_file.read()

        if not video_bytes:
            raise APIError(400, "Video upload is empty.")

    # Require at least one emotion modality.
    if audio_bytes is None and video_bytes is None:
        raise APIError(
            400,
            "Upload audio, video, or both for emotion recognition."
        )

    history = session_history.get(session_id, [])

    # ---------------- MELD emotion inference ----------------
    result = infer_emotion(
        question,
        audio_bytes,
        video_bytes,
        history,
        tokenizer,
        text_model,
        audio_processor,
        wav2vec,
        mfcc_transform,
        video_backbone,
        video_lstm,
        attention_model,
        reliability_model,
        temporal_model,
        svm_model,
        temporal_scale,
        audio_filename=audio_filename,
        video_filename=video_filename
    )

    history.append(result.pop("reliability_embedding"))
    session_history[session_id] = history[-30:]

    temporal_state = (
        f"session_sequence_length={len(session_history[session_id])}; "
        "trajectory is provisional and not a validated learning diagnosis"
    )

    # ---------------- Ollama adaptive response ----------------
    answer, strategy = generate_answer(
        question,
        result["emotion"],
        result["confidence"],
        result["needs_confirmation"],
        temporal_state
    )

    return jsonify({
        "answer": answer,

        # Actual backend emotion output
        "emotion": result["emotion"],

        "confidence_score": result["confidence"],
        "confidence_is_calibrated": False,
        "needs_confirmation": result["needs_confirmation"],

        "reliability_weights": result["reliability_weights"],

        "strategy": strategy.get("strategy"),
        "strategy_instructions": strategy.get("instructions"),
        "strategy_reason": strategy.get("reason"),

        "temporal_state": temporal_state,
        "warning": result["warning"],

        "emotion_source": "trained_multimodal_models_and_temporal_svm",
        "session_id": session_id
    })


@app.route("/reset-session", methods=["POST", "OPTIONS"])
def reset_session():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    if not session_id:
        raise APIError(400, "session_id is required.")
    session_history.pop(session_id, None)
    return jsonify({"status": "cleared", "session_id": session_id})

if __name__ == "__main__":
    # Run with: python backend.py
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)

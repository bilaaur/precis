import io
import os

import pytextrank  # noqa: F401  (registers the "textrank" spaCy component)
import spacy
import speech_recognition as sr
from flask import Flask, jsonify, render_template, request
from rouge_score import rouge_scorer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads

# Loaded once at startup, not per request.
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Pegasus is optional: ~2 GB model plus torch, so it only loads when ENABLE_PEGASUS=1.
PEGASUS_ENABLED = os.getenv("ENABLE_PEGASUS") == "1"
PEGASUS_MODEL = "google/pegasus-xsum"
_pegasus = {}
_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def pegasus_summary(text):
    """Abstractive summary. Gets the raw transcript, not lowercased or lemmatized text."""
    if not _pegasus:  # lazy load on first request
        import torch
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer

        _pegasus["torch"] = torch
        _pegasus["tok"] = PegasusTokenizer.from_pretrained(PEGASUS_MODEL)
        _pegasus["model"] = PegasusForConditionalGeneration.from_pretrained(PEGASUS_MODEL).eval()
    tok, model, torch = _pegasus["tok"], _pegasus["model"], _pegasus["torch"]
    truncated = len(tok(text, truncation=False)["input_ids"]) > tok.model_max_length
    batch = tok(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**batch, min_length=10, max_length=64, num_beams=4)
    return tok.decode(ids[0], skip_special_tokens=True), truncated


def rouge(reference, summary):
    scores = _scorer.score(reference, summary)  # (target, prediction)
    return {k: {"p": round(v.precision, 3), "r": round(v.recall, 3), "f": round(v.fmeasure, 3)}
            for k, v in scores.items()}


@app.get("/")
def index():
    return render_template("index.html", pegasus=PEGASUS_ENABLED)


@app.post("/api/summarize")
def summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    try:
        n = max(1, min(int(data.get("sentences", 3)), 10))
    except (TypeError, ValueError):
        n = 3

    if len(text.split()) < 20:
        return jsonify(error="Add at least 20 words before summarizing."), 400

    method = data.get("method", "textrank")
    reference = (data.get("reference") or "").strip()
    if method == "pegasus" and not PEGASUS_ENABLED:
        return jsonify(error="Pegasus is not enabled on this server."), 400

    doc = nlp(text[:30000])
    truncated = False
    if method == "pegasus":
        try:
            summary, truncated = pegasus_summary(text)
        except ImportError:
            return jsonify(error="Pegasus needs: pip install -r requirements-pegasus.txt"), 500
    else:
        sentences = [s.text.strip() for s in doc._.textrank.summary(limit_phrases=15, limit_sentences=n)]
        summary = " ".join(sentences)

    return jsonify(
        summary=summary,
        method=method,
        truncated=truncated,
        keywords=[p.text for p in doc._.phrases[:8]],
        words_in=len(text.split()),
        words_out=len(summary.split()),
        rouge=rouge(reference, summary) if reference else None,
    )


@app.post("/api/transcribe")
def transcribe():
    f = request.files.get("audio")
    if not f:
        return jsonify(error="No audio file received."), 400

    lang = request.form.get("language", "en-US")
    if lang not in ("en-US", "id-ID"):
        lang = "en-US"
    rec = sr.Recognizer()
    parts = []
    try:
        with sr.AudioFile(io.BytesIO(f.read())) as source:
            # Google's free endpoint is limited per request, so send ~50 s chunks.
            for _ in range(12):  # up to about 10 minutes
                audio = rec.record(source, duration=50)
                if not audio.frame_data:
                    break
                try:
                    parts.append(rec.recognize_google(audio, language=lang))
                except sr.UnknownValueError:
                    continue
    except ValueError:
        return jsonify(error="Unsupported audio. Use a WAV, FLAC, or AIFF file."), 400
    except sr.RequestError:
        return jsonify(error="Google Speech Recognition is unreachable. Try again shortly."), 502

    if not parts:
        return jsonify(error="Google could not recognise any speech. Check that the language setting matches the recording and that the speech is clear."), 422
    return jsonify(text=" ".join(parts))


if __name__ == "__main__":
    app.run(debug=True)
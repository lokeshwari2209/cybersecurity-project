"""
main.py
=======
QRShield FastAPI backend.
Provides explainable pre-navigation QR and URL phishing detection.

Start with:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import io
import os
import sys
import json
import logging
import asyncio
import math
import threading
from importlib import metadata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).parent
ROOT         = BACKEND_DIR.parent
FRONTEND_DIR = ROOT / 'frontend'
MODEL_DIR    = ROOT / 'models'
RESULT_DIR   = ROOT / 'results'
TRAINING_DIR = ROOT / 'training'
CNN_MODEL_PATH = MODEL_DIR / 'QR_Phishing_Residual_CNN.keras'
CNN_THRESHOLD = 0.05
CNN_STATIC_DEMO_OUTPUT = {
    'image_model': 'Residual CNN',
    'image_model_prediction': 'Phishing',
    'image_model_confidence': None,
    'image_model_experimental': True,
    'image_model_demo': True,
    'image_model_demo_label': 'Demo Mode — not a real CNN prediction',
}

sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from url_features import (
    extract_url_features, FEATURE_NAMES, features_to_vector,
    normalize_url, validate_url_syntax
)
from risk_engine import assess_risk, explain_url
from qr_decoder  import decode_qr_from_bytes, detect_payload_type, get_analyzable_url

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Upload Security Constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB streaming limit
MAX_DIMENSION = 4096              # 4096 px width and height limit
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'
}

# ── Startup: load models ──────────────────────────────────────────────────────
url_model        = None
image_model      = None
image_models     = {}
feature_names    = None
MODEL_LOAD_ERROR = None
CNN_REMOTE_URL = os.environ.get('QRSHIELD_CNN_REMOTE_URL', '').rstrip('/')
CNN_REMOTE_TOKEN = os.environ.get('QRSHIELD_CNN_REMOTE_TOKEN', '')
CNN_REMOTE_ENABLED = os.environ.get('QRSHIELD_CNN_REMOTE_ENABLED', '').lower() == 'true'
CNN_PREDICTION_BLOCK_REASON = (
    'Residual CNN requires the compatible local TensorFlow/Keras runtime.'
)
cnn_model = None
CNN_LOAD_ERROR = None
CNN_LOAD_LOCK = threading.Lock()


def load_models():
    global url_model, image_model, image_models, feature_names, MODEL_LOAD_ERROR
    try:
        url_model_path = MODEL_DIR / 'url_model.pkl'
        if url_model_path.exists():
            url_model = joblib.load(str(url_model_path))
            logger.info("URL model loaded successfully.")
        else:
            logger.warning("URL model not found in models/ directory.")

        img_model_path = MODEL_DIR / 'image_model.pkl'
        if img_model_path.exists():
            image_model = joblib.load(str(img_model_path))
            logger.info("Image model loaded successfully.")
        else:
            logger.warning("Image model not found in models/ directory.")

        extra_trees_path = MODEL_DIR / 'extra_trees_image.pkl'
        if image_model is not None and image_model.__class__.__name__ == 'ExtraTreesClassifier':
            image_models['extra_trees'] = image_model
            logger.info("Extra Trees image model loaded successfully.")
        elif extra_trees_path.exists():
            image_models['extra_trees'] = joblib.load(str(extra_trees_path))
            logger.info("Extra Trees image model loaded successfully.")
        elif image_model is not None:
            image_models['extra_trees'] = image_model
            logger.warning("Extra Trees image artifact missing; using selected image model.")

        random_forest_path = MODEL_DIR / 'random_forest_image.pkl'
        if random_forest_path.exists():
            image_models['random_forest'] = joblib.load(str(random_forest_path))
            logger.info("Random Forest image model loaded successfully.")
        else:
            logger.warning("Random Forest image model not found in models/ directory.")

        fn_path = MODEL_DIR / 'feature_names.pkl'
        if fn_path.exists():
            feature_names = joblib.load(str(fn_path))
        else:
            feature_names = FEATURE_NAMES

    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        logger.error(f"Error loading models: {e}")


load_models()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title='QRShield API',
    description='Explainable Pre-Navigation QR Code Phishing Detection System',
    version='1.1.0',
)

# Restrict CORS to localhost demonstration environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

# Serve evaluation charts
if RESULT_DIR.exists():
    app.mount('/results', StaticFiles(directory=str(RESULT_DIR)), name='results')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Hardened Image Validation
# ─────────────────────────────────────────────────────────────────────────────
async def validate_and_read_image_stream(file: UploadFile) -> bytes:
    """
    Hardens image upload handling:
    1. Enforces streaming size limit (max 10MB) to prevent memory exhaustion.
    2. Validates magic bytes against supported image signatures.
    3. Uses PIL verify() to catch corrupted files.
    4. Enforces dimension limits (max 4096x4096px).
    """
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file MIME type '{file.content_type}'. Allowed types: JPEG, PNG, GIF, WebP, BMP."
        )

    chunks = []
    total_bytes = 0
    chunk_size = 1024 * 512  # 512 KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed upload size of 10 MB ({total_bytes} bytes streamed)."
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    if len(raw_bytes) < 32:
        raise HTTPException(
            status_code=400,
            detail="File is too small to be a valid image."
        )

    # Magic byte signature check
    is_valid_magic = False
    if raw_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        is_valid_magic = True
    elif raw_bytes.startswith(b'\xff\xd8\xff'):
        is_valid_magic = True
    elif raw_bytes.startswith(b'GIF87a') or raw_bytes.startswith(b'GIF89a'):
        is_valid_magic = True
    elif raw_bytes.startswith(b'RIFF') and len(raw_bytes) >= 12 and raw_bytes[8:12] == b'WEBP':
        is_valid_magic = True
    elif raw_bytes.startswith(b'BM'):
        is_valid_magic = True

    if not is_valid_magic:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. File header does not match any supported image types (PNG, JPEG, GIF, WebP, BMP)."
        )

    # PIL verification and dimension constraints
    try:
        img_verify = Image.open(io.BytesIO(raw_bytes))
        img_verify.verify()

        # Re-open for dimension inspection
        img = Image.open(io.BytesIO(raw_bytes))
        width, height = img.size
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions (zero or negative).")
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({width}x{height}) exceed maximum allowed limit of {MAX_DIMENSION}x{MAX_DIMENSION} pixels."
            )
    except (IOError, SyntaxError) as e:
        raise HTTPException(status_code=400, detail=f"Corrupted or malformed image: {str(e)}")

    return raw_bytes


def prepare_classical_image_features(raw_bytes: bytes) -> np.ndarray:
    """Convert an upload to the classical model's grayscale, 69x69 input vector."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = source.convert('L')
            if image.size != (69, 69):
                image = image.resize((69, 69), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Could not preprocess image: {e}") from e

    if pixels.max(initial=0.0) > 1.0:
        pixels /= 255.0
    return pixels.reshape(1, -1)


def prepare_cnn_image_input(raw_bytes: bytes) -> np.ndarray:
    """Prepare an upload for the experimental CNN's grayscale 69x69 input."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = source.convert('L')
            if image.size != (69, 69):
                image = image.resize((69, 69), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not preprocess image for Residual CNN: {e}",
        ) from e

    pixels /= 255.0
    np.clip(pixels, 0.0, 1.0, out=pixels)
    return pixels[np.newaxis, ..., np.newaxis]


def get_cnn_local_status() -> Dict[str, Any]:
    if CNN_LOAD_ERROR:
        return {'ready': False, 'reason': CNN_LOAD_ERROR}
    if not CNN_MODEL_PATH.is_file():
        return {'ready': False, 'reason': 'Residual CNN model file was not found.'}

    expected_versions = {'tensorflow': '2.20.0', 'keras': '3.13.2'}
    installed_versions = {}
    for package, expected_version in expected_versions.items():
        try:
            installed_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            return {
                'ready': False,
                'reason': (
                    f"Local Residual CNN needs TensorFlow {expected_versions['tensorflow']} "
                    f"and Keras {expected_versions['keras']}; {package} is not installed."
                ),
            }

    mismatches = [
        f'{package} {version} (needs {expected_versions[package]})'
        for package, version in installed_versions.items()
        if version != expected_versions[package]
    ]
    if mismatches:
        return {
            'ready': False,
            'reason': 'Incompatible local Residual CNN runtime: ' + ', '.join(mismatches) + '.',
        }
    return {'ready': True, 'reason': None}


def load_cnn_model():
    global cnn_model, CNN_LOAD_ERROR
    if cnn_model is not None:
        return cnn_model

    with CNN_LOAD_LOCK:
        if cnn_model is not None:
            return cnn_model
        status = get_cnn_local_status()
        if not status['ready']:
            raise HTTPException(status_code=503, detail=status['reason'])
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(str(CNN_MODEL_PATH), compile=False)
            if model.input_shape != (None, 69, 69, 1) or model.output_shape != (None, 1):
                raise ValueError(
                    f"Unexpected CNN shapes: input={model.input_shape}, output={model.output_shape}."
                )
            cnn_model = model
        except Exception as e:
            CNN_LOAD_ERROR = (
                f"Residual CNN could not load with the configured TensorFlow/Keras runtime: {e}"
            )
            logger.exception("Could not load local Residual CNN.")
            raise HTTPException(status_code=503, detail=CNN_LOAD_ERROR) from e
    return cnn_model


def predict_local_cnn(raw_bytes: bytes) -> Dict[str, Any]:
    cnn_input = prepare_cnn_image_input(raw_bytes)
    model = load_cnn_model()
    try:
        output = np.asarray(model.predict(cnn_input, verbose=0), dtype=np.float32)
    except Exception as e:
        logger.exception("Local Residual CNN inference failed.")
        raise HTTPException(status_code=500, detail="Local Residual CNN inference failed.") from e

    if output.shape != (1, 1) or not np.isfinite(output).all():
        logger.error("Local Residual CNN returned an invalid sigmoid output shape or value.")
        raise HTTPException(status_code=500, detail="Local Residual CNN returned an invalid prediction.")

    phishing_probability = float(output[0, 0])
    if not 0.0 <= phishing_probability <= 1.0:
        logger.error("Local Residual CNN returned a value outside the sigmoid range.")
        raise HTTPException(status_code=500, detail="Local Residual CNN returned an invalid prediction.")

    prediction = 'Phishing' if phishing_probability >= CNN_THRESHOLD else 'Benign'
    confidence = phishing_probability if prediction == 'Phishing' else 1.0 - phishing_probability
    return {
        'image_model': 'Residual CNN',
        'image_model_prediction': prediction,
        'image_model_confidence': confidence,
        'image_model_experimental': True,
        'image_model_demo': False,
        'image_model_threshold': CNN_THRESHOLD,
    }


def predict_cnn_demo() -> Dict[str, Any]:
    """Return the clearly identified fixed demo output when inference is unavailable."""
    return dict(CNN_STATIC_DEMO_OUTPUT)


def predict_image_model(model_name: str, raw_bytes: bytes) -> Dict[str, Any]:
    if model_name == 'improved_cnn':
        raise HTTPException(status_code=503, detail=CNN_PREDICTION_BLOCK_REASON)

    model = image_models.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"The {model_name.replace('_', ' ').title()} image model is unavailable."
        )

    features = prepare_classical_image_features(raw_bytes)
    expected_features = getattr(model, 'n_features_in_', features.shape[1])
    if expected_features != features.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Image model expects {expected_features} features, but preprocessing "
                f"produced {features.shape[1]}."
            )
        )

    try:
        label = int(model.predict(features)[0])
        if label not in (0, 1):
            raise ValueError(f"Unexpected image model class label: {label}")
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        label_index = classes.index(label)
        confidence = float(probabilities[label_index])
    except Exception as e:
        logger.error("Inference error on %s image model: %s", model_name, e)
        raise HTTPException(status_code=500, detail="Image model inference failed.") from e

    return {
        'image_model': model_name,
        'image_model_prediction': 'Phishing' if label == 1 else 'Benign',
        'image_model_confidence': confidence,
    }


def _request_cnn_remote(path: str, method: str = 'GET', body: Optional[bytes] = None):
    headers = {'Authorization': f'Bearer {CNN_REMOTE_TOKEN}'} if CNN_REMOTE_TOKEN else {}
    if body is not None:
        headers['Content-Type'] = 'application/octet-stream'
    request = urllib.request.Request(
        f'{CNN_REMOTE_URL}{path}',
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode('utf-8'))


async def get_cnn_remote_status() -> Dict[str, Any]:
    if not CNN_REMOTE_ENABLED:
        return {'ready': False, 'reason': 'Remote CNN inference has not been enabled.'}
    if not CNN_REMOTE_URL or not CNN_REMOTE_TOKEN:
        return {'ready': False, 'reason': 'Remote CNN URL or server-side token is not configured.'}
    parsed_url = urllib.parse.urlparse(CNN_REMOTE_URL)
    if parsed_url.scheme != 'https' and parsed_url.hostname not in ('localhost', '127.0.0.1', '::1'):
        return {'ready': False, 'reason': 'Remote CNN URL must use HTTPS.'}

    try:
        status = await asyncio.to_thread(_request_cnn_remote, '/health')
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        logger.warning("Remote CNN health check failed: %s", e)
        return {'ready': False, 'reason': 'Remote CNN service is unreachable or returned invalid health data.'}

    if not isinstance(status, dict):
        return {'ready': False, 'reason': 'Remote CNN service returned invalid health data.'}
    verified_fields = (
        'model_loaded',
        'preprocessing_verified',
        'class_mapping_verified',
        'threshold_verified',
        'prediction_ready',
    )
    ready = status.get('status') == 'ok' and all(status.get(field) is True for field in verified_fields)
    reason = status.get('reason')
    return {
        'ready': ready,
        'reason': None if ready else (
            reason if isinstance(reason, str) and reason
            else 'Remote CNN service is not fully deployed and verified.'
        ),
    }


async def predict_remote_cnn(
    raw_bytes: bytes,
    remote_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if remote_status is None:
        remote_status = await get_cnn_remote_status()
    if not remote_status['ready']:
        raise HTTPException(status_code=503, detail=remote_status['reason'] or CNN_PREDICTION_BLOCK_REASON)

    try:
        result = await asyncio.to_thread(_request_cnn_remote, '/predict', 'POST', raw_bytes)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        logger.error("Remote CNN inference request failed: %s", e)
        raise HTTPException(status_code=502, detail="Remote CNN inference service failed.") from e

    if not isinstance(result, dict):
        logger.error("Remote CNN returned an invalid prediction response.")
        raise HTTPException(status_code=502, detail="Remote CNN returned an invalid prediction response.")
    prediction = result.get('prediction')
    confidence = result.get('confidence')
    if (
        prediction not in ('Benign', 'Phishing')
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        logger.error("Remote CNN returned an invalid prediction response.")
        raise HTTPException(status_code=502, detail="Remote CNN returned an invalid prediction response.")

    return {
        'image_model': 'Residual CNN',
        'image_model_prediction': prediction,
        'image_model_confidence': float(confidence),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Safe URL Prediction & Risk Scoring
# ─────────────────────────────────────────────────────────────────────────────
def run_url_analysis(url_input: str) -> Dict[str, Any]:
    """
    Safely normalize bare hostnames, extract 30 static features,
    run ML inference, and generate explainable threat report.
    Never visits or fetches the destination URL.
    """
    normalized_url, scheme_assumed, assumed_scheme = normalize_url(url_input)

    features = extract_url_features(normalized_url)
    feat_vec = np.array(features_to_vector(normalized_url), dtype=np.float32).reshape(1, -1)

    model_available = (url_model is not None)
    model_label = None
    model_prob  = None
    model_used  = 'none'

    if model_available:
        try:
            model_label = int(url_model.predict(feat_vec)[0])
            if hasattr(url_model, 'predict_proba'):
                probs = url_model.predict_proba(feat_vec)[0]
                model_prob = float(probs[1])  # P(Phishing)
            else:
                model_prob = float(model_label)
            model_used = getattr(url_model, '__class__', type(url_model)).__name__
        except Exception as e:
            logger.error(f"Inference error on URL model: {e}")
            model_available = False
            model_label = None
            model_prob  = None

    # Risk evaluation
    risk_report = assess_risk(
        features=features,
        model_prob=model_prob,
        model_label=model_label,
        model_available=model_available,
        url_str=normalized_url
    )

    explanation = explain_url(normalized_url, features, risk_report)

    indicators_clean = [
        {'label': i['label'], 'severity': i['severity']}
        for i in risk_report['indicators']
    ]

    return {
        'original_url':           url_input,
        'url':                    normalized_url,
        'scheme_assumed':         scheme_assumed,
        'assumed_scheme':         assumed_scheme,
        'features': {
            'url_length':         features['url_len'],
            'subdomain_count':    features['subdomain_count'],
            'is_ip':              bool(features['is_ip']),
            'has_https':          bool(features['has_https']),
            'suspicious_tld':     bool(features['suspicious_tld']),
            'brand_impersonation':bool(features['brand_impersonation']),
            'is_shortener':       bool(features['is_shortener']),
            'host_entropy':       features['host_entropy'],
        },
        'model_prediction':       risk_report['model_prediction'],
        'phishing_probability':   risk_report['phishing_probability'],
        'prediction_confidence':  risk_report['prediction_confidence'],
        'model_used':             model_used,
        'model_available':        model_available,
        'risk_level':             risk_report['risk_level'],
        'risk_score':             risk_report['risk_score'],
        'indicators':             indicators_clean,
        'consequences':           risk_report['consequences'],
        'safeguards':             risk_report['safeguards'],
        'recommendation':         risk_report['recommendation'],
        'explanation':            explanation,
        'model_note':             risk_report['model_note'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/')
async def serve_index():
    index = FRONTEND_DIR / 'index.html'
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({'message': 'QRShield API running. Frontend static files not found.'})


@app.get('/health')
async def health():
    return {
        'status':             'ok',
        'url_model_loaded':   url_model is not None,
        'image_model_loaded': image_model is not None,
        'image_models_loaded': sorted(image_models.keys()),
        'cnn_prediction_enabled': False,
        'cnn_status_reason':  'Residual CNN is in demo mode; its fixed output is not a model prediction.',
        'cnn_experimental':   True,
        'cnn_demo_mode':      True,
        'model_error':        MODEL_LOAD_ERROR,
        'feature_count':      len(feature_names) if feature_names else 0,
    }


@app.post('/api/analyze')
async def analyze_qr(
    file: UploadFile = File(...),
    image_model_name: str = Form('extra_trees'),
):
    """
    Decodes uploaded QR code locally and performs pre-navigation threat detection.
    Never navigates to or opens the destination URL.
    """
    if image_model_name not in ('extra_trees', 'random_forest', 'improved_cnn'):
        raise HTTPException(status_code=400, detail="Unsupported image model selection.")
    raw_bytes = await validate_and_read_image_stream(file)
    if image_model_name == 'improved_cnn':
        image_prediction = predict_cnn_demo()
    else:
        image_prediction = predict_image_model(image_model_name, raw_bytes)

    payload, payload_type, decode_method = decode_qr_from_bytes(raw_bytes)

    if payload is None:
        return JSONResponse({
            'success': False,
            'error':   'No valid QR code detected in the uploaded image.',
            'tips': [
                'Ensure the QR code is clearly focused and adequately illuminated.',
                'The QR code should occupy the central region of the image.',
                'Try uploading a higher-resolution image file.',
            ]
        }, status_code=422)

    url_to_analyze = get_analyzable_url(payload, payload_type)

    if url_to_analyze is None:
        return {
            'success':               True,
            'payload':               payload,
            'payload_type':          payload_type,
            'decode_method':         decode_method,
            'is_url':                False,
            'url':                   payload,
            'original_url':          payload,
            'scheme_assumed':        False,
            'assumed_scheme':        None,
            'risk_level':            'SAFE',
            'risk_score':            0,
            'model_prediction':      'Non-URL Content',
            'phishing_probability':  0.0,
            'prediction_confidence': 1.0,
            'recommendation':        'SAFE TO VIEW — Non-URL payload',
            'explanation': (
                f"QR code decoded successfully via **{decode_method}**. "
                f"The content is **{payload_type}** (`{payload}`). "
                f"Because this payload does not contain a web URL, there is no pre-navigation web phishing threat. "
                f"No browser redirects or network requests will occur."
            ),
            'indicators':            [],
            'consequences':          ['No external web navigation or browser redirection risks detected.'],
            'safeguards':            [
                f"Payload type is {payload_type}. Content is displayed as plain text without execution.",
                "Verify the displayed content matches what you expect (e.g. serial number, Wi-Fi code, or text)."
            ],
            'model_note':            'Static analysis confirmed payload contains no navigable web URLs.',
            **image_prediction,
        }

    analysis = run_url_analysis(url_to_analyze)

    return {
        'success':       True,
        'payload':       payload,
        'payload_type':  payload_type,
        'decode_method': decode_method,
        'is_url':        True,
        **image_prediction,
        **analysis
    }


@app.post('/api/analyze-url')
async def analyze_url_direct(request: dict):
    """
    Directly analyze a URL without QR decoding.
    Validates syntax, normalizes bare hostnames, and extracts features.
    """
    raw_url = request.get('url', '').strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="No URL provided.")

    is_valid, err_msg = validate_url_syntax(raw_url)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    analysis = run_url_analysis(raw_url)

    return {
        'success': True,
        'is_url':  True,
        **analysis
    }


@app.get('/api/performance')
async def get_performance():
    """Return genuine saved model metrics and evaluation charts without retraining."""
    url_path = RESULT_DIR / 'url_model_metrics.json'
    img_path = RESULT_DIR / 'image_model_metrics.json'

    result = {}

    if url_path.exists():
        with open(str(url_path), 'r') as f:
            result['url_model'] = json.load(f)
    else:
        result['url_model'] = {'error': 'URL model evaluation metrics not found.'}

    if img_path.exists():
        with open(str(img_path), 'r') as f:
            result['image_model'] = json.load(f)
    else:
        result['image_model'] = {'error': 'Image model evaluation metrics not found.'}

    chart_names = [
        'cm_extra_trees.png', 'cm_random_forest.png',
        'roc_comparison.png', 'feature_importance_rf.png',
        'feature_importance_et.png', 'model_comparison.png',
        'cm_image_extra_trees.png', 'cm_image_random_forest.png',
        'roc_image_comparison.png', 'model_comparison_image.png'
    ]

    result_images = {}
    for fname in chart_names:
        if (RESULT_DIR / fname).exists():
            result_images[fname] = f'/results/{fname}'

    result['charts'] = result_images
    return result

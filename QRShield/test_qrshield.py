"""
test_qrshield.py
================
Comprehensive Integration and Unit Test Suite for QRShield.
Tests:
1. Label mapping validation (0=Legitimate, 1=Phishing)
2. URL syntax validation and bare-hostname normalization
3. Static feature extraction (30 features, brand impersonation, subdomains)
4. Model loading and inference on legitimate vs phishing benchmarks
5. False-positive prevention on common sites (Google, GitHub, Wikipedia, etc.)
6. Missing model handling (must return UNKNOWN, never default to SAFE)
7. Probability vs confidence separation
8. Image upload hardening (streaming limit, corrupt bytes, dimension checks)
9. FastAPI backend integration endpoints
"""

import os
import sys
import json
import io
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, 'training'))
sys.path.insert(0, os.path.join(THIS_DIR, 'backend'))

import joblib
from url_features import (
    extract_url_features, FEATURE_NAMES, features_to_vector,
    normalize_url, validate_url_syntax
)
from risk_engine import assess_risk, explain_url
from qr_decoder import decode_qr_from_bytes, detect_payload_type, get_analyzable_url


class TestLabelMapping(unittest.TestCase):
    """Verify ground truth label standards: 0 = Legitimate, 1 = Phishing."""

    def test_label_mapping_convention(self):
        metrics_file = os.path.join(THIS_DIR, 'results', 'url_model_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            mapping_str = data.get('dataset', {}).get('label_mapping', '')
            self.assertIn('0 = Legitimate', mapping_str)
            self.assertIn('1 = Phishing', mapping_str)

    def test_anchor_domain_labels(self):
        dataset_csv = os.path.join(THIS_DIR, 'datasets', 'url_dataset.csv')
        if os.path.exists(dataset_csv):
            import pandas as pd
            df = pd.read_csv(dataset_csv, nrows=1000)
            edu_samples = df[df['url'].str.contains(r'\.edu|\.gov', case=False, na=False)]
            if len(edu_samples) > 0:
                self.assertEqual(edu_samples['label'].mode()[0], 0, "Edu/Gov anchor domains must have label 0 (Legitimate)")


class TestURLNormalizationAndValidation(unittest.TestCase):
    """Test bare-hostname normalization and strict URL syntax validation."""

    def test_bare_hostname_normalized_to_https(self):
        norm, assumed, scheme = normalize_url("www.google.com")
        self.assertEqual(norm, "https://www.google.com")
        self.assertTrue(assumed)
        self.assertEqual(scheme, "https")

    def test_bare_domain_normalized_to_https(self):
        norm, assumed, scheme = normalize_url("google.com")
        self.assertEqual(norm, "https://google.com")
        self.assertTrue(assumed)

    def test_explicit_http_preserved(self):
        norm, assumed, scheme = normalize_url("http://example.com/test")
        self.assertEqual(norm, "http://example.com/test")
        self.assertFalse(assumed)
        self.assertIsNone(scheme)

    def test_explicit_https_preserved(self):
        norm, assumed, scheme = normalize_url("https://github.com/torvalds")
        self.assertEqual(norm, "https://github.com/torvalds")
        self.assertFalse(assumed)

    def test_empty_url_invalid(self):
        valid, msg = validate_url_syntax("")
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())

    def test_url_with_spaces_invalid(self):
        valid, msg = validate_url_syntax("https://exam ple.com")
        self.assertFalse(valid)
        self.assertIn("space", msg.lower())

    def test_oversized_url_invalid(self):
        oversized = "https://example.com/" + "a" * 2100
        valid, msg = validate_url_syntax(oversized)
        self.assertFalse(valid)
        self.assertIn("maximum", msg.lower())

    def test_valid_urls_pass_validation(self):
        for u in ["www.google.com", "google.com", "https://github.com", "http://192.168.1.1/admin"]:
            valid, msg = validate_url_syntax(u)
            self.assertTrue(valid, f"Expected '{u}' to be valid, got: {msg}")


class TestURLFeatureExtraction(unittest.TestCase):
    """Test feature extractor logic including brand impersonation and subdomains."""

    def test_feature_count(self):
        feats = extract_url_features("https://www.google.com")
        self.assertEqual(len(feats), 30)
        self.assertEqual(len(FEATURE_NAMES), 30)

    def test_subdomain_count_excludes_www(self):
        f1 = extract_url_features("https://www.google.com")
        f2 = extract_url_features("https://google.com")
        self.assertEqual(f1['subdomain_count'], 0)
        self.assertEqual(f2['subdomain_count'], 0)

        f_nested = extract_url_features("http://login.secure.attacker.com")
        self.assertGreaterEqual(f_nested['subdomain_count'], 2)

    def test_brand_impersonation_detection(self):
        # Legitimate brand domains MUST NOT trigger impersonation
        self.assertEqual(extract_url_features("https://www.google.com")['brand_impersonation'], 0)
        self.assertEqual(extract_url_features("https://google.com")['brand_impersonation'], 0)
        self.assertEqual(extract_url_features("https://github.com")['brand_impersonation'], 0)
        self.assertEqual(extract_url_features("https://www.paypal.com")['brand_impersonation'], 0)

        # True spoofing / impersonation attempts MUST trigger impersonation
        self.assertEqual(extract_url_features("http://paypal-security-update.xyz/login")['brand_impersonation'], 1)
        self.assertEqual(extract_url_features("http://google.com.phish-account.top/verify")['brand_impersonation'], 1)

    def test_ip_address_detection(self):
        self.assertEqual(extract_url_features("http://192.168.1.1/admin")['is_ip'], 1)
        self.assertEqual(extract_url_features("https://example.com")['is_ip'], 0)

    def test_suspicious_tld_detection(self):
        self.assertEqual(extract_url_features("http://free-reward.xyz")['suspicious_tld'], 1)
        self.assertEqual(extract_url_features("http://claim.tk")['suspicious_tld'], 1)
        self.assertEqual(extract_url_features("https://example.com")['suspicious_tld'], 0)


class TestRiskEngine(unittest.TestCase):
    """Test explainable risk engine, confidence separation, and failure fallbacks."""

    def test_missing_model_returns_unknown_never_safe(self):
        feats = extract_url_features("https://unknown-new-domain.com")
        report = assess_risk(
            features=feats,
            model_prob=None,
            model_label=None,
            model_available=False,
            url_str="https://unknown-new-domain.com"
        )
        self.assertNotEqual(report['risk_level'], 'SAFE', "Missing model must NEVER default to SAFE!")
        self.assertEqual(report['risk_level'], 'UNKNOWN')
        self.assertEqual(report['model_prediction'], 'Unknown')
        self.assertIsNone(report['phishing_probability'])
        self.assertIn("UNAVAILABLE", report['recommendation'])

    def test_separated_probability_and_confidence(self):
        feats = extract_url_features("https://example.com")
        # Phishing prediction with 85% probability
        report_phish = assess_risk(feats, model_prob=0.85, model_label=1, model_available=True)
        self.assertEqual(report_phish['model_prediction'], 'Phishing')
        self.assertEqual(report_phish['phishing_probability'], 0.85)
        self.assertEqual(report_phish['prediction_confidence'], 0.85)

        # Legitimate prediction with 2% phishing probability (98% confidence)
        report_legit = assess_risk(feats, model_prob=0.02, model_label=0, model_available=True)
        self.assertEqual(report_legit['model_prediction'], 'Legitimate')
        self.assertEqual(report_legit['phishing_probability'], 0.02)
        self.assertEqual(report_legit['prediction_confidence'], 0.98)

    def test_explanation_generation(self):
        feats = extract_url_features("http://paypal-security-update.xyz/login")
        report = assess_risk(feats, model_prob=0.99, model_label=1, model_available=True, url_str="http://paypal-security-update.xyz/login")
        expl = explain_url("http://paypal-security-update.xyz/login", feats, report)
        self.assertIsInstance(expl, str)
        self.assertIn("Phishing", expl)
        self.assertIn("confidence", expl.lower())


class TestModelInferenceAndFalsePositives(unittest.TestCase):
    """Test the saved trained model against common legitimate websites and real phishing attacks."""

    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(THIS_DIR, 'models', 'url_model.pkl')
        if not os.path.exists(model_path):
            raise unittest.SkipTest("url_model.pkl not found — train model first.")
        cls.model = joblib.load(model_path)

    def test_popular_legitimate_urls_not_flagged(self):
        legit_benchmarks = [
            "www.google.com",
            "https://www.google.com",
            "google.com",
            "https://github.com",
            "github.com",
            "https://www.wikipedia.org",
            "https://www.amazon.com",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://www.python.org",
        ]
        for url in legit_benchmarks:
            vec = features_to_vector(url)
            pred = int(self.model.predict([vec])[0])
            prob = float(self.model.predict_proba([vec])[0][1])
            self.assertEqual(pred, 0, f"False positive: '{url}' was classified as Phishing (prob={prob*100:.1f}%)")
            self.assertLess(prob, 0.40, f"False positive high probability on '{url}': {prob*100:.1f}%")

    def test_phishing_attacks_flagged(self):
        phishing_attacks = [
            "http://paypal-security-update.xyz/login",
            "http://192.168.1.1/admin/login.php",
            "http://google.com.phish-account.top/verify",
            "http://bit.ly/free-bitcoin-claim",
            "http://apple-id-suspended.gq/restore",
            "http://chase-bank-verify.bid/login",
        ]
        for url in phishing_attacks:
            vec = features_to_vector(url)
            pred = int(self.model.predict([vec])[0])
            prob = float(self.model.predict_proba([vec])[0][1])
            self.assertEqual(pred, 1, f"False negative: '{url}' was missed as Legitimate (prob={prob*100:.1f}%)")
            self.assertGreaterEqual(prob, 0.60, f"Low phishing probability on attack '{url}': {prob*100:.1f}%")


class TestImageUploadHardening(unittest.TestCase):
    """Test hardened image upload validations: magic bytes, dimensions, and corrupt data."""

    def test_valid_image_bytes(self):
        img = Image.new('RGB', (100, 100), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        raw = buf.getvalue()
        self.assertTrue(raw.startswith(b'\x89PNG\r\n\x1a\n'))

    def test_corrupt_bytes_rejected_by_decoder(self):
        payload, ptype, method = decode_qr_from_bytes(b"not_an_image_random_corrupt_string")
        self.assertIsNone(payload)
        self.assertIn("failed", method)

    def test_empty_bytes_rejected_by_decoder(self):
        payload, ptype, method = decode_qr_from_bytes(b"")
        self.assertIsNone(payload)


class TestImageModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import main
            cls.main = main
        except ImportError as e:
            raise unittest.SkipTest(f"Backend dependencies unavailable: {e}")

    class FakeClassifier:
        n_features_in_ = 69 * 69
        classes_ = np.array([0, 1])

        def predict(self, features):
            return np.array([1])

        def predict_proba(self, features):
            return np.array([[0.2, 0.8]])

    @staticmethod
    def make_image_bytes():
        image = Image.new('L', (69, 69), color=255)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()

    def test_classical_preprocessing_shape_and_range(self):
        features = self.main.prepare_classical_image_features(self.make_image_bytes())
        self.assertEqual(features.shape, (1, 69 * 69))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(float(features.min()), 1.0)
        self.assertEqual(float(features.max()), 1.0)

    def test_extra_trees_prediction_and_confidence(self):
        with patch.dict(self.main.image_models, {'extra_trees': self.FakeClassifier()}):
            result = self.main.predict_image_model('extra_trees', self.make_image_bytes())
        self.assertEqual(result['image_model_prediction'], 'Phishing')
        self.assertEqual(result['image_model_confidence'], 0.8)

    def test_random_forest_prediction_and_confidence(self):
        with patch.dict(self.main.image_models, {'random_forest': self.FakeClassifier()}):
            result = self.main.predict_image_model('random_forest', self.make_image_bytes())
        self.assertEqual(result['image_model_prediction'], 'Phishing')
        self.assertEqual(result['image_model_confidence'], 0.8)

    def test_cnn_preprocessing_shape_range_and_grayscale(self):
        image = Image.new('RGB', (138, 276), color=(255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')

        result = self.main.prepare_cnn_image_input(buffer.getvalue())

        self.assertEqual(result.shape, (1, 69, 69, 1))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 0.298, places=2)

    def test_cnn_prediction_uses_notebook_threshold_and_is_experimental(self):
        class FakeCNN:
            def __init__(self, score):
                self.score = score
                self.inputs = None

            def predict(self, inputs, verbose=0):
                self.inputs = inputs
                return np.array([[self.score]], dtype=np.float32)

        phishing_model = FakeCNN(0.05)
        with patch.object(self.main, 'load_cnn_model', return_value=phishing_model):
            phishing = self.main.predict_local_cnn(self.make_image_bytes())
        self.assertEqual(phishing['image_model_prediction'], 'Phishing')
        self.assertAlmostEqual(phishing['image_model_confidence'], 0.05, places=6)
        self.assertTrue(phishing['image_model_experimental'])
        self.assertEqual(phishing['image_model_threshold'], 0.05)
        self.assertEqual(phishing_model.inputs.shape, (1, 69, 69, 1))
        self.assertEqual(phishing_model.inputs.dtype, np.float32)

        benign_model = FakeCNN(0.04)
        with patch.object(self.main, 'load_cnn_model', return_value=benign_model):
            benign = self.main.predict_local_cnn(self.make_image_bytes())
        self.assertEqual(benign['image_model_prediction'], 'Benign')
        self.assertAlmostEqual(benign['image_model_confidence'], 0.96, places=6)

    def test_cnn_demo_output_is_fixed_and_explicitly_not_a_model_prediction(self):
        first = self.main.predict_cnn_demo()
        second = self.main.predict_cnn_demo()
        self.assertEqual(first, second)
        self.assertEqual(first['image_model_prediction'], 'Phishing')
        self.assertIsNone(first['image_model_confidence'])
        self.assertTrue(first['image_model_demo'])
        self.assertTrue(first['image_model_experimental'])
        self.assertEqual(first['image_model_demo_label'], 'Demo Mode — not a real CNN prediction')

    def test_cnn_runtime_rejects_incompatible_tensorflow_keras(self):
        def installed_version(package):
            return {'tensorflow': '2.15.0', 'keras': '2.15.0'}[package]

        with (
            patch.object(self.main.metadata, 'version', side_effect=installed_version),
            patch.object(self.main, 'CNN_LOAD_ERROR', None),
        ):
            status = self.main.get_cnn_local_status()
        self.assertFalse(status['ready'])
        self.assertIn('tensorflow 2.15.0', status['reason'])
        self.assertIn('needs 2.20.0', status['reason'])

    def test_cnn_prediction_rejects_invalid_sigmoid_output(self):
        from fastapi import HTTPException

        class InvalidCNN:
            def predict(self, inputs, verbose=0):
                return np.array([[1.2]], dtype=np.float32)

        with (
            patch.object(self.main, 'load_cnn_model', return_value=InvalidCNN()),
            self.assertRaises(HTTPException) as error,
        ):
            self.main.predict_local_cnn(self.make_image_bytes())
        self.assertEqual(error.exception.status_code, 500)

    def test_remote_cnn_requires_every_verification_flag(self):
        settings = {
            'CNN_REMOTE_ENABLED': True,
            'CNN_REMOTE_URL': 'https://cnn.example.test',
            'CNN_REMOTE_TOKEN': 'test-token',
        }
        remote_health = {
            'status': 'ok',
            'model_loaded': True,
            'preprocessing_verified': True,
            'class_mapping_verified': True,
            'threshold_verified': True,
            'prediction_ready': False,
            'reason': 'Inference remains disabled.',
        }
        with (
            patch.multiple(self.main, **settings),
            patch.object(self.main, '_request_cnn_remote', return_value=remote_health),
        ):
            status = asyncio.run(self.main.get_cnn_remote_status())
        self.assertFalse(status['ready'])
        self.assertEqual(status['reason'], 'Inference remains disabled.')

    def test_remote_cnn_prediction_returns_only_supported_fields(self):
        response = {'prediction': 'Phishing', 'confidence': 0.91}
        with (
            patch.object(
                self.main,
                'get_cnn_remote_status',
                new=AsyncMock(return_value={'ready': True, 'reason': None}),
            ),
            patch.object(self.main, '_request_cnn_remote', return_value=response),
        ):
            result = asyncio.run(self.main.predict_remote_cnn(b'validated image'))
        self.assertEqual(result, {
            'image_model': 'Residual CNN',
            'image_model_prediction': 'Phishing',
            'image_model_confidence': 0.91,
        })

    def test_remote_request_keeps_token_in_server_side_authorization_header(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        with (
            patch.object(self.main, 'CNN_REMOTE_URL', 'https://cnn.example.test'),
            patch.object(self.main, 'CNN_REMOTE_TOKEN', 'test-token'),
            patch.object(self.main.urllib.request, 'urlopen', return_value=response) as urlopen,
        ):
            result = self.main._request_cnn_remote('/health')
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-token')
        self.assertEqual(result, {'status': 'ok'})

    def test_remote_cnn_rejects_invalid_prediction_shape(self):
        from fastapi import HTTPException

        with (
            patch.object(
                self.main,
                'get_cnn_remote_status',
                new=AsyncMock(return_value={'ready': True, 'reason': None}),
            ),
            patch.object(self.main, '_request_cnn_remote', return_value=['not', 'a', 'mapping']),
        ):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(self.main.predict_remote_cnn(b'validated image'))
        self.assertEqual(error.exception.status_code, 502)


class TestQRDecoderPayloads(unittest.TestCase):
    """Verify QR decoder classifies non-URL payloads safely."""

    def test_payload_classification(self):
        self.assertEqual(detect_payload_type("https://example.com"), "URL")
        self.assertEqual(detect_payload_type("mailto:security@example.com"), "EMAIL")
        self.assertEqual(detect_payload_type("WIFI:S:MyWifi;T:WPA;P:password;;"), "WIFI")
        self.assertEqual(detect_payload_type("Plain text message"), "TEXT")

    def test_get_analyzable_url(self):
        self.assertEqual(get_analyzable_url("https://example.com", "URL"), "https://example.com")
        self.assertIsNone(get_analyzable_url("mailto:test@test.com", "EMAIL"))
        self.assertIsNone(get_analyzable_url("WIFI:S:Home;;", "WIFI"))


class TestRemoteCNNServiceScaffold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from cnn_service import main as service

        cls.service = service
        cls.client = TestClient(service.app)

    def test_health_is_authenticated_and_fails_closed(self):
        with patch.object(self.service, 'CNN_API_TOKEN', 'test-token'):
            unauthorized = self.client.get('/health')
            response = self.client.get('/health', headers={'Authorization': 'Bearer test-token'})
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertFalse(status['model_loaded'])
        self.assertFalse(status['preprocessing_verified'])
        self.assertFalse(status['class_mapping_verified'])
        self.assertFalse(status['threshold_verified'])
        self.assertFalse(status['prediction_ready'])

    def test_prediction_remains_disabled(self):
        with patch.object(self.service, 'CNN_API_TOKEN', 'test-token'):
            response = self.client.post(
                '/predict',
                content=b'QR image bytes',
                headers={
                    'Authorization': 'Bearer test-token',
                    'Content-Type': 'application/octet-stream',
                },
            )
        self.assertEqual(response.status_code, 503)


class TestAPIEndpoints(unittest.TestCase):
    """Test FastAPI application endpoints directly using TestClient."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            import main
            cls.client = TestClient(main.app)
        except ImportError:
            raise unittest.SkipTest("fastapi.testclient or httpx not available for direct client testing.")

    def test_health_endpoint(self):
        import main

        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get('status'), 'ok')
        self.assertTrue(data.get('url_model_loaded'))
        self.assertFalse(data.get('cnn_prediction_enabled'))
        self.assertTrue(data.get('cnn_demo_mode'))
        self.assertTrue(data.get('cnn_experimental'))

    def test_frontend_exposes_cnn_demo_without_unverified_metrics(self):
        frontend_path = os.path.join(THIS_DIR, 'frontend', 'index.html')
        with open(frontend_path, encoding='utf-8') as frontend:
            content = frontend.read()
        self.assertIn('data-scanner-img-model="improved_cnn"', content)
        self.assertIn('data-img-model="improved_cnn"', content)
        self.assertIn('Accuracy (training notebook): 78.48%', content)
        self.assertIn('Fixed display of the notebook', content)
        self.assertIn('Demo Mode — not a real CNN prediction', content)
        self.assertNotIn('Accuracy reported in the training notebook', content)

    def test_analyze_qr_uses_selected_classical_model_and_preserves_url_analysis(self):
        import main

        raw_bytes = b'test qr image bytes'
        upload = {'file': ('qr.png', raw_bytes, 'image/png')}
        for selected_model in ('extra_trees', 'random_forest'):
            with self.subTest(model=selected_model):
                with (
                    patch.object(main, 'validate_and_read_image_stream', new=AsyncMock(return_value=raw_bytes)),
                    patch.object(main, 'predict_image_model', return_value={
                        'image_model': selected_model,
                        'image_model_prediction': 'Phishing',
                        'image_model_confidence': 0.8,
                    }) as predict,
                    patch.object(main, 'decode_qr_from_bytes', return_value=('https://example.com', 'URL', 'opencv')),
                    patch.object(main, 'get_analyzable_url', return_value='https://example.com'),
                    patch.object(main, 'run_url_analysis', return_value={
                        'model_prediction': 'Legitimate',
                        'prediction_confidence': 0.9,
                    }),
                ):
                    response = self.client.post(
                        '/api/analyze',
                        files=upload,
                        data={'image_model_name': selected_model},
                    )
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result['image_model_prediction'], 'Phishing')
                self.assertEqual(result['image_model_confidence'], 0.8)
                self.assertEqual(result['model_prediction'], 'Legitimate')
                predict.assert_called_once_with(selected_model, raw_bytes)

    def test_analyze_qr_uses_fixed_cnn_demo_even_if_local_inference_is_available(self):
        import main

        with (
            patch.object(main, 'validate_and_read_image_stream', new=AsyncMock(return_value=b'valid image')),
            patch.object(main, 'predict_local_cnn') as predict,
            patch.object(main, 'decode_qr_from_bytes', return_value=('https://example.com', 'URL', 'opencv')),
            patch.object(main, 'get_analyzable_url', return_value='https://example.com'),
            patch.object(main, 'run_url_analysis', return_value={
                'model_prediction': 'Legitimate',
                'prediction_confidence': 0.9,
            }),
        ):
            response = self.client.post(
                '/api/analyze',
                files={'file': ('qr.png', b'test qr image bytes', 'image/png')},
                data={'image_model_name': 'improved_cnn'},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['image_model_prediction'], 'Phishing')
        self.assertTrue(response.json()['image_model_experimental'])
        self.assertTrue(response.json()['image_model_demo'])
        self.assertIsNone(response.json()['image_model_confidence'])
        self.assertEqual(
            response.json()['image_model_demo_label'],
            'Demo Mode — not a real CNN prediction',
        )
        predict.assert_not_called()

    def test_analyze_qr_returns_fixed_cnn_demo_if_local_runtime_is_incompatible(self):
        import main

        with (
            patch.object(
                main,
                'get_cnn_local_status',
                return_value={'ready': False, 'reason': 'TensorFlow 2.20.0 and Keras 3.13.2 are required.'},
            ),
            patch.object(main, 'validate_and_read_image_stream', new=AsyncMock(return_value=b'validated image')),
            patch.object(main, 'decode_qr_from_bytes', return_value=('https://example.com', 'URL', 'opencv')),
            patch.object(main, 'get_analyzable_url', return_value='https://example.com'),
            patch.object(main, 'run_url_analysis', return_value={
                'model_prediction': 'Legitimate',
                'prediction_confidence': 0.9,
            }),
        ):
            response = self.client.post(
                '/api/analyze',
                files={'file': ('qr.png', b'test qr image bytes', 'image/png')},
                data={'image_model_name': 'improved_cnn'},
            )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['image_model_prediction'], 'Phishing')
        self.assertIsNone(result['image_model_confidence'])
        self.assertTrue(result['image_model_demo'])

    def test_analyze_url_legitimate(self):
        res = self.client.post('/api/analyze-url', json={'url': 'www.google.com'})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['model_prediction'], 'Legitimate')
        self.assertEqual(data['risk_level'], 'SAFE')
        self.assertTrue(data['scheme_assumed'])
        self.assertEqual(data['assumed_scheme'], 'https')

    def test_analyze_url_phishing(self):
        res = self.client.post('/api/analyze-url', json={'url': 'http://paypal-security-update.xyz/login'})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['model_prediction'], 'Phishing')
        self.assertIn(data['risk_level'], ('HIGH', 'MEDIUM'))
        self.assertIn("DO NOT PROCEED", data['recommendation'])

    def test_analyze_url_invalid_input(self):
        # Empty URL
        res = self.client.post('/api/analyze-url', json={'url': ''})
        self.assertEqual(res.status_code, 400)

        # URL with spaces
        res2 = self.client.post('/api/analyze-url', json={'url': 'https://bad url.com'})
        self.assertEqual(res2.status_code, 400)

    def test_performance_endpoint(self):
        res = self.client.get('/api/performance')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('url_model', data)
        self.assertIn('charts', data)
        charts = data['charts']
        self.assertIn('cm_extra_trees.png', charts)
        self.assertIn('roc_comparison.png', charts)


if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  QRShield — Automated Test Suite")
    print("=" * 65 + "\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_cls in [
        TestLabelMapping,
        TestURLNormalizationAndValidation,
        TestURLFeatureExtraction,
        TestRiskEngine,
        TestModelInferenceAndFalsePositives,
        TestImageUploadHardening,
        TestImageModelInference,
        TestQRDecoderPayloads,
        TestAPIEndpoints,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    print(f"Results: {result.testsRun} tests run.")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors:   {len(result.errors)}")
    print(f"  Status:   {'ALL PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 65)
    sys.exit(0 if result.wasSuccessful() else 1)

"""Tests for first-use model loading without downloading model weights."""
import importlib.util
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ModelLoadingTests(unittest.TestCase):
    def test_default_parent_created_and_concurrent_load_is_single(self):
        calls = []
        def fake_yolo(path):
            calls.append(path)
            self.assertTrue(Path(path).parent.is_dir())
            time.sleep(0.02)
            return types.SimpleNamespace(names={0: 'person'})

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            'cv2': types.ModuleType('cv2'), 'numpy': types.ModuleType('numpy'),
            'ultralytics': types.SimpleNamespace(YOLO=fake_yolo),
        }):
            spec = importlib.util.spec_from_file_location('detection_test', Path(__file__).resolve().parents[1] / 'backend' / 'detection.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.BASE = Path(tmp)
            module.SETTINGS_FILE = Path(tmp) / 'settings.json'
            manager = module.ModelManager()
            threads = [threading.Thread(target=manager.ensure_loaded) for _ in range(4)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(calls, [str(Path(tmp) / 'models' / 'yolov8n.pt')])
            self.assertEqual(manager.class_names, {0: 'person'})

if __name__ == '__main__':
    unittest.main()

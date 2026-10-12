import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from nudenet_regions import load_nudenet_regions


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class NudeNetRegionCacheTests(unittest.TestCase):
    def test_detections_are_cached_and_invalidated(self) -> None:
        calls = []

        class FakeYOLO:
            names = {4: "FEMALE_GENITALIA_EXPOSED", 14: "MALE_GENITALIA_EXPOSED"}

            def __init__(self, weights):
                calls.append(weights)

            def predict(self, paths, **kwargs):
                result = types.SimpleNamespace(
                    names=self.names,
                    boxes=types.SimpleNamespace(
                        xyxy=FakeTensor([[10, 20, 30, 40]]),
                        cls=FakeTensor([4]),
                    ),
                )
                return [result for _ in paths]

        fake_module = types.ModuleType("ultralytics")
        fake_module.YOLO = FakeYOLO
        prefix = Path(__file__).parent / f".nudenet-test-{uuid.uuid4().hex}"
        image = prefix.with_suffix(".png")
        weights = prefix.with_suffix(".pt")
        cache = prefix.with_suffix(".json")
        try:
            image.write_bytes(b"original")
            weights.write_bytes(b"weights")
            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                first = load_nudenet_regions([image], weights, cache)
                second = load_nudenet_regions([image], weights, cache)
                self.assertEqual(first[str(image.resolve())], [(10.0, 20.0, 30.0, 40.0)])
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)

                image.write_bytes(b"changed image")
                load_nudenet_regions([image], weights, cache)
                self.assertEqual(len(calls), 2)
        finally:
            for path in (image, weights, cache, cache.with_suffix(".json.tmp")):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

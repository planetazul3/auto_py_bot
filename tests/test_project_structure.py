import importlib
from pathlib import Path
import unittest


class TestProfessionalStructure(unittest.TestCase):
    def test_expected_directories_exist(self):
        for path in ["src/auto_py_bot", "scripts", "docs", "tests", "deploy"]:
            self.assertTrue(Path(path).exists(), f"Falta directorio: {path}")

    def test_expected_key_files_exist(self):
        for path in [
            "src/auto_py_bot/trading_bot.py",
            "src/auto_py_bot/enhanced_bot.py",
            "scripts/trading_bot.py",
            "docs/ROADMAP_TASKS.md",
            "deploy/Dockerfile",
        ]:
            self.assertTrue(Path(path).exists(), f"Falta archivo: {path}")

    def test_import_core_modules(self):
        modules = [
            "auto_py_bot.ml_model",
            "auto_py_bot.data_collector",
            "auto_py_bot.risk_manager",
            "auto_py_bot.trading_bot",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()

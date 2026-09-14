"""Survey catalogue integrity and Slick operation ownership, without model calls."""

import ast
import re
import unittest
from pathlib import Path

from jinja2 import Environment, nodes

ROOT = Path(__file__).resolve().parents[1]


class SurveyLayoutTests(unittest.TestCase):
    def test_catalogue_and_local_operation_templates(self):
        catalogue = (ROOT / "prompt_optimization/README.md").read_text()
        folders = re.findall(r"\| \[([^\]]+)\]\(\.\./\1/README\.md\) \|", catalogue)
        self.assertEqual(len(folders), 53)
        self.check_folders(folders)

    def test_full_bibliography_coverage(self):
        catalogue = (ROOT / "prompt_optimization/REPOSITORY.md").read_text()
        folders = re.findall(r"\| \[([^\]]+)\]\(\.\./\1/README\.md\) \|", catalogue)
        self.assertEqual(len(folders), 45)
        self.assertEqual(len(set(folders)), 45)
        survey = re.findall(
            r"\| \[([^\]]+)\]\(\.\./\1/README\.md\) \|",
            (ROOT / "prompt_optimization/README.md").read_text(),
        )
        self.assertEqual(len(set(survey) | set(folders)), 86)
        self.check_folders(folders)

    def check_folders(self, folders):
        environment = Environment()
        for folder in folders:
            with self.subTest(folder=folder):
                directory = ROOT / folder
                self.assertTrue((directory / "README.md").is_file())
                self.assertTrue((directory / "__init__.py").is_file())
                tree = ast.parse((directory / "agent.py").read_text())
                for call in ast.walk(tree):
                    if not isinstance(call, ast.Call):
                        continue
                    if not isinstance(call.func, ast.Name) or call.func.id != "prompt":
                        continue
                    for keyword in call.keywords:
                        if keyword.arg == "template":
                            template = directory / "prompts" / ast.literal_eval(keyword.value)
                            self.assertTrue(template.is_file(), str(template))
                for template in (directory / "prompts").glob("*.j2"):
                    parsed = environment.parse(template.read_text())
                    self.assertFalse(
                        list(parsed.find_all((nodes.If, nodes.CondExpr))), str(template)
                    )

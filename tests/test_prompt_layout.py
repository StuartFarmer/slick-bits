"""Keep operation selection in Python instead of conditional Jinja templates."""

import unittest
from pathlib import Path

from jinja2 import Environment, nodes

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (
    "ael",
    "apex",
    "decomp",
    "eoh",
    "evoprompt",
    "llm_gp",
    "multiagent_debate",
    "optimizer",
    "pot",
    "qube",
    "reevo",
)


class PromptLayoutChecks(unittest.TestCase):
    def test_templates_have_no_conditional_branches(self):
        roots = [ROOT / name / "prompts" for name in AGENTS]
        roots.append(ROOT / "skills/slick-development/assets/prompts")
        for root in roots:
            for template in root.glob("*.j2"):
                with self.subTest(template=template.relative_to(ROOT)):
                    parsed = Environment().parse(template.read_text())
                    self.assertEqual(list(parsed.find_all((nodes.If, nodes.CondExpr))), [])


if __name__ == "__main__":
    unittest.main()

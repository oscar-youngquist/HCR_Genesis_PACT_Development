"""Presentation labels/macros must not rename implementation tasks."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from b1z1_paper_names import DISPLAY_NAMES, LATEX_METHOD_NAMES
from exp1_analysis import paper_latex


@pytest.mark.parametrize("task,label,macro", [
    ("b1z1_pact", "PILOT", "pilot"),
    ("b1z1_ppo_pos", "Pos", "posppo"),
    ("b1z1_unifp_original", "UniFP", "unifp"),
    ("b1z1_pact_ab4_coupled_none", "A4", "pilotnofilm"),
    ("b1z1_pact_ab5_coupled_concat", "A5", "pilotconcat"),
    ("b1z1_pact_ab6_position_film", "A1", "pilotposnopinn"),
    ("b1z1_pact_ab7_position_full", "A2", "pilotpospinn"),
    ("b1z1_pact_ab8_coupled_film", "A3", "pilotnopinn"),
    ("b1z1_pact_ab9_representation", "A6", "pilotencpinn"),
    ("b1z1_pact_ab10_actor", "A7", "pilotactpinn"),
])
def test_paper_mapping_and_latex(task, label, macro):
    assert DISPLAY_NAMES[task] == label
    assert LATEX_METHOD_NAMES[task] == "\\" + macro
    frame = pd.DataFrame({"error_m": [1.]}, index=pd.MultiIndex.from_tuples([(label, "nominal")]))
    tex = paper_latex(frame, DISPLAY_NAMES, LATEX_METHOD_NAMES)
    assert "\\" + macro in tex and "PAPERMETHOD" not in tex
    assert frame.index[0][0] == label  # Export conversion must not mutate displayed tables.


def test_all_notebooks_share_paper_names():
    paths = sorted((ROOT / "notebooks").glob("exp*.ipynb"))
    assert len(paths) == 5
    for path in paths:
        source = "\n".join("".join(c["source"]) for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code")
        tree = ast.parse(source)
        assert "from b1z1_paper_names import DISPLAY_NAMES, LATEX_METHOD_NAMES" in source
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
        config = eval(compile(ast.Expression(assignment.value), str(path), "eval"),
                      dict(ROOT=ROOT, REPO=ROOT, np=np, DISPLAY_NAMES=DISPLAY_NAMES,
                           LATEX_METHOD_NAMES=LATEX_METHOD_NAMES))
        if not path.name.startswith("exp3"):
            assert config["display_names"] == DISPLAY_NAMES
            if "compare" in path.name:
                assert config["latex_method_names"] == LATEX_METHOD_NAMES
        else:
            assert 'method["display_name"] = DISPLAY_NAMES[method["task"]]' in source

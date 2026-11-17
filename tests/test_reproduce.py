"""The one-command reproduction (possession/reproduce.py) must stay in step with run_all and make."""

import os
import re
import shlex

from possession import ROOT, reproduce


def test_rule_settings_match_run_all_and_the_makefile():
    run_all = open(os.path.join(ROOT, "scripts", "run_all.sh")).read()
    old = re.search(r"^OLD=\((.*)\)$", run_all, re.M).group(1)
    assert shlex.split(old) == reproduce.V3_SET
    makefile = open(os.path.join(ROOT, "Makefile")).read()
    v3 = re.search(r"^V3_SET = (.*)$", makefile, re.M).group(1)
    assert shlex.split(v3) == reproduce.V3_SET


def test_steps_in_the_documented_order():
    names = [s[0] for s in reproduce.steps("x.csv")]
    assert names == ["detector self-check", "ball detection", "our ATD", "prediction", "format check"]
    assert [s[0] for s in reproduce.steps("x.csv", skip_video=True)] == names[2:]
    selfcheck = reproduce.steps("x.csv")[0]
    assert 3 in selfcheck[3]                          # gate not passed is expected, not a failure
    prediction = reproduce.steps("x.csv")[3]
    assert prediction[2] == {"MTAH_DATA_DIR": reproduce.OURS}

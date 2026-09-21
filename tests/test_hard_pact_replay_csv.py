"""Joint dynamics summaries survive replay CSV export, with explicit populations."""
import csv
import torch

from test_hard_pact_qp_diagnose import owner
from test_hard_pact_reduced_qp import inputs
from rsl_rl.algorithms.hard_pact_qp_diagnose import QPCapture, candidate_assessment
from scripts.diagnose_hard_pact_qp import write_replay_csv


def test_csv_joint_metrics_and_unavailable_rows(tmp_path):
    qp,d=owner(),inputs(2)
    d["joint_position"][0,0]=3.
    m=qp._build(d)
    packet=QPCapture(tmp_path/"capture").before(qp,m,d,"primary",torch.arange(2))
    assessment=candidate_assessment(packet,torch.zeros_like(m.p))
    path=tmp_path/"replay.csv"
    write_replay_csv(path,[dict(variant="baseline",assessment=assessment),dict(variant="old",assessment={"joint_unavailable":"v1 packet"})])
    with path.open() as stream: rows=list(csv.DictReader(stream))
    row=rows[0]
    assert row["joint_metrics_available"]=="True"
    assert row["joint/all/rows"]=="2" and row["joint/accepted/rows"]=="1"
    assert float(row["joint/all/violation/position_rad/max"])==1.
    assert float(row["joint/all/violation/position_rad/joint_0/max"])==1.
    assert float(row["joint/accepted/violation/position_rad/max"])==0.
    assert float(row["joint/all/joint_0/q_next_rad/max"])==3.
    assert row["joint/all/empty_intersection_rows"]=="1"
    assert float(row["joint/all/recovery_slack_rad_s2/max"])==0.
    assert row["joint/all/acceleration_rad_s2/p95"]!=""
    assert rows[1]["joint_metrics_available"]=="False"
    assert rows[1]["joint_metrics_unavailable_reason"]=="v1 packet"
    assert rows[1]["joint/all/violation/position_rad/max"]==""


def test_csv_empty_accepted_population_is_not_reported_as_zero(tmp_path):
    qp,d=owner(),inputs(1)
    m=qp._build(d)
    packet=QPCapture(tmp_path/"capture").before(qp,m,d,"primary",torch.arange(1))
    z=torch.full_like(m.p,float("nan"))
    assessment=candidate_assessment(packet,z)
    path=tmp_path/"replay.csv"
    write_replay_csv(path,[dict(assessment=assessment)])
    with path.open() as stream: row=next(csv.DictReader(stream))
    assert row["joint/accepted/rows"]=="0"
    assert row["joint/accepted/acceleration_rad_s2/count"]=="0"
    assert row["joint/accepted/acceleration_rad_s2/mean"]==""
    assert row["joint/all/acceleration_rad_s2/nonfinite_count"]=="12"

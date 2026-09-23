import numpy as np
import pandas as pd
import torch

from directionlab.data_contract import validate_frame
from directionlab.features import build_current_features, make_causal_windows
from directionlab.models import CausalConv1d, DirectionalThreatModel


def test_contract_rejects_reverse_flow_columns():
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="s"),
            "duration_s": [1.0, 2.0],
            "bytes": [100, 200],
            "packets": [2, 4],
            "transport_protocol": ["TCP", "UDP"],
            "src_port": [1234, 53],
            "dst_port": [443, 5353],
            "label": [0, 1],
            "reverse_bytes": [10, 20],
        }
    )
    report = validate_frame(frame)
    assert not report.valid
    assert "reverse_bytes" in report.forbidden_columns


def test_causal_windows_never_read_future_rows():
    features = np.arange(5, dtype=np.float32).reshape(-1, 1)
    windows, _ = make_causal_windows(features, np.zeros(5), history=3)
    assert windows[0, -1, 0] == 0
    assert windows[2, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert windows[2].max() == 2.0


def test_model_forward_shapes_and_causal_prefix_invariance():
    torch.manual_seed(7)
    model = DirectionalThreatModel(input_dim=10, hidden_dim=16, classes=3)
    model.eval()
    current = torch.randn(2, 10)
    history = torch.randn(2, 8, 10)
    with torch.no_grad():
        result = model(current, history)
    assert result["binary_logit"].shape == (2,)
    assert result["multiclass_logits"].shape == (2, 3)
    assert result["severity"].shape == (2,)
    assert result["embedding"].shape == (2, 16)
    assert result["gate"].min() >= 0 and result["gate"].max() <= 1
    assert not torch.isnan(result["binary_logit"]).any()


def test_causal_convolution_prefix_ignores_future_values():
    torch.manual_seed(7)
    layer = CausalConv1d(2, 3, kernel_size=3, dilation=2)
    layer.eval()
    values = torch.randn(1, 2, 10)
    changed = values.clone()
    changed[:, :, 6:] += 1000
    with torch.no_grad():
        original = layer(values)
        altered = layer(changed)
    torch.testing.assert_close(original[:, :, :6], altered[:, :, :6])


def test_feature_builder_is_allowlisted():
    frame = pd.DataFrame(
        {
            "duration_s": [0, 2], "bytes": [10, 20], "packets": [1, 2],
            "src_port": [1, 2], "dst_port": [80, 443], "transport_protocol": ["tcp", "udp"],
            "payload_length": [100, 200],
        }
    )
    output = build_current_features(frame)
    assert "payload_length" not in output.columns
    assert "reverse_bytes" not in output.columns
    assert output.shape == (2, 10)

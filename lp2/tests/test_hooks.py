import pytest
from lpa2.core.models import load_pipe
from lpa2.core.hooks import attach_lpa_hooks, detach_lpa_hooks

@pytest.mark.skip(reason="requires model weights; smoke tested in runtime")
def test_attach_detach():
    pipe = load_pipe("runwayml/stable-diffusion-v1-5", "euler_a", "cpu", "fp32")
    attach_lpa_hooks(pipe, ["up2"], ["down1"], (200,700), "naive")
    assert hasattr(pipe, "_lpa_saved_processors")
    detach_lpa_hooks(pipe)

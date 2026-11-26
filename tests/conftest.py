import pytest

from physengine.modules import nbodymodule


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "step_budget: keep NBodyModule's wall-clock step limit (the test is about it)"
    )


@pytest.fixture(autouse=True)
def no_wall_clock_step_budget(request, monkeypatch):
    """Let every N-body step run to its end.

    NBodyModule cuts a step short after STEP_BUDGET seconds of wall-clock
    time, to keep the window responsive. In a test that makes the result
    depend on how busy the machine is, so the tests run without the limit,
    except those marked `step_budget`, which test what happens at the limit.
    """
    if request.node.get_closest_marker("step_budget") is None:
        monkeypatch.setattr(nbodymodule, "STEP_BUDGET", float("inf"))

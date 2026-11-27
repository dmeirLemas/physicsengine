from physengine.core.recorder import frames_due


def test_frames_follow_wall_time():
    # 60 fps output, window drawing at 144 Hz: on average 60 frames per second.
    written = 0
    elapsed = 0.0
    for _ in range(144):
        elapsed += 1.0 / 144.0
        written += frames_due(elapsed, 60, written)
    assert written in (59, 60)


def test_slow_frames_are_repeated():
    # A 0.1 s frame at 30 fps covers three output frames.
    assert frames_due(0.1 + 1e-9, 30, 0) == 3
    assert frames_due(0.1, 30, 3) == 0

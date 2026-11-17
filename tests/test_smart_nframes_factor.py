from helios.videoalign.vision_process import FRAME_FACTOR, smart_nframes


def test_clamping_to_total_frames_keeps_frame_factor():
    # When the requested frame count exceeds the video length, smart_nframes
    # clamps to total_frames. If that value is odd, the FRAME_FACTOR=2
    # invariant is broken and torch.linspace samples an odd number of frames.
    nframes = smart_nframes({"nframes": 81}, total_frames=41, video_fps=24)
    assert nframes % FRAME_FACTOR == 0
    assert nframes <= 41


def test_clamping_small_odd_video_keeps_frame_factor():
    nframes = smart_nframes({"nframes": 20}, total_frames=11, video_fps=24)
    assert nframes % FRAME_FACTOR == 0
    assert nframes <= 11


def test_even_total_frames_unchanged():
    assert smart_nframes({"nframes": 20}, total_frames=12, video_fps=24) == 12

"""Record the 3D view to a video file, without the ImGui windows.

The app draws the scene first and the ImGui windows over it, so a frame read
back from the screen between the two holds only the scene. Everything drawn
with ImGui is left out: the menus, and also the overlays drawn with ImGui's
draw lists (the scale bar, the particle labels, the tour's captions).

Frames go to an ffmpeg process through a pipe and are encoded as H.264 MP4.
ffmpeg comes from the imageio-ffmpeg package (it ships a binary), or from
the PATH if that package is missing.

The video plays in real time: the window does not draw at a steady rate, so
each read-back frame is written as many times as the output frame rate asks
for in the wall time it covers (zero times when the window draws faster, so
then the screen is not read at all).
A writer thread feeds ffmpeg, so a slow encode does not stall the frame
unless the queue fills up.

Reading the screen does not stall the frame either; see FrameReader.
"""

import ctypes
import datetime
import queue
import shutil
import subprocess
import threading
from pathlib import Path

RECORDINGS_DIR = Path("recordings")
QUEUE_FRAMES = 8


def ffmpeg_exe():
    """Path to an ffmpeg binary, or None when there is none."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg")


class FrameReader:
    """Reads the window's back buffer without waiting for the GPU.

    A plain glReadPixels makes the CPU wait until the GPU has finished the
    frame, then copies it: about 11 ms per frame for a 2560x1600 window. Here
    glReadPixels writes into a pixel buffer object (PBO) instead, so it only
    queues the copy, and the GPU does it while the CPU moves on. The bytes are
    collected on the next capture, a frame or more later, when the copy is
    done. Two PBOs take turns.

    With scale 0.5, the frame is first shrunk on the GPU (a linear blit, which
    averages each 2x2 block) into a framebuffer of its own: 4x fewer pixels to
    copy and to encode.

    It uses PyOpenGL (imported once the window exists, see core/imgui_renderer.py)
    and not moderngl: moderngl's read of the window's own framebuffer selects
    GL_COLOR_ATTACHMENT0 as the read buffer, which only offscreen framebuffers
    have, and leaves GL_INVALID_OPERATION behind for ImGui's renderer to raise
    on. Every piece of GL state changed here is put back, so moderngl's view of
    the state stays right. Needs the GL context current in every method.
    """

    def __init__(self, source_size, scale=1.0):
        from OpenGL import GL

        self.GL = GL
        self.source_size = tuple(source_size)
        self.size = (max(int(source_size[0] * scale), 2), max(int(source_size[1] * scale), 2))
        self.nbytes = self.size[0] * self.size[1] * 3
        self.pbos = [int(b) for b in GL.glGenBuffers(2)]
        self.next_pbo = 0
        # (pbo, how many video frames it fills), oldest first.
        self.pending = []

        pack = GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING)
        for pbo in self.pbos:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, self.nbytes, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pack)

        self.fbo = self.rbo = None
        if self.size != self.source_size:
            renderbuffer = GL.glGetIntegerv(GL.GL_RENDERBUFFER_BINDING)
            draw_fbo = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
            self.rbo = int(GL.glGenRenderbuffers(1))
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGB8, *self.size)
            self.fbo = int(GL.glGenFramebuffers(1))
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
            GL.glFramebufferRenderbuffer(
                GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.rbo
            )
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, renderbuffer)

    def request(self, count):
        """Start copying the back buffer; its bytes come out of collect() later."""
        from OpenGL.raw.GL.VERSION.GL_1_0 import glReadPixels

        GL = self.GL
        read_fbo = GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING)
        draw_fbo = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
        pack = GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING)
        alignment = GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT)
        scissor = GL.glIsEnabled(GL.GL_SCISSOR_TEST)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        read_buffer = GL.glGetIntegerv(GL.GL_READ_BUFFER)
        GL.glReadBuffer(GL.GL_BACK)
        try:
            if self.fbo is not None:
                GL.glDisable(GL.GL_SCISSOR_TEST)  # it would clip the blit
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
                GL.glBlitFramebuffer(
                    0, 0, *self.source_size, 0, 0, *self.size,
                    GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR,
                )  # fmt: skip
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            pbo = self.pbos[self.next_pbo]
            self.next_pbo = (self.next_pbo + 1) % len(self.pbos)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            # With a PBO bound, the last argument is an offset into it.
            glReadPixels(0, 0, *self.size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            self.pending.append((pbo, count))
        finally:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pack)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, alignment)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            GL.glReadBuffer(read_buffer)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
            if scissor:
                GL.glEnable(GL.GL_SCISSOR_TEST)

    def collect(self):
        """[(bytes, count)] of the frames requested before, oldest first: RGB,
        rows bottom to top. Waits for the GPU only if a copy is not done yet."""
        GL = self.GL
        frames = []
        pack = GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING)
        try:
            for pbo, count in self.pending:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
                address = GL.glMapBufferRange(
                    GL.GL_PIXEL_PACK_BUFFER, 0, self.nbytes, GL.GL_MAP_READ_BIT
                )
                address = getattr(address, "value", address)
                frames.append((ctypes.string_at(address, self.nbytes), count))
                GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
        finally:
            self.pending = []
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pack)
        return frames

    def release(self):
        GL = self.GL
        GL.glDeleteBuffers(len(self.pbos), self.pbos)
        if self.fbo is not None:
            GL.glDeleteFramebuffers(1, [self.fbo])
            GL.glDeleteRenderbuffers(1, [self.rbo])


def frames_due(elapsed, fps, written):
    """How many output frames to write so that `written` reaches elapsed * fps."""
    return max(int(elapsed * fps) - written, 0)


class Recorder:
    def __init__(self):
        self.fps = 30
        # 1.0: the video has the window's full size in pixels. 0.5: half the
        # width and height, 4x fewer pixels to copy and encode.
        self.scale = 1.0
        # None: record until stopped. Otherwise seconds of wall time.
        self.duration: float | None = None

        self.path: Path | None = None
        # The window's framebuffer size when the recording started.
        self.source_size: tuple[int, int] | None = None
        self.elapsed = 0.0
        self.frames_written = 0
        # While paused, frames are not read and the clock stands still, so the
        # video continues from the frame where it was paused.
        self.paused = False
        # The last thing worth telling the user: where a file went, or what failed.
        self.message = ""

        self._process: subprocess.Popen | None = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._reader: FrameReader | None = None

    @property
    def recording(self):
        return self._process is not None

    def start(self, size, path=None):
        """Start recording the window, whose framebuffer is size (width, height)
        pixels, to path (an .mp4). Needs the GL context current."""
        if self.recording:
            return
        exe = ffmpeg_exe()
        if exe is None:
            self.message = "No ffmpeg found: pip install imageio-ffmpeg"
            return

        if path is None:
            RECORDINGS_DIR.mkdir(exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = RECORDINGS_DIR / f"recording_{stamp}.mp4"
            n = 2
            while path.exists():  # two recordings in the same second
                path = RECORDINGS_DIR / f"recording_{stamp}_{n}.mp4"
                n += 1
        self.path = Path(path)
        self.source_size = (int(size[0]), int(size[1]))
        reader = FrameReader(self.source_size, self.scale)
        width, height = reader.size

        command = [
            exe,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",
            # OpenGL rows run bottom to top; H.264 in yuv420p needs even sizes.
            "-vf", "vflip,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.path),
        ]  # fmt: skip
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as error:
            self.message = f"Could not start ffmpeg: {error}"
            reader.release()
            return

        self.elapsed = 0.0
        self.frames_written = 0
        self.paused = False
        self._error = None
        self._reader = reader
        self._queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._write_frames, daemon=True)
        self._thread.start()
        self.message = f"Recording to {self.path} ({width}x{height})"

    def capture(self, size, frame_time):
        """Read the screen and queue it for the video. Call after the scene is
        drawn and before the ImGui windows are. size: the framebuffer's size in
        pixels (on a HiDPI screen, larger than the window's size)."""
        if not self.recording:
            return
        if self._error is not None:
            self.stop()
            return
        self._send(self._reader.collect())
        if tuple(size) != self.source_size:
            # A video cannot change size halfway; end this one.
            self.stop()
            self.message += " (stopped: the window was resized)"
            return
        if self.paused:
            return

        self.elapsed += frame_time
        if self.duration is not None:
            self.elapsed = min(self.elapsed, self.duration)
        count = frames_due(self.elapsed, self.fps, self.frames_written)
        if count:
            self._reader.request(count)
            self.frames_written += count

        if self.duration is not None and self.elapsed >= self.duration:
            self.stop()

    def toggle_pause(self):
        if self.recording:
            self.paused = not self.paused

    def discard(self):
        """Stop and delete the file instead of keeping it."""
        if not self.recording:
            return
        self.stop()
        self.path.unlink(missing_ok=True)
        self.message = "Recording discarded"

    def stop(self):
        """Finish the file. Blocks until ffmpeg has written it."""
        if not self.recording:
            return
        self.paused = False
        self._send(self._reader.collect())
        self._reader.release()
        self._reader = None
        self._queue.put(None)
        self._thread.join()
        process = self._process
        try:
            process.stdin.close()
        except OSError:
            pass
        stderr = process.stderr.read().decode(errors="replace").strip()
        code = process.wait()
        self._process = self._queue = self._thread = None

        if code != 0 or self._error is not None:
            detail = stderr.splitlines()[-1] if stderr else self._error
            self.message = f"Recording failed: {detail}"
        else:
            self.message = f"Saved {self.path} ({self.elapsed:.1f} s)"

    def _send(self, frames):
        for pixels, count in frames:
            for _ in range(count):
                self._queue.put(pixels)

    def _write_frames(self):
        stdin = self._process.stdin
        while True:
            pixels = self._queue.get()
            if pixels is None:
                return
            if self._error is not None:
                continue  # keep draining so capture() never blocks on a dead pipe
            try:
                stdin.write(pixels)
            except OSError as error:
                self._error = str(error)

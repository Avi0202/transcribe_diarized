from strands_agents.tools.toolkit import Toolkit
import os
import tempfile
import asyncio
import logging
import sys
import traceback
import yt_dlp
import ffmpeg
import glob
from logging.handlers import RotatingFileHandler
from fastapi import HTTPException
from openai import AsyncOpenAI


class WhisperDiarizedTool(Toolkit):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.register(
            self.transcribe_diarized,
            name="transcribe_diarized",
            description="Downloads YouTube audio, compresses if needed, and returns diarized transcript using OpenAI’s gpt-4o-transcribe-diarize."
        )

    # ---------------- LOGGER ---------------
    async def _get_logger(self) -> logging.Logger:
        LOG_DIR = "logs"
        os.makedirs(LOG_DIR, exist_ok=True)
        LOG_FILE = os.path.join(LOG_DIR, "whisper_diarized.log")
        LOG_FORMAT = "%(levelname)s | %(asctime)s | %(name)s | %(message)s"

        logger = logging.getLogger("whisper_diarized")
        if not logger.handlers:
            file_handler = RotatingFileHandler(
                LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            file_handler.setLevel(logging.INFO)

            try:
                console_stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
            except Exception:
                console_stream = sys.stdout

            console_handler = logging.StreamHandler(console_stream)
            console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            console_handler.setLevel(logging.INFO)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger

    # ---------------- HELPERS ---------------
    async def _download_audio(self, youtube_url: str) -> str:
        """
        Tries yt-dlp → youtube_dl → pytube → ffmpeg fallback to pull YouTube audio as WAV.
        """
        import subprocess, shlex, shutil, youtube_dl
        from pytube import YouTube

        tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
        output_template = os.path.join(tmpdir, "audio")

        def _run():
            # 1️⃣ yt-dlp
            try:
                ydl_opts = {
                    "format": "bestaudio[ext=m4a]/bestaudio/best",
                    "outtmpl": output_template,
                    "postprocessors": [
                        {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
                    ],
                    "noplaylist": True,
                    "quiet": True,
                    "source_address": "0.0.0.0",
                    "retries": 1,
                    "nocheckcertificate": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                for f in os.listdir(tmpdir):
                    if f.endswith(".wav"):
                        return os.path.join(tmpdir, f)
                raise FileNotFoundError("yt-dlp finished but no .wav found")
            except Exception:
                pass

            # 2️⃣ youtube_dl fallback
            try:
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": output_template,
                    "postprocessors": [
                        {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
                    ],
                    "noplaylist": True,
                    "quiet": True,
                }
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                for f in os.listdir(tmpdir):
                    if f.endswith(".wav"):
                        return os.path.join(tmpdir, f)
            except Exception:
                pass

            # 3️⃣ pytube fallback
            try:
                yt = YouTube(youtube_url)
                stream = yt.streams.filter(only_audio=True).first()
                if not stream:
                    raise ValueError("No audio streams found via pytube.")
                out_file = stream.download(output_path=tmpdir, filename="audio.mp4")
                wav_path = os.path.join(tmpdir, "audio.wav")
                (
                    ffmpeg.input(out_file)
                    .output(wav_path, ac=1, ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
                if os.path.exists(wav_path):
                    return wav_path
            except Exception:
                pass

            # 4️⃣ ffmpeg direct stream
            try:
                ffmpeg_bin = shutil.which("ffmpeg")
                if not ffmpeg_bin:
                    raise RuntimeError("ffmpeg not installed or not in PATH")
                fallback_out = os.path.join(tmpdir, "fallback_audio.wav")
                cmd = f'{ffmpeg_bin} -y -i "{youtube_url}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{fallback_out}"'
                result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and os.path.exists(fallback_out):
                    return fallback_out
            except Exception:
                pass

            raise RuntimeError("All extraction methods failed: yt-dlp, youtube_dl, pytube, ffmpeg")

        return await asyncio.to_thread(_run)

    async def _compress_audio(self, path: str, br: str = "64k", sr: int = 16000) -> str:
        base, _ = os.path.splitext(path)
        out = base + "_compressed.wav"

        def _run():
            (
                ffmpeg.input(path)
                .output(out, **{"b:a": br, "ac": 1, "ar": sr, "y": None})
                .overwrite_output()
                .run(quiet=True)
            )
            return out

        return await asyncio.to_thread(_run)

    # ---------------- TOOL FUNCTION ---------------
    async def transcribe_diarized(self, youtube_url: str, flag: bool = True):
        """
        Download → Compress → Transcribe (split/stream if >25 MB)
        """
        logger = await self._get_logger()
        MAX_MB = 25

        logger.info(f"=== Starting {'diarized' if flag else 'standard'} transcription ===")
        logger.info(f"Input URL: {youtube_url}")

        try:
            # 1. Download
            logger.info("Step 1: Downloading audio…")
            audio_path = await self._download_audio(youtube_url)
            logger.info(f"Downloaded to {audio_path}")

            size = os.path.getsize(audio_path) / 1e6
            logger.info(f"Downloaded file size: {size:.2f} MB")

            # 2. Compress
            logger.info("Step 2: Compressing audio…")
            audio_path = await self._compress_audio(audio_path)
            size = os.path.getsize(audio_path) / 1e6
            logger.info(f"Compressed size: {size:.2f} MB")

            # 3. Initialize OpenAI client
            client = AsyncOpenAI(api_key=self.api_key)
            model_name = "gpt-4o-transcribe-diarize" if flag else "gpt-4o-transcribe"

            # 4. If file too large → segment and stream
            if size > MAX_MB:
                logger.info("⚠️ File too large, splitting into chunks (3 min each)…")
                chunk_dir = tempfile.mkdtemp(prefix="chunks_")
                seg_pattern = os.path.join(chunk_dir, "part_%03d.wav")
                segment_seconds = 180  # 3 min chunks

                def _split():
                    (
                        ffmpeg
                        .input(audio_path)
                        .output(
                            seg_pattern,
                            f="segment",
                            segment_time=segment_seconds,
                            reset_timestamps=1,
                            ac=1,
                            ar=16000,
                            **{"map": "0"}
                        )
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    return sorted(glob.glob(os.path.join(chunk_dir, "part_*.wav")))

                chunks = await asyncio.to_thread(_split)
                if not chunks:
                    raise RuntimeError("No chunks produced")

                all_segments = []
                logger.info(f"Split into {len(chunks)} chunks")

                for i, part in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} …")
                    with open(part, "rb") as f:
                        resp = await client.audio.transcriptions.create(
                            model=model_name,
                            file=f,
                            response_format="diarized_json" if flag else "json",
                        )
                    if flag:
                        all_segments.extend([
                            {
                                "speaker": getattr(s, "speaker", None),
                                "start": getattr(s, "start", None),
                                "end": getattr(s, "end", None),
                                "text": getattr(s, "text", "").strip(),
                            } for s in resp.segments
                        ])
                    else:
                        all_segments.append({
                            "speaker": None,
                            "start": None,
                            "end": None,
                            "text": getattr(resp, "text", "").strip(),
                        })

                summary = " ".join(
                    f"{s['speaker']}: {s['text']}" if s["speaker"] else s["text"]
                    for s in all_segments if s["text"]
                )
                segments = all_segments

            # 5. Otherwise normal transcription
            else:
                logger.info("Step 3: Processing normally …")
                with open(audio_path, "rb") as f:
                    resp = await client.audio.transcriptions.create(
                        model=model_name,
                        file=f,
                        response_format="diarized_json" if flag else "json",
                        chunking_strategy="auto",
                    )

                if flag:
                    segments = [{
                        "speaker": getattr(s, "speaker", None),
                        "start": getattr(s, "start", None),
                        "end": getattr(s, "end", None),
                        "text": getattr(s, "text", "").strip(),
                    } for s in resp.segments]
                    summary = " ".join(f"{s['speaker']}: {s['text']}" for s in segments)
                else:
                    text = getattr(resp, "text", "").strip()
                    segments = [{"speaker": None, "start": None, "end": None, "text": text}]
                    summary = text

            logger.info(f"=== Completed {'diarized' if flag else 'standard'} transcription ===")
            return {"model": model_name, "segments": segments, "summary": summary}

        except Exception as e:
            logger.error(f"FAILED: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))


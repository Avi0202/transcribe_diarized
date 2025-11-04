from strands_agents.tools.toolkit import Toolkit
import os
import tempfile
import asyncio
import logging
import sys
import traceback
import yt_dlp
import ffmpeg
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
            description="Downloads YouTube audio, compresses if needed, and returns diarized transcript using OpenAI’s gpt‑4o‑transcribe‑diarize."
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
        tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
        output_template = os.path.join(tmpdir, "audio")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
        }

        def _run():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            for f in os.listdir(tmpdir):
                if f.endswith(".wav"):
                    return os.path.join(tmpdir, f)
            raise FileNotFoundError("Audio file not found after download.")
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
    async def transcribe_diarized(self, youtube_url: str):
        """
        Download → Compress → Diarize transcript using OpenAI API
        """
        logger = await self._get_logger()
        MAX_MB = 25

        logger.info("=== Starting diarized transcription ===")
        logger.info(f"Input URL: {youtube_url}")

        try:
            # 1. Download
            logger.info("Step 1: Downloading audio…")
            audio_path = await self._download_audio(youtube_url)
            logger.info(f"Downloaded to {audio_path}")

            size = os.path.getsize(audio_path) / 1e6
            logger.info(f"Downloaded file size: {size:.2f} MB")

            # 2. Compress if needed
            if size > MAX_MB:
                logger.info("Step 2: Compressing audio…")
                audio_path = await self._compress_audio(audio_path)
                size = os.path.getsize(audio_path) / 1e6
                logger.info(f"Compressed size: {size:.2f} MB")
            else:
                logger.info("Compression not needed")

            # 3. Diarize
            logger.info("Step 3: Calling OpenAI gpt‑4o‑transcribe‑diarize API…")
            client = AsyncOpenAI(api_key=self.api_key)
            with open(audio_path, "rb") as f:
                response = await client.audio.transcriptions.create(
                    model="gpt-4o-transcribe-diarize",
                    file=f,
                    response_format="diarized_json",
                    chunking_strategy="auto"
                )

            # 4. Parse
            if not hasattr(response, "segments"):
                raise HTTPException(status_code=500, detail="No 'segments' in response")

            segments = [{
                "speaker": getattr(s, "speaker", None),
                "start": getattr(s, "start", None),
                "end": getattr(s, "end", None),
                "text": getattr(s, "text", "").strip(),
            } for s in response.segments]

            summary = " ".join(f"{s['speaker']}: {s['text']}" for s in segments)
            logger.info("=== Completed diarized transcription ===")

            return {
                "model": "gpt‑4o‑transcribe‑diarize",
                "segments": segments,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"FAILED: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
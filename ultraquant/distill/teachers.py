"""On-demand local teachers with a small spec/ask surface and bounded cleanup."""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path

from ultraquant.cloud import supervisor
from ultraquant.convert import gguf
from ultraquant.interpreter.llmls import ModelCard


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    gguf: Path


def gguf_card(path) -> ModelCard:
    """Use header metadata only; never infer identity from an artifact filename.

    ModelCard has no ancestry field. Base-model identity is a fallback where
    artifact identity is missing, not a reason to overwrite the reported arch
    or publisher. Missing publishers stay unknown rather than being guessed.
    """
    metadata = gguf.read(path).metadata
    bases = [f"general.base_model.{index}."
             for index in range(int(metadata.get("general.base_model.count", 0)))]

    def identity(field):
        return next((str(metadata[prefix + field]) for prefix in ["general.", *bases]
                     if metadata.get(prefix + field)), "")

    name = (metadata.get("general.name") or metadata.get("general.basename")
            or identity("name") or identity("basename"))
    if not name:
        raise ValueError("GGUF has no model name or basename metadata")
    arch = str(metadata.get("general.architecture", ""))
    # GGUF general.file_type is a model-level quantization enum, distinct
    # from tensor type numbers. Unknown types remain unclaimed.
    quantizations = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
                     8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
                     12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
                     16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K"}
    return ModelCard(
        id=str(name), arch=arch, publisher=identity("organization"),
        loaded=False, quantization=quantizations.get(metadata.get("general.file_type"), ""),
        context=int(metadata.get(f"{arch}.context_length", 0)),
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LlamaServerTeacher:
    def __init__(self, spec, *, server_exe, port=8095, parallel=8, context=4096,
                 gpu_layers=99, startup_timeout=900.0, request_timeout=120.0):
        if not 1 <= port <= 65535 or parallel < 1 or context < 1:
            raise ValueError("Invalid port, parallelism or context size")
        if any(not math.isfinite(value) or value <= 0
               for value in (startup_timeout, request_timeout)):
            raise ValueError("Timeouts must be finite and positive")
        self.spec = spec
        self.server_exe = Path(server_exe)
        self.port = port
        self.parallel = parallel
        self.context = context
        self.gpu_layers = gpu_layers
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process = None
        self._log = None
        self._ready = False

    def _open(self, request, timeout):
        # Neither environment proxies nor redirects may send local prompts
        # elsewhere. Each worker owns its opener.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        return opener.open(request, timeout=timeout)

    def _port_open(self, timeout=3.0):
        # Windows may take about two seconds to report an actual refusal.
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout):
                return True
        except ConnectionRefusedError:
            return False

    def _tail(self):
        if self._log is None:
            return "(no server log)"
        size = os.fstat(self._log.fileno()).st_size
        self._log.seek(max(0, size - 8000))
        return self._log.read(min(size, 8000)).decode("utf-8", errors="replace")

    def __enter__(self):
        if self._process is not None:
            raise RuntimeError("Teacher is already running")
        if self._port_open():
            raise RuntimeError(f"Port {self.port} is already occupied")
        self._log = tempfile.TemporaryFile(mode="w+b")
        try:
            argv = [str(self.server_exe), "-m", str(self.spec.gguf),
                    "--host", "127.0.0.1", "--port", str(self.port),
                    "-ngl", str(self.gpu_layers), "-c", str(self.context),
                    "-np", str(self.parallel), "--reasoning", "off", "--no-webui"]
            self._process = supervisor.launch(argv, dict(os.environ), self._log)
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {self._process.returncode}")
                try:
                    with self._open(f"http://127.0.0.1:{self.port}/health",
                                    min(1.0, max(0.001, deadline - time.monotonic()))) as response:
                        if response.status == 200 and self._process.poll() is None:
                            self._ready = True
                            return self
                except urllib.error.HTTPError as exc:
                    exc.close()
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
            raise TimeoutError("Server startup timed out")
        except BaseException as exc:
            cleanup_error = None
            try:
                self._stop()
            except Exception as error:
                cleanup_error = error
            finally:
                tail = self._tail()
                self._log.close()
                self._log = None
            message = f"{exc}\nServer log tail:\n{tail}"
            if cleanup_error:
                message += f"\nCleanup failed: {cleanup_error}"
            if not isinstance(exc, Exception):
                exc.add_note(message)
                raise
            raise RuntimeError(message) from exc

    def _stop(self):
        self._ready = False
        process = self._process
        if process is None:
            return
        try:
            supervisor.kill_tree(process, 5.0)
        finally:
            supervisor.close_job(process)
        try:
            process.wait(timeout=5.0)
            deadline = time.monotonic() + 5.0
            while self._port_open():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server port {self.port} did not close")
                time.sleep(0.1)
        finally:
            self._process = None

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._stop()
        except Exception as error:
            if exc is None:
                raise RuntimeError(f"Server cleanup failed: {error}\n{self._tail()}") from error
            exc.add_note(f"Server cleanup failed: {error}\n{self._tail()}")
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
        return False

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens,
            seeds) -> list[list[str]]:
        if not self._ready or self._process is None or self._process.poll() is not None:
            raise RuntimeError("Teacher must be running in its context manager")
        questions, seeds = list(questions), tuple(seeds)
        if samples < 1 or samples != len(seeds):
            raise ValueError("Exactly one seed is required for each sample")

        def request(pair):
            question, seed = pair
            payload = json.dumps({
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": question}],
                "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "seed": seed, "stream": False,
            }).encode("utf-8")
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{self.port}/v1/chat/completions",
                        data=payload, headers={"Content-Type": "application/json"}, method="POST")
                    with self._open(req, self.request_timeout) as response:
                        data = json.load(response)
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("Completion content is not a string")
                    return content
                except (urllib.error.URLError, OSError, HTTPException) as error:
                    if isinstance(error, urllib.error.HTTPError):
                        error.close()
                    if attempt:
                        raise RuntimeError(f"Completion failed twice for seed {seed}: {error}") from error
                    time.sleep(0.2)

        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            flat = list(pool.map(request, ((question, seed) for question in questions for seed in seeds)))
        return [flat[start:start + samples] for start in range(0, len(flat), samples)]

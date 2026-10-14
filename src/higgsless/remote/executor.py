"""Run commands and move files on the workstation, over SSH (asyncssh) or locally."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from higgsless.core.config import WorkstationConfig
from higgsless.core.project import remote_join
from higgsless.errors import JobFailed, RemoteError

LineCallback = Callable[[str], None]


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "stdout_tail": self.stdout[-2000:],
            "stderr_tail": self.stderr[-2000:],
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class FileStat:
    size: int
    mtime: float
    is_dir: bool


def _raise_failed(result: CommandResult) -> None:
    raise JobFailed(
        f"command failed with exit code {result.returncode}: {shlex.join(result.argv)[:300]}",
        details=result.to_dict(),
    )


class Executor(ABC):
    kind = "abstract"

    def __init__(self, cfg: WorkstationConfig) -> None:
        self.cfg = cfg

    def join(self, root: str, *parts: str) -> str:
        return remote_join(root, *parts)

    @abstractmethod
    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
        on_line: LineCallback | None = None,
    ) -> CommandResult: ...

    @abstractmethod
    def put(self, local: Path, remote: str) -> None: ...

    @abstractmethod
    def get(self, remote: str, local: Path) -> None: ...

    @abstractmethod
    def stat(self, remote: str) -> FileStat | None: ...

    @abstractmethod
    def listdir(self, remote: str) -> list[str]: ...

    @abstractmethod
    def read_text(self, remote: str) -> str: ...

    @abstractmethod
    def write_text(self, remote: str, text: str) -> None: ...

    @abstractmethod
    def makedirs(self, remote: str) -> None: ...

    @abstractmethod
    def remove(self, remote: str, recursive: bool = False) -> None: ...

    def exists(self, remote: str) -> bool:
        return self.stat(remote) is not None

    def put_tree(self, local: Path, remote: str, *, skip: Callable[[Path], bool] | None = None) -> int:
        """Copy a local directory tree to the remote side. Returns the number of files copied."""
        count = 0
        for path in sorted(local.rglob("*")):
            rel = path.relative_to(local)
            if skip and skip(rel):
                continue
            target = self.join(remote, *rel.parts)
            if path.is_dir():
                self.makedirs(target)
            else:
                self.makedirs(self.join(remote, *rel.parts[:-1]) if len(rel.parts) > 1 else remote)
                self.put(path, target)
                count += 1
        return count

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class LocalExecutor(Executor):
    """Runs on this machine. Used in tests and when the CLI runs on the workstation itself."""

    kind = "local"

    def run(self, argv, *, cwd=None, env=None, timeout=None, check=True, on_line=None) -> CommandResult:
        full_env = {**os.environ, **(env or {})}
        start = time.monotonic()
        try:
            if on_line is None:
                proc = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=full_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
                result = CommandResult(argv, proc.returncode, proc.stdout, proc.stderr, time.monotonic() - start)
            else:
                result = self._stream(argv, cwd, full_env, timeout, on_line, start)
        except FileNotFoundError as exc:
            raise JobFailed(f"could not start {argv[0]}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise JobFailed(f"command timed out after {timeout}s: {shlex.join(argv)[:300]}") from exc
        if check and not result.ok:
            _raise_failed(result)
        return result

    @staticmethod
    def _stream(argv, cwd, env, timeout, on_line, start) -> CommandResult:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            on_line(line.rstrip("\n"))
            if timeout and time.monotonic() - start > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(argv, timeout)
        proc.wait()
        return CommandResult(argv, proc.returncode, "".join(lines), "", time.monotonic() - start)

    def put(self, local: Path, remote: str) -> None:
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)

    def get(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote, local)

    def stat(self, remote: str) -> FileStat | None:
        try:
            st = os.stat(remote)
        except FileNotFoundError:
            return None
        return FileStat(st.st_size, st.st_mtime, os.path.isdir(remote))

    def listdir(self, remote: str) -> list[str]:
        try:
            return sorted(os.listdir(remote))
        except FileNotFoundError:
            return []

    def read_text(self, remote: str) -> str:
        return Path(remote).read_text(encoding="utf-8")

    def write_text(self, remote: str, text: str) -> None:
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        Path(remote).write_text(text, encoding="utf-8", newline="\n")

    def makedirs(self, remote: str) -> None:
        Path(remote).mkdir(parents=True, exist_ok=True)

    def remove(self, remote: str, recursive: bool = False) -> None:
        path = Path(remote)
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink(missing_ok=True)


class SshExecutor(Executor):
    """One persistent asyncssh connection, driven from a background event loop thread."""

    kind = "ssh"

    def __init__(self, cfg: WorkstationConfig) -> None:
        super().__init__(cfg)
        if cfg.ssh is None:
            raise RemoteError("ssh executor needs an ssh section in the config")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="higgsless-ssh", daemon=True)
        self._thread.start()
        self._conn: Any = None
        self._sftp: Any = None

    def _sync(self, coro: Any, timeout: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connection(self) -> Any:
        if self._conn is None:
            import asyncssh

            ssh = self.cfg.ssh
            assert ssh is not None
            kwargs: dict[str, Any] = {"host": ssh.host, "port": ssh.port, "connect_timeout": ssh.connect_timeout_s}
            if ssh.user:
                kwargs["username"] = ssh.user
            if ssh.key_path:
                kwargs["client_keys"] = [os.path.expanduser(ssh.key_path)]
            if ssh.known_hosts == "ignore":
                kwargs["known_hosts"] = None
            try:
                self._conn = await asyncssh.connect(**kwargs)
            except (TimeoutError, OSError, asyncssh.Error) as exc:
                raise RemoteError(
                    f"cannot connect to {ssh.host}:{ssh.port} over SSH: {exc}",
                    hint="check Tailscale, `ssh <host>` from a terminal, and ssh.host/user/key_path",
                ) from exc
        return self._conn

    async def _sftp_client(self) -> Any:
        if self._sftp is None:
            conn = await self._connection()
            self._sftp = await conn.start_sftp_client()
        return self._sftp

    def run(self, argv, *, cwd=None, env=None, timeout=None, check=True, on_line=None) -> CommandResult:
        command = shlex.join(argv)
        if env:
            command = "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + command
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"

        async def go() -> tuple[int, str, str]:
            conn = await self._connection()
            if on_line is None:
                res = await conn.run(command, check=False, timeout=timeout)
                status = res.exit_status if res.exit_status is not None else -1
                return status, str(res.stdout or ""), str(res.stderr or "")
            proc = await conn.create_process(command, stderr=asyncio.subprocess.STDOUT)
            chunks: list[str] = []

            async def pump() -> None:
                async for line in proc.stdout:
                    chunks.append(line)
                    on_line(line.rstrip("\n"))

            await asyncio.wait_for(pump(), timeout)
            await proc.wait()
            status = proc.exit_status if proc.exit_status is not None else -1
            return status, "".join(chunks), ""

        start = time.monotonic()
        try:
            status, out, err = self._sync(go())
        except TimeoutError as exc:
            raise JobFailed(f"remote command timed out after {timeout}s: {command[:300]}") from exc
        except RemoteError:
            raise
        except Exception as exc:
            if type(exc).__module__.startswith("asyncssh"):
                raise RemoteError(f"SSH command failed: {exc}") from exc
            raise
        result = CommandResult(argv, status, out, err, time.monotonic() - start)
        if check and not result.ok:
            _raise_failed(result)
        return result

    def _sftp_call(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        async def go() -> Any:
            sftp = await self._sftp_client()
            return await getattr(sftp, fn_name)(*args, **kwargs)

        return self._sync(go())

    def put(self, local: Path, remote: str) -> None:
        self._sftp_call("put", str(local), remote)

    def get(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._sftp_call("get", remote, str(local))

    def stat(self, remote: str) -> FileStat | None:
        import asyncssh

        try:
            attrs = self._sftp_call("stat", remote)
        except asyncssh.SFTPNoSuchFile:
            return None
        except asyncssh.SFTPError as exc:
            if "No such file" in str(exc):
                return None
            raise
        is_dir = attrs.type == asyncssh.FILEXFER_TYPE_DIRECTORY
        return FileStat(int(attrs.size or 0), float(attrs.mtime or 0), is_dir)

    def listdir(self, remote: str) -> list[str]:
        import asyncssh

        try:
            names = self._sftp_call("listdir", remote)
        except asyncssh.SFTPNoSuchFile:
            return []
        return sorted(n for n in names if n not in (".", ".."))

    def read_text(self, remote: str) -> str:
        async def go() -> str:
            sftp = await self._sftp_client()
            async with sftp.open(remote, "rb") as fh:
                return (await fh.read()).decode("utf-8")

        return self._sync(go())

    def write_text(self, remote: str, text: str) -> None:
        async def go() -> None:
            sftp = await self._sftp_client()
            parent = remote.rsplit("/", 1)[0]
            if parent:
                await sftp.makedirs(parent, exist_ok=True)
            async with sftp.open(remote, "wb") as fh:
                await fh.write(text.encode("utf-8"))

        self._sync(go())

    def makedirs(self, remote: str) -> None:
        self._sftp_call("makedirs", remote, exist_ok=True)

    def remove(self, remote: str, recursive: bool = False) -> None:
        st = self.stat(remote)
        if st is None:
            return
        if st.is_dir:
            self._sftp_call("rmtree" if recursive else "rmdir", remote)
        else:
            self._sftp_call("remove", remote)

    def close(self) -> None:
        async def go() -> None:
            if self._sftp is not None:
                self._sftp.exit()
            if self._conn is not None:
                self._conn.close()
                await self._conn.wait_closed()

        try:
            if self._conn is not None:
                self._sync(go(), timeout=5)
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def make_executor(cfg: WorkstationConfig) -> Executor:
    return SshExecutor(cfg) if cfg.executor == "ssh" else LocalExecutor(cfg)

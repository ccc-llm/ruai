"""py0 package `subprocess`: `run` with a command string (split like
`shlex.split`, or given to /bin/sh with `shell=True`). A timeout raises
`subprocess.TimeoutExpired` (catch it as `Exception`)."""

from rust import Result

class CompletedProcess:
    @property
    def args(self) -> str: ...
    @property
    def returncode(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...

def run(args: str, shell: bool = False, cwd: str | None = None, capture_output: bool = False,
        text: bool = False, timeout: float | None = None,
        input: str | None = None) -> Result[CompletedProcess, OSError | Exception]: ...

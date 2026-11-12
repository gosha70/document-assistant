"""Guards the CUDA-tag coherence of the PyTorch stack.

torch, torchvision and torchaudio ship CUDA-tagged Linux wheels (``2.15.0+cu132``) and
torchaudio validates its tag against torch at import time. Their metadata declares no
torch pin, so a single-package bump resolves cleanly and then fails at runtime — and the
failure surfaces through transformers' lazy loader as
``ModuleNotFoundError: Could not import module 'AutoProcessor'``, which names neither
package. These tests report the real mismatch instead.

Versions are read from each package's ``version.py`` *without importing it*: importing
torchaudio is what raises, so an import-based check would reproduce the masked error.
"""

import re
from importlib.util import find_spec
from pathlib import Path

import pytest

COMPANIONS = ("torchvision", "torchaudio")


def _version_file(package: str) -> Path | None:
    """Locate <package>/version.py without executing the package."""
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = Path(next(iter(spec.submodule_search_locations))) / "version.py"
    return candidate if candidate.is_file() else None


def _declared_version(package: str) -> str | None:
    """Return the ``__version__`` string (may carry a ``+cuXXX`` local tag)."""
    path = _version_file(package)
    if path is None:
        return None
    match = re.search(r"""^__version__\s*=\s*['"](.+?)['"]""", path.read_text(), re.M)
    return match.group(1) if match else None


def _cuda_tag(version: str | None) -> str | None:
    """Extract the ``cuXXX`` local tag, or None for a CPU-only build."""
    if not version or "+" not in version:
        return None
    local = version.split("+", 1)[1]
    return local if local.startswith("cu") else None


def _torch_cuda_version() -> str | None:
    """torch's CUDA version (e.g. '13.2'), or None for a CPU-only build.

    Imported rather than parsed: torch itself imports cleanly (torchaudio is the package
    that raises), and ``torch/version.py`` writes the value as an annotated assignment
    (``cuda: Optional[str] = '13.2'``) that is easy to misparse.
    """
    import torch  # noqa: PLC0415 - kept local so collection does not pay the import

    return torch.version.cuda


def _expected_tag(cuda_version: str) -> str:
    """'13.2' -> 'cu132'."""
    return "cu" + cuda_version.replace(".", "")


@pytest.mark.parametrize("package", COMPANIONS)
def test_cuda_tag_matches_torch(package: str):
    """A CUDA-tagged companion must carry the same cuXXX tag as torch."""
    if _version_file(package) is None:
        pytest.skip(f"{package} is not installed (legacy group absent)")

    torch_cuda = _torch_cuda_version() or ""
    if not torch_cuda:
        pytest.skip("torch is a CPU-only build; no CUDA tags to reconcile")

    companion_version = _declared_version(package)
    companion_tag = _cuda_tag(companion_version)
    if companion_tag is None:
        pytest.skip(f"{package} {companion_version} is a CPU-only build")

    expected = _expected_tag(torch_cuda)
    assert companion_tag == expected, (
        f"{package} {companion_version} was built for {companion_tag}, but torch "
        f"reports CUDA {torch_cuda} ({expected}). Bump torch, torchvision and torchaudio "
        f"together — their metadata declares no torch pin, so the resolver cannot catch this. "
        f"Left unfixed, importing torchaudio raises and transformers reports it as "
        f"\"Could not import module 'AutoProcessor'\"."
    )


def test_torchaudio_imports():
    """The import that actually breaks: torchaudio checks its CUDA tag on import."""
    if _version_file("torchaudio") is None:
        pytest.skip("torchaudio is not installed (legacy group absent)")

    import torchaudio  # noqa: PLC0415 - deliberate: this import is the thing under test

    assert torchaudio.__version__

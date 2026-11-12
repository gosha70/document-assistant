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
from _pytest.outcomes import Failed

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


def _parse_version(text: str) -> str | None:
    """Extract a ``__version__`` literal from version.py source. None if unreadable.

    Tolerates an optional type annotation (``__version__: str = "..."``) because that is
    exactly the format drift that made the first torch parser skip silently.
    """
    match = re.search(r"""^__version__\s*(?::[^=\n]+)?=\s*['"](.+?)['"]""", text, re.M)
    return match.group(1) if match else None


def _declared_version(package: str) -> str | None:
    """Return the ``__version__`` string (may carry a ``+cuXXX`` local tag).

    None means *unreadable*, not CPU-only: callers must fail rather than skip, or a
    parser that stops matching would silently disable this guard.
    """
    path = _version_file(package)
    if path is None:
        return None
    return _parse_version(path.read_text())


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
    if companion_version is None:
        pytest.fail(
            f"{package} is installed but its version could not be read from "
            f"{_version_file(package)}. Fail rather than skip: an unreadable version would "
            f"otherwise look identical to a CPU-only build and silently disable this guard "
            f"(see _parse_version and test_parse_version_formats). Update the parser."
        )

    companion_tag = _cuda_tag(companion_version)
    if companion_tag is None:
        # Parsed successfully and genuinely carries no local tag -> CPU-only wheel.
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


class TestVersionParsing:
    """Probes for the parser itself.

    The guard is only as good as its ability to read a version string: a format it cannot
    parse must surface as a failure, never as a skip. Both the original torch parser and
    the companion parser broke on annotated assignments, so pin the accepted formats.
    """

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('__version__ = "0.29.2+cu132"', "0.29.2+cu132"),
            ("__version__ = '0.29.2+cu132'", "0.29.2+cu132"),
            ('__version__: str = "0.29.3+cu134"', "0.29.3+cu134"),  # annotated: the drift case
            ('__version__ : str="2.11.1+cu132"', "2.11.1+cu132"),  # odd spacing
            ('from x import y\n__version__ = "2.11.1"\ncuda = None', "2.11.1"),
            ("__version__ = get_version()", None),  # computed -> unreadable
            ("", None),
            ("version = '1.0'", None),
        ],
    )
    def test_parse_version_formats(self, source: str, expected: str | None):
        assert _parse_version(source) == expected

    def test_unreadable_version_fails_instead_of_skipping(self, tmp_path, monkeypatch):
        """An installed package with an unparseable version must fail the guard."""
        fake = tmp_path / "version.py"
        fake.write_text("__version__ = get_version()  # computed at build time\n")
        monkeypatch.setattr("tests.test_torch_stack._version_file", lambda package: fake)
        monkeypatch.setattr("tests.test_torch_stack._torch_cuda_version", lambda: "13.2")

        with pytest.raises(Failed, match="version could not be read"):
            test_cuda_tag_matches_torch("torchvision")

    def test_annotated_mismatch_is_detected(self, tmp_path, monkeypatch):
        """An annotated version with a stale cu tag must fail, not skip."""
        fake = tmp_path / "version.py"
        fake.write_text('__version__: str = "0.29.1+cu130"\n')
        monkeypatch.setattr("tests.test_torch_stack._version_file", lambda package: fake)
        monkeypatch.setattr("tests.test_torch_stack._torch_cuda_version", lambda: "13.2")

        with pytest.raises(AssertionError, match="was built for cu130"):
            test_cuda_tag_matches_torch("torchvision")


def test_torchvision_imports():
    """Companion smoke test: torchvision must load against the installed torch."""
    if _version_file("torchvision") is None:
        pytest.skip("torchvision is not installed (legacy group absent)")

    import torchvision  # noqa: PLC0415 - deliberate: this import is the thing under test

    assert torchvision.__version__

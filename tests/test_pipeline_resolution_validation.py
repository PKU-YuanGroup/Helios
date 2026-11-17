"""Pipeline resolution checks must use the pyramid's real alignment, not 16."""
import ast
from pathlib import Path

FILES = [
    "helios/pipelines/pipeline_helios.py",
    "helios/pipelines/pipeline_helios_ode.py",
    "helios/diffusers_version/pipeline_helios_diffusers.py",
]


def check_inputs_source(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check_inputs":
            return ast.get_source_segment(Path(path).read_text(encoding="utf-8"), node) or ""
    raise AssertionError(f"no check_inputs in {path}")


def test_all_pipelines_require_pyramid_alignment():
    # VAE spatial factor 8 * 4 (three-stage latent pyramid) * patch size 2 = 64.
    for path in FILES:
        src = check_inputs_source(path)
        assert "% 64" in src, f"{path} must validate multiples of 64"
        assert "% 16 !=" not in src, f"{path} still uses the too-permissive 16"


if __name__ == "__main__":
    test_all_pipelines_require_pyramid_alignment()
    print("ok")

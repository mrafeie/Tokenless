from pathlib import Path


def test_gpt_oss_kernel_templates_compile():
    root = Path(__file__).resolve().parents[1]
    templates = [
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "run.template.py",
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "serve.template.py",
    ]

    for template in templates:
        compile(template.read_text(encoding="utf-8"), str(template), "exec")

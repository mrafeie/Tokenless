# Contributing

Thanks for taking the time to improve Tokenless.

## Development Setup

```bash
git clone https://github.com/mrafeie/tokenless.git
cd tokenless
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m py_compile tokenless/client.py tokenless/notebook.py
```

## Secrets

Never commit Kaggle credentials, `.env` files, or local tunnel URLs. Use
environment variables such as `KAGGLE_USERNAME` and `KAGGLE_KEY` during local
development.

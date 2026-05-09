# Contributing to ClimateShield AI

Thank you for your interest in contributing! ClimateShield AI is an open-source project built for UNICEF's Climate & Health mission. We welcome contributions from developers, epidemiologists, public health researchers, and climate scientists.

---

## Ways to Contribute

- **Add new country support** — extend data pipelines and admin boundaries for new UNICEF programme countries
- **Improve AI models** — better dengue/malaria/respiratory forecasting models
- **Add disease modules** — malaria, cholera, typhoid after floods
- **Mobile app features** — symptom logging, SMS alerts for feature phones
- **Translations** — Urdu, Bengali, Hausa, Swahili support
- **Documentation** — deployment guides, dataset tutorials
- **Bug fixes** — see open issues on GitHub

---

## Development Setup

```bash
git clone https://github.com/YOUR_ORG/climateshield-ai.git
cd climateshield-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Start API in dev mode
uvicorn api.main:app --reload --port 8000
```

---

## Code Standards

- Python: Black formatting, type hints, docstrings
- Tests required for new model code
- No hardcoded API keys — use `.env`
- All data sources must be free and publicly accessible

---

## Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Write tests
4. Run `pytest` and ensure all pass
5. Submit a pull request with a clear description

---

## Community

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment.

Questions? Open an issue or start a discussion on GitHub.

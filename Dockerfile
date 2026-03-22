# ── Stage: runtime ────────────────────────────────────────────────────────────
# python:3.12-slim keeps the image lean.
# NOTE: numpy and scikit-learn need gcc to compile C extensions — so we
#       install build-essential first, then clean up apt cache to save space.
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE → no .pyc files inside image
# PYTHONUNBUFFERED       → validation output appears line-by-line, not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install C build tools needed by numpy/scikit-learn, then purge apt cache.
# Doing this in a single RUN keeps the image layer small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies BEFORE copying source — layer caching means
# pip install is only re-run when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default: run the full validation report (produces the 3/3 PASS output)
CMD ["python", "validate.py"]

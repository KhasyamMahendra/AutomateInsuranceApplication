# ─── Stage 1: Build & train the model ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and train the model to produce insurance_model.pkl
COPY . .
RUN python train_model.py

# ─── Stage 2: Runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source and trained model
COPY --from=builder /app/app.py .
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/insurance_model.pkl .

# Switch to non-root user
USER appuser

EXPOSE 5001

# Run with gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "60", "app:app"]

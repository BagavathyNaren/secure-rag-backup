# ============================================================
# STAGE 1: Builder
# Install dependencies here so we can throw away the build tools
# ============================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Set environment variables to save space
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only requirements first (better caching)
COPY requirements.txt .

# Upgrade pip first to use the faster resolver
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies to user local directory
# --no-cache-dir: Don't store pip cache
# --user: Install to ~/.local (easy to copy later)
RUN pip install --no-cache-dir --user -r requirements.txt --default-timeout=100


# ============================================================
# STAGE 2: Final Image
# This is the actual container that runs. It's tiny.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy your application code
COPY . .

# Add local bin to PATH so we can run uvicorn/python
ENV PATH=/root/.local/bin:$PATH

# Expose the port your app runs on
EXPOSE 7860

# Command to run the ap
# Make sure 'app.server:app' matches your file structure
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
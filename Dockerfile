# Keep Devel for your hardware compilation needs
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. System Setup & Certificates (Consolidated to save space)
COPY nscacert.crt /usr/local/share/ca-certificates/nscacert.crt

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget vim ca-certificates openjdk-11-jdk && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set SSL Environment variables
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# 2. Python Tooling & Cert Injection
# Combining these prevents extra layers for minor file edits
RUN pip3 install --no-cache-dir --break-system-packages certifi && \
    cat /usr/local/share/ca-certificates/nscacert.crt >> $(python3 -m certifi) && \
    pip3 install --no-cache-dir --break-system-packages --upgrade pip

# 3. Java Certificate Import
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    keytool -import -trustcacerts -alias nscacert \
    -file /usr/local/share/ca-certificates/nscacert.crt \
    -keystore $JAVA_HOME/lib/security/cacerts \
    -storepass changeit -noprompt

# 4. Dependency Management (The "Live-Sync" Strategy)
WORKDIR /workspace

# Copy ONLY the dependency file first
COPY pyproject.toml .

# Install dependencies using the toml file
# --no-cache-dir is critical for keeping size down
RUN pip3 install --no-cache-dir --break-system-packages .

# 5. Final Source Copy
# This is "baked in" for portability, but your bind mount 
# will override this at runtime for live-sync editing.
#COPY . .

CMD ["/bin/bash"]
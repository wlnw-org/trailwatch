#!/bin/bash
# User data for an Amazon Linux 2023 arm64 instance on AWS Graviton (t4g.medium, 4 GB, 30 GB gp3).
# Installs Docker, builds TrailWatch with SpeciesNet and MegaDetector, and serves it on port 80.
set -euxo pipefail
# A 4 GB swap file gives PyTorch headroom while the models load.
if [ ! -f /swapfile ]; then fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile; fi
dnf install -y docker git
systemctl enable --now docker
git clone --depth 1 https://github.com/wlnw-org/trailwatch /opt/trailwatch
cd /opt/trailwatch
docker build -t trailwatch .
docker run -d --restart unless-stopped --name trailwatch -p 80:8080 \
  -e TRAILWATCH_MAX_BYTES=52428800 --memory 3500m trailwatch

#!/bin/bash
# User data for an Amazon Linux 2023 instance on AWS Graviton (t4g.small).
# Installs Docker, builds TrailWatch from the public repo and serves it on port 80.
set -euxo pipefail
dnf install -y docker git
systemctl enable --now docker
git clone --depth 1 https://github.com/wlnw-org/trailwatch /opt/trailwatch
cd /opt/trailwatch
docker build -t trailwatch .
docker run -d --restart unless-stopped --name trailwatch -p 80:8080 \
  -e TRAILWATCH_MAX_BYTES=52428800 trailwatch

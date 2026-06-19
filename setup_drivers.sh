#!/bin/bash
set -e

echo "=== Installation des drivers NVIDIA ==="
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

echo ""
echo "=== Drivers installés. REDÉMARRE le PC puis lance : ==="
echo "    cd ~/WorldVideo && bash train.sh"

#!/bin/bash

# ==========================================
# Administrative Privileges
# ==========================================

# Stop the script execution after an error
set -e

# Save the project path
PROJECT_ROOT=$(pwd)

# Ask for user password
echo "Please enter your password to authorize the installation:"
sudo -v
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# ==========================================
# Initial Setup & Environment Detection
# ==========================================

# Detect the current environment
if grep -qi microsoft /proc/version; then
    ENV="WSL"
else
    ENV="LINUX"
fi

# Show a message
echo "Environment detected: $ENV"

# ==========================================
# Essential Build Tools & Dependencies
# ==========================================

# Update the package lists from the repositories
echo "Updating package lists..."
sudo apt-get update

# Install essentials build tools
sudo apt-get install -y git gcc g++ cmake make wget libgsl-dev
echo "Essential tools installed successfully"

# ==========================================
# CUDA Toolkit Installation (WSL & Native)
# ==========================================

# Install CUDA depending on environment
if [ "$ENV" = "WSL" ]; then

    # Verify if the GPU driver is installed
    if ! nvidia-smi &> /dev/null; then
        # Show a message
        echo "WARNING: NVIDIA GPU driver is not detected. Install NVIDIA GPU driver with NVIDIA APP or equivalent"
        # Stop the script execution
        exit 1
    else 
        # Show a message
        echo "NVIDIA driver is detected"
    fi

    # Show a message
    echo "Installing CUDA toolkit for WSL..."

    # This is needed for older version of cuda toolkit
    wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.2-0ubuntu2_amd64.deb
    sudo dpkg -i libtinfo5_6.2-0ubuntu2_amd64.deb

    # Install cuda toolkit 12.4 in WSL
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
else
    # Verify exact OS (Ubuntu 24.04 only)
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        if [ "$ID" != "ubuntu" ] || [ "$VERSION_ID" != "24.04" ]; then
            echo "=========================================================================="
            echo "                                 WARNING                                  "
            echo "=========================================================================="
            echo " This script is strictly configured for Ubuntu 24.04.                     "
            echo " Your system is detected as: $PRETTY_NAME                                 "
            echo "                                                                          "
            echo " To prevent breaking your system packages, execution has been stopped.    "
            echo " Please modify this script with the correct NVIDIA repository for your OS."
            echo " You can find the exact installation commands here:                       "
            echo " https://developer.nvidia.com/cuda-12-4-0-download-archive"
            echo "=========================================================================="
            exit 1
        fi
    fi

    # Show a message
    echo "Configuring NVIDIA Oficial repository..."

    # Configure the NVIDIA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update

    # Verify if the GPU driver is installed
    if ! nvidia-smi &> /dev/null; then
        # Show a message
        echo "The NVIDIA GPU driver is not detected. Installing NVIDIA drivers + CUDA..."
        
        # Install the NVIDA driver and CUDA toolkit
        sudo apt-get -y install cuda-12-4
    else 
        # Show a message
        echo "NVIDIA driver is detected. Installing the CUDA toolkit..."

        # Install cuda toolkit 12.4 in Linux
        sudo apt-get -y install cuda-toolkit-12-4
    fi
fi

# ==========================================
# Environment Variables Configuration
# ==========================================

# Remove the previous varibles to avoid duplicated
sed -i '/\/usr\/local\/cuda/d' ~/.bashrc
sed -i '/CUDA_HOME=/d' ~/.bashrc

# Set the global CUDA paths
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

# Set the CUDA path to execute this script
export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Show a message
echo "CUDA was installed successfully"

# ==========================================
# Post-Installation GPU Verification
# ==========================================

# Verify GPU
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: GPU not detected after install"
    exit 1
else
    echo "GPU detected correctly"
fi

# ==========================================
# Task Spooler GPU (tsgpu) Installation
# ==========================================

# Installing Task Spooler GPU
echo "Downloading Task Spooler GPU from Github..."

# Move to the temporal folder 
cd /tmp

# Clone the Task Spooler GPU project
if [ ! -d "task-spooler" ]; then
    git clone https://github.com/justanhduc/task-spooler
fi

# Show a message
echo "Compiling Task Spooler GPU..."

# Compile the project
cd task-spooler
./install_cmake

# Show a message
echo "Task Spooler GPU was installed successfully"

# ==========================================
# GPU Queues & Alias Configuration
# ==========================================

# Create a folder to contain the queues
mkdir -p ~/.tsockets

# Detect the number of GPUs
num_gpus=$(nvidia-smi -L | wc -l)

# Clean previous Task Spooler GPU config
sed -i '/tsockets/d' ~/.profile
sed -i '/tsgpu/d' ~/.bashrc

# Loop over the GPUs
for ((i=0; i<num_gpus; i++)); do  
  # Create a queue for the current GPU
  echo 'TS_MAXFINISHED=0 ts -S 1 "$HOME/.tsockets/ts'"${i}"'.sock"' >> ~/.profile
  # Create an alias for the current GPU
  echo 'alias tsgpu'"${i}"'='\''TS_SOCKET=$HOME/.tsockets/ts'"${i}"'.sock TS_VISIBLE_DEVICE='"${i}"' ts -G 1'\''' >> ~/.bashrc
done

# Show a message
echo "A queue was created for each GPU (total queues=${num_gpus})"

# ==========================================
# Dynamic Makefile modification for IonWake
# ==========================================

# Show a message
echo "Configuring GPU targets in IonWake/Makefile..."

# Path to the Makefile (adjust if the relative path is different from where Setup.sh is executed)
MAKEFILE_PATH="$PROJECT_ROOT/IonWake/Makefile"

if [ -f "$MAKEFILE_PATH" ]; then
    # Clean any previous GPU configurations injected by this script
    # This deletes everything from our marker to the end of the file
    sed -i '/# === AUTO-GENERATED GPU TARGETS ===/,$d' "$MAKEFILE_PATH"

    # Insert a black space
    echo "" >> "$MAKEFILE_PATH"

    # Insert the marker
    echo "# === AUTO-GENERATED GPU TARGETS ===" >> "$MAKEFILE_PATH"

    # Inject SHELL variables for each GPU
    for ((i=0; i<num_gpus; i++)); do
        echo "run_local_gpu${i}: SHELL := /bin/bash" >> "$MAKEFILE_PATH"
        echo "run_local_gpu${i}: .SHELLFLAGS := -i -c" >> "$MAKEFILE_PATH"
    done

    # Insert a black space
    echo "" >> "$MAKEFILE_PATH"

    # Inject targets (execution rules) for each GPU
    # DO NOT ALIGN IT
    for ((i=0; i<num_gpus; i++)); do
cat <<EOF >> "$MAKEFILE_PATH"
run_local_gpu${i}:
	@read -p "Name of Job: " name && \\
	cd ../DRIAD_run_scripts && \\
	./setup_run.sh \$\$name && \\
	tsgpu${i} ./runLocal.sh \$\$name && \\
	echo "------Code running in GPU ${i}------"
EOF
done

    echo "Makefile updated successfully with ${num_gpus} GPU targets."
else
    echo "WARNING: Makefile not found at $MAKEFILE_PATH. Please verify the path."
    exit 1
fi

# Show a message
echo "IMPORTANT: Restart the computer."
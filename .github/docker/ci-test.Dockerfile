FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install only the absolute minimum system dependencies for PySide6/Qt6 GUI testing
# and basic GHA requirements (git, ca-certificates for checkout/downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libglib2.0-0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libx11-xcb1 \
    libxcb-randr0 \
    libxcb-xtest0 \
    libasound2t64 \
    libnss3 \
    libgbm1 \
    ca-certificates \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up environment variables for Qt/GUI
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

CMD ["/bin/bash"]

# 基礎映像檔是為 NVIDIA Jetson JetPack 6 設計的 (Ubuntu 22.04 aarch64)
FROM ultralytics/ultralytics:latest-jetson-jetpack6

# 修正了錯字：noninteractive
# 設定為非互動模式，避免在建置過程中卡住
ENV DEBIAN_FRONTEND=noninteractive

# 設定工作目錄，讓後續指令更簡潔
WORKDIR /

# 將所有操作合併到一個 RUN 指令中，以優化映像檔大小
RUN \
    echo "Starting Librealsense build process..." && \
    # 步驟 1: 更新軟體源並安裝從源碼編譯所需的核心依賴套件
    # 注意：我們不再需要 software-properties-common
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        freeglut3-dev \
        software-properties-common \
        build-essential \
        libssl-dev \
        libusb-1.0-0-dev \
        pkg-config \
        libgtk-3-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        python3-dev \
	libopencv-dev \
    && \
    # 步驟 2: 建立暫存目錄並複製您指定的專案
    # 這個專案應包含在 Jetson 上編譯 Librealsense 的腳本
    mkdir installLibrealsense && \
    cd installLibrealsense && \
    git clone https://github.com/ProgrammingCarrot/Jetson-Project.git . && \
    chmod +x installLibrealsense.sh && chmod +x buildLibrealsense.sh && \
    ./installLibrealsense.sh && \
    # 步驟 3: 直接執行編譯腳本
    # 這個腳本應該會處理下載 Librealsense 源碼並進行編譯安裝
    echo "Dependencies installed, executing build script..." && \
    ./buildLibrealsense.sh && \
    # 步驟 4: 清理工作
    # 返回根目錄，然後刪除暫存資料夾和 apt 快取
    echo "Build script finished, cleaning up..." && \
    cd / && \
    # 修正了錯字：installLibrealsense
    rm -rf installLibrealsense && \
    rm -rf /var/lib/apt/lists/* 

# 將工作目錄設回一個常用位置
WORKDIR /home

# 預設啟動指令
CMD ["bash"]

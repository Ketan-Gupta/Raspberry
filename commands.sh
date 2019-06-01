#!bin/bash

sudo apt-get update -y
sudo apt-get install gvim git -y
cd ~
mkdir tf
sudo apt-get install libatlas-base-dev -y
pip3 install tensorflow
sudo pip3 install pillow lxml jupyter matplotlib cython -y
sudo apt-get install python-tk -y
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
sudo apt-get install qt4-dev-tools -y
sudo apt install libgtk-3-dev libcanberra-gtk3-dev -y
sudo apt install libtiff-dev zlib1g-dev -y
sudo apt install libjpeg-dev libpng-dev -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
cd ~
git clone https://github.com/Ketan-Gupta/OpenCV-Rpi
cd OpenCV-Rpi
tar xf opencv-4.0.0-armhf.tar.bz2
sudo mv opencv-4.0.0 /opt
sudo mv opencv.pc /usr/lib/arm-linux-gnueabihf/pkgconfig
echo 'export LD_LIBRARY_PATH=/opt/opencv-4.0.0/lib:$LD_LIBRARY_PATH' >> .bashrc
source .bashrc
cd ~
cd /opt/opencv-4.0.0/python
sudo python setup.py develop
sudo python3 setup.py develop
cd ~
cd tf
sudo apt-get install autoconf automake libtool curl -y
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protobuf-all-3.8.0.tar.gz
tar -zxvf protobuf-all-3.8.0.tar.gz
cd protobuf-3.8.0
./configure
make
make check











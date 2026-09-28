Record, display, and stream high def video, using an 477P Motorized Focus High Quality Camera for NVIDIA® Jetson AGX Orin Nano, Jetpack 7.2.
Update ffmpeg from source.
./configure --prefix=usr --enable-indev=pulse --enable-gpl
make
sudo make install
sudo ldconfig

Enable the device tree for the imx744.
cd /boot
sudo mkdir dtb
sudo cp *dtb dtb
sudo cp *dtbo dtb

sudo  python /opt/nvidia/jetson-io/jetson-io.py
select Configure Jetson 22pin CSI Connector

Choose Camera IMX477-A 
Save and reboot.

ls /dev/v*
Look for /dev/video0.

sudo dmesg | grep imx

[    0.027618] /bus@0/cam_i2cmux/i2c@0/rbpcv3_imx477_a@1a: Fixed dependency cycle(s) with /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0
[    0.028006] /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0: Fixed dependency cycle(s) with /bus@0/cam_i2cmux/i2c@0/rbpcv3_imx477_a@1a
[    0.028104] /bus@0/cam_i2cmux/i2c@0/rbpcv3_imx477_a@1a: Fixed dependency cycle(s) with /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0
[    0.033888] /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0: Fixed dependency cycle(s) with /bus@0/cam_i2cmux/i2c@0/rbpcv3_imx477_a@1a
[    4.545361] /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0: Fixed dependency cycle(s) with /bus@0/cam_i2cmux/i2c@0/rbpcv3_imx477_a@1a
[    4.556153] /bus@0/cam_i2cmux/camera_autofocus -h 2160 -w 3840 -f 30@0/rbpcv3_imx477_a@1a: Fixed dependency cycle(s) with /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0
[   10.136300] /bus@0/host1x@13e00000/nvcsi@15a00000/channel@0: Fixed dependency cycle(s) with /bus@0/cam_i2cmux/i2c@0/rbpcv3_imx477_a@1a
[   11.008503] imx477 9-001a: tegracam sensor driver:imx477_v2.0.6
[   11.322863] tegra-camrtc-capture-vi tegra-capture-vi: subdev imx477 9-001a bound

Clone the resository
cd ~/imx744_autofocus_orin_nano
mkdir build
cd build
cmake ..
make

camera_autofocus -h 2160 -w 3840 -f 30
vlc will display streaming video.
Close the display window to stop recoding/streaming.
Look for the video in the Videos directory.

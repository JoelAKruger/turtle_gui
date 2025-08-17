#!/bin/bash

# Trap Ctrl+C and kill both libcamera-vid and ffmpeg
cleanup() {
  echo "Stopping streaming..."
  pkill libcamera-vid
  pkill ffmpeg
  exit 0
}

trap cleanup INT TERM

# Run the pipeline
libcamera-vid -t 0 --inline --flush --width 1280 --height 720 --framerate 30 --codec h264 --bitrate 5000000 --profile high -o - | \
ffmpeg -fflags +genpts -i - -c copy -f mpegts -fflags nobuffer+flush_packets+genpts -flags low_delay \
udp://192.168.0.103:1234?pkt_size=1316&buffer_size=65535&fifo_size=500

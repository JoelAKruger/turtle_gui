# Cheatsheet
SSH:
- `ssh turtle@192.168.1.100`

Video:
- `gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! rtpjpegpay ! udpsink host=0.0.0.0 port=1234`
- `gst-launch-1.0 v4l2src device=/dev/video0 ! queue ! videoconvert ! x264enc tune=zerolatency key-int-max=15 ! video/x-h264,profile=main,width=1920,height=1080,framerate=30/1 ! rtph264pay pt=96 config-interval=-1 ! udpsink host=192.168.1.1 port=5000`

Audio:
- https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/User/Modules/
- TurtlePi (sender):
  - `ffmpeg -ar 8000 -acodec pcm_s32le -ac 1 -f alsa -i plughw:0  -acodec mp2 -b:a 128k -f rtp rtp://192.168.1.102:4444`
  - `pactl load-module module-null-sink sink_name=rtp`
    - `format=s32le channels=2 rate=48000`
  - `pactl load-module module-rtp-send source=rtp.monitor`
    - `destination_ip=192.168.1.102 port=46000`
- Desktop
  - `ffplay -protocol_whitelist file,udp,rtp -i rtp://0.0.0.0:4444`
  - `pactl load-module module-rtp-recv`
    - `sap_address=0.0.0.0`

# Troubleshooting
```
Connection failure: Connection refused
pa_context_connect() failed: Connection refused
```
- `pulseaudio --start`

```
ERROR: pipeline could not be constructed: no element "v4l2src".
```
- `nix-shell -p gst_all_1.gstreamer gst_all_1.gst-plugins-good`
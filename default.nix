{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation {
  pname = "turtle_gui";
  version = "1.0";

  src = ./.;

  buildInputs = [
    pkgs.gcc
    pkgs.imgui
    pkgs.glfw
    pkgs.libGL
    pkgs.pkg-config
    pkgs.ffmpeg_6
    pkgs.opencv4
    pkgs.onnxruntime
    pkgs.pulseaudio
  ];

  buildPhase = ''
    g++ -O0 -g src/main.cpp -o turtle_gui \
      -limgui \
      $(pkg-config --cflags --libs opencv4 glfw3 libonnxruntime) \
      -lGL \
      -lavformat -lavcodec -lavutil -lswscale -lpulse-simple \
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp turtle_gui $out/bin/
  '';

  dontStrip = true;
}
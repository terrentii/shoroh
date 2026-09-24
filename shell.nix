{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [ pkgs.portaudio ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.portaudio ];
  shellHook = "source .venv/bin/activate";
}

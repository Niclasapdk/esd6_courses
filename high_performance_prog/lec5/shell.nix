{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = [
    pkgs.python3Packages.numpy
    pkgs.python3Packages.numba
    pkgs.cudaPackages.cudatoolkit
  ];
}

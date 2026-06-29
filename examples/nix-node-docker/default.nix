# Classic (non-flake) entry point.
#
#   nix-build            # uses your <nixpkgs> channel
#   docker load < result
#
# Pin nixpkgs explicitly if you want reproducibility without flakes, e.g.
#   nix-build --arg pkgs 'import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/<rev>.tar.gz") {}'
{ pkgs ? import <nixpkgs> { } }:

import ./image.nix { inherit pkgs; }

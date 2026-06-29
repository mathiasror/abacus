{
  description = "Minimal, rootless Docker image for running a Node.js app, built with Nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      # Container images are Linux-only; build for the common arches.
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forEach = nixpkgs.lib.genAttrs systems;
    in
    {
      # `nix build .#default` (or just `nix build`) produces ./result, a
      # `docker load`-able image tarball for the host's architecture.
      packages = forEach (system:
        let pkgs = import nixpkgs { inherit system; };
        in { default = import ./image.nix { inherit pkgs; }; });
    };
}

# Minimal, rootless Node.js base image, built with Nix.
#
# The result is a Docker/OCI image that contains ONLY:
#   * the `node` runtime and its shared-library closure
#   * CA certificates (so the app can speak TLS)
#   * /etc/passwd + /etc/group for a single unprivileged user
#   * empty, user-owned /app, /home/node and a sticky /tmp
#
# There is deliberately NO shell, NO npm/corepack, NO coreutils and NO
# package manager in the image. The only thing it can do is run `node`.
#
# Imported by both flake.nix (flakes) and default.nix (classic nix-build).
{ pkgs }:

let
  # `nodejs-slim` is the Node.js runtime WITHOUT npm and corepack, so no
  # package manager is ever baked into the image.
  node = pkgs.nodejs-slim_22;

  # The unprivileged user the container runs as.
  user = "node";
  uid = 1000;
  gid = 1000;
  home = "/home/${user}";

  # dockerTools images have no /etc/passwd by default. Node resolves its own
  # uid/gid and $HOME at startup, so provide a single-user passwd/group.
  # The login shell is intentionally a path that does not exist.
  passwd = pkgs.writeText "passwd" ''
    root:x:0:0:root:/root:/noshell
    ${user}:x:${toString uid}:${toString gid}:${user}:${home}:/noshell
  '';
  group = pkgs.writeText "group" ''
    root:x:0:
    ${user}:x:${toString gid}:
  '';
in
pkgs.dockerTools.buildLayeredImage {
  name = "node-minimal";
  tag = "latest";

  # Referencing `node` and `pkgs.cacert` here pulls their full runtime
  # closures into the image automatically — nothing more, nothing less.
  contents = [ node pkgs.cacert ];

  # Create the directory skeleton and identity files. These commands run in
  # the build sandbox against the image root, so they don't add host tools
  # to the image itself.
  extraCommands = ''
    mkdir -p app etc tmp .${home}
    cp ${passwd} etc/passwd
    cp ${group}  etc/group
    chmod 1777 tmp
  '';

  # fakeRootCommands runs under fakeroot, so ownership changes are recorded
  # in the layer. Give the app dir and HOME to the unprivileged user.
  fakeRootCommands = ''
    chown -R ${toString uid}:${toString gid} app .${home}
  '';

  config = {
    # Run as the unprivileged user — never root.
    User = "${toString uid}:${toString gid}";
    WorkingDir = "/app";
    Env = [
      "HOME=${home}"
      "NODE_ENV=production"
      "PATH=${node}/bin"
      # Let Node and OpenSSL find the bundled CA bundle.
      "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      "NODE_EXTRA_CA_CERTS=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    ];
    # Entrypoint is `node`; the app just supplies the script path via Cmd,
    # so child images only need `CMD ["/app/your-entry.js"]`.
    Entrypoint = [ "${node}/bin/node" ];
    Cmd = [ "/app/index.js" ];
  };
}

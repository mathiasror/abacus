# Minimal, rootless Node.js Docker image with Nix

A prebuilt base image that contains **only** what a Node.js app needs to run:
the `node` runtime, its shared-library closure, and CA certificates. There is
**no shell, no npm/corepack, no coreutils, no package manager** — the image
can do exactly one thing: run `node`. It also runs as an **unprivileged user**
(uid/gid `1000`), never root.

You build the base once, then drop your application on top with a two-line
`Dockerfile`.

## Why Nix instead of a `Dockerfile`?

`dockerTools.buildLayeredImage` builds the image from the Nix store, so the
image contains the *exact* runtime closure of `node` and nothing incidental —
no base distro, no `apt`, no `/bin/sh`. That is what makes "no shell, no
package managers, nothing else" actually true rather than aspirational.

## Files

| File | Purpose |
| --- | --- |
| `image.nix` | The image definition (shared by both entry points below). |
| `flake.nix` | Flake entry point: `nix build`. |
| `default.nix` | Classic entry point: `nix-build`. |
| `Dockerfile` | Thin layer that copies your app onto the base image. |
| `app/index.js` | A tiny example app — replace with your own. |

## Build the base image

With flakes:

```bash
nix build            # produces ./result (a docker-load-able tarball)
docker load < result # -> node-minimal:latest
```

Or without flakes:

```bash
nix-build            # produces ./result
docker load < result
```

## Insert your app

The base is prebuilt; your app goes in a thin child image. The base has no
shell, but `FROM` / `COPY` / `CMD` don't need one, so this just works:

```dockerfile
FROM node-minimal:latest
COPY --chown=1000:1000 app/ /app/
CMD ["/app/index.js"]
```

```bash
docker build -t my-node-app .
docker run --rm -p 3000:3000 my-node-app
# {"message":"Hello ...","node":"v22.x","uid":1000,"gid":1000}
```

`ENTRYPOINT` is already `node`, so a child image only needs to point `CMD` at
its entry script.

### Dependencies

Because there is no `npm` in the image, **bring your own `node_modules`**:
install/build them before the `COPY` (e.g. `npm ci --omit=dev` on your build
host, or vendor them with Nix) and copy the result into `app/`.

## What's inside (and what isn't)

Included:

- `node` (from `nodejs-slim`, i.e. without npm/corepack) and its closure
- `cacert` CA bundle, wired up via `SSL_CERT_FILE` / `NODE_EXTRA_CA_CERTS`
- `/etc/passwd` + `/etc/group` for the single `node` user
- empty, user-owned `/app` and `/home/node`, plus a sticky `/tmp`

Not included: any shell, `npm`/`corepack`, coreutils, or a package manager.

## Customizing

- **Node version:** change `pkgs.nodejs-slim_22` in `image.nix`
  (e.g. `nodejs-slim_20`).
- **No outbound TLS needed?** Drop `pkgs.cacert` from `contents` and remove
  the `SSL_CERT_FILE` / `NODE_EXTRA_CA_CERTS` env vars for an even smaller image.
- **Different user/uid:** edit `user` / `uid` / `gid` in `image.nix` (and the
  `--chown` in your `Dockerfile`).

## Pure-Nix alternative (no `Dockerfile`)

If you'd rather not use Docker's `COPY` at all, add your app straight into the
image by listing it in `contents` and pointing `Cmd` at it — for example, in
`image.nix`:

```nix
contents = [ node pkgs.cacert ./app ];
config.Cmd = [ "/index.js" ];   # ./app is unpacked at the image root
```

Then `nix build && docker load < result` gives you a ready-to-run image with
the app already inside.

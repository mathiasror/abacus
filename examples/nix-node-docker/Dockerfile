# Thin application layer on top of the Nix-built base image.
#
# The base has no shell, so you cannot RUN anything here — but COPY, FROM,
# CMD and ENTRYPOINT are handled by the builder/daemon, not by a shell
# inside the image, so they all work fine.
#
#   nix build                       # build the base (produces ./result)
#   docker load < result            # -> node-minimal:latest
#   docker build -t my-node-app .   # copy your app on top
#   docker run --rm -p 3000:3000 my-node-app
FROM node-minimal:latest

# Copy your already-built app (sources + any vendored node_modules) and make
# the unprivileged user own it. Bring your own dependencies: there is no npm
# in the image, so install/build them beforehand (e.g. `npm ci --omit=dev`)
# and copy node_modules in, or vendor them via Nix.
COPY --chown=1000:1000 app/ /app/

# Entrypoint is already `node`; just point at your entry script.
CMD ["/app/index.js"]

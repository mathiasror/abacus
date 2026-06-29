// Tiny dependency-free example app so the image can be run end to end.
// Replace this whole `app/` directory with your own Node.js application.
const http = require("node:http");

const port = process.env.PORT || 3000;

http
  .createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        message: "Hello from a minimal, rootless, Nix-built Node.js image!",
        node: process.version,
        uid: process.getuid(),
        gid: process.getgid(),
      }),
    );
  })
  .listen(port, () => console.log(`listening on :${port}`));

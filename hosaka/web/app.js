// Minimal hosaka web client: open a WebSocket, send a line of text, play the
// streamed float32 PCM through an AudioWorklet. The audio contract is 24 kHz
// mono float32 LE everywhere, so the AudioContext is pinned to 24 kHz and
// binary frames are fed to the worklet verbatim.
const $ = (id) => document.getElementById(id);
const setStatus = (s) => {
  $("status").textContent = s;
};

let ctx = null; // AudioContext
let node = null; // AudioWorkletNode (pcm-player)
let ws = null; // WebSocket

async function ensureAudio() {
  if (ctx) return;
  ctx = new AudioContext({ sampleRate: 24000 });
  await ctx.audioWorklet.addModule("/app/pcm-player.js");
  node = new AudioWorkletNode(ctx, "pcm-player");
  node.connect(ctx.destination);
}

function openSocket() {
  return new Promise((resolve, reject) => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/v1/audio/stream`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      setStatus("connected");
      resolve();
    };
    ws.onerror = () => {
      setStatus("connection error");
      reject(new Error("ws error"));
    };
    ws.onclose = () => setStatus("disconnected");
    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.type === "error") setStatus("error: " + msg.detail);
        else if (msg.type === "start") setStatus("speaking...");
        else if (msg.type === "end") setStatus("done");
        return;
      }
      node.port.postMessage(new Float32Array(e.data)); // PCM frame
    };
  });
}

async function speak() {
  await ensureAudio();
  if (ctx.state === "suspended") await ctx.resume(); // gesture-unlock audio
  if (!ws || ws.readyState !== WebSocket.OPEN) await openSocket();
  node.port.postMessage(null); // flush any tail from a previous line
  ws.send(
    JSON.stringify({
      input: $("text").value,
      backend: $("backend").value,
      voice: $("voice").value.trim() || "af_heart",
    }),
  );
}

window.addEventListener("DOMContentLoaded", () => {
  $("speak").addEventListener("click", () => {
    speak().catch((err) => setStatus("error: " + err.message));
  });
});

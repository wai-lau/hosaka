// hosaka web client: WebSocket -> AudioWorklet PCM playback (24 kHz mono
// float32). Voices come from GET /v1/voices (each carries its backend +
// description); the synth knobs are sent as native param values; volume is a
// client-side gain applied in the worklet.
const $ = (id) => document.getElementById(id);
const setStatus = (s) => {
  $("status").textContent = s;
};

let ctx = null; // AudioContext
let node = null; // AudioWorkletNode (pcm-player)
let ws = null; // WebSocket
let gain = 1.0; // client playback gain

const PARAM_IDS = ["exaggeration", "cfg_weight", "temperature", "speed"];

async function loadVoices() {
  const sel = $("voice");
  let voices;
  try {
    voices = await (await fetch("/v1/voices")).json();
  } catch {
    sel.innerHTML = '<option value="af_heart">af_heart</option>';
    return;
  }
  // group options by backend
  const groups = {};
  for (const v of voices) (groups[v.backend] ||= []).push(v);
  const labels = {
    kokoro: "kokoro (realtime)",
    chatterbox: "chatterbox (clone)",
    piper: "piper (character)",
  };
  sel.innerHTML = "";
  for (const backend of Object.keys(groups)) {
    const og = document.createElement("optgroup");
    og.label = labels[backend] || backend;
    for (const v of groups[backend]) {
      const o = document.createElement("option");
      o.value = v.id;
      o.dataset.backend = v.backend;
      o.textContent = v.description ? `${v.id} - ${v.description}` : v.id;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  sel.addEventListener("change", reflectBackend);
  reflectBackend();
}

function selectedBackend() {
  const o = $("voice").selectedOptions[0];
  return o ? o.dataset.backend : "kokoro";
}

// Dim the chatterbox-only knobs for non-chatterbox voices (they do nothing
// there -- kokoro and piper only honor speed).
function reflectBackend() {
  $("cb-knobs").classList.toggle("off", selectedBackend() !== "chatterbox");
}

function wireKnobs() {
  for (const id of [...PARAM_IDS, "volume"]) {
    const el = $(id);
    const out = $(id + "-val");
    const show = () => {
      out.textContent = parseFloat(el.value).toFixed(2);
    };
    el.addEventListener("input", () => {
      show();
      if (id === "volume") {
        gain = parseFloat(el.value);
        if (node) node.port.postMessage({ gain });
      }
    });
    show();
  }
}

function params() {
  const p = {};
  for (const id of PARAM_IDS) p[id] = parseFloat($(id).value);
  return p;
}

async function ensureAudio() {
  if (ctx) return;
  ctx = new AudioContext({ sampleRate: 24000 });
  await ctx.audioWorklet.addModule("/app/pcm-player.js");
  node = new AudioWorkletNode(ctx, "pcm-player");
  node.port.postMessage({ gain });
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
      backend: selectedBackend(),
      voice: $("voice").value,
      params: params(),
    }),
  );
}

window.addEventListener("DOMContentLoaded", () => {
  wireKnobs();
  loadVoices();
  $("speak").addEventListener("click", () => {
    speak().catch((err) => setStatus("error: " + err.message));
  });
});

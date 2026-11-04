import { Reactor, type ReactorStatus } from "@reactor-team/js-sdk";
import "./style.css";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<header><a class="brand" href="https://reactor.inc" target="_blank" rel="noreferrer"><img src="/reactor-symbol.svg" alt="Reactor" width="32" height="23"><strong>Helios</strong><span>on Reactor</span></a>
<div class="runtime-picker"><label for="endpoint">Runtime URL</label><input id="endpoint" type="url" value="http://localhost:8090" spellcheck="false"></div>
<div class="connection"><span id="status" role="status" data-status="disconnected">Offline</span><button id="connect">Connect</button></div></header>
<main><aside class="scene-panel"><h1>Create a video</h1><p class="hint scene-intro">Choose a starting point, then guide what happens next.</p>
<label for="mode">Start from</label><select id="mode"><option value="text">Text</option><option value="image">Image + text</option><option value="video">Video + text</option></select>
<div id="reference-field" hidden><div class="reference-picker"><label for="reference">Reference</label><input id="reference" type="file" accept="image/png,image/jpeg,image/webp"><p id="file-help" class="hint"></p></div><img id="preview" alt="Selected reference" hidden></div>
<label for="prompt">Scene prompt</label><textarea id="prompt" rows="5" maxlength="4096" placeholder="Describe the scene, subjects and action…">A vibrant tropical fish swims gracefully among colorful coral reefs in clear turquoise water. Its blue and yellow scales shimmer in the sunlight as its fins move fluidly. The camera follows the fish closely through the reef.</textarea>
<div class="prompt-actions"><button id="start" class="primary" disabled>Start video</button><button id="update" class="primary secondary" disabled>Update prompt</button></div>
<p id="error" role="alert" hidden></p><p class="hint prompt-help">Update replaces the scene description. Keep the setting and subjects; describe the next action.</p><details class="generation-settings"><summary>Generation settings</summary><div class="settings-fields">
<div><label for="seed">Seed for new video</label><input id="seed" type="number" min="0" max="2147483647" step="1" value="42"></div>
<div><label for="chunks">Video length</label><select id="chunks"><option value="10">10 chunks · about 14 seconds</option><option value="44" selected>44 chunks · about 1 minute</option><option value="88">88 chunks · about 2 minutes</option><option value="132">132 chunks · about 3 minutes</option></select></div></div><p class="hint">Durations shown at 24 FPS for reference. Live playback adapts to generation speed.</p></details></aside>
<section class="output-panel"><div class="heading"><h2>Live preview</h2><span class="tag">640 × 384 · Adaptive playback</span></div>
<div class="screen"><video id="video" muted playsinline aria-label="Generated Helios video"></video><div id="placeholder"><span>☼</span><h3 id="waiting">Your story starts here</h3><p>Connect, set the scene, then Start video.</p></div></div>
<div class="player-bar"><span id="progress" role="status">No video generated yet.</span><div><button id="play" hidden>Play video</button><button id="pause" disabled>Pause</button><button id="reset" disabled>Reset video</button><button id="stop" disabled>Stop</button><button id="fullscreen" disabled>Fullscreen</button></div></div>
<div class="controls"><h3>Current prompt</h3><p id="accepted-prompt" class="hint">Your accepted prompt will appear here.</p><p class="hint">Prompt changes are experimental. Each update replaces the full scene description and blends over three chunks. Short commands can cause visual drift or collapse. Already buffered video plays first.</p></div></section>

<footer><span>Helios-Distilled · PKU YuanGroup</span><a href="https://github.com/PKU-YuanGroup/Helios" target="_blank" rel="noreferrer">Model & research ↗</a></footer></main>`;

type State = {
  world_id: number; prompt: string; seed: number; mode: string;
  reference_name: string; active: boolean; paused: boolean; limit_reached: boolean;
  completed_chunks: number; output_frames: number; max_chunks: number;
  last_chunk_seconds: number | null;
};
const el = <T extends HTMLElement>(id: string) => document.getElementById(id)! as T;
const button = (id: string) => el<HTMLButtonElement>(id);
const input = (id: string) => el<HTMLInputElement>(id);
const mode = el<HTMLSelectElement>("mode");
const prompt = el<HTMLTextAreaElement>("prompt");
const video = el<HTMLVideoElement>("video");
type StartingMode = "text" | "image" | "video";
const promptDrafts: Record<StartingMode, string> = {
  text: prompt.value,
  image: "Continue the scene naturally with subtle, realistic motion. Preserve the subjects, composition, lighting, and overall appearance of the reference image.",
  video: "Continue the reference video naturally. Preserve the subjects, setting, lighting, and visual style, with smooth, coherent motion.",
};
let draftMode: StartingMode = "text";
const fromLink = new URLSearchParams(location.search).get("endpoint");
if (fromLink) input("endpoint").value = fromLink;
let client: Reactor | undefined;
let state: State | undefined;
let ready = false, busy = false, playing = false;
let stream = new MediaStream();
let preview: string | undefined;
let frameCallback: number | undefined;
let mediaEpoch = 0;

function error(message: string) {
  el("error").textContent = message;
  el("error").hidden = !message;
}
function clearVideo() {
  mediaEpoch++;
  if (frameCallback !== undefined) video.cancelVideoFrameCallback(frameCallback);
  frameCallback = undefined;
  video.pause(); video.srcObject = null; playing = false;
  button("play").hidden = true;
}
function refresh() {
  button("connect").disabled = busy;
  input("endpoint").disabled = !!client || busy;
  button("start").disabled = !ready || busy || !prompt.value.trim() || (mode.value !== "text" && !input("reference").files?.length);
  button("start").textContent = state?.active ? "Start new video" : "Start video";
  button("update").disabled = !ready || busy || !state?.active || state.limit_reached || !prompt.value.trim();
  button("pause").disabled = !ready || busy || !state?.active || state.limit_reached;
  button("pause").textContent = state?.paused ? "Resume" : "Pause";
  button("reset").disabled = !ready || busy || !state?.prompt;
  button("stop").disabled = !ready || busy || !state?.active;
  button("fullscreen").disabled = !playing;
  for (const id of ["mode", "chunks", "seed", "reference"]) (el(id) as HTMLInputElement).disabled = busy;
  prompt.disabled = busy;
  el("placeholder").hidden = playing;
  el("waiting").textContent = state?.active ? "Generating your first view…" : ready ? "Ready when you are" : "Your story starts here";
  const seconds = state?.last_chunk_seconds;
  const timing = seconds ? ` · ${seconds.toFixed(2)} s / chunk` : "";
  el("progress").textContent = state?.prompt
    ? `Video ${state.world_id} · ${state.completed_chunks}/${state.max_chunks} chunks · ${state.output_frames} frames${timing}${state.limit_reached ? " · Complete" : state.paused ? " · Paused" : !state.active ? " · Stopped" : ""}`
    : "No video generated yet.";
  el("accepted-prompt").textContent = state?.prompt || "Your accepted prompt will appear here.";
}
async function play() {
  const epoch = mediaEpoch, current = video.srcObject;
  try {
    await video.play();
    if (epoch === mediaEpoch && video.srcObject === current) button("play").hidden = true;
  } catch {
    if (epoch === mediaEpoch && video.srcObject === current) button("play").hidden = false;
  }
}
function attach() {
  if (!ready || !state?.active || state.completed_chunks < 1 || !stream.getVideoTracks().length || video.srcObject === stream) return;
  video.srcObject = stream;
  const epoch = mediaEpoch;
  frameCallback = video.requestVideoFrameCallback(() => {
    if (epoch !== mediaEpoch || video.srcObject !== stream) return;
    frameCallback = undefined; playing = true; refresh();
  });
  void play();
}
function setStatus(status: ReactorStatus) {
  const labels = { disconnected: "Offline", connecting: "Connecting", waiting: "Waiting", ready: "Online" };
  el("status").textContent = labels[status];
  el("status").dataset.status = status;
}
function disconnected() {
  ready = false; state = undefined; clearVideo(); stream = new MediaStream();
  setStatus("disconnected"); button("connect").textContent = "Connect"; refresh();
}
async function act(fn: () => Promise<void>) {
  if (busy) return;
  busy = true; error(""); refresh();
  try { await fn(); }
  catch (cause) { error(cause instanceof Error ? cause.message : String(cause)); }
  finally { busy = false; refresh(); }
}
async function command(name: string, data: Record<string, unknown> = {}) {
  if (!client || !ready) throw new Error("Connect before sending commands.");
  const reply = await client.sendCommand(name, data);
  if (reply === undefined) throw new Error(el("error").textContent || "The command was refused.");
  return reply;
}
button("connect").onclick = () => void act(async () => {
  if (client) {
    // SDK disconnect ends the session. Await teardown before clearing its UI.
    const previous = client;
    await previous.disconnect();
    if (client === previous) client = undefined;
    disconnected();
    return;
  }
  const endpoint = input("endpoint").value.trim();
  if (!["http:", "https:"].includes(new URL(endpoint).protocol)) throw new Error("Use an HTTP or HTTPS Runtime endpoint.");
  const connection = new Reactor({
    modelName: "your-org/helios-distilled", apiUrl: endpoint, local: true,
    controlRequestTimeoutMs: 60000,
    modelTracks: [{ name: "main_video", kind: "video", direction: "recvonly" }],
  });
  client = connection;
  connection.on("statusChanged", status => {
    if (client !== connection) return;
    ready = status === "ready"; setStatus(status);
    button("connect").textContent = status === "disconnected" ? "Connect" : "Disconnect";
    if (status === "disconnected") { client = undefined; disconnected(); }
    refresh(); attach();
  });
  connection.on("error", cause => { if (client === connection) error(cause.message); });
  connection.on("trackReceived", (_name, track) => {
    if (client !== connection || track.kind !== "video") return;
    if (!stream.getTracks().some(t => t.id === track.id)) stream.addTrack(track);
    attach();
  });
  connection.on("message", message => {
    if (client !== connection || message.type !== "state_update") return;
    const next = message.data as State;
    if (next.world_id !== state?.world_id || !next.active) clearVideo();
    if (!state?.prompt && next.prompt) {
      prompt.value = next.prompt; input("seed").value = String(next.seed);
    }
    state = next; refresh(); attach();
  });
  try { await connection.connect(); }
  catch (cause) {
    await connection.disconnect();
    if (client === connection) client = undefined;
    disconnected(); throw cause;
  }
});
function referenceChanged() {
  if (preview) URL.revokeObjectURL(preview);
  preview = undefined;
  const file = input("reference").files?.[0];
  if (file && mode.value === "image") preview = URL.createObjectURL(file);
  const image = el<HTMLImageElement>("preview"); image.hidden = !preview;
  if (preview) image.src = preview; else image.removeAttribute("src");
  refresh();
}
mode.onchange = () => {
  promptDrafts[draftMode] = prompt.value;
  draftMode = mode.value as StartingMode;
  prompt.value = promptDrafts[draftMode];
  input("reference").value = "";
  input("reference").accept = mode.value === "image" ? "image/png,image/jpeg,image/webp" : "video/mp4,video/webm,video/quicktime";
  el("reference-field").hidden = mode.value === "text";
  el("file-help").textContent = mode.value === "image" ? "PNG, JPEG or WebP · up to 20 MiB" : "33–330 frames · up to 100 MiB. The generated video continues the reference.";
  referenceChanged();
};
input("reference").onchange = referenceChanged;
prompt.oninput = refresh;
button("start").onclick = () => void act(async () => {
  const seed = Number(input("seed").value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) throw new Error("Seed must be an integer between 0 and 2147483647.");
  const file = input("reference").files?.[0];
  let reference;
  if (mode.value !== "text") {
    if (!file || !client) throw new Error("Select a reference first.");
    const limit = mode.value === "image" ? 20 : 100;
    if (file.size > limit * 1024 * 1024) throw new Error(`Choose a reference smaller than ${limit} MiB.`);
    reference = await client.uploadFile(file);
  }
  await command("start", { prompt: prompt.value.trim(), seed, mode: mode.value,
    chunks: Number(el<HTMLSelectElement>("chunks").value), ...(reference ? { reference } : {}) });
});
button("update").onclick = () => void act(async () => { await command("update_prompt", { prompt: prompt.value.trim() }); });
button("pause").onclick = () => void act(async () => { await command(state?.paused ? "resume" : "pause"); });
button("reset").onclick = () => void act(async () => { await command("reset"); });
button("stop").onclick = () => void act(async () => { await command("stop"); });
button("play").onclick = () => void play();
button("fullscreen").onclick = () => void el("video").requestFullscreen();
refresh();

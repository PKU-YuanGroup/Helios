# Helios browser demo

A local browser client using `@reactor-team/js-sdk` 3.3.0, with the WorldCrafter
Reactor demo's layout. Start the model as described in the parent README, then:

```sh
npm ci
npm run dev
```

Open the printed Vite URL (normally http://localhost:5173). Enter the Runtime
URL (normally http://localhost:8090), click **Connect**, select Text, Image or
Video, enter a prompt and click **Start video**. File selection waits for Start.
**Update prompt** keeps the current history and blends the text condition over
three chunks. Pause holds generation; Reset uses the accepted inputs; Stop ends
generation. Disconnect ends the shared Runtime session.

Each update replaces the complete scene description. Keep the setting and
subjects you want to retain, and describe the new action. Upstream prompt
interpolation is experimental: short commands or abrupt changes can cause
color saturation and scene collapse, including in official inference without
Reactor Runtime. Starting a new video clears an already degraded history.

Start video and Update prompt are below the prompt, with a fixed bottom action
bar on phones. Expand **Generation settings** under the left-hand prompt controls to change Seed
or Video length. Image and video modes use neutral continuation prompts by
default and retain their own edited drafts when you switch modes or files.

Build and serve the production client with `npm run build` and `npm run preview`.
The browser and Runtime need working HTTP and WebRTC connectivity. SSH HTTP
forwarding also needs a configured TURN/TCP media path; see the parent README.
`?endpoint=http%3A%2F%2Flocalhost%3A8090` pre-fills the endpoint.

This demo uses local mode. A hosted client needs server-side API-key exchange
for a stable session JWT and an `owner/name` model identifier; never put an API
key into browser code. Command replies are awaited; only `state_update` is
subscribed to. Refused commands leave the accepted video and pause state intact.

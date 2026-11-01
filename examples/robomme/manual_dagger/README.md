# RoboTTT Manual DAgger

This service lets a human decide when a RoboTTT rollout must switch from the
policy to the task's existing official expert. The browser is only a control and
display surface: the simulator, policy server, expert, HDF5 recorder, and MP4
encoder all run in the RTX 5090 CCI.

## Controls

- `s`: start the prepared episode.
- `i`: latch takeover immediately and run the official expert for one current
  subtask, then pause. Press `z` to resume the policy or `i` for another expert
  subtask; repeated interventions are supported. If the entire task finishes,
  review and save with `c` as before. If policy inference is in flight, its result is discarded
  before another action can execute.
- `z`: pause or resume only the enabled environment. The independent control
  environment keeps running while enabled is paused.
- `r`: discard and reset only enabled on the same task, episode, and seed.
  Control keeps running and its filmstrip is not cleared.
- `p`: stop the episode and discard all of its staged data.
- `c`: after the episode ends, publish its HDF5, MP4, and metadata.
- `d`: after the episode ends, discard enabled while retaining the completed
  control MP4 for audit.
- `b`: block the current seed persistently and advance to another legal seed.
- `+` / `-`: increase or decrease playback FPS. The default is 5 FPS.

The page renders two full-width base-and-wrist camera rows. The independent
control environment is on top, always stays on policy, and starts as soon as the
episode is prepared without waiting for `s`; the enabled environment remains on
its initial frame until `s` and accepts `i`, `z`, and `p`. Both environments use
the same task, episode, seed, and prompt but have separate simulator and policy
client state. The complete task prompt and status are ordinary DOM text on a
fixed white layout; they are not burned into the video frames.

As soon as both environments and their initial frames are ready, the server
starts prefetching the first policy action chunk in the background. `s` never
waits for prefetch before being accepted: if the chunk is ready execution starts
immediately, otherwise execution begins as soon as the in-flight inference
finishes. An `i`, `z`, or `p` received during that inference is latched before
any inferred action can execute.

Both rows have browser-side frame scrubbers. New frames keep appending while an
older frame is selected, so the operator can review the current rollout before
pressing `c` or `d`. A control frame can also be selected as the intervention
point. The enabled environment is then discarded, reset on the same episode and
seed, started automatically, and switched to the official expert at that exact
control action step. Live keyboard commands are processed before this scheduled
takeover. The task selector optionally accepts an official seed for the next
pair. An invalid seed is rejected instead of silently changing the environment.

`collection_progress.json` lives at the persistent output root. It records
blocked seeds, allocation cursors, unique take numbers, retained counts, and a
manually selected next task. The default 100-per-task target determines where
`c` goes next: the current task remains selected until it reaches 100, then the
next incomplete task is selected. After all 16 targets are met, collection
continues without a 1,600-sample cap. Restarts reconcile counts from published
enabled JSON artifacts.

In persistent collection mode, background control screening covers every legal
train episode once. Successful controls are remembered in
`control_screening.json` and never enter the annotation queue; failed, timeout,
and error controls form the only selectable pool. Because each task has 100
legal train episodes but may have fewer than 100 failures, failed seeds cycle
for additional intervention takes until the task has 100 retained enabled
samples. A scene-generation failure abandons only that control candidate and
does not stop the browser service. The default task order starts with
`VideoPlaceOrder`.

## Process and storage boundaries

The async web server receives keys and streams JPEG previews. Each simulator
is serialized on its own dedicated thread. MP4 encoding and final artifact
publication happen in one separate child process, so disk and AOSS copies do
not block the intervention loop.

Each episode is staged under `/tmp/<session>/<episode-id>`. `p` deletes both
staging directories; `d` deletes enabled and publishes control; `c` publishes
both. Persistent artifacts use this layout:

```text
<output-root>/
  manifest.jsonl
  collection_progress.json
  <Task>/
    <Task>_ep<E>_seed<S>_take<N>_enabled.h5
    <Task>_ep<E>_seed<S>_take<N>_enabled.mp4
    <Task>_ep<E>_seed<S>_take<N>_enabled.json
    <Task>_ep<E>_seed<S>_take<N>_control.mp4
    <Task>_ep<E>_seed<S>_take<N>_control.json
```

The CCI launch uses an AFS output root for the authoritative dataset and may
mirror the same files to AOSS. No runtime bundle, encoded fragment, or other
intermediate output is placed under `/workspace`.

## Launch

On a GPU CCI with the repository, checkpoint, AFS, and AOSS mounts available:

```bash
cd /workspace/ttt_ws/robomme_policy_learning/examples/robomme
/workspace/ttt_ws/robomme_policy_learning/.venv/bin/python \
  -m manual_dagger.launch \
  --host 0.0.0.0 \
  --port 8765 \
  --tasks all \
  --target-per-task 100 \
  --fps 5 \
  --staging-root /tmp/robottt-manual-dagger \
  --output-root /workspace/ttt_ws/runs/robottt/manual_dagger/<session> \
  --mirror-root /aoss/ttt_runs/robottt/manual_dagger/<session> \
  --access-token <unguessable-token>
```

`manual_dagger.launch` keeps the CCI container alive and runs the actual
policy/browser workload in the `robottt-manual-dagger` screen session. The
policy uses GPU 0; after its health endpoint is ready, the browser/simulator is
started with `third_party/robomme_benchmark/.venv`. The two processes
communicate only through the local policy port. From the CCI WebTerminal,
restart or inspect only the workload without restarting the CCI:

```bash
cd /workspace/ttt_ws/robomme_policy_learning/examples/robomme
/workspace/ttt_ws/robomme_policy_learning/.venv/bin/python -m manual_dagger.launch status
/workspace/ttt_ws/robomme_policy_learning/.venv/bin/python -m manual_dagger.launch restart
screen -r robottt-manual-dagger
```

Expose the declared CCI browser port with a private SSH or VS Code port
forward; the URL additionally requires the access token.

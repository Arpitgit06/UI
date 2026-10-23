# Rebuilt Forge Hub example

A reviewed reconstruction of the supplied recording, built from React components,
CSS panels, extracted individual icon artwork, and separately preserved sphere artwork.
There are no full-frame image or recorded-video layers. `scene.json` owns tool
names, positions and colors. Icon crops are reviewed source assets extracted with
`tools/extract_reference_icons.py`. The transparent sphere sprite atlas contains
only isolated artwork frames; panels, text, inputs and cards are independently
rendered. Nebula and floor geometry remain procedural approximations.

This is an authored reference implementation, not output automatically inferred by
the extraction pipeline. It establishes a component/animation target for improving
that pipeline. The original 3D geometry is unknown and is not reconstructed.

`motion.json` contains 120 observations across the source's ten seconds. The sphere
atlas preserves observed artwork motion; smooth 2D ellipses are fitted to observed
gold ring pixels, and glow/wave amplitude follow measured brightness envelopes.
Ellipse fits and floor geometry are approximate, not recovered 3D motion.
Pause/resume retains the animation phase, reduced motion is respected, and playback
holds the final observation. Replay source timing restarts it. `?time=4.5` selects
an observation for review. With a matching reference recording, regenerate
measurements and the atlas from the project root:

    .venv/Scripts/python.exe tools/analyze_reference_motion.py "path/to/reference.mp4" examples/forge-hub/motion.json

The clock starts at the reference's 23:58 and advances as a local demo clock.

Tool buttons open accessible local workspaces. Task submission updates a local
activity feed. Attachments remain in browser memory and voice input uses optional
browser speech recognition after a click. No external tool APIs are connected.

From the project root, build and verify with:

    node tools/frontend/forge-example.mjs

Then run OmniUI and open `/examples/forge-hub/`.

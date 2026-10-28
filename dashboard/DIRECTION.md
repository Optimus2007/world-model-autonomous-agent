# Dashboard direction (Impresario skill, app/dashboard scope)

Source: github.com/Shushant17711/impresario (MIT), `SKILL.md` and `references/colour.md`.
Only the parts that apply to an operational dashboard are used. That scope says to keep the work
surface clear, to use motion only to explain state changes, and to avoid cinematic or scroll effects.

- **Subject:** a lab instrument replaying one agent run, read like an engineering plot.
- **Palette structure:** a tonal world in one drafting-film blue hue (L 0.17–0.995, hue about 240–250).
  The only other colours are data marks, each carrying a meaning:
  - residual/error: red-orange
  - disagreement: teal
  - dynamics changed: amber band
  - alarm: red
  - model update and goal: green
  - prediction: blue

  There is no paper + ember and no neon. Dark mode follows `prefers-color-scheme`.
- **Motion:** only the step cursor advances when Play is pressed. Nothing animates on its own.
- **Data rule:** every mark comes from the selected run's `results/<run>/demo_run.json` and
  `summary.json`, written by the experiment scripts. If a file is missing, the page shows an error rather
  than placeholder data.
- **Not used:** Impresario's motion kit, the GSAP/Lenis scroll machinery and the screenshot review
  loop (`scripts/shoot.mjs` needs Node and Playwright, which are not installed here).

"""Capture the composition frame by frame: render(t) is a pure function of time."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FPS, DUR = 30, 20.0
out = HERE / "frames"
out.mkdir(exist_ok=True)
stills = [float(x) for x in sys.argv[1:]]  # optional: only these times -> stills/

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.goto((HERE / "index.html").as_uri())
    pg.evaluate("document.fonts.ready")
    if stills:
        (HERE / "stills").mkdir(exist_ok=True)
        for t in stills:
            pg.evaluate(f"render({t})")
            pg.screenshot(path=str(HERE / "stills" / f"t{t:05.2f}.png"))
    else:
        n = int(FPS * DUR)
        for f in range(n):
            pg.evaluate(f"render({f / FPS})")
            pg.screenshot(path=str(out / f"f{f:04d}.png"))
            if f % 60 == 0:
                print(f, "/", n, flush=True)
    b.close()

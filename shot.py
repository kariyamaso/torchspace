"""Screenshot harness for the PoC viewer (verification + design-doc figures)."""
import asyncio
import sys

from playwright.async_api import async_playwright

URL = "file:///home/claude/torchspace_poc/out/torchspace_viewer.html"


async def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "default"
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/opt/pw-browsers/chromium",
            args=["--use-gl=angle", "--use-angle=swiftshader",
                  "--enable-unsafe-swiftshader"])
        page = await browser.new_page(viewport={"width": 1680, "height": 1000})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text)
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        await page.goto(URL)
        await page.wait_for_timeout(1800)

        if scenario == "resnet":
            await page.select_option("#modelSel", "1")
            await page.wait_for_timeout(1200)
        elif scenario == "resnet_expand":
            await page.select_option("#modelSel", "1")
            await page.wait_for_timeout(800)
            await page.evaluate(
                "expanded.add('layer1.0'); expanded.add('layer2.0');"
                "rebuildScene(); fitCamera();")
            await page.wait_for_timeout(800)
        elif scenario == "ortho":
            await page.click("#btnOrtho")
            await page.wait_for_timeout(600)
        elif scenario == "tensor":
            await page.select_option("#modelSel", "1")
            await page.wait_for_timeout(600)
            await page.check("#mTensor")
            await page.wait_for_timeout(900)
        elif scenario == "replay":
            await page.evaluate(
                "$('timeline').value = Math.floor(tlFrames.length*0.35);"
                "applyTimeline($('timeline').value);"
                "const f = tlFrames[$('timeline').value-1];"
                "const own = displayOwner(f.node, false);"
                "const m = nodeMesh[own];"
                "controls.target.copy(m.position); controls.r = 22;"
                "controls.theta = 0.35; controls.phi = 1.1; controls.apply();"
                "spawnPulses(f);")
            await page.wait_for_timeout(260)
        elif scenario == "select":
            await page.evaluate(
                "select('backbone.4', true); controls.r = 26; controls.apply();")
            await page.wait_for_timeout(500)

        await page.screenshot(path=f"shots/{scenario}.png")
        print("errors:", [e for e in errors if "GPU" not in e and "WebGL" not in e][:8]
              or "none")
        await browser.close()


asyncio.run(main())

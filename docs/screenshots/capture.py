"""Screenshots of the app for docs/screenshots/ (Playwright, a stub LLM server answering with the rule-based FakeLLM).

    python docs/screenshots/capture.py            # 1440×900 light, plus a 1280 px and a dark-mode check
The LLM in these screenshots is NOT a real model.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments'))
from fake_llm import RuleBasedFakeLLM  # noqa: E402

LLM_PORT, APP_PORT, DARK_PORT = 11556, 8656, 8657
FAKE = RuleBasedFakeLLM()


class StubLLM(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ctx = json.loads(body['messages'][-1]['content']) if body['messages'][-1]['content'].startswith('{') else {}
        if 'window_facts' in ctx:
            out = FAKE.evaluate(ctx)
        else:
            bad = bool(ctx.get('suspicious'))
            out = {'trust_score': 0.2 if bad else 0.9, 'action': 'BLOCK' if bad else 'ALLOW', 'risk_score': 0.8 if bad else 0.1,
                   'confidence': 0.7, 'reason': 'stand-in rule', 'factors': ['stand-in'], 'behavior_assessment': [],
                   'required_validation': 'ZK proof'}
        time.sleep(0.05)
        data = json.dumps({'message': {'content': json.dumps(out)}}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


async def session(port: int, width: int, suffix: str, full: bool):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': width, 'height': 900})

        async def key(k, wait=700):
            await page.locator(f'.st-key-{k} button').first.click()
            await page.wait_for_timeout(wait)

        async def pick(label, option):
            await page.get_by_test_id('stSidebar').get_by_test_id('stSelectbox').filter(has_text=label).first.click()
            await page.wait_for_timeout(400)
            await page.get_by_role('option', name=option, exact=True).click()
            await page.wait_for_timeout(1200)

        async def steps(n):
            for _ in range(n):
                await key('live_step', 350)
            await page.wait_for_timeout(1500)

        async def shot(name):
            await page.screenshot(path=str(OUT / f'{name}{suffix}.png'))
            if full:   # Streamlit scrolls inside the page: a tall viewport shows everything for review
                await page.set_viewport_size({'width': width, 'height': 3200})
                await page.wait_for_timeout(1500)
                await page.screenshot(path=str(OUT / f'{name}{suffix}_full.png'))
                await page.set_viewport_size({'width': width, 'height': 900})
                await page.wait_for_timeout(800)
            print('screenshot', name + suffix)

        await page.goto(f'http://localhost:{port}'); await page.wait_for_timeout(6000)
        await key('live_reset', 1500)
        await steps(30)
        await shot('01_live_normal_day_t30')

        await pick('Scenario', 'compromised_valid_token')
        await key('live_reset', 1500)
        await steps(42)
        await shot('02_live_compromised_t42')

        await page.get_by_test_id('stSidebar').get_by_text('Both — isolated comparison').click(); await page.wait_for_timeout(1200)
        await key('live_reset', 1500)
        await steps(45)
        await shot('03_both_fake_llm')
        await page.get_by_role('tab', name='Bayesian vs LLM').click(); await page.wait_for_timeout(5000)
        await shot('04_bayes_vs_llm_tab')
        await page.get_by_role('tab', name='Trust calculation').click(); await page.wait_for_timeout(4000)
        await shot('05_trust_calculation_tab')

        await page.goto(f'http://localhost:{port}'); await page.wait_for_timeout(6000)
        await page.get_by_text('Classic evaluator', exact=True).click(); await page.wait_for_timeout(2500)
        await key('classic_run', 4000)
        await shot('06_classic')
        await browser.close()


def start_app(port, extra):
    env = dict(os.environ, OLLAMA_ENDPOINT=f'http://127.0.0.1:{LLM_PORT}/api/chat')
    return subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'), '--server.headless', 'true',
                             '--server.port', str(port), '--browser.gatherUsageStats', 'false', *extra],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    server = ThreadingHTTPServer(('127.0.0.1', LLM_PORT), StubLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    light = start_app(APP_PORT, [])
    # Streamlit's built-in dark theme (what a user gets from Settings → Dark)
    dark = start_app(DARK_PORT, ['--theme.base', 'dark', '--theme.backgroundColor', '#0E1117', '--theme.secondaryBackgroundColor', '#262730',
                                 '--theme.textColor', '#FAFAFA'])
    try:
        time.sleep(8)
        what = sys.argv[1:] or ['light', 'narrow', 'dark']
        if 'light' in what:
            asyncio.run(session(APP_PORT, 1440, '', full=True))
        if 'narrow' in what:
            asyncio.run(session(APP_PORT, 1280, '_1280', full=False))
        if 'dark' in what:
            asyncio.run(session(DARK_PORT, 1440, '_dark', full=False))
    finally:
        light.terminate(); dark.terminate(); server.shutdown()


if __name__ == '__main__':
    main()

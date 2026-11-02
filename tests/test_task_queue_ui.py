"""Exercise the actual shared queue widget in headless Edge, with a fixture API."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]


def test_queue_dialog_cancel_and_failure_recovery():
    rows=[dict(id='a'*32,project_title='正在处理的歌曲',profile='stems',status='running',stage_label='正在分离'),
          dict(id='b'*32,project_title='<b>等待歌曲</b>',profile='piano',status='queued',queue_position=1)]
    calls=[];failed=[False]
    html='<html><head><meta charset="utf-8"><link rel="stylesheet" href="/assets/task-queue.css"><script defer src="/assets/task-queue.js"></script></head><body><header>Chord Studio</header></body></html>'
    def route(request):
        url=request.request.url.split('http://studio.test')[-1]
        if url.startswith('/assets/'):
            p=ROOT/'frontend'/url.rsplit('/',1)[-1]
            request.fulfill(body=p.read_bytes(),content_type='text/css' if p.suffix=='.css' else 'application/javascript');return
        if url=='/api/jobs':
            request.fulfill(json=dict(active=[r for r in rows if r['status'] in {'running','queued'}],recent=[r for r in rows if r['status']=='cancelled']));return
        if url.endswith('/cancel'):
            assert request.request.method=='POST'
            assert request.request.headers['x-harmony-client']=='studio'
            jid=url.split('/')[3];calls.append(jid)
            if jid=='a'*32 and not failed[0]:
                failed[0]=True;request.fulfill(status=503,json={'detail':'临时错误，请重试'});return
            row=next(r for r in rows if r['id']==jid);row.update(status='cancelled',stage_label='已取消')
            request.fulfill(json=row);return
        request.fulfill(body=html,content_type='text/html')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport={'width':1280,'height':850});page.route('http://studio.test/**',route)
            page.goto('http://studio.test/');page.get_by_role('button',name='任务队列 · 2').click()
            expect(page.get_by_role('dialog')).to_be_visible()
            expect(page.locator('.tq-row').nth(1)).to_contain_text('<b>等待歌曲</b>')
            assert page.locator('.tq-content b').count()==0
            page.get_by_role('button',name='取消排队').click()
            expect(page.locator('.tq-active .tq-row')).to_have_count(1)
            page.get_by_role('button',name='停止任务').click()
            expect(page.locator('.tq-message')).to_contain_text('临时错误')
            expect(page.get_by_role('button',name='停止任务')).to_be_enabled()
            page.get_by_role('button',name='停止任务').click()
            expect(page.locator('.tq-active')).to_contain_text('没有进行中')
            page.locator('summary').click()
            expect(page.locator('.tq-recent .tq-row')).to_have_count(2)
            page.keyboard.press('Escape');expect(page.get_by_role('dialog')).not_to_be_visible()
            assert calls==['b'*32,'a'*32,'a'*32]
        finally:browser.close()

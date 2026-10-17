"""Fast pass/discard, expired inference, and the fresh-window screenshot button."""
from pathlib import Path
from types import SimpleNamespace
import json
import threading

import cv2
import numpy as np
import pytest

from app.advice import ScreenshotAdvice
from app.capture.frames import Frame,load_image
from app.ui.advice import AdvicePanel
from app.vision.config import load_config
from app.vision.trigger import TimerEdge

ROOT=Path(__file__).resolve().parents[1]
TIMER=[0,0,.1,.1]
HAND=[0,.8,.8,.2]
RIVER=[.3,.4,.3,.3]
CONFIG={'action_timer':{'rect':TIMER},'seat_mapping':{'self':0,'right':1,'top':2,'left':3},
        'regions':{'hand':{'kind':'hand','player':0,'hand_layout':{},'rect':HAND},
                   'river':{'kind':'river','player':0,'grid':{},'rect':RIVER}}}


def test_fast_pass_restarts_with_unchanged_seconds_and_no_missing_timer_probe():
    edge=TimerEdge()
    assert edge.observe(True,299,(13,False,5))=='start'
    assert edge.observe(True,299,(13,True,5)) is None
    assert edge.observe(True,299,(13,True,5))=='start'
    assert edge.serial==2
    assert [edge.observe(True,s,(13,True,5)) for s in (298,299,297)]==[None]*3


def test_fast_discard_and_next_draw_restarts_even_with_the_same_hand_layout():
    edge=TimerEdge()
    assert edge.observe(True,299,(13,True,5))=='start'
    assert edge.observe(True,295,(13,True,6)) is None
    assert edge.observe(True,295,(13,True,6))=='start'
    assert edge.serial==2


def test_unread_or_transient_layout_does_not_rearm_the_same_action():
    edge=TimerEdge()
    edge.observe(True,299,(13,True,5))
    assert edge.observe(True,298,(13,False,5)) is None
    assert edge.observe(True,298,None) is None
    assert edge.observe(True,298,(13,False,5)) is None
    assert edge.observe(True,297,(13,True,5)) is None
    assert edge.observe(True,296,(13,True,None)) is None
    assert edge.observe(True,296,(13,True,5)) is None
    assert edge.serial==1


def test_missing_timer_value_breaks_an_unconfirmed_restart_sequence():
    edge=TimerEdge()
    edge.observe(True,280,(13,True,5))
    assert edge.observe(True,299,(13,True,5)) is None
    assert edge.observe(True,None,(13,True,5)) is None
    assert edge.observe(True,299,(13,True,5)) is None
    assert edge.observe(True,299,(13,True,5))=='start'


def test_cancelled_capture_does_not_consume_a_timer_that_never_goes_missing_twice():
    edge=TimerEdge()
    edge.observe(True,299,(13,False,5))
    edge.observe(False)
    edge.cancel()
    assert edge.observe(True,299,(13,True,5))=='start'


@pytest.mark.parametrize('size',[(1920,1080),(1366,768)])
def test_real_user_frames_distinguish_draw_and_response_without_the_network(size):
    service=ScreenshotAdvice.__new__(ScreenshotAdvice)
    service.config=load_config(ROOT/'configs/window-steam-1366-v2.json')
    markers=[]
    for name in ('draw-with-stale-pass','pon-with-stale-discard'):
        pixels=load_image(ROOT/'data/fixtures/action-transition'/f'{name}-1920.png')
        if size[0]!=1920:pixels=cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
        markers.append(service.action_marker(pixels))
    assert markers==[(13,True,11),(13,False,15)]


def live_harness(tmp_path,monkeypatch,scenario):
    import app.ui.advice as ui
    own_to_own=scenario=='next_draw_during_inference'
    contexts={1:(13,True,5) if own_to_own else (13,False,5),
              2:(13,True,6) if own_to_own else (13,True,5)}
    state=SimpleNamespace(phase=1,active=True,seconds=299,probes=0,boards=0,calls=0,closed=0,waits=[])
    stop=threading.Event()
    def wait(delay):
        state.waits.append(delay)
        if scenario=='cancel_during_delay' and delay==.3 and state.phase==1:state.active=False
        elif scenario in {'cancel_during_delay','end_during_inference'} and delay==.15 and not state.active:
            state.phase,state.active=2,True
        return stop.is_set()
    stop.wait=wait
    def probe(rect):
        state.probes+=1
        if state.probes>60:stop.set()
        return SimpleNamespace(pixels=np.array([int(state.active),state.seconds]) if rect==TIMER else np.array([state.phase]))
    def grab():
        state.boards+=1
        return Frame(np.full((32,64,3),state.phase,np.uint8),{'frame_id':state.boards})
    capture=SimpleNamespace(grab_region=probe,grab=grab,close=lambda:setattr(state,'closed',state.closed+1))
    service=SimpleNamespace(clear=lambda:None,timer_seconds=None)
    def active(pixels,**kwargs):
        service.timer_seconds=int(pixels[1]);return bool(pixels[0])
    service.timer_active=active
    service.action_marker=lambda pixels,**kwargs:contexts[int(pixels.flat[0])]
    def process(pixels,metadata):
        state.calls+=1
        phase=int(pixels.flat[0])
        if phase==1:
            if scenario=='end_during_inference':state.active=False
            else:state.phase,state.seconds=2,295
        return {'status':'ready','observation':None,
                'recommendation':{'semantic':'pass' if phase==1 else 'discard',
                                  'event':{'type':'none'} if phase==1 else {'type':'dahai','pai':'N'}}}
    service.process=process
    monkeypatch.setattr(ui,'WindowCapture',lambda hwnd,cfg:capture)
    panel=AdvicePanel.__new__(AdvicePanel)
    panel.config,panel.output=CONFIG,tmp_path
    panel.get_service=lambda stable:service
    published=[]
    def publish(generation,result,pixels):
        published.append(result)
        if result.get('status')=='ready':stop.set()
    panel.publish=publish
    panel.live_worker(1,stop,123)
    return state,published,capture


@pytest.mark.parametrize('scenario',['pass_during_inference','next_draw_during_inference',
                                     'end_during_inference','cancel_during_delay'])
def test_early_action_never_publishes_the_old_result_and_does_analyze_the_next(tmp_path,monkeypatch,scenario):
    state,published,capture=live_harness(tmp_path,monkeypatch,scenario)
    ready=[r for r in published if r['status']=='ready']
    assert len(ready)==1 and ready[0]['recommendation']['event']['pai']=='N'
    assert state.calls==state.boards==(1 if scenario=='cancel_during_delay' else 2)
    assert state.closed==1 and capture.window_content_fallback
    assert state.waits.count(.3)==2
    assert json.loads((tmp_path/'latest.json').read_text(encoding='utf-8'))['recommendation']['event']['pai']=='N'
    saved=[json.loads(p.read_text(encoding='utf-8')) for p in tmp_path.glob('run-*/action-*/advice.json')]
    assert len([r for r in saved if r['status']=='ready'])==1
    assert len(list(tmp_path.glob('run-*/action-*/[0-9]*.png')))==state.boards


@pytest.mark.parametrize('resume',[False,True])
@pytest.mark.parametrize('cancel',[False,True])
def test_manual_button_takes_a_fresh_window_frame_and_respects_stop(tmp_path,monkeypatch,resume,cancel):
    import app.ui.advice as ui
    stop=threading.Event();stop.wait=lambda delay:stop.is_set()
    state=SimpleNamespace(boards=0,processed=0,closed=False)
    capture=SimpleNamespace(close=lambda:setattr(state,'closed',True),
                            grab_region=lambda rect:SimpleNamespace(pixels=np.array([1,299])))
    def grab():
        state.boards+=1
        return Frame(np.full((32,64,3),87,np.uint8),{'frame_id':state.boards})
    capture.grab=grab
    service=SimpleNamespace(clear=lambda:None,timer_seconds=299,timer_active=lambda *a,**k:True)
    def process(pixels,metadata):
        assert pixels[0,0,0]==87
        state.processed+=1
        if cancel:stop.set()
        return {'status':'ready','observation':None,'recommendation':{'semantic':'discard','event':{'pai':'N'}}}
    service.process=process
    handles=[]
    monkeypatch.setattr(ui,'WindowCapture',lambda hwnd,cfg:handles.append(hwnd) or capture)
    panel=AdvicePanel.__new__(AdvicePanel)
    panel.output,panel.config,panel.generation=tmp_path,{'action_timer':{'rect':TIMER}},7
    panel.get_service=lambda count:service
    published=[];resumed=[]
    panel.publish=lambda *args:published.append(args)
    original_save=Frame.save
    def save_after_publishing(frame,directory):
        assert published, '必须先显示建议，再保存图片'
        assert published[-1][1]['latency']['analysis_count']==1
        return original_save(frame,directory)
    monkeypatch.setattr(Frame,'save',save_after_publishing)
    panel.live_worker=lambda *args,**kwargs:resumed.append(kwargs)
    panel.capture_worker(7,stop,4321,resume=resume)
    assert handles==[4321] and state.boards==state.processed==1 and state.closed
    assert capture.window_content_fallback
    assert bool(published) is (not cancel)
    assert bool(resumed) is (resume and not cancel)
    assert bool(list(tmp_path.glob('manual-*/[0-9]*.png'))) is (not cancel)
    if not cancel:
        metadata=json.loads(next(tmp_path.glob('manual-*/*.frame.json')).read_text(encoding='utf-8'))
        assert metadata['capture_trigger']=='manual_screenshot_button'
    if resumed:
        assert resumed[0]['initial_edge'].observe(True,299) is None  # no duplicate auto analysis


def test_queued_manual_capture_from_an_older_request_does_not_read_the_window(tmp_path,monkeypatch):
    import app.ui.advice as ui
    panel=AdvicePanel.__new__(AdvicePanel)
    panel.generation,panel.output=8,tmp_path
    monkeypatch.setattr(ui,'WindowCapture',lambda *args:pytest.fail('旧请求不能读取窗口'))
    panel.capture_worker(7,threading.Event(),4321,resume=True)
    assert not (tmp_path/'latest.json').exists()


def test_manual_capture_replaces_the_live_job_and_keeps_automatic_mode(monkeypatch):
    from unittest.mock import Mock
    panel=AdvicePanel.__new__(AdvicePanel)
    panel.generation,panel.live,panel.result=3,True,{'status':'ready'}
    panel.stop_event=threading.Event();old_stop=panel.stop_event
    panel.areas=[SimpleNamespace(hwnd=123),SimpleNamespace(hwnd=456)]
    panel.selector=SimpleNamespace(current=lambda:1)
    panel.status,panel.advice,panel.table,panel.executor=Mock(),Mock(),Mock(),Mock()
    panel.table.get_children.return_value=()
    panel.capture_now()
    assert old_stop.is_set() and not panel.stop_event.is_set()
    assert panel.generation==4 and panel.live and panel.result is None
    panel.executor.submit.assert_called_once_with(panel.capture_worker,4,panel.stop_event,456,True)


def test_manual_result_is_dropped_if_the_hand_phase_changes_during_analysis(tmp_path,monkeypatch):
    import app.ui.advice as ui
    stop=threading.Event();stop.wait=lambda delay:False
    state=SimpleNamespace(marker=(13,False,5))
    capture=SimpleNamespace(grab=lambda:Frame(np.full((32,64,3),1,np.uint8),{'frame_id':1}),
                            grab_region=lambda rect:SimpleNamespace(pixels=np.array([1,299]) if rect==TIMER else np.array([2])),
                            close=lambda:None)
    service=SimpleNamespace(clear=lambda:None,timer_seconds=299,timer_active=lambda *a,**k:True)
    service.action_marker=lambda pixels,**kwargs:state.marker
    def process(*args):
        state.marker=(13,True,5)
        return {'status':'needs_confirmation','observation':None,'reason':'旧牌河不完整','recommendation':None}
    service.process=process
    monkeypatch.setattr(ui,'WindowCapture',lambda *args:capture)
    panel=AdvicePanel.__new__(AdvicePanel)
    panel.config,panel.output,panel.generation=CONFIG,tmp_path,7
    panel.get_service=lambda count:service
    published=[];resumed=[]
    panel.publish=lambda generation,result,pixels:published.append(result)
    panel.live_worker=lambda *args,**kwargs:resumed.append(kwargs)
    panel.capture_worker(7,stop,4321,resume=True)
    assert published[0]['status']=='waiting_turn' and '行动已变化' in published[0]['reason']
    assert published[0]['recommendation'] is None and resumed[0]['pending_start']


def test_four_panel_buttons_are_bound_and_fit_the_small_panel(tmp_path,monkeypatch):
    import tkinter as tk
    from tkinter import ttk
    import app.ui.advice as ui
    monkeypatch.setattr(ui,'windows',lambda *args:[])
    root=tk.Tk();root.withdraw()
    panel=AdvicePanel(root,ROOT/'configs/window-steam-1366-v2.json',tmp_path)
    try:
        root.geometry('450x760');root.attributes('-alpha',0)
        root.deiconify();root.update()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        widgets=list(descendants(root))
        buttons=[w for w in widgets if isinstance(w,ttk.Button)]
        assert [b.cget('text') for b in buttons]==['刷新雀魂窗口','整理窗口','开始自动建议','刷新截图']
        assert not any(isinstance(w,ttk.Checkbutton) for w in widgets)
        assert not any('不追踪临时振听' in str(w.cget('text')) for w in widgets if isinstance(w,ttk.Label))
        for button in buttons:
            assert button.winfo_width()>=button.winfo_reqwidth()
            assert button.winfo_x()+button.winfo_width() <= button.master.winfo_width()
            button.invoke()
        panel.capture_button.invoke()
        assert panel.status.get()=='请先选择雀魂窗口。'
    finally:panel.close()

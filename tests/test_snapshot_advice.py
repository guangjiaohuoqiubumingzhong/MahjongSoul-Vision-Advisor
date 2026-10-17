from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest
from libriichi.state import PlayerState

from app.advice import ScreenshotAdvice, load_screenshot, format_advice
from app.backend import Backend
from app.engine.snapshot import SnapshotBackend
from app.state.snapshot import build_snapshot, SnapshotError, digest
from app.vision.config import load_config
from app.vision.detector import concealed_back_mask
from app.vision.turn import parse_timer, detect_timer, WindReader
from app.ui.advice import AdvicePanel, parse_melds, meld_text
from app.ui.tile_text import parse_tiles
from conftest import ROOT, load_sample


def from_events(session):
    """Test-only reference projection. Production never loads these fixtures."""
    start=next(e for e in reversed(session.history) if e['type']=='start_kyoku')
    rivers=[[] for _ in range(4)]; dora=[start['dora_marker']]
    scores=list(start['scores']); kyotaku=start['kyotaku']; declared=set(); accepted=set()
    for e in session.history[session.history.index(start)+1:]:
        if e['type']=='dahai':
            rivers[e['actor']].append({'tile':e['pai'],'riichi':e['actor'] in declared and e['actor'] not in accepted,
                                        'tedashi':not e['tsumogiri']})
        elif e['type'] in {'chi','pon','daiminkan'}:
            assert rivers[e['target']][-1]['tile']==e['pai']
            rivers[e['target']].pop()
        elif e['type']=='dora':dora.append(e['dora_marker'])
        elif e['type']=='reach':declared.add(e['actor'])
        elif e['type']=='reach_accepted':
            scores[e['actor']]-=1000;kyotaku+=1;accepted.add(e['actor'])
    player=session.player_id; l=session.ledger
    hand=list(l.hand.elements());draw=None;last_call=None;target=None;reaction=None
    if l.pending_discard==player:
        if session.state.last_self_tsumo() is not None:
            phase='draw';draw=session.state.last_self_tsumo();hand.remove(draw)
        else:
            phase='call_discard';last_call=l.melds[player][-1]
    else:
        phase='reaction';target=session.state.last_cans.target_actor;reaction=session.state.last_kawa_tile()
    return {'player_id':player,'oya':start['oya'],'bakaze':start['bakaze'],'kyoku':start['kyoku'],
            'honba':start['honba'],'kyotaku':kyotaku,'scores':scores,'remaining':70-l.draw_count,
            'hand':hand,'draw':draw,'dora':dora,'rivers':rivers,'melds':deepcopy(l.melds),
            'riichi':[i in l.reach_accepted for i in range(4)],
            'riichi_declared':[i in l.reach_declared for i in range(4)],
            'phase':phase,'target':target,'reaction_tile':reaction,'last_call':last_call}


@pytest.mark.gpu
@pytest.mark.parametrize('name', ['discard_red','chi_red_choices','chi_then_discard','pon','daiminkan',
                                 'ankan','two_ankan_choices','kakan','riichi','riichi_confirmation',
                                 'tsumo','ron','kyushukyuhai','ankan_rinshan_dora'])
def test_snapshot_rules_match_event_engine(runtime,feed,name):
    session=Backend(runtime).initialize('snapshot-'+name,0)
    expectations=json.loads((ROOT/'data/fixtures/manifest.json').read_text(encoding='utf-8'))
    points={e['event_seq'] for row in expectations if row['name']==name for e in row['expectations']}
    for seq,e in enumerate(load_sample(name),1):
        response=feed(session,e)
        if seq not in points:continue
        snap=from_events(session)
        state=PlayerState.from_snapshot(json.dumps(snap))
        _,actual=state.encode_obs(3,False)
        _,expected=session.state.encode_obs(3,False)
        assert np.array_equal(actual,expected), (name,seq,actual,expected)
        before=deepcopy(snap)
        result=SnapshotBackend(runtime).decide(snap)
        assert result['status']=='ready' and result['execution_provider']=='CUDAExecutionProvider'
        state.validate_reaction(json.dumps(result['recommendation']['event']))
        assert snap==before
        assert set(np.flatnonzero(actual))=={c['action_id'] for c in result['legal_candidates']}


@pytest.mark.gpu
def test_first_draw_features_and_q_match_exact_backend(runtime,feed):
    session=Backend(runtime).initialize('snapshot-parity',0)
    for e in load_sample('discard_red'):expected=feed(session,e)
    s=from_events(session); state=PlayerState.from_snapshot(json.dumps(s))
    expected_obs,expected_mask=session.state.encode_obs(3,False)
    obs,mask=state.encode_obs(3,False)
    assert np.array_equal(obs,expected_obs) and np.array_equal(mask,expected_mask)
    actual=SnapshotBackend(runtime).decide(s)
    assert actual['observation_sha256']==expected['observation_sha256']
    assert actual['recommendation']['event']==expected['recommendation']['event']
    assert actual['recommendation']['model_score']==pytest.approx(expected['recommendation']['model_score'],abs=1e-6)


@pytest.mark.gpu
def test_reaction_buttons_restrict_model_actions(runtime,feed):
    session=Backend(runtime).initialize('buttons',0)
    for e in load_sample('chi_red_choices'):feed(session,e)
    s=from_events(session)
    result=SnapshotBackend(runtime).decide(s,['pass'])
    assert result['recommendation']['semantic']=='pass'
    assert np.flatnonzero(result['action_mask']).tolist()==[45]
    assert np.flatnonzero(result['rule_mask']).tolist()!=[45]


def unit_observation():
    start=load_sample('discard_red')[1]
    regions={}
    for kind,labels in [('hand',start['tehais'][0]),('draw',['W']),('dora',[start['dora_marker']])]:
        regions[kind]={'kind':kind,'player':0 if kind!='dora' else None,'complete':True,
                       'items':[{'label':t} for t in labels]}
    for actor in range(4):
        regions[f'river_{actor}']={'kind':'river','player':actor,'complete':True,'items':[]}
        regions[f'meld_{actor}']={'kind':'meld','player':actor,'complete':True,'items':[],'groups':[]}
        regions[f'score_{actor}']={'kind':'text','player':actor,'field':'score','complete':True,'text':{'value':25000}}
    for field,value in [('round',{'bakaze':'E','kyoku':1}),('remaining',69),('honba',0),('kyotaku',0)]:
        regions[field]={'kind':'text','player':None,'field':field,'complete':True,'text':{'value':value}}
    return {'regions':regions,'image_sha256':'unit-image','self_wind':{'value':'E'},'action_timer':{'active':True},
            'issues':[{'reason':'uncalibrated_regions'}]}


def config():return {'seat_mapping':{'self':0,'right':1,'top':2,'left':3}}


@pytest.mark.parametrize('kind', ['ankan','daiminkan','kakan'])
def test_automatically_confirmed_kan_is_converted_to_its_own_event_schema(kind):
    observation = unit_observation()
    group = {'type':kind,'status':'confirmed','labels':['2s']*4,
             'target':3 if kind=='daiminkan' else None,
             'called_positions':[0] if kind=='daiminkan' else [],'added_position':1 if kind=='kakan' else None}
    observation['regions']['meld_1']['groups'] = [group]
    snapshot,_ = build_snapshot(observation,config())
    event = snapshot['melds'][1][0]
    assert event['actor'] == 1 and event['type'] == kind
    assert event['consumed'] == ['2s']*(4 if kind=='ankan' else 3)
    assert ('pai' in event) is (kind!='ankan')
    assert ('target' in event) is (kind=='daiminkan')
    if kind=='daiminkan': assert event['target'] == 3


@pytest.mark.parametrize('kind', ['ankan','daiminkan','kakan'])
def test_incomplete_kan_source_or_added_tile_does_not_enter_a_snapshot(kind):
    observation = unit_observation()
    observation['regions']['meld_1']['groups'] = [{'type':kind,'status':'confirmed','labels':['2s']*4,
                                               'target':None,'called_positions':[0] if kind=='ankan' else []}]
    with pytest.raises(SnapshotError,match='副露类型和来源'):
        build_snapshot(observation,config())


@pytest.mark.parametrize('mutation', ['unknown','incomplete','wrong_count','physical_conflict','bad_score','missing_timer','unknown_wind'])
def test_incomplete_or_conflicting_snapshot_is_blocked(mutation):
    o=unit_observation()
    if mutation=='unknown':o['regions']['hand']['items'][0]['label']='?'
    elif mutation=='incomplete':o['regions']['meld_1']['complete']=False
    elif mutation=='wrong_count':o['regions']['hand']['items'].pop()
    elif mutation=='physical_conflict':o['regions']['hand']['items'][:5]=[{'label':'E'}]*5
    elif mutation=='bad_score':o['regions']['score_1']['text']['value']=None
    elif mutation=='missing_timer':o['action_timer']['active']=False
    else:o['self_wind']['value']=None
    with pytest.raises((SnapshotError,TypeError)):build_snapshot(o,config())


def test_explicit_current_correction_and_seat_context():
    o=unit_observation();o['regions']['meld_1']['complete']=False;o['self_wind']['value']=None
    s,_=build_snapshot(o,config(),{'meld_1':[]},{'oya':2})
    assert s['oya']==2 and s['melds']==[[],[],[],[]]
    assert not o['regions']['meld_1']['complete']


def test_reaction_uses_current_highlight_and_missing_buttons_never_mean_call_discard():
    o=unit_observation();o['regions']['draw']['items']=[]
    o['regions']['river_3']['items']=[{'label':'S'}]
    o['regions']['buttons']={'kind':'buttons','items':[{'label':'pon','clickable':True}]}
    o['reaction_discard']={'confirmed':True,'target':3,'tile':'S'}
    snapshot,allowed=build_snapshot(o,config())
    assert snapshot['phase']=='reaction' and snapshot['target']==3 and snapshot['last_call'] is None
    assert allowed==['pass','pon']
    o['regions']['buttons']['items']=[]
    with pytest.raises(SnapshotError,match='响应操作按钮'):
        build_snapshot(o,config())
    o['reaction_discard']['confirmed']=False
    with pytest.raises(SnapshotError,match='待响应弃牌'):
        build_snapshot(o,config())


@pytest.mark.parametrize('text,value',[('4 +20',{'fixed_seconds':4,'reserve_seconds':20}),
                                     ('0+3',{'fixed_seconds':0,'reserve_seconds':3}),
                                     ('0+0',{'fixed_seconds':0,'reserve_seconds':0}),
                                     ('没有数字',None),('5',{'fixed_seconds':5,'reserve_seconds':None}),
                                     ('15',{'fixed_seconds':15,'reserve_seconds':None}),
                                     ('0',{'fixed_seconds':0,'reserve_seconds':None}),
                                     ('293',{'fixed_seconds':293,'reserve_seconds':None}),
                                     ('25000',{'fixed_seconds':25000,'reserve_seconds':None}),
                                     ('4+99',{'fixed_seconds':4,'reserve_seconds':99}),
                                     ('15F',{'fixed_seconds':15,'reserve_seconds':None}),
                                     ('１２３４５６７８９',{'fixed_seconds':123456789,'reserve_seconds':None})])
def test_timer_grammar(text,value):assert parse_timer(text)==value


def test_timer_accepts_single_colour_digits_and_requires_confident_ocr():
    crop=np.zeros((64,128,3),np.uint8);cfg={'rect':[0,0,1,1],'threshold':.95}
    reader=SimpleNamespace(_infer=lambda p:('4+20',.99))
    assert not detect_timer(crop,cfg,reader)['active']
    reader._infer=lambda p:('15',.99)
    crop[30:55,60:100]=(20,160,255)
    assert detect_timer(crop,cfg,reader)['active']
    crop[:]=0;crop[5:30,10:25]=255
    reader._infer=lambda p:('5',.99)
    assert detect_timer(crop,cfg,reader)['active']
    crop[5:30,10:25]=255;crop[30:55,60:100]=(20,160,255)
    assert detect_timer(crop,cfg,reader)['active']
    reader._infer=lambda p:('4+20',.5)
    assert not detect_timer(crop,cfg,reader)['active']


@pytest.mark.parametrize('directory,number',[('user-single-countdown-example',15),('user-three-digit-countdown-example',293)])
def test_single_countdown_from_user_panel_preview(directory,number):
    # User supplied a panel screenshot, not an original game client capture.
    # Resize its preview only for a timer regression, not board accuracy claims.
    panel=cv2.imread(str(ROOT/f'data/screenshots/{directory}/panel.png'))
    pixels=cv2.resize(panel[261:553,21:541],(1366,768))
    from app.vision.text import TextReader
    result=detect_timer(pixels,load_config(ROOT/'configs/window-steam-1366-v2.json')['action_timer'],TextReader())
    assert result['active'] and result['raw_text']==str(number)
    assert result['value']=={'fixed_seconds':number,'reserve_seconds':None}
    from app.vision.turn import detect_timer_crop
    assert detect_timer_crop(pixels[543:631,1000:1230],load_config(ROOT/'configs/window-steam-1366-v2.json')['action_timer'],TextReader())['active']


def test_concealed_corridor_alone_does_not_remove_white_meld():
    corridor=np.ones((70,80),np.uint8)*255
    crop=np.zeros((70,80,3),np.uint8);crop[5:60,10:65]=240
    crop[15:45,30:40]=(40,100,40)
    assert not concealed_back_mask(crop,corridor).any()
    crop[15:45,30:40]=(210,190,170)
    assert concealed_back_mask(crop,corridor).any()
    assert not concealed_back_mask(crop,np.zeros_like(corridor)).any()


class FakeBackend:
    def __init__(self):self.calls=0
    def decide(self,snapshot,allowed):
        self.calls+=1
        return {'status':'ready','recommendation':{'semantic':'discard','event':{'type':'dahai','pai':'N'}},'snapshot_sha256':digest(snapshot)}


def fake_service(stable_frames=1):
    o=unit_observation(); backend=FakeBackend()
    r=SimpleNamespace(recognize=lambda p,m:deepcopy(o))
    return ScreenshotAdvice(config(),recognizer=r,backend=backend,stable_frames=stable_frames),o,backend


def test_live_stability_cache_and_timer_loss_clear_recommendation():
    s,o,b=fake_service(2)
    assert s.process(None)['status']=='confirming_frame'
    assert s.process(None)['status']=='ready' and b.calls==1
    assert s.process(None)['status']=='ready' and b.calls==1
    o['action_timer']['active']=False
    result=s.process(None)
    assert result['status']=='waiting_turn' and result['recommendation'] is None
    o['action_timer']['active']=True
    assert s.process(None)['status']=='confirming_frame'


def test_changed_or_bad_frame_cannot_reuse_old_decision_or_review():
    s,o,b=fake_service(2)
    s.process(None);s.process(None)
    o['regions']['draw']['items'][0]['label']='S'
    assert s.process(None)['status']=='confirming_frame' and b.calls==1
    assert s.process(None)['status']=='ready' and b.calls==2
    o['regions']['hand']['complete']=False
    assert s.process(None)['recommendation'] is None
    assert s.process(None,review={'image_sha256':'another-image'})['status']=='needs_confirmation'


def test_tile_and_meld_review_roundtrip():
    assert parse_tiles('1万 5筒赤 东 白')==['1m','5pr','E','P']
    text='碰 对家 2万 | 2万 2万; 暗杠 5筒 5筒 5筒 5筒赤'
    assert parse_melds(meld_text(parse_melds(text,0)),0)==parse_melds(text,0)


@pytest.mark.parametrize('checks,expected_frames,expected_ready,incomplete,expected_delays',[
    ([False,False,True,True,True,True,True,False,False,True,True,True,True],2,2,False,2),
    ([True,True,True,True,True,True],1,1,False,1),
    ([True,True,True,False,True,True,True],1,1,False,1),
    ([True,True,False,False],1,0,False,1),
    ([True]*8,1,0,True,1),
    ([True,False,False],0,0,False,1),
    ([True,False,False,True,True,True],1,1,False,2),
])
def test_live_worker_captures_once_per_digit_appearance(tmp_path,monkeypatch,checks,expected_frames,expected_ready,incomplete,expected_delays):
    import app.ui.advice as ui
    from app.capture.frames import Frame
    stop=threading.Event();waits=[]
    def wait(seconds):waits.append(seconds);return False
    stop.wait=wait
    sequence=iter(checks);captured=[];closed=[];analyses=[]
    def probe(rect):
        value=next(sequence,None)
        if value is None:stop.set();value=False
        return SimpleNamespace(pixels=np.array([value]))
    def grab():
        assert waits[-1] in (.3,.4)
        captured.append(True)
        return Frame(np.zeros((128,256,3),np.uint8),{'frame_id':len(captured)})
    capture=SimpleNamespace(grab=grab,grab_region=probe,close=lambda:closed.append(True))
    service,o,backend=fake_service()
    for region in o['regions'].values():
        region['bbox']=[0,0,10,10]
        for item in region.setdefault('items',[]):item['bbox']=[0,0,5,5]
    if incomplete:o['regions']['hand']['complete']=False
    original=service.process
    def process(p,m):analyses.append(True);return original(p,m)
    service.process=process
    service.timer_active=lambda p,*,cropped:bool(p[0])
    monkeypatch.setattr(ui,'WindowCapture',lambda hwnd,cfg:capture)
    panel=AdvicePanel.__new__(AdvicePanel);panel.output=tmp_path;panel.config={'action_timer':{'rect':[0,0,1,1]}}
    def get_service(n):assert n==1;return service
    panel.get_service=get_service;published=[]
    panel.publish=lambda g,r,p:published.append(r)
    panel.live_worker(1,stop,123)
    # An incomplete board is recaptured (0.4 s apart) while the timer is still on;
    # only the kept frame is saved.
    recaptures=sum(r.get('recaptures',0) for r in published)
    assert recaptures==(3 if incomplete else 0)
    assert len(captured)==len(analyses)==expected_frames+recaptures
    assert waits.count(.3)==expected_delays and waits.count(.4)==recaptures
    assert backend.calls==(0 if incomplete else expected_frames)
    assert sum(r['status']=='ready' for r in published)==expected_ready
    assert closed==[True] and all(r['status']!='error' for r in published)
    assert len(list(tmp_path.glob('run-*/action-*/[0-9]*.png')))==expected_frames
    if expected_ready==0 and not incomplete:
        assert published[-1]['recommendation'] is None
        assert json.loads((tmp_path/'latest.json').read_text(encoding='utf-8'))['recommendation'] is None


def test_stop_during_capture_delay_does_not_grab_or_analyze(tmp_path,monkeypatch):
    import app.ui.advice as ui
    stop=threading.Event();closed=[];waits=[]
    def wait(seconds):waits.append(seconds);stop.set();return True
    stop.wait=wait
    def forbidden(*args):pytest.fail('停止后不应获取完整截图或分析')
    capture=SimpleNamespace(grab_region=lambda rect:SimpleNamespace(pixels=np.ones(1)),grab=forbidden,close=lambda:closed.append(True))
    service=SimpleNamespace(clear=lambda:None,timer_active=lambda p,**kwargs:True,process=forbidden)
    monkeypatch.setattr(ui,'WindowCapture',lambda hwnd,cfg:capture)
    panel=AdvicePanel.__new__(AdvicePanel);panel.output=tmp_path;panel.config={'action_timer':{'rect':[0,0,1,1]}}
    panel.get_service=lambda n:service
    published=[]
    panel.publish=lambda g,r,p:published.append(r)
    panel.live_worker(1,stop,123)
    assert waits==[.3] and closed==[True]
    assert all(r['recommendation'] is None for r in published)
    assert not (tmp_path/'latest.json').exists()


def test_window_timer_probe_grabs_only_small_region(monkeypatch):
    import app.capture.frames as frames
    area=SimpleNamespace(pid=123,left=50,top=20,width=1366,height=768,asdict=lambda:{'pid':123})
    calls=[]
    capture=frames.WindowCapture.__new__(frames.WindowCapture)
    capture.hwnd=1;capture.initial=area;capture.annotation_overlays=False;capture.frame_id=0
    capture.config=load_config(ROOT/'configs/window-steam-1366-v2.json');capture.config_sha256='test'
    capture._validate_geometry=lambda _:None
    capture.sct=SimpleNamespace(grab=lambda rect:(calls.append(rect) or np.zeros((rect['height'],rect['width'],4),np.uint8)))
    monkeypatch.setattr(frames,'client_area',lambda _:area)
    monkeypatch.setattr(frames,'ensure_unoccluded',lambda _:[])
    probe=capture.grab_region(capture.config['action_timer']['rect'])
    assert calls==[{'left':1050,'top':563,'width':230,'height':88}]
    assert probe.pixels.shape==(88,230,3)
    assert probe.metadata['capture_rect']==[1000,543,230,88]


def test_advice_panel_ignores_late_results_and_clears_old_candidates(tmp_path,monkeypatch):
    import tkinter as tk
    import app.ui.advice as ui
    monkeypatch.setattr(ui,'windows',lambda title:[])
    root=tk.Tk();root.withdraw()
    panel=AdvicePanel(root,ROOT/'configs/window-steam-1366-v2.json',tmp_path)
    ready={'status':'ready','request_id':'test','recommendation':{'semantic':'discard','event':{'pai':'S'}},
           'decision':{'legal_candidates':[]},'observation':{'action_timer':{'value':{'fixed_seconds':4,'reserve_seconds':20}}}}
    try:
        panel.publish(panel.generation,ready,np.zeros((100,200,3),np.uint8));panel.poll()
        assert panel.advice.get()=='打出 南'
        assert '4 +20' in panel.status.get()
        ready['decision']['legal_candidates']=[
            {'semantic':'discard','model_score':score,'variants':[{'pai':tile}]}
            for score,tile in zip([100,-1,-1,-2,-2,-3],['1m','2m','3m','4m','5m','6m'])]
        ready['observation']['action_timer']['value']={'fixed_seconds':15,'reserve_seconds':None}
        panel.publish(panel.generation,ready,None);panel.poll()
        assert '15；' in panel.status.get() and '+None' not in panel.status.get()
        rows=panel.table.rows
        assert len(rows)==5
        assert [row[1] for row in rows]==['99.6%','0.1%','0.1%','0.1%','0.1%']
        panel.stop();generation=panel.generation
        panel.publish(generation-1,ready,None);panel.poll()
        assert '打出' not in panel.advice.get()
        panel.publish(generation,{'status':'waiting_turn','reason':'等待计时器','recommendation':None},None);panel.poll()
        assert panel.advice.get()=='等待计时器' and not panel.table.rows
    finally:panel.close()


@pytest.mark.gpu
def test_supplied_countdown_screenshot_end_to_end(runtime):
    cfg=load_config(ROOT/'configs/window-steam-1366-v2.json')
    path=ROOT/'data/screenshots/user-countdown-example/window.png'
    pixels,metadata=load_screenshot(path,cfg)
    service=ScreenshotAdvice(cfg,backend=SnapshotBackend(runtime))
    result=service.process(pixels,metadata)
    assert result['status']=='ready',result.get('reason')
    assert result['corrections']=={} and result['context']=={}
    assert result['observation']['action_timer']['value']=={'fixed_seconds':4,'reserve_seconds':20}
    assert result['snapshot']['oya']==2 and result['snapshot']['draw']=='3s'
    assert result['snapshot']['hand']==['1m','2m','3m','4m','5p','7p','5sr','5s','6s','7s','8s','S','P']
    assert result['snapshot']['dora']==['F'] and result['snapshot']['remaining']==67
    assert result['snapshot']['melds']==[[],[],[],[]]
    assert result['snapshot']['rivers'][2][0]['tile']=='W'
    assert result['snapshot']['rivers'][3][0]['tile']=='E'
    PlayerState.from_snapshot(json.dumps(result['snapshot'])).validate_reaction(json.dumps(result['recommendation']['event']))
    # A replay frame with 14 tiles is not an action unless a countdown exists.
    replay,metadata=load_screenshot(ROOT/'data/screenshots/user-251228-2240-test/kyoku04-start/00000001.png',cfg)
    result=service.process(replay,metadata)
    assert result['status']=='waiting_turn' and result['recommendation'] is None


@pytest.mark.parametrize('name', ['kyoku01','kyoku04','kyoku05','kyoku06'])
def test_wind_reader_uses_verified_training_crops(name):
    from app.capture.frames import load_image
    cfg=load_config(ROOT/'configs/window-steam-1366-v2.json')
    p=load_image(ROOT/f'data/screenshots/user-250819-train/{name}/00000001.png')
    assert WindReader(cfg['self_wind']).read(p)['value']=={'kyoku01':'E','kyoku04':'N','kyoku05':'W','kyoku06':'S'}[name]

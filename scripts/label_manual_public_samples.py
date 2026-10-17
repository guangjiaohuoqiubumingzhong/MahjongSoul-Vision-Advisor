"""Manual transcription of five human-triggered captures, including four new scenes.

Saved detector rectangles are location proposals only. Every identity below was
read from the original pixels and enlarged atlases; predictions are not labels.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

from app.capture.frames import load_image
from app.vision.config import write_json
from scripts.label_reviewed_samples import hand_targets,text_targets

ROOT=Path(__file__).resolve().parents[1]
NAMES=["capture-20261009T114430Z-4bc2965c","capture-20261009T115837Z-d6259b6c",
       "capture-20261009T115847Z-86db5e30","capture-20261009T115913Z-0aa10107",
       "capture-20261009T115919Z-01b71bd7"]
HANDS=[['2s','2s','4s','4s','4s','5s','6s','7s','7s','S'],
       ['2s','2s','4s','4s','4s','5s','6s','F'],
       ['2s','2s','4s','4s','4s','5s','6s','7s','7s','S','W'],
       ['3m','4m','4m','5mr','5m','5m','E'],
       ['3m','4m','4m','5mr','5m','5m','E','C']]
RIVERS=[
    {'river_self':['9m','3m','3p','C','N','F','E','5m','W'],
     'river_right':['2s','8s','7m','8m','2m','P','3m','S','C','9m','8s'],
     'river_top':['1p','C','E','F','P','8s','6s','9m','9s','9m','5p'],
     'river_left':['P','W','S','C','5p','N','1p','2p','8p','9p']},
    {'river_self':['9m','3m','3p','C','N','F','E','5m','W','S'],
     'river_right':['2s','8s','7m','8m','2m','P','3m','S','C','9m','8s'],
     'river_top':['1p','C','E','F','P','8s','9m','9s','9m','5p'],
     'river_left':['P','W','S','C','5p','N','1p','2p','8p','9p']},
    {'river_self':['9m','3m','3p','C','N','F','E','5m'],
     'river_right':['2s','8s','7m','8m','2m','P','3m','S','C','9m'],
     'river_top':['1p','C','E','F','P','8s','6s','9m','9s','9m'],
     'river_left':['P','W','S','C','5p','N','1p','2p','8p']},
    {'river_self':['1s','9s','F','C'],'river_right':['7p','1m','4p','S','5m'],
     'river_top':['2s','9p','6s','4s','7p'],'river_left':['C','3m','7m','8p']},
    {'river_self':['1s','9s','F'],'river_right':['7p','1m','4p','S'],
     'river_top':['2s','9p','6s','4s'],'river_left':['C','3m','7m']},
]


def main():
    totals=[]
    for number,name in enumerate(NAMES,1):
        i=number-1
        image=ROOT/'data/screenshots/manual-captures'/name/'00000001.png'
        load_image(image)
        observation_path=ROOT/f'reports/vision/manual-public-review/{number:02d}/observation.json'
        observation=json.loads(observation_path.read_text(encoding='utf-8'))
        regions=observation['regions']
        targets=hand_targets(HANDS[i])
        if i in {0,3}:
            box=[854,655,68,108] if i==0 else [651,655,67,108]
            targets.append({'id':'draw-0','kind':'tile','label':'F' if i==0 else '3m',
                            'bbox':box,'region':'draw','roles':['hand','draw'],'rotation':0,'player':0})
        targets.append({'id':'dora-0','kind':'tile','label':'3p' if i<3 else '6s',
                        'bbox':[20,32,38,56],'region':'dora','roles':['dora'],'rotation':0})
        for name,labels in RIVERS[i].items():
            items=regions[name]['items'][:len(labels)]
            if len(items)!=len(labels):
                raise ValueError('Location proposal inventory changed; repeat pixel review.')
            for j,(item,label) in enumerate(zip(items,labels)):
                targets.append({'id':f'{name}-{j}','kind':'tile','label':label,'bbox':item['bbox'],
                                'region':name,'roles':[name],'rotation':regions[name]['rotation'],
                                'player':regions[name]['player'],'horizontal':False,'slot':item['slot'],
                                'source':'manual_review_of_enlarged_original_pixels'})
        if i==1:
            # This tile is clearly present in the original but the detector missed it.
            targets.append({'id':'river_top-missed-six','kind':'tile','label':'6s','bbox':[757,141,38,31],
                            'region':'river_top','roles':['river_top'],'rotation':180,'player':2,
                            'horizontal':False,'slot':[1,0],'source':'manual_review_of_original_not_a_detector_box'})
        meld_labels={'meld_right':['4p','7p','8p','3p','9p','2p']} if i<3 else {
            'meld_right':['2s','3s','1s'],'meld_left':['8s']*3}
        meld_labels['meld_self']=(['1s']*3 if i in {0,2} else
                                 ['7s','7s','1s','1s','7s','1s'] if i==1 else
                                 ['5pr','7p','2p','3p','6p','4p'])
        for name,labels in meld_labels.items():
            for j,(item,label) in enumerate(zip(regions[name]['items'],labels)):
                sideways=(name=='meld_self' and j in ({2} if i in {0,2} else {4,5}) or
                          name=='meld_right' and j in ({4,5} if i<3 else {2}) or
                          name=='meld_left' and j==2)
                group=f'{name}-{label}' if i<3 and name=='meld_self' else (
                    f'{name}-234' if i<3 and name=='meld_right' and label in {'2p','3p','4p'} else
                    f'{name}-789' if i<3 and name=='meld_right' else
                    f'{name}-567' if name=='meld_self' and label in {'5pr','6p','7p'} else
                    f'{name}-234' if name=='meld_self' else name)
                kind='pon' if i<3 and name=='meld_self' or name=='meld_left' else 'chi'
                # Chi's supplier is fixed by the four-player rule. Pon suppliers
                # are retained as unknown here without separate call-sequence evidence.
                source=0 if name=='meld_right' else 3 if kind=='chi' else None
                targets.append({'id':f'{name}-{j}','kind':'tile','label':label,'bbox':item['bbox'],
                                'region':name,'roles':['meld_self','meld_right','meld_top','meld_left'],
                                'rotation':(regions[name]['rotation']+270*sideways)%360,
                                'horizontal':sideways,'player':regions[name]['player'],
                                'group_id':group,'meld_type':kind,'target':source,
                                'source':'manual_review_of_enlarged_original_pixels'})
        scores=[25000]*4 if i<3 else [37000,13000,25000,25000]
        values=[(name,str(score),score) for name,score in zip(['score_self','score_right','score_top','score_left'],scores)]
        values += [('round','東1局',{'bakaze':'E','kyoku':1}),('remaining',f'余{[28,27,32,51,55][i]}',[28,27,32,51,55][i]),
                   ('honba',str(0 if i<3 else 1),0 if i<3 else 1),('kyotaku','0',0)]
        targets += text_targets(values)
        path=image.with_suffix('.annotation.json')
        old=json.loads(path.read_text(encoding='utf-8'))
        annotation={**old,'reviewed':True,'reviewer':'manual_visual_transcription_by_developer',
                    'frame_quality':'stable','targets':targets,
                    'complete_regions':['hand','draw','dora','river_self','river_right','river_top','river_left',
                                        'meld_self','meld_right','meld_top','meld_left',*[v[0] for v in values]],
                    'coverage_tags':['chi','pon']+(['red_in_meld'] if i>=3 else []),
                    'review_evidence':[str(observation_path.parent/'atlas.png'),str(image)],
                    'image_file_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                    'notes':'Only visible self slots and public tiles transcribed. Ordinary 5s and 7s contain red accents and are not red fives. All new river tiles are upright, including cropped 1p. Opponent hands hidden. No button/clickability or kan evidence. Pon supplier not asserted. Adjacent sorted slots do not prove which tile was just drawn.'}
        if i==4:
            annotation.update(count_as_distinct_scene=False,
                              duplicate_of=str(ROOT/'data/screenshots/user-250819-train/kyoku02-turn06/00000001.annotation.json'),
                              use='additional_visual_variant_of_existing_training_scene')
        if not old.get('reviewed'):
            with path.with_suffix('.corrections.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'reason':'Original pixels and enlarged atlases manually reviewed; detector labels were not accepted as truth.',
                                         'before':old,'after':annotation},ensure_ascii=False)+'\n')
        write_json(path,annotation)
        totals.append({'image':str(image),'known_tiles':sum(t['kind']=='tile' for t in targets),
                       'distinct_scene':annotation.get('count_as_distinct_scene',True)})
    write_json(ROOT/'reports/vision/manual-public-label-review.json',{'human_triggered':True,'game_inputs_sent':0,
               'split':'train','game_id':'user-250819-2005','samples':totals,
               'additional_captures':4,'total_captures':5,'new_distinct_scenes':4})
    print(totals)


if __name__=='__main__':
    main()

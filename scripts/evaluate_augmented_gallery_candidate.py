"""Frozen train-only multiscale proposal experiment; does not change defaults.

Only otherwise unknown front-tile proposals may be accepted. Known/conflicting
results and blank dragons stay under the existing policy. Three frozen SVMs
must agree with a strong generated-template proposal. Test is diagnosis only.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

import cv2
import numpy as np

from app.vision.config import load_config,write_json
from app.vision.faces import FaceGallery,tile_face,unrotate_quad
from app.vision.templates import TemplateBank,rotate
from app.vision.recognizer import StaticRecognizer
from app.vision.evaluate import evaluate
from app.vision.text import TextReader
from scripts.prototype_augmented_face_gallery import records,SCALES


def training_gallery(bank):
    labels,features,sources=[],[],[]
    for path,annotation,target,patch in records('train'):
        if target['kind']!='tile' or target['region'] not in {'hand','draw'}:
            continue
        sources.append({'annotation':str(path),'game_id':annotation['game_id'],'id':target['id'],
                        'image_sha256':hashlib.sha256(Path(annotation['image']).read_bytes()).hexdigest(),
                        'split':'train','reviewed':True})
        for scale in SCALES:
            candidate=patch if scale==1 else cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            face=tile_face(candidate,bank.face_parameters)
            if face is not None:
                labels.append(target['label'])
                features.append(face.feature)
    values=np.stack(features)
    digest=hashlib.sha256(values.tobytes()+json.dumps(labels).encode()).hexdigest()
    output=Path('models/vision-augmented-gallery')
    output.mkdir(parents=True,exist_ok=True)
    filename=output/f'features-{digest}.npz'
    if not filename.exists():
        np.savez_compressed(filename,features=values,labels=np.array(labels))
    manifest={'scope':'frozen_train_only_proposal_experiment','scales':SCALES,'feature_sha256':digest,
              'file':str(filename),'file_sha256':hashlib.sha256(filename.read_bytes()).hexdigest(),
              'face_parameters':bank.face_parameters,'sources':sources,
              'policy':{'minimum_score':.9,'minimum_class_margin':.04,'never_override_known_or_conflicting':True,
                        'never_promote_blank_white_dragon':True,'required_models':'all_three_existing_train_svms'}}
    write_json(output/f'manifest-{digest}.json',manifest)
    return labels,FaceGallery(values),manifest


class ExperimentalGalleryBank(TemplateBank):
    def __init__(self,config):
        super().__init__(config['templates'],config.get('classifier_manifest'))
        if self.verifier is None:
            raise ValueError('This candidate requires the existing frozen SVM verifier.')
        self.aug_labels,self.aug_gallery,self.aug_manifest=training_gallery(self)
        self.stats=Counter()

    def classify(self,pixels,role,threshold,margin,rotations=(0,),*,kinds=('tile','back')):
        current=super().classify(pixels,role,threshold,margin,rotations,kinds=kinds)
        if (current['label']!='?' or 'tile' not in kinds or role=='choices' or
                current.get('reason')=='conflicting_tile_features' or current.get('best_label')=='back'):
            return current
        scores={}
        for degrees in rotations:
            face=tile_face(rotate(pixels,degrees),self.face_parameters)
            if face is None:
                continue
            for label,value in zip(self.aug_labels,self.aug_gallery.scores(face.feature)):
                # The scale-only pilot matched some blue backs to P. No new P
                # acceptance is permitted by this experiment, even with votes.
                if label=='P' and not face.blank:
                    continue
                if label not in scores or value>scores[label][0]:
                    scores[label]=(float(value),degrees,face)
        if not scores:
            return current
        ranked=sorted(scores.items(),key=lambda pair:pair[1][0],reverse=True)
        label,(value,degrees,face)=ranked[0]
        if label=='P':
            return current
        difference=value-ranked[1][1][0] if len(ranked)>1 else 0
        if value<max(threshold,.9) or difference<max(margin,.04):
            return current
        proposal={'best_label':label,'score':value,'blank_face':face.blank}
        verifier=self.verifier.verify(rotate(pixels,degrees),proposal)
        if not verifier['accepted']:
            self.stats['strong_proposal_rejected_by_svm']+=1
            return current
        self.stats['accepted_augmented_proposal']+=1
        return {**current,'label':label,'best_label':label,'score':value,'margin':difference,'reason':None,
                'face_rotation':degrees,'face_quad_local':unrotate_quad(face.quad,pixels.shape[1],pixels.shape[0],degrees),
                'previous_unknown_proposal':current,'classifier_evidence':verifier,
                'augmentation_feature_sha256':self.aug_manifest['feature_sha256'],
                'feature_scope':'train_only_multiscale_glyph',
                'acceptance_policy':'strong_train_multiscale_proposal_and_unanimous_frozen_svm'}


def main():
    config=load_config('configs/window-steam-1366-manual-public-candidate.json')
    bank=ExperimentalGalleryBank(config)
    # Training negatives and public poses are checked before reading held-out GT.
    rows=[]
    for path,annotation,target,patch in records('train'):
        if target['region'] in {'hand','draw'}:
            continue
        result=bank.classify(patch,target['region'],.9,.04,(0,))
        rows.append({'annotation':str(path),'id':target['id'],'kind':target['kind'],
                     'expected':target['label'],'label':result['label']})
    write_json('reports/vision/training-augmented-gallery-candidate.json',{
        'scope':'train_cross_role_crop_diagnosis_not_acceptance','manifest':bank.aug_manifest,
        'wrong':[r for r in rows if r['kind']=='tile' and r['label'] not in {'?',r['expected']}],
        'back_false_tiles':[r for r in rows if r['kind']=='back' and r['label'] not in {'?','back'}],
        'rows':rows,'stats':dict(bank.stats)})
    reviewed=[]
    for path in sorted(Path('data').rglob('*.annotation.json')):
        annotation=json.loads(path.read_text(encoding='utf-8-sig'))
        if annotation.get('reviewed'):
            reviewed.append(path)
    bank.stats.clear()
    report=evaluate(StaticRecognizer(config,bank,TextReader()),reviewed)
    report.update(experiment_scope='frozen_train_multiscale_proposals_not_default',
                  augmentation_manifest=bank.aug_manifest,augmentation_stats=dict(bank.stats))
    write_json('reports/vision/augmented-gallery-candidate-acceptance.json',report)
    print({key:report[key] for key in ['regions','melds','text','uncertain_frames','gate_passed','augmentation_stats']})


if __name__=='__main__':
    main()

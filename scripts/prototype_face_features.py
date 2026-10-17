"""Inspect a tile-face rectifier using reviewed training images only.

This experiment never modifies source images, annotations or the active bank.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from app.capture.frames import load_image, save_image
from app.vision.templates import rotate


def rectify(pixels):
    background = np.percentile(pixels,85,axis=(0,1)).astype(np.float32)
    normalized = np.clip(pixels.astype(np.float32)/np.maximum(background,64),0,1)
    mask = (np.min(normalized,axis=2) >= .94).astype(np.uint8)*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None,None
    contour = max(contours,key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    polygon = cv2.approxPolyDP(hull,.03*cv2.arcLength(hull,True),True).reshape(-1,2)
    if len(polygon)!=4 or cv2.contourArea(polygon) < pixels.shape[0]*pixels.shape[1]*.2:
        return None,None
    corners = polygon.astype(np.float32)
    sums,differences = corners.sum(axis=1),corners[:,0]-corners[:,1]
    quad = np.array([corners[np.argmin(sums)],corners[np.argmax(differences)],
                     corners[np.argmax(sums)],corners[np.argmin(differences)]],np.float32)
    if len({tuple(p) for p in quad}) != 4:
        return None,None
    transform = cv2.getPerspectiveTransform(quad,np.array([[0,0],[39,0],[39,59],[0,59]],np.float32))
    face = cv2.warpPerspective(normalized,transform,(40,60),flags=cv2.INTER_LINEAR)
    return face[3:-3,3:-3],quad


def main():
    samples=[]
    for path in sorted(Path('data').rglob('*.annotation.json')):
        annotation=json.loads(path.read_text(encoding='utf-8'))
        if annotation.get('split')!='train' or not annotation.get('reviewed'):
            continue
        pixels=load_image(annotation['image'])
        for target in annotation['targets']:
            if target.get('kind')!='tile':continue
            x,y,w,h=target['bbox'];patch=rotate(pixels[y:y+h,x:x+w],target.get('rotation',0))
            face,quad=rectify(patch)
            samples.append((target,patch,face,quad))
    print('training tile faces',len(samples),'rectified',sum(s[2] is not None for s in samples))
    # Show several actual training poses of shared identities, including white.
    selected=[]
    for label in ['1s','2s','6s','8s','P','5pr','8m','S']:
        seen=set()
        for sample in samples:
            target=sample[0]
            role=target['region']
            if target['label']==label and role not in seen:
                selected.append(sample);seen.add(role)
    rows=[]
    for target,patch,face,quad in selected:
        canvas=np.zeros((145,460,3),np.uint8)
        original=cv2.resize(patch,(95,115),interpolation=cv2.INTER_NEAREST)
        canvas[25:140,0:95]=original
        if face is not None:
            display=cv2.resize((face*255).astype(np.uint8),(95,115),interpolation=cv2.INTER_NEAREST)
            canvas[25:140,110:205]=display
        cv2.putText(canvas,target['label']+' '+target['region'],(0,18),0,.45,(255,255,255),1)
        cv2.putText(canvas,'quad' if quad is not None else 'no quad',(225,75),0,.6,(255,255,255),1)
        rows.append(canvas)
    save_image('reports/vision/training-face-rectifier-prototype.png',np.vstack(rows))


if __name__=='__main__':main()

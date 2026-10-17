"""Retain the user frames showing advice from an earlier action."""
from pathlib import Path
import hashlib
import json
import shutil
import cv2
from app.advice import ScreenshotAdvice
from app.capture.frames import load_image,save_image
from app.vision.config import load_config,write_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/fixtures/action-transition'
SOURCES={
    'draw-with-stale-pass':'codex-clipboard-22ac224d-b6be-4615-b0ba-ec26d8e84f4b.png',
    'pon-with-stale-discard':'codex-clipboard-7b6a045c-9719-4a2c-a56e-e91a1282f738.png',
}

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    service=ScreenshotAdvice.__new__(ScreenshotAdvice)
    service.config=load_config(ROOT/'configs/window-steam-1366-v2.json')
    rows=[];sources={}
    for name,filename in SOURCES.items():
        target=DATA/(name+'-desktop.png')
        if not target.exists():
            shutil.copyfile(Path('C:/Users/16bits_love/AppData/Local/Temp')/filename,target)
        source=load_image(target)
        pixels=source[31:1111,1:1921].copy()
        save_image(DATA/(name+'-1920.png'),pixels)
        sources[name]={'source':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                       'crop_xywh':[1,31,1920,1080]}
        for size in ((1920,1080),(1366,768)):
            image=pixels if size[0]==1920 else cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
            row={'name':name,'size':list(size),'action_marker':service.action_marker(image)}
            rows.append(row);print(json.dumps(row,ensure_ascii=False))
    write_json(DATA/'sources.json',sources)
    write_json(ROOT/'reports/action-transition-fix/markers.json',rows)

if __name__=='__main__':main()

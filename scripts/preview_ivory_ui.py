"""Render the real panel in controlled states; never connect to or move a game."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tkinter as tk
from unittest.mock import patch

from app.capture.frames import save_image
from app.capture.window import ClientArea, _api
from app.ui.advice import AdvicePanel
from scripts.capture_native_series import print_window


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/ui-ivory')
    parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # Establish the same native coordinates used by production window capture.
    user=_api()
    root=tk.Tk();root.withdraw();root.attributes('-alpha',0)
    area=ClientArea(1,'雀魂麻将',1,0,0,1920,1080,96)
    with patch('app.ui.advice.windows',return_value=[area]):
        panel=AdvicePanel(root,ROOT/'configs/window-steam-1366-v2.json',args.output/'unused')
    sample={'status':'ready','recommendation':{'semantic':'discard','event':{'pai':'6m'}},
            'observation':None,'decision':{'legal_candidates':[
                {'semantic':'discard','model_score':score,'variants':[{'pai':value}]}
                for value,score in [('6m',4.23),('2p',2.04),('3p',1.38),('4p',.55),('8s',0)]]}}
    ready=json.loads(args.result.read_text(encoding='utf-8-sig')) if args.result else sample
    cases=[('ready',ready),('waiting',{'status':'waiting_turn','reason':'等待自己的行动计时器。','recommendation':None}),
           ('confirming',{'status':'confirming_frame','reason':'画面还在变化（牌河），正在补截确认……','recommendation':None}),
           ('confirmation',{'status':'needs_confirmation','reason':'请核对上家牌河：识别不完整。','recommendation':None}),
           ('error',{'status':'error','reason':'游戏窗口已最小化。正在重试，恢复后自动继续。','recommendation':None})]
    long_action={'status':'ready','recommendation':{'semantic':'riichi','event':{'type':'reach'},
                 'after_riichi_declaration':{'event':{'pai':'5mr'}}},'observation':None,
                 'decision':{'legal_candidates':[
                     {'semantic':'chi','model_score':3,'variants':[{'pai':'3m','consumed':['1m','2m']}]},
                     {'semantic':'pon','model_score':2,'variants':[{'pai':'5pr','consumed':['5p','5p']}]},
                     {'semantic':'kakan','model_score':1,'variants':[{'pai':'5sr','consumed':['5s','5s','5s']}]},
                     {'semantic':'pass','model_score':0,'variants':[{}]}]}}
    cases.append(('long-actions',long_action))
    report=[]
    try:
        for width in (560,450):
            root.geometry(f'{width}x640+40+40');root.deiconify();root.update()
            for name,result in cases:
                panel.publish(panel.generation,deepcopy(result),None);panel.poll();root.update()
                root.after(40,root.quit);root.mainloop()
                # All capture pixels belong to this test window, even when transparent.
                hwnd=user.GetAncestor(root.winfo_id(),2)
                pixels,_=print_window(hwnd)
                filename=f'{name}-{width}.png'
                save_image(args.output/filename,pixels)
                bounds={key:[widget.winfo_x(),widget.winfo_y(),widget.winfo_width(),widget.winfo_height()]
                        for key,widget in [('advice',panel.hero),('candidates',panel.table)]}
                assert all(y+h<=640 for _,y,_,h in bounds.values()),bounds
                assert panel.table.winfo_height()>=222,bounds
                assert len(panel.buttons)==4 and not hasattr(panel,'preview')
                report.append({'image':filename,'geometry':root.geometry(),'bounds':bounds,
                               'message':panel.advice.get(),'rows':panel.table.rows})
        for selector in (panel.selector,panel.size_selector):
            selector.event_generate('<Button-1>',x=selector.winfo_width()-8,y=selector.winfo_height()//2)
            root.update()
            popup=root.tk.call('ttk::combobox::PopdownWindow',str(selector))
            assert int(root.tk.call('winfo','ismapped',popup)),str(selector)
            root.tk.call('ttk::combobox::Unpost',str(selector))
            root.update()
        panel.size_selector.current(0)
        assert panel.client_size.get()=='1366×768'
        # A missing window must not retain a stale connected badge or a preview.
        with patch('app.ui.advice.windows',return_value=[]):panel.refresh()
        root.update()
        assert panel.connection.cget('text')=='● 未连接'
        panel.buttons['开始自动建议'].invoke()
        assert panel.status.get()=='请先选择雀魂窗口。'
        panel.stop()
        assert not panel.table.rows and panel.hero.result is None
        (args.output/'visual-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'status':'passed','renders':len(report),'output':str(args.output)},ensure_ascii=False))
    finally:panel.close()


if __name__=='__main__':main()

"""Compare the former pin layout with the game-referenced production layout."""
import json
from pathlib import Path
import tkinter as tk
from unittest.mock import patch

from app.capture.frames import save_image
from app.capture.window import _api
from app.ui import ivory
from scripts.capture_native_series import print_window


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/ui-pin-traditional'
OLD = {
    1:[(0,0)],2:[(0,-.5),(0,.5)],3:[(-.5,-.6),(0,0),(.5,.6)],
    4:[(-.5,-.5),(.5,-.5),(-.5,.5),(.5,.5)],
    5:[(-.5,-.6),(.5,-.6),(0,0),(-.5,.6),(.5,.6)],
    6:[(a,b) for b in (-.65,0,.65) for a in (-.5,.5)],
    7:[(-.5,-.7),(0,-.45),(.5,-.2),(-.5,.25),(.5,.25),(-.5,.7),(.5,.7)],
    8:[(a,b) for b in (-.75,-.25,.25,.75) for a in (-.5,.5)],
    9:[(a,b) for b in (-.65,0,.65) for a in (-.6,0,.6)],
}


def old_tile(canvas,x,y,width,height,value):
    for bounds,radius,fill,outline in (
        ((x+3,y+4,x+width+4,y+height+5),8,'#DAD8CB',''),
        ((x+2,y+3,x+width+2,y+height+2),8,ivory.GREEN,''),
        ((x,y,x+width,y+height-3),8,'#FCFAF1','#D8D3C5'),
        ((x+3,y+3,x+width-3,y+height-6),6,'#FFFEF9','#EEE9DE')):
        ivory.rounded(canvas,bounds,radius,fill=fill,outline=outline)
    number=int(value[0]);r=width*.092
    for a,b in OLD[number]:
        px,py=x+width/2+a*width*.59,y+(height-3)/2+b*height*.52
        color=ivory.RED if value.endswith('r') or (a==0 and number in (1,3,5,7)) else ivory.INK
        canvas.create_oval(px-r,py-r,px+r,py+r,outline=color,width=max(1,width*.025))
        canvas.create_oval(px-r*.43,py-r*.43,px+r*.43,py+r*.43,fill=color,outline='')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    user=_api()
    root=tk.Tk();root.withdraw();root.attributes('-alpha',0)
    root.title('筒子牌排布调整');root.geometry('1220x694+10+10')
    ivory.IvoryTheme(root)
    heading=tk.Canvas(root,height=90,background=ivory.PAPER,highlightthickness=0)
    heading.pack(fill='x')
    heading.create_text(24,30,text='筒子牌 · 雀魂排布 + A 传统日麻',anchor='w',fill=ivory.INK,
                        font=(ivory.UI,-26,'bold'))
    heading.create_text(24,66,text='蓝绿双环与红心点缀，沿用已调整的圆点位置、大小及上下分组。',
                        anchor='w',fill=ivory.MUTED,font=(ivory.UI,-14))
    holder=tk.Frame(root,background=ivory.PAPER);holder.pack(fill='both',expand=True,padx=14,pady=(0,14))
    for column in (0,1):holder.columnconfigure(column,weight=1,uniform='column')
    holder.rowconfigure(0,weight=1)
    current_tile=ivory.tile
    def draw(canvas,*args):
        (old_tile if getattr(canvas,'_old_pins',False) else current_tile)(canvas,*args)
    recommendation={'status':'ready','recommendation':{'semantic':'riichi','event':{'type':'reach'},
                    'after_riichi_declaration':{'event':{'pai':'7p'}}}}
    panels=[]
    with patch.object(ivory,'tile',draw):
        for column,legacy in enumerate((True,False)):
            panel=tk.Frame(holder,background=ivory.CARD,highlightbackground=ivory.BORDER,highlightthickness=1)
            panel.grid(row=0,column=column,sticky='nsew',padx=8)
            panels.append(panel)
            label=tk.Label(panel,text='最初版本' if legacy else '当前 · A 传统日麻',background=ivory.CARD,
                           foreground=ivory.MUTED if legacy else ivory.GREEN,font=(ivory.UI,-22,'bold'),anchor='w')
            label.pack(fill='x',padx=16,pady=(14,12))
            card=ivory.AdviceCard(panel);card._old_pins=legacy
            card.configure(background=ivory.CARD);card.pack(fill='x',padx=12)
            card.show(recommendation,'立直，宣言后打出 七筒')
            strip=tk.Canvas(panel,height=114,background=ivory.CARD,highlightthickness=0)
            strip._old_pins=legacy;strip.pack(fill='x',padx=12)
            for i,value in enumerate([f'{v}p' for v in range(1,10)]+['5pr']):
                before=len(strip.find_all());draw(strip,5+i*55,12,44,62,value)
                created=strip.find_all()[before:]
                if legacy:assert sum(strip.type(item)=='oval' for item in created)==2*int(value[0])
                else:
                    assert len(ivory.pin_dots(int(value[0]),44,62))==int(value[0])
                    assert sum(strip.type(item)=='image' for item in created)==1
                strip.create_text(27+i*55,96,text='赤五' if value=='5pr' else '一二三四五六七八九'[i]+'筒',
                                  fill=ivory.RED if value=='5pr' else ivory.MUTED,font=(ivory.UI,-11))
            zoom=tk.Canvas(panel,height=242,background=ivory.CARD,highlightthickness=0)
            zoom._old_pins=legacy;zoom.pack(fill='both',expand=True,padx=12)
            zoom.create_text(8,17,text='六筒、七筒、九筒放大对比',anchor='w',fill=ivory.MUTED,font=(ivory.UI,-12))
            for i,n in enumerate((6,7,9)):
                x=57+i*170;draw(zoom,x,43,90,126,f'{n}p')
                caption=({6:'三排等距',7:'上下同宽',9:'点小、间距大'} if legacy else
                         {6:'上二、下四',7:'上三、下四',9:'三行三列'})[n]
                zoom.create_text(x+45,196,text=caption,
                                 fill=ivory.MUTED,font=(ivory.UI,-12))
        root.deiconify();root.update()
        root.after(80,root.quit);root.mainloop()
        pixels,_=print_window(user.GetAncestor(root.winfo_id(),2))
        save_image(OUT/'comparison.png',pixels)
        for name,panel in zip(('before','after'),panels):
            x,y=holder.winfo_x()+panel.winfo_x(),holder.winfo_y()+panel.winfo_y()
            save_image(OUT/f'{name}.png',pixels[y:y+panel.winfo_height(),x:x+panel.winfo_width()].copy())
        (OUT/'visual-check.json').write_text(json.dumps({
            'status':'passed','scope':'Real desktop components, example riichi action; no game connection.',
            'reference':'data/public-mahjong/pjura/dataset/train/{1..9}p/{1..9}p.png',
            'checked_tiles':[f'{n}p' for n in range(1,10)]+['5pr'],
            'pip_counts_correct':True,'production_applied':True,'style':'A-traditional',
            'patterns':ivory.PIN_PATTERNS},ensure_ascii=False,indent=2),encoding='utf-8')
    root.destroy()
    print(json.dumps({'status':'passed','comparison':str(OUT/'comparison.png')},ensure_ascii=False))


if __name__=='__main__':main()

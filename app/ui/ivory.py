"""Compact ivory/forest desktop components, drawn locally with no theme assets."""
from functools import lru_cache
import math
import re
import tkinter as tk
from tkinter import font, ttk

from PIL import Image, ImageDraw, ImageTk


PAPER = '#F5F1E9'
CARD = '#FFFCF7'
INK = '#262C27'
MUTED = '#85867E'
GREEN = '#205B46'
GREEN_HOVER = '#2C7157'
PALE_GREEN = '#E7F0E8'
BORDER = '#E4DED2'
GOLD = '#C6AA76'
RED = '#B1362D'
UI = 'Microsoft YaHei UI'
SERIF = 'Noto Serif SC'


# Centres and relative radii referenced against the existing Mahjong Soul
# default-skin crops (88 x 142). Fit each complete glyph with a uniform scale:
# the displayed tile is wider, but the pips must remain circular.
PIN_PATTERNS = {
    1: ([(44,81)],39),
    2: ([(44,53.5),(44,108.5)],22.5),
    3: ([(24,47.5),(44,80),(65,113)],15.5),
    4: ([(27,49.5),(61,49.5),(27,114.5),(61,114.5)],15.5),
    5: ([(26,49),(62,49),(44,81),(26,113),(62,113)],15.5),
    6: ([(29,44),(59,44),(29,88),(59,88),(29,117),(59,117)],14),
    7: ([(21,41),(44,55),(66,70),(30,93),(58,93),(30,120),(58,120)],13),
    8: ([(x,y) for y in (41,67,93,120) for x in (31,57)],12),
    9: ([(x,y) for y in (45,81,117) for x in (19,44,69)],12),
}
PIN_RED_DOTS = {3:{1},5:{2},6:{2,3,4,5},7:{3,4,5,6},9:{3,4,5}}


def pin_dots(number,width,height):
    """Centre the game-style groups inside the existing ivory tile face."""
    points,radius=PIN_PATTERNS[number]
    left,right=min(x for x,y in points)-radius,max(x for x,y in points)+radius
    top,bottom=min(y for x,y in points)-radius,max(y for x,y in points)+radius
    scale=min(width*.78/(right-left),(height-3)*.82/(bottom-top))
    cx,cy=(left+right)/2,(top+bottom)/2
    return [(width/2+(px-cx)*scale,(height-3)/2+(py-cy)*scale,radius*scale)
            for px,py in points]


@lru_cache(maxsize=64)
def pin_face(number,width,height,red=False):
    """Style A double rings, antialiased at the current game-style positions."""
    scale=5
    image=Image.new('RGBA',(width*scale,height*scale))
    draw=ImageDraw.Draw(image)
    face='#FFFEF9';blue='#245B71';crimson='#B83A36'
    def circle(cx,cy,radius,color):
        draw.ellipse(tuple(v*scale for v in (cx-radius,cy-radius,cx+radius,cy+radius)),fill=color)
    for index,(px,py,r) in enumerate(pin_dots(number,width,height)):
        color=crimson if red or index in PIN_RED_DOTS.get(number,set()) else blue
        circle(px,py,r,color)
        circle(px,py,r*.72,face)
        circle(px,py,r*.43,color)
        circle(px,py,r*.19,face)
        if number==1:
            for petal in range(8):
                angle=petal*math.pi/4
                circle(px+math.cos(angle)*r*.49,py+math.sin(angle)*r*.49,r*.13,face)
            circle(px,py,r*.27,crimson)
            circle(px,py,r*.10,'#FFF2D8')
    return image.resize((width,height),Image.Resampling.LANCZOS)


def rounded(canvas, bounds, radius=10, **kwargs):
    x1, y1, x2, y2 = bounds
    r = min(radius, (x2-x1)/2, (y2-y1)/2)
    return canvas.create_polygon(
        x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2,
        x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1,
        smooth=True, splinesteps=24, **kwargs)


def chinese_tiles(text):
    return re.sub(r'[1-9](?=[万筒索])', lambda m: '一二三四五六七八九'[int(m[0])-1], text)


def tile(canvas, x, y, width, height, value=None):
    """A small local vector tile; its face always comes from the real action."""
    rounded(canvas, (x+3,y+4,x+width+4,y+height+5), 8, fill='#DAD8CB', outline='')
    rounded(canvas, (x+2,y+3,x+width+2,y+height+2), 8, fill=GREEN, outline='')
    rounded(canvas, (x,y,x+width,y+height-3), 8, fill='#FCFAF1', outline='#D8D3C5')
    rounded(canvas, (x+3,y+3,x+width-3,y+height-6), 6, fill='#FFFEF9', outline='#EEE9DE')
    cx, cy = x+width/2, y+(height-3)/2
    serif=getattr(canvas.winfo_toplevel(),'_majan_serif','SimSun')
    if not value:
        canvas.create_text(cx, cy, text='發', fill=GREEN, font=(serif,-int(height*.45),'bold'))
        return
    if value[0].isdigit() and value[1:2] in 'mps':
        number, suit = int(value[0]), value[1]
        if suit == 'p':
            if not hasattr(canvas,'_pin_images'):canvas._pin_images={}
            key=(value,int(width),int(height))
            if key not in canvas._pin_images:
                canvas._pin_images[key]=ImageTk.PhotoImage(
                    pin_face(number,key[1],key[2],value.endswith('r')),master=canvas)
            canvas.create_image(x,y,image=canvas._pin_images[key],anchor='nw')
        else:
            color = RED if suit == 'm' or value.endswith('r') else GREEN
            canvas.create_text(cx,y+height*.28,text='一二三四五六七八九'[number-1],fill=color if value.endswith('r') else INK,
                               font=(serif,-int(height*.34),'bold'))
            canvas.create_text(cx,y+height*.67,text='萬' if suit == 'm' else '索',fill=color,
                               font=(serif,-int(height*.36),'bold'))
    else:
        text = {'E':'東','S':'南','W':'西','N':'北','P':'白','F':'發','C':'中'}.get(value,value)
        canvas.create_text(cx,cy,text=text,fill=RED if value == 'C' else GREEN if value == 'F' else INK,
                           font=(serif,-int(height*.47),'bold'))


class IvoryTheme:
    """Native ttk controls retain keyboard, focus, disabled and hover behavior."""
    def __init__(self, root):
        self.style = s = ttk.Style(root)
        s.theme_use('clam')
        self.images = []
        self.icons={}
        for title in ('刷新雀魂窗口','整理窗口','开始自动建议','刷新截图'):
            icon=Image.new('RGBA',(48,48))
            draw=ImageDraw.Draw(icon)
            color='white' if title == '开始自动建议' else GREEN
            if title == '刷新雀魂窗口':
                draw.arc((5,5,42,42),35,320,fill=color,width=4)
                draw.polygon([(40,3),(42,18),(28,11)],fill=color)
            elif title == '整理窗口':
                draw.rounded_rectangle((6,8,42,40),radius=4,outline=color,width=4)
                draw.line((7,18,41,18),fill=color,width=3)
                draw.line((19,19,19,39),fill=color,width=3)
            elif title == '开始自动建议':draw.polygon([(13,5),(41,24),(13,43)],fill=color)
            else:
                draw.rounded_rectangle((5,14,43,40),radius=5,outline=color,width=4)
                draw.rounded_rectangle((16,8,31,17),radius=3,fill=color)
                draw.ellipse((18,20,31,33),outline=color,width=3)
            photo=ImageTk.PhotoImage(icon.resize((16,16),Image.Resampling.LANCZOS),master=root)
            self.images.append(photo);self.icons[title]=photo
        self.serif=SERIF if SERIF in font.families(root) else 'SimSun'
        root._majan_serif=self.serif
        root.configure(background=PAPER)
        root.option_add('*TCombobox*Listbox.background', CARD)
        root.option_add('*TCombobox*Listbox.foreground', INK)
        root.option_add('*TCombobox*Listbox.selectBackground', PALE_GREEN)
        root.option_add('*TCombobox*Listbox.selectForeground', GREEN)
        root.option_add('*TCombobox*Listbox.font', (UI,-13))
        s.configure('Ivory.TCombobox', foreground=INK, fieldbackground=CARD, background=CARD,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, arrowcolor=GREEN,
                    padding=(8,4), arrowsize=13, font=(UI,-13))
        s.map('Ivory.TCombobox', fieldbackground=[('readonly',CARD)],
              foreground=[('readonly',INK)], bordercolor=[('focus',GREEN)],
              selectbackground=[('readonly',CARD)], selectforeground=[('readonly',INK)])
        fields=[]
        for color in (BORDER,GREEN):
            image=Image.new('RGB',(180,90),PAPER)
            ImageDraw.Draw(image).rounded_rectangle((1,1,178,88),radius=21,fill=CARD,outline=color,width=3)
            photo=ImageTk.PhotoImage(image.resize((60,30),Image.Resampling.LANCZOS),master=root)
            self.images.append(photo);fields.append(photo)
        s.element_create('Ivory.field','image',fields[0],('focus',fields[1]),border=5,sticky='nsew')
        s.layout('Ivory.TCombobox',[('Ivory.field',{'sticky':'nsew','children':[
            ('Combobox.downarrow',{'side':'right','sticky':'ns'}),
            ('Combobox.padding',{'sticky':'nsew','children':[('Combobox.textarea',{'sticky':'nsew'})]})]})])
        s.configure('Ivory.TCombobox',bordercolor=CARD,lightcolor=CARD,darkcolor=CARD,padding=(6,1))
        for name, primary in (('Ivory.TButton',False),('Ivory.Primary.TButton',True)):
            fills = (GREEN,GREEN_HOVER,'#164633','#D9DED5') if primary else (CARD,'#EFEBE1','#E3E9DF','#F0EDE6')
            images = []
            for index, color in enumerate(fills):
                image = Image.new('RGB',(180,60),PAPER)
                draw = ImageDraw.Draw(image)
                draw.rounded_rectangle((1,1,178,58),radius=12,fill=color,
                                       outline=GREEN if index == 2 else color if primary else BORDER,width=3)
                photo = ImageTk.PhotoImage(image.resize((60,20),Image.Resampling.LANCZOS),master=root)
                self.images.append(photo); images.append(photo)
            element = name+'.surface'
            s.element_create(element,'image',images[0],('disabled',images[3]),('pressed',images[2]),
                             ('active',images[1]),('focus',images[2]),border=4,sticky='nsew')
            s.layout(name,[(element,{'sticky':'nsew','children':[
                ('Button.padding',{'sticky':'nsew','children':[('Button.label',{'sticky':'nsew'})]})]})])
            s.configure(name,foreground='#FFFFFF' if primary else GREEN,font=(UI,-13,'bold'),
                        padding=(12,5),anchor='center',background=PAPER)
            s.map(name,foreground=[('disabled','#9B9F95')])


class Card(tk.Canvas):
    def __init__(self,parent,*,padding=12,**kwargs):
        super().__init__(parent,background=PAPER,highlightthickness=0,borderwidth=0,**kwargs)
        self.padding=padding
        self.body=tk.Frame(self,background=CARD)
        self.body_id=self.create_window(padding,padding,window=self.body,anchor='nw')
        self.bind('<Configure>',self._resize)
        self.body.bind('<Configure>',self._height)

    def _height(self,event=None):
        height=self.body.winfo_reqheight()+self.padding*2
        if int(self.cget('height')) != height:self.configure(height=height)

    def _resize(self,event):
        self.delete('surface')
        rounded(self,(1,1,event.width-1,event.height-1),12,fill=CARD,outline=BORDER,tags='surface')
        self.tag_lower('surface')
        self.itemconfigure(self.body_id,width=max(1,event.width-self.padding*2))


class AdviceCard(tk.Canvas):
    def __init__(self,parent):
        super().__init__(parent,height=148,background=PAPER,highlightthickness=0)
        self.result=None
        self.message='尚无建议'
        self.bind('<Configure>',self.draw)

    def show(self,result,message):
        self.result,self.message=result,message
        self.draw()

    def draw(self,event=None):
        self.delete('all')
        w,h=self.winfo_width(),self.winfo_height()
        if w < 50:return
        serif=getattr(self.winfo_toplevel(),'_majan_serif','SimSun')
        rounded(self,(1,1,w-1,h-1),12,fill=CARD,outline=GOLD)
        self.create_polygon(15,23,20,18,25,23,20,28,fill=GOLD,outline='')
        self.create_text(32,23,text='当前建议',anchor='w',fill=INK,font=(serif,-19,'bold'))
        state=(self.result or {}).get('status')
        ready=state == 'ready'
        preparing=state is None and self.message.startswith('正在')
        badge={'ready':'● 识别完成','needs_confirmation':'● 需要确认','error':'● 暂未就绪',
               'confirming_frame':'● 正在分析'}.get(state,'● 等待行动')
        if preparing:badge='● 正在准备'
        self.create_text(w-16,23,text=badge,anchor='e',fill=GREEN if ready else MUTED,font=(UI,-11))
        if ready:
            rec=self.result['recommendation']
            value=rec.get('event',{}).get('pai')
            if rec['semantic'] == 'riichi':
                value=rec.get('after_riichi_declaration',{}).get('event',{}).get('pai')
            # Subtle corner motif behind the actual recommended tile.
            for offset in (0,16):
                self.create_polygon(w-47+offset,40,w-19+offset,68,w-47+offset,96,w-75+offset,68,
                                    fill='',outline='#EEE6D6',width=1)
            if value:tile(self,w-106,45,62,87,value)
            main,_,detail=chinese_tiles(self.message).partition('（使用 ')
            if '，宣言后打出 ' in main:
                main,detail=main.split('，宣言后打出 ',1);detail='宣言后打出 '+detail
            elif '；' in main:
                main,detail=main.split('；',1)
            elif detail:detail='使用 '+detail.rstrip('）')
            available=w-140 if value else w-36
            size=34
            face=font.Font(self,family=serif,size=-size,weight='bold')
            while face.measure(main) > available and size > 19:
                size-=1;face.configure(size=-size)
            y=82 if detail else 91
            prefix,separator,suffix=main.partition(' ')
            if separator and face.measure(main)<=available:
                self.create_text(20,y,text=prefix+' ',anchor='w',fill=INK,font=(serif,-size,'bold'))
                self.create_text(20+face.measure(prefix+' '),y,text=suffix,anchor='w',fill=GREEN,font=(serif,-size,'bold'))
            else:self.create_text(20,y,text=main,anchor='w',fill=GREEN,font=(serif,-size,'bold'),width=available)
            if detail:self.create_text(20,119,text=detail,anchor='w',fill=MUTED,font=(UI,-11),width=available)
        else:
            title={'needs_confirmation':'等待局面确认','error':'暂时无法获取建议',
                   'confirming_frame':'正在识别当前局面','waiting_turn':'等待自己的行动'}.get(state,'尚无当前建议')
            if preparing:title='正在准备建议'
            self.create_text(20,66,text=title,anchor='w',fill=GREEN,font=(serif,-23,'bold'))
            message=self.message if self.message != '尚无建议' else '选择游戏窗口，开始自动建议。'
            self.create_text(20,94,text=message,anchor='nw',fill=MUTED,font=(UI,-12),width=w-40)


class CandidateList(tk.Canvas):
    """Five real candidate rows; scores are supplied by the existing Q conversion."""
    def __init__(self,parent):
        super().__init__(parent,height=222,background=PAPER,highlightthickness=0)
        self.rows=[]
        self.bind('<Configure>',self.draw)

    def clear(self):
        self.rows=[];self.draw()

    def set_rows(self,rows):
        self.rows=list(rows)[:5];self.draw()

    def draw(self,event=None):
        self.delete('all')
        w,h=self.winfo_width(),self.winfo_height()
        if w < 50:return
        serif=getattr(self.winfo_toplevel(),'_majan_serif','SimSun')
        rounded(self,(1,1,w-1,h-1),12,fill=CARD,outline=BORDER)
        self.create_polygon(15,23,20,18,25,23,20,28,fill=GOLD,outline='')
        self.create_text(32,23,text='候选操作 · Top 5',anchor='w',fill=INK,font=(serif,-17,'bold'))
        self.create_text(w-18,23,text='相对评分',anchor='e',fill=MUTED,font=(UI,-11))
        if not self.rows:
            self.create_text(w/2,115,text='暂无候选操作',fill=MUTED,font=(UI,-15))
            self.create_text(w/2,143,text='识别完成后，在这里比较建议',fill=MUTED,font=(UI,-11))
            return
        for i,(label,percentage) in enumerate(self.rows):
            top=44+i*33;cy=top+15
            if i == 0:rounded(self,(12,top,w-12,top+31),7,fill=PALE_GREEN,outline='')
            elif i < len(self.rows)-1:self.create_line(16,top+32,w-16,top+32,fill='#F0ECE3')
            self.create_oval(21,cy-10,41,cy+10,fill=GREEN if i == 0 else '#B0B5AB',outline='')
            self.create_text(31,cy,text=str(i+1),fill='white',font=(UI,-12,'bold'))
            label=chinese_tiles(label)
            # Consumed tiles stay visible on a second line instead of shrinking the action.
            text_width=int(w*.51)-53
            label,separator,detail=label.partition('（使用 ')
            if separator:detail='使用 '+detail.rstrip('）')
            elif '；' in label:label,detail=label.split('；',1)
            text_font=font.Font(self,family=UI,size=-13,weight='bold' if i == 0 else 'normal')
            size=13
            while text_font.measure(label) > text_width and size > 10:
                size-=1;text_font.configure(size=-size)
            if text_font.measure(label) > text_width:
                self.create_text(53,cy-6 if detail else cy,text=label,anchor='w',fill=INK,font=(UI,-10),width=text_width)
            else:self.create_text(53,cy-6 if detail else cy,text=label,anchor='w',fill=INK,font=(UI,-size,'bold' if i == 0 else 'normal'))
            if detail:
                detail_font=font.Font(self,family=UI,size=-10)
                while detail_font.measure(detail)>text_width and len(detail)>1:
                    detail=detail.rstrip('…')[:-1]+'…'
                self.create_text(53,cy+9,text=detail,anchor='w',fill=MUTED,font=(UI,-10))
            start,end=int(w*.53),w-89
            rounded(self,(start,cy-4,end,cy+4),4,fill='#E8E7DF',outline='')
            share=float(percentage.rstrip('%'))/100
            if share > 0:rounded(self,(start,cy-4,start+max(2,(end-start)*share),cy+4),4,
                                fill=GREEN if i == 0 else '#7EA18B',outline='')
            self.create_text(w-18,cy,text=percentage,anchor='e',fill=GREEN if i == 0 else INK,
                             font=(UI,-15,'bold' if i == 0 else 'normal'))

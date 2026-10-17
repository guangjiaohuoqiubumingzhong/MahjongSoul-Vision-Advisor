"""Code-drawn pin-tile candidates; production UI is unchanged until selection."""
from functools import lru_cache
import math

from PIL import Image, ImageDraw


STYLES = {
    'A': ('传统日麻', '蓝绿双环，红心点缀，接近实体日麻牌'),
    'B': ('清爽扁平', '玉绿色实心圆，简洁醒目，与米白界面协调'),
    'C': ('珐琅立体', '深蓝釉色与柔和高光，牌面更有质感'),
    'D': ('极简线描', '墨绿细环与留白，视觉更轻盈'),
}
PATTERNS = {
    1: [(0, 0)], 2: [(0, -.56), (0, .56)],
    3: [(-.54, -.62), (0, 0), (.54, .62)],
    4: [(-.54, -.54), (.54, -.54), (-.54, .54), (.54, .54)],
    5: [(-.54, -.64), (.54, -.64), (0, 0), (-.54, .64), (.54, .64)],
    6: [(a, b) for b in (-.68, 0, .68) for a in (-.54, .54)],
    7: [(-.54, -.76), (0, -.51), (.54, -.26),
        (-.54, .25), (.54, .25), (-.54, .74), (.54, .74)],
    8: [(a, b) for b in (-.76, -.25, .25, .76) for a in (-.54, .54)],
    9: [(a, b) for b in (-.68, 0, .68) for a in (-.63, 0, .63)],
}


def mix(a, b, amount):
    a = tuple(int(a[i:i+2], 16) for i in (1, 3, 5))
    b = tuple(int(b[i:i+2], 16) for i in (1, 3, 5))
    return tuple(round(x+(y-x)*amount) for x, y in zip(a, b))


@lru_cache(maxsize=160)
def render_tile(value, width=62, height=87, style='B'):
    number = int(value[0])
    assert value in [f'{n}p' for n in range(1, 10)] + ['5pr']
    assert style in STYLES
    scale = 5
    image = Image.new('RGBA', ((width+7)*scale, (height+7)*scale))
    draw = ImageDraw.Draw(image)

    def box(bounds): return tuple(v*scale for v in bounds)
    def rounded(bounds, radius, fill, outline=None, line=.8):
        draw.rounded_rectangle(box(bounds), radius=radius*scale, fill=fill,
                               outline=outline, width=max(1, round(line*scale)))
    def circle(x, y, radius, fill=None, outline=None, line=.8):
        draw.ellipse(box((x-radius, y-radius, x+radius, y+radius)), fill=fill,
                     outline=outline, width=max(1, round(line*scale)))

    # Keep the existing tile dimensions, ivory face and forest-green base.
    rounded((3, 4, width+4, height+5), 5, '#DAD8CB')
    rounded((2, 3, width+2, height+2), 5, '#205B46')
    rounded((0, 0, width, height-3), 5, '#FCFAF1', '#D8D3C5')
    rounded((3, 3, width-3, height-6), 3.5, '#FFFEF9', '#EEE9DE', .55)
    cx, cy = width/2, (height-3)/2
    base = {'A': '#245B71', 'B': '#277961', 'C': '#275B88', 'D': '#386557'}[style]
    red = '#B83A36'
    for index, (a, b) in enumerate(PATTERNS[number]):
        x, y = cx+a*width*.50, cy+b*height*.43
        r = width*(.24 if number == 1 else .093 if number < 9 else .083)
        accent = number in (3, 5) and a == b == 0 or number == 7 and index == 1
        color = red if value.endswith('r') or accent else base
        if style == 'A':
            circle(x, y, r, fill=color)
            circle(x, y, r*.72, fill='#FFFEF9')
            circle(x, y, r*.43, fill=color)
            circle(x, y, r*.19, fill='#FFFEF9')
            if number == 1:
                for petal in range(8):
                    angle = petal*math.pi/4
                    circle(x+math.cos(angle)*r*.49, y+math.sin(angle)*r*.49,
                           r*.13, fill='#FFFEF9')
                circle(x, y, r*.27, fill=red)
                circle(x, y, r*.10, fill='#FFF2D8')
        elif style == 'B':
            circle(x, y, r, fill=color)
            circle(x, y, r*.32, fill='#FFFEF9')
            if number == 1:
                circle(x, y, r*.58, fill='#FFFEF9')
                circle(x, y, r*.30, fill=red)
        elif style == 'C':
            circle(x+.2, y+.45, r+.35, fill='#E5E3D9')
            circle(x, y, r, fill=mix(color, '#102B3C', .25))
            for step in range(20, 0, -1):
                radius = r*.89*step/20
                circle(x, y-.1, radius, fill=mix(color, '#A8D7E3' if color != red else '#F4A18C',
                                               (1-step/20)*.52))
            circle(x, y, r*.57, outline=mix(color, '#E8F2EF', .55), line=max(.45, r*.1))
            circle(x-r*.28, y-r*.30, r*.16, fill='#EAF6F5')
            circle(x, y+r*.12, r*.18, fill=color)
            if number == 1:
                circle(x, y, r*.27, fill=red)
                circle(x-r*.06, y-r*.07, r*.07, fill='#FFE3CE')
        else:
            circle(x, y, r, outline=color, line=max(.9, width*.018))
            circle(x, y, r*.28, fill=color)
            if number == 1:
                circle(x, y, r*.66, outline=color, line=width*.016)
                circle(x, y, r*.23, fill=red)
    return image.resize((width+7, height+7), Image.Resampling.LANCZOS)

"""Render selectable pin icons inside the real advice card, without changing it."""
import importlib.util
import json
from pathlib import Path
import tkinter as tk
from unittest.mock import patch

from PIL import ImageTk

from app.capture.frames import save_image
from app.capture.window import _api
from app.ui import ivory
from scripts.capture_native_series import print_window


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'design/ui-concepts/2026-10-17-pin-styles'
spec = importlib.util.spec_from_file_location('pin_styles', OUT / 'render_styles.py')
styles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    user = _api()
    root = tk.Tk(); root.withdraw(); root.attributes('-alpha', 0)
    root.title('筒子牌样式候选'); root.geometry('1360x900+10+10')
    ivory.IvoryTheme(root)
    photos = []
    original_tile = ivory.tile

    def candidate(canvas, x, y, width, height, value=None):
        style = getattr(canvas, '_pin_style', 'B')
        if value and value[1:2] == 'p':
            photo = ImageTk.PhotoImage(styles.render_tile(value, width, height, style), master=root)
            photos.append(photo)
            canvas.create_image(x, y, image=photo, anchor='nw')
        else: original_tile(canvas, x, y, width, height, value)

    heading = tk.Canvas(root, height=100, background=ivory.PAPER, highlightthickness=0)
    heading.pack(fill='x')
    heading.create_text(30, 31, text='筒子牌 · 四种样式', anchor='w', fill=ivory.INK,
                        font=(ivory.UI, -29, 'bold'))
    heading.create_text(30, 70, text='同一米白界面 · 七筒为实际显示大小 · 下方展示一至九筒与赤五筒',
                        anchor='w', fill=ivory.MUTED, font=(ivory.UI, -15))
    grid = tk.Frame(root, background=ivory.PAPER)
    grid.pack(fill='both', expand=True, padx=20, pady=(0, 18))
    for row in range(2): grid.rowconfigure(row, weight=1, uniform='row')
    for column in range(2): grid.columnconfigure(column, weight=1, uniform='column')
    recommendation = {'status': 'ready', 'recommendation': {
        'semantic': 'riichi', 'event': {'type': 'reach'},
        'after_riichi_declaration': {'event': {'pai': '7p'}}}}
    panels = []
    with patch.object(ivory, 'tile', candidate):
        for index, (key, (name, description)) in enumerate(styles.STYLES.items()):
            panel = tk.Frame(grid, background=ivory.CARD, highlightbackground=ivory.BORDER,
                             highlightthickness=1)
            panel.grid(row=index//2, column=index%2, sticky='nsew', padx=8, pady=8)
            panels.append((key, panel))
            top = tk.Canvas(panel, height=77, background=ivory.CARD, highlightthickness=0)
            top.pack(fill='x')
            top.create_text(18, 26, text=f'{key}  {name}' + ('  · 推荐' if key == 'B' else ''),
                            anchor='w', fill=ivory.GREEN, font=(ivory.UI, -22, 'bold'))
            top.create_text(18, 57, text=description, anchor='w', fill=ivory.MUTED,
                            font=(ivory.UI, -13))
            card = ivory.AdviceCard(panel); card._pin_style = key
            card.configure(background=ivory.CARD)
            card.pack(fill='x', padx=14)
            card.show(recommendation, '立直，宣言后打出 七筒')
            strip = tk.Canvas(panel, height=142, background=ivory.CARD, highlightthickness=0)
            strip.pack(fill='both', expand=True)
            for n, value in enumerate([f'{v}p' for v in range(1, 10)] + ['5pr']):
                photo = ImageTk.PhotoImage(styles.render_tile(value, 49, 69, key), master=root)
                photos.append(photo)
                x = 16+n*61
                strip.create_image(x, 16, image=photo, anchor='nw')
                strip.create_text(x+24, 109, text='赤五' if value == '5pr' else '一二三四五六七八九'[n]+'筒',
                                  fill=ivory.RED if value == '5pr' else ivory.MUTED,
                                  font=(ivory.UI, -12))
            for value in [f'{v}p' for v in range(1, 10)] + ['5pr']:
                styles.render_tile(value, 62, 87, key).save(OUT / f'{key}-{value}.png')
        root.deiconify(); root.update()
        root.after(100, root.quit); root.mainloop()
        hwnd = user.GetAncestor(root.winfo_id(), 2)
        pixels, _ = print_window(hwnd)
        save_image(OUT / 'comparison.png', pixels)
        for key, panel in panels:
            x, y = panel.winfo_x()+grid.winfo_x(), panel.winfo_y()+grid.winfo_y()
            w, h = panel.winfo_width(), panel.winfo_height()
            save_image(OUT / f'{key}-preview.png', pixels[y:y+h, x:x+w].copy())
        (OUT / 'preview-info.json').write_text(json.dumps({
            'scope': 'Design choices rendered with the real advice card; example action, not a model result.',
            'production_changed': False, 'card_tile_size': [62, 87], 'antialias_scale': 5,
            'styles': styles.STYLES}, ensure_ascii=False, indent=2), encoding='utf-8')
    root.destroy()
    print(json.dumps({'status': 'rendered', 'comparison': str(OUT / 'comparison.png')}, ensure_ascii=False))


if __name__ == '__main__': main()

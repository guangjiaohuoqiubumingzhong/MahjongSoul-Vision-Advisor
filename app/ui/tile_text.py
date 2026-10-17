from app.state.events import TILES

NAMES = {t:(t[0]+{"m":"万","p":"筒","s":"索"}[t[1]]+("赤" if t.endswith("r") else ""))
         for t in TILES if t[0].isdigit()}
NAMES.update(dict(zip("ESWNPFC",["东","南","西","北","白","发","中"])))
REVERSE = {name:tile for tile,name in NAMES.items()}


def tile_name(tile):
    return NAMES.get(tile,tile)


def parse_tiles(text):
    result = []
    for word in text.replace("，"," ").replace(","," ").split():
        tile = REVERSE.get(word,word)
        if tile not in TILES:
            raise ValueError(f"牌面无效：{word}；例如 1万、5筒赤、东。")
        result.append(tile)
    return result

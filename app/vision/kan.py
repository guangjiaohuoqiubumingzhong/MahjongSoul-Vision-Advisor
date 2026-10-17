"""Quad structure from observed tile orientations and the public meld layout."""
import numpy as np


def row_geometry(items):
    """Distinct neighbouring bodies sharing a top or bottom baseline."""
    boxes=np.asarray([i['canonical_bbox'] for i in items],dtype=float)
    boxes=boxes[np.argsort(boxes[:,0])]
    if len(boxes)<3 or not np.all(np.isfinite(boxes)) or np.any(boxes[:,2:]<=0):return False
    widths,heights=boxes[:,2],boxes[:,3]
    steps=np.diff(boxes[:,0])
    if (np.any(steps<.6*np.minimum(widths[:-1],widths[1:]))
            or np.any(steps>1.4*np.maximum(widths[:-1],widths[1:]))):return False
    tolerance=.4*float(np.min(heights))
    if min(np.ptp(boxes[:,1]),np.ptp(boxes[:,1]+heights))<=tolerance:return True
    for baseline in (boxes[:,1],boxes[:,1]+heights):
        slope,intercept=np.polyfit(boxes[:,0],baseline,1)
        # Side-view projection can tilt a coherent row beyond 0.5; retain the
        # tight residual and distinct-body checks instead of rejecting its tilt.
        if abs(slope)<=.75 and np.max(np.abs(baseline-(slope*boxes[:,0]+intercept)))<=.1*min(heights):
            return True
    return False


def called_positions(items):
    """Orientation evidence, with the same conservative body/blank-face fallback."""
    known=[i for i,item in enumerate(items) if item.get('horizontal') is True]
    ratios=[item['canonical_bbox'][2]/max(1,item['canonical_bbox'][3]) for item in items]
    wide=[i for i,r in enumerate(ratios) if r>=1.15]
    if (len(wide) in (1,2) and all(r<=1 for i,r in enumerate(ratios) if i not in wide)
            and set(known).issubset(wide)):
        return wide,'tile_aspect'
    if known:return known,'tile_orientation'
    from .meld_geometry import relative_called_positions
    blank=all(i['label']=='P' for i in items)
    found=relative_called_positions(items,maximum=2,blank=blank)
    if found is not None:
        return found[0],'blank_face_relative_geometry' if blank else 'face_relative_geometry'
    return [],'unconfirmed'


def kan_structure(items,region):
    """Do not infer a kan from four matching identities alone."""
    if len(items)!=4:return None
    boxes=np.asarray([i['canonical_bbox'] for i in items],dtype=float)
    if boxes.shape!=(4,4) or not np.all(np.isfinite(boxes)) or np.any(boxes[:,2:]<=0):return None
    called,evidence=called_positions(items)
    if len(called)==1 and row_geometry(items):
        target=region.get('kan_source_by_called_position',{}).get(str(called[0]))
        if target is None:return None
        return {'type':'daiminkan','target':target,'called_positions':called,
                'called_position_evidence':evidence,'structure_evidence':'four_in_row_one_called_tile'}
    if len(called)!=2 or called[1]!=called[0]+1:return None
    first,second=(items[i]['canonical_bbox'] for i in called)
    x,y,w,h=first;bx,by,bw,bh=second
    overlap_x=max(0,min(x+w,bx+bw)-max(x,bx))/min(w,bw)
    overlap_y=max(0,min(y+h,by+bh)-max(y,by))/min(h,bh)
    if (max(overlap_x,overlap_y)<.55 or
            abs((x+w/2)-(bx+bw/2))>1.5*max(w,bw) or
            abs((y+h/2)-(by+bh/2))>1.5*max(h,bh)):
        return None
    possibilities=[i for i in called if row_geometry([item for j,item in enumerate(items) if j!=i])]
    if len(possibilities)==1:
        added=possibilities[0];added_evidence='unique_three_tile_base_row'
    elif len(possibilities)==2 and items[called[0]]['label']==items[called[1]]['label']:
        # Either layer yields the same exact pai/consumed event. No red/normal
        # distinction is guessed from the seat's apparent screen direction.
        added=called[0];added_evidence='called_pair_has_identical_exact_tile_labels'
    else:return None
    base_called=called[0]  # collapsing the adjacent pair leaves one pon position
    target=region.get('pon_source_by_called_position',{}).get(str(base_called))
    return {'type':'kakan','target':target,'called_positions':called,'added_position':added,
            'added_position_evidence':added_evidence,'called_position_evidence':evidence,
            'structure_evidence':'two_adjacent_called_tiles_and_three_tile_base',
            'stack_geometry':{'overlap_x':float(overlap_x),'overlap_y':float(overlap_y)}}

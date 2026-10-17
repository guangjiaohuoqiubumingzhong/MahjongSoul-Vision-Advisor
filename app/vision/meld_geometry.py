"""Physical meld-face orientation, independent of a glyph's winning rotation."""
import cv2
import numpy as np

RATIO_SEPARATION = 1.5
UPRIGHT_SPREAD = 1.25


def screen_quad(points):
    """Put already unrotated corners in screen-axis order, or reject the fit."""
    quad = np.asarray(points,dtype=float)
    if quad.shape != (4,2) or not np.isfinite(quad).all():
        return None
    sums,differences = quad.sum(axis=1),quad[:,0]-quad[:,1]
    quad = quad[[np.argmin(sums),np.argmax(differences),np.argmax(sums),np.argmin(differences)]]
    if len({tuple(p) for p in quad}) != 4 or not cv2.isContourConvex(quad.astype(np.float32)):
        return None
    return quad


def face_ratio(item, source, *, blank=False):
    if source == "legacy":
        evidence = item.get("legacy_evidence") or {}
        if "face_quad_local" not in evidence:
            evidence = evidence.get("face_evidence") or {}
        if blank and evidence.get("blank_face") is not True:
            return None
        points = evidence.get("face_quad_local",[])
    else:
        evidence = item.get("pjura_evidence") or {}
        if evidence.get("face_fitted") is not True:
            return None
        points = evidence.get("rectified_quad",[])
    quad = screen_quad(points)
    if quad is None:
        return None
    width = (np.linalg.norm(quad[1]-quad[0])+np.linalg.norm(quad[2]-quad[3]))/2
    height = (np.linalg.norm(quad[3]-quad[0])+np.linalg.norm(quad[2]-quad[1]))/2
    return float(width/height) if min(width,height)>=4 else None


def relative_called_positions(items, *, maximum=1, blank=False):
    """Require two similar upright peers and a separated called-face cluster.

    Use screen coordinates from either fitted outline. A missing legacy fit
    may use the network's geometric outline; its label/rotation is never used.
    If two complete outline sources prove different positions, abstain.
    """
    if len(items) not in (3,4) or len(items)-maximum < 2:
        return None
    proposals = []
    all_ratios = {}
    candidates = {"legacy":[face_ratio(item,"legacy",blank=blank) for item in items]}
    if not blank:
        candidates["pjura"] = [face_ratio(item,"pjura") for item in items]
        candidates["available"] = [old if old is not None else new
                                   for old,new in zip(candidates["legacy"],candidates["pjura"])]
    for source,ratios in candidates.items():
        if any(r is None for r in ratios):
            continue
        all_ratios[source] = ratios
        order = np.argsort(ratios)
        for count in range(1,maximum+1):
            upright = [ratios[i] for i in order[:-count]]
            sideways = [ratios[i] for i in order[-count:]]
            if (min(sideways)<RATIO_SEPARATION*max(upright)
                    or max(upright)>UPRIGHT_SPREAD*min(upright)
                    or max(sideways)>UPRIGHT_SPREAD*min(sideways)):
                continue
            called = tuple(sorted(int(i) for i in order[-count:]))
            if not blank and any(items[i].get("horizontal") is False for i in called):
                continue
            proposals.append((called,source,min(sideways)/max(upright)))
    if not proposals or len({p[0] for p in proposals}) != 1:
        return None
    called,source,separation = proposals[0]
    return list(called), {"face_width_height_ratios":all_ratios[source],
                          "called_to_upright_ratio":float(separation),
                          "outline_source":source,"available_outline_ratios":all_ratios}

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


FACE_PARAMETERS = {"background_percentile":85,"minimum_background":64,"minimum_crop_side":8,
                   "white_min_channel":.94,"polygon_epsilon":.03,"minimum_area_fraction":.2,
                   "warp_size":[40,60],"face_trim":3,"glyph_min_channel":.7,
                   "blank_ink_min_channel":.96,"blank_max_ink_fraction":.01,
                   "glyph_size":[30,50],"glyph_padding":2,"blur_kernel":[3,3],"blur_sigma":1.0,
                   "ink_offset":.15,"ink_contrast":.45,"red_contrast":.3,"green_contrast":.2,
                   "correlation_weight":.65,"mse_weight":.35,"mse_scale":6,"constant_mse_scale":5}

# Preserve the original recipe for existing immutable v3 manifests. The new
# fallback is used only when the original bright-face geometry is unusable.
HIGHLIGHT_FACE_PARAMETERS = {**FACE_PARAMETERS,"fallback_white_min_channel":.84,
                           "fallback_minimum_area_fraction":.5}


@dataclass
class FaceFeature:
    feature: np.ndarray
    quad: np.ndarray
    ink_fraction: float
    area_fraction: float
    blank: bool


def _face_quad(normalized: np.ndarray, threshold: float, minimum_area: float, polygon_epsilon: float):
    mask = (np.min(normalized,axis=2) >= threshold).astype(np.uint8)*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours,key=cv2.contourArea))
    polygon = cv2.approxPolyDP(hull,polygon_epsilon*cv2.arcLength(hull,True),True).reshape(-1,2)
    area = cv2.contourArea(polygon)/(normalized.shape[0]*normalized.shape[1])
    if len(polygon)!=4 or area < minimum_area:
        return None
    points = polygon.astype(np.float32)
    sums,differences = points.sum(axis=1),points[:,0]-points[:,1]
    quad = np.array([points[np.argmin(sums)],points[np.argmax(differences)],
                     points[np.argmax(sums)],points[np.argmin(differences)]],np.float32)
    if len({tuple(p) for p in quad}) != 4:
        return None
    return quad,area


def tile_face(pixels: np.ndarray, parameters: dict | None = None) -> FaceFeature | None:
    """Rectify a visible white face and compare its coloured glyph, not its base.

    The quadrilateral comes entirely from this crop. It carries no tile identity,
    seat information or reference event. A failed extraction supplies no evidence.
    """
    parameters = FACE_PARAMETERS if parameters is None else parameters
    if min(pixels.shape[:2]) < parameters["minimum_crop_side"]:
        return None
    background = np.percentile(pixels,parameters["background_percentile"],axis=(0,1)).astype(np.float32)
    normalized = np.clip(pixels.astype(np.float32)/np.maximum(background,parameters["minimum_background"]),0,1)
    geometry = _face_quad(normalized,parameters["white_min_channel"],parameters["minimum_area_fraction"],parameters["polygon_epsilon"])
    if geometry is None and "fallback_white_min_channel" in parameters:
        geometry = _face_quad(normalized,parameters["fallback_white_min_channel"],parameters["fallback_minimum_area_fraction"],parameters["polygon_epsilon"])
    if geometry is None:
        return None
    quad,area = geometry
    width,height = parameters["warp_size"]
    trim = parameters["face_trim"]
    transform = cv2.getPerspectiveTransform(quad,np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32))
    face = cv2.warpPerspective(normalized,transform,(width,height),flags=cv2.INTER_LINEAR)[trim:-trim,trim:-trim]
    ink_fraction = float(np.mean(np.min(face,axis=2) < parameters["blank_ink_min_channel"]))
    ys,xs = np.where(np.min(face,axis=2) < parameters["glyph_min_channel"])
    feature = np.ones_like(face)
    if len(xs):
        x1,x2 = max(0,xs.min()-1),min(face.shape[1],xs.max()+2)
        y1,y2 = max(0,ys.min()-1),min(face.shape[0],ys.max()+2)
        glyph = face[y1:y2,x1:x2]
        ink = np.clip((1-np.min(glyph,axis=2)-parameters["ink_offset"])/parameters["ink_contrast"],0,1)
        red = np.clip((glyph[:,:,2]-np.maximum(glyph[:,:,0],glyph[:,:,1]))/parameters["red_contrast"],0,1)
        green = np.clip((glyph[:,:,1]-np.maximum(glyph[:,:,0],glyph[:,:,2]))/parameters["green_contrast"],0,1)
        # Keep red/green/black occupancy while removing lighting differences.
        colour = np.repeat((1-ink)[:,:,None],3,axis=2)
        colour[:,:,1] += ink*green
        colour[:,:,2] += ink*red
        padding = parameters["glyph_padding"]
        feature[padding:-padding,padding:-padding] = cv2.resize(colour,tuple(parameters["glyph_size"]),interpolation=cv2.INTER_AREA)
        feature = cv2.GaussianBlur(feature,tuple(parameters["blur_kernel"]),parameters["blur_sigma"])
    return FaceFeature(feature,quad,ink_fraction,float(area),ink_fraction <= parameters["blank_max_ink_fraction"])


def unrotate_quad(quad: np.ndarray, width: int, height: int, degrees: int) -> list[list[float]]:
    """Map pixel centres back to the classifier's original, unrotated crop."""
    x,y = quad[:,0],quad[:,1]
    if degrees == 90:
        x,y = width-1-y,x
    elif degrees == 180:
        x,y = width-1-x,height-1-y
    elif degrees == 270:
        x,y = y,height-1-x
    return np.column_stack((x,y)).astype(float).tolist()


def batch_similarity(features: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Same correlation/MSE scoring as whole-body templates, in one CPU batch."""
    vectors = features.reshape(len(features),-1)
    q = query.reshape(-1)
    mse = np.mean((vectors-q)**2,axis=1)
    centred = vectors-vectors.mean(axis=1,keepdims=True)
    cq = q-q.mean()
    denominator = np.linalg.norm(centred,axis=1)*np.linalg.norm(cq)
    correlation = np.divide(np.einsum('ij,j->i',centred,cq),denominator,
                            out=1-mse*FACE_PARAMETERS["constant_mse_scale"],where=denominator>1e-8)
    return np.clip(FACE_PARAMETERS["correlation_weight"]*correlation+
                   FACE_PARAMETERS["mse_weight"]*(1-np.minimum(1,mse*FACE_PARAMETERS["mse_scale"])),-1,1)


class FaceGallery:
    """Precompute immutable template statistics; each crop only supplies a query."""
    def __init__(self, features: np.ndarray):
        self.vectors = features.reshape(len(features),-1)
        self.centred = self.vectors-self.vectors.mean(axis=1,keepdims=True)
        self.norms = np.linalg.norm(self.centred,axis=1)
        self.squared_means = np.mean(self.vectors*self.vectors,axis=1)

    def scores(self, query: np.ndarray) -> np.ndarray:
        q = query.reshape(-1)
        cq = q-q.mean()
        product = np.einsum('ij,j->i',self.vectors,q)/len(q)
        mse = np.maximum(0,self.squared_means+np.mean(q*q)-2*product)
        denominator = self.norms*np.linalg.norm(cq)
        correlation = np.divide(np.einsum('ij,j->i',self.centred,cq),denominator,
                                out=1-mse*FACE_PARAMETERS["constant_mse_scale"],where=denominator>1e-8)
        return np.clip(FACE_PARAMETERS["correlation_weight"]*correlation+
                       FACE_PARAMETERS["mse_weight"]*(1-np.minimum(1,mse*FACE_PARAMETERS["mse_scale"])),-1,1)

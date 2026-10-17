"""One analysis per visible countdown, rearmed after confirmed disappearance.

A new action can also follow the previous one without the countdown ever
disappearing between two probes: skipping a chi/pon offered on the upper
player's discard leads straight into the player's own draw, and the game just
restarts the countdown. A countdown that jumps back up, or a changed hand/draw
layout (confirmed by two consecutive reads), therefore starts a new action.
"""


def hand_action_marker(pixels, region):
    """Cheap public action context from tile bodies, without tile classification.

    Skipping a call changes a 13-tile response into 13 tiles plus a separate
    draw. Discarding removes that draw; a completed chi/pon changes the regular
    hand count. Ambiguous/hovered/animated layouts supply no marker.
    """
    from .hand import hand_layout
    layout = hand_layout(pixels, region)
    if layout["reasons"] or not layout["hand"]:
        return None
    return len(layout["hand"]), bool(layout["draw"])


def river_action_count(pixels, region):
    """Count evidenced own-river lattice cells, without reading any glyphs."""
    import numpy as np
    from .detector import tile_candidates,river_slots
    detected=tile_candidates(pixels,region)
    if detected['unresolved']:
        return None
    grid=region['grid']
    area=np.linalg.norm(grid['basis'][0])*np.linalg.norm(grid['basis'][1])
    boxes=[box for box in detected['boxes'] if box[2]*box[3] >= .5*area]
    slots=river_slots(boxes,grid)
    cells=[]
    columns=grid.get('columns',6)
    for (row,column),residual in slots:
        if (row<0 or column<0 or row>=grid.get('rows',4) or column>=columns
                or residual>grid.get('tolerance',10)):
            return None
        cells.append(row*columns+column)
    if sorted(cells) != list(range(len(cells))):
        return None
    return len(cells)


def same_marker(first,second):
    """An unread river never changes an otherwise identical hand context."""
    if isinstance(first,tuple) and isinstance(second,tuple) and len(first)==len(second)==3:
        return first[:2]==second[:2] and (first[2] is None or second[2] is None or first[2]==second[2])
    return first==second


class TimerEdge:
    # Seconds the countdown must rise by; one-second OCR jitter is ignored.
    RESET_MIN_RISE = 2

    def __init__(self, absence_checks=2, reset_checks=2):
        self.visible = False
        self.absent = 0
        self.absence_checks = absence_checks
        self.reset_checks = reset_checks
        self.lowest = None          # lowest countdown seen in the current action
        self.rising = 0
        self.marker = None
        self.pending_marker = None
        self.marker_checks = 0
        self.serial = 0

    def cancel(self):
        """A cancelled analysis must not consume the next visible action."""
        self.visible, self.absent = False, 0
        self.lowest, self.rising = None, 0
        self.marker, self.pending_marker, self.marker_checks = None, None, 0

    def bind(self, marker, seconds=None):
        """Bind a captured frame to its own current layout, after the delay."""
        if marker is not None:
            self.marker = marker
        self.pending_marker, self.marker_checks = None, 0
        if seconds is not None:
            self.lowest = seconds
        self.rising = 0

    def _start(self, seconds, marker):
        self.visible, self.absent = True, 0
        self.lowest, self.rising = seconds, 0
        self.marker, self.pending_marker, self.marker_checks = marker, None, 0
        self.serial += 1
        return "start"

    def observe(self, active, seconds=None, marker=None):
        if active:
            self.absent = 0
            if not self.visible:
                return self._start(seconds, marker)
            if marker is not None:
                if self.marker is None:
                    self.marker = marker
                elif not same_marker(marker,self.marker):
                    self.marker_checks = self.marker_checks+1 if same_marker(marker,self.pending_marker) else 1
                    self.pending_marker = marker
                    if self.marker_checks >= self.reset_checks:
                        return self._start(seconds, marker)
                else:
                    self.pending_marker, self.marker_checks = None, 0
                    if (isinstance(self.marker,tuple) and len(self.marker)==3 and self.marker[2] is None
                            and isinstance(marker,tuple) and len(marker)==3 and marker[2] is not None):
                        self.marker=marker
            else:
                self.pending_marker, self.marker_checks = None, 0
            if seconds is None:
                self.rising = 0
                return None
            if self.lowest is None or seconds <= self.lowest:
                self.lowest, self.rising = seconds, 0
            elif seconds >= self.lowest + self.RESET_MIN_RISE:
                self.rising += 1
                if self.rising >= self.reset_checks:
                    return self._start(seconds, marker if marker is not None else self.marker)
            else:
                self.rising = 0
        elif self.visible:
            self.absent += 1
            self.rising = 0
            self.pending_marker, self.marker_checks = None, 0
            if self.absent >= self.absence_checks:
                self.cancel()
                return "end"
        return None

---
license: agpl-3.0
library_name: pytorch
tags:
- mahjong
- riichi
- japanese-mahjong
- reinforcement-learning
- game-ai
- mortal
---

# mortal3-b40c192-t23032505 — Japanese mahjong bot

A [Mortal](https://github.com/Equim-chan/Mortal) v3 model for
four-player Japanese mahjong (riichi), trained on Tenhou Houou (鳳凰卓) game
records. It picks a discard or call from the current game state; it is a full
playing agent, not a scoring or evaluation model.

Of 6 checkpoints from this training lineage, this one measured
strongest in a 5-way head-to-head evaluation — see below.

## Architecture

| | |
|---|---|
| Backbone | ResNet, 40 blocks × 192 channels, with channel attention |
| Head | dueling DQN (separate value and advantage heads) |
| Parameters | 11,310,592 (10,773,713 backbone + 536,879 head) |
| Obs version | v3 (`libriichi` feature encoding) |
| Checkpoint | 43.4 MB, fp32 |

## Training

- **Data:** 191,132 Tenhou Houou game records in mjai format, one epoch, no augmentation.
  A superset of that corpus is published as [`hhim8826/tenhou-houou-mjai`](https://huggingface.co/datasets/hhim8826/tenhou-houou-mjai).
- **Steps:** 107,200 (batch size 2048)
- **Finished:** 2023-03-25

## Evaluation

Each challenger played **one seat against three copies of this model** using
`libriichi`'s `OneVsThree` arena under Tenhou hanchan rules (25,000 start, tobi,
西入). Every challenger saw the same deal seeds, and each seed was played four
times with the challenger in each seat. pt is scored +90/+45/0/-135
by placement. Intervals are 95% CIs clustered by seed.

**A challenger equal in strength to this model scores 0 pt by construction**, so
each interval containing 0 means "not distinguishable from this model".

| challenger | what it is | hanchan | avg rank ↓ | avg pt ↑ (95% CI) | result |
|---|---|---:|---:|---:|---|
| mortal_model/best.pth | sibling checkpoint, v3, 100k steps | 4,000 | 2.4943 | -0.24 (-2.33, +1.86) | not separable |
| mortal-120000.pth | sibling checkpoint, v3, 144.8k steps | 4,000 | 2.4992 | -0.55 (-2.62, +1.52) | not separable |
| mortal_model/mortal_online.pth | online-trained checkpoint, v3, 50.8k steps | 4,000 | 2.5445 | -1.67 (-3.93, +0.60) | tied on pt, worse on rank |
| best.pth | earlier v4 run, 16k steps | 1,000 | 2.5900 | -6.84 (-11.44, -2.24) | clearly weaker |
| mortal_model/mortal4/best.pth | earlier v4 run, 2.4k steps | 1,000 | 2.6940 | -15.03 (-20.07, -9.99) | clearly weaker |

No checkpoint beat it. Two are statistically indistinguishable from it, one is
worse on placement, and the two early v4 runs are clearly weaker.

### Harness control

Running this model against three copies of itself returned avg rank
2.5005 and +0.045 pt over 4,000 hanchan, with
placements [999, 997, 1007, 997] — the expected exact-symmetry result, which
confirms the arena has no seat or scoring bias.

## Limitations

- **Relative, not absolute.** These numbers compare checkpoints from one training
  run. The model has not been measured against the official Mortal releases,
  akochan, or human players, so its absolute strength is unknown.
- **Rule scope.** Trained and evaluated on four-player hanchan with red fives.
  Not evaluated on three-player (sanma) or tonpuusen.
- The `best_perf` field recorded inside the original training checkpoint
  ({'avg_rank': 2.416, 'avg_pt': 6.3}) refers to whatever opponent was current
  during training and is **not** comparable to the table above.

## Usage

Needs [Mortal](https://github.com/Equim-chan/Mortal) and its `libriichi`
extension. Point the config at this checkpoint:

```python
import torch
from model import Brain, DQN
from engine import MortalEngine
from libriichi.mjai import Bot

state = torch.load("mortal.pth", map_location="cpu", weights_only=True)
cfg = state["config"]
version = cfg["control"]["version"]

brain = Brain(version=version, **cfg["resnet"]).eval()
dqn = DQN(version=version).eval()
brain.load_state_dict(state["mortal"])
dqn.load_state_dict(state["current_dqn"])

engine = MortalEngine(
    brain, dqn, version=version, is_oracle=False,
    device=torch.device("cpu"), enable_amp=False,
    enable_quick_eval=True, enable_rule_based_agari_guard=True, name="mortal",
)
bot = Bot(engine, player_id=0)  # then feed it mjai events
```

This checkpoint holds only what inference needs — weights plus `version` and the
ResNet shape. Optimizer, scheduler and scaler state were removed, which is why
it is 43 MB rather than
142 MB. It cannot be used to resume training.

## Files

| file | what it is |
|---|---|
| `mortal.pth` | PyTorch weights (43 MB) — needs `torch` plus Mortal's `model.py` |
| `mortal.onnx` | The same network exported as one graph, `(obs, mask) -> q values`, with illegal actions already `-inf`. Runs on `onnxruntime` alone, no torch. |
| `grp.onnx` | The global reward predictor: one feature row per kyoku in, a distribution over the 24 finishing orders out. Used for placement probabilities, not for choosing moves. |
| `tenpai.onnx` | Reads the other three hands from one seat's own view. Does not play; see below. |
| `tenpai.onnx.json` | Its shapes, its held-out scores, and what it was trained on |
| `model_meta.json` | Full training hyper-parameters |

The ONNX graph was checked against the PyTorch model on real game observations:
maximum absolute difference on legal actions is around `1e-5` (float32 rounding),
and the chosen action never differs. Serving from ONNX rather than torch cut a
demo app's peak memory from 633 MB to 366 MB.

## Reading the other hands

`tenpai.onnx` answers a different question from the player: given what one seat
can see, is each of the other three tenpai, and which tiles are they waiting on.
It never chooses a move.

    obs (n, 934, 34) float32, the v3 observation from that seat
      -> tenpai   (n, 3)      each opponent, in turn order from the observer
         waits    (n, 3, 34)  is that opponent waiting on this tile
         any_wait (n, 34)     is any of the three
         furiten  (n, 3)

All four are probabilities; the sigmoids are inside the graph. 986k parameters,
4 MB, and a whole hanchan reads in about 2 s on two CPU threads.

Its labels are facts rather than judgements: four `PlayerState`s fed one log
know exactly who was tenpai and on what, so the hidden hands make the labels
and never enter the input. Trained on 1.6M Tenhou houou games (20,000 steps,
batch 1024), held out by whole row groups so that no game -- and no other
seat's view of one -- is on both sides.

It is measured on where the old reads run out, since the average over a table
is dominated by positions nobody could call either way. Among **non-suji tiles
against a riichi**, where suji and genbutsu have spent everything they know, a
rule baseline of riichi/genbutsu/suji/turn separates a safest tenth at 6.2%
waits from a riskiest at 9.0%; this model separates **0.13% from 17.2%**.
Against a **called hand**, which offers no discards to read suji off at all,
0.01% against 8.4% where the rules manage 0.02% against 4.0%.

The numbers are meant to be read as numbers. Over 4.2M held-out (seat, tile)
pairs, tiles it calls 13% are waits 12.7% of the time and tiles it calls 23%,
20.7%; above 30% it is somewhat over-confident on thin samples. The tenpai head
scores a Brier of 0.0427 against a base rate of 0.147.

Waiting on a tile is not the same as being able to win on it -- a hand with no
yaku cannot ron, and a furiten one may not -- so read `any_wait` as "somebody is
waiting on this", which is what it was trained on.

## Provenance

Built with [Mortal](https://github.com/Equim-chan/Mortal) (AGPL-3.0-or-later).
Training data came from [Tenhou](https://tenhou.net/) game records; Tenhou
publishes terms governing 牌譜 use on its
[log listing page](https://tenhou.net/sc/raw/?old). Review those terms for your
use case.

SHA-256 of `mortal.pth`: `ec679c428766af25fe2c0520938fffc7047f6c8016e3930f3b6c61fdabe48559`

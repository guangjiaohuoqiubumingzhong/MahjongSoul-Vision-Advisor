//! Local Majan extension: import a visible single-frame decision state.
//! No event history or hidden opponents' hands are reconstructed.
use super::{ActionCandidate, PlayerState};
use super::item::{KawaItem, Sutehai};
use super::update::MoveType;
use crate::mjai::Event;
use crate::tile::Tile;
use crate::{must_tile, tu8};
use anyhow::{Result, bail, ensure};
use pyo3::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RiverTile {
    tile: Tile,
    #[serde(default)]
    riichi: bool,
    // Unknown tedashi has no positive feature, rather than invented history.
    #[serde(default)]
    tedashi: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    player_id: u8, oya: u8, bakaze: Tile, kyoku: u8,
    honba: u8, kyotaku: u8, scores: [i32; 4], remaining: u8,
    hand: Vec<Tile>, draw: Option<Tile>, dora: Vec<Tile>,
    rivers: [Vec<RiverTile>; 4], melds: [Vec<Event>; 4],
    riichi: [bool; 4], phase: String,
    #[serde(default)]
    riichi_declared: Option<[bool; 4]>,
    target: Option<u8>, reaction_tile: Option<Tile>, last_call: Option<Event>,
}

#[pymethods]
impl PlayerState {
    /// Build a fresh state from visible snapshot data. Same-cycle furiten,
    /// ippatsu and rinshan history are intentionally not reconstructed.
    #[staticmethod]
    pub fn from_snapshot(snapshot_json: &str) -> Result<Self> {
        let snap: Snapshot = serde_json::from_str(snapshot_json)?;
        ensure!(snap.player_id < 4 && snap.oya < 4, "invalid snapshot seat");
        ensure!(matches!(snap.bakaze.as_u8(), 27 | 28) && (1..=4).contains(&snap.kyoku), "unsupported round");
        ensure!(snap.remaining <= 69 && (1..=5).contains(&snap.dora.len()), "invalid wall/dora count");
        ensure!(snap.melds.iter().all(|m| m.len() <= 4) && snap.rivers.iter().all(|r| r.len() <= 24), "snapshot capacity exceeded");
        let own = snap.player_id as usize;
        let n = snap.melds[own].len();
        let count = if snap.phase == "call_discard" {14 - 3*n} else {13 - 3*n};
        ensure!(snap.hand.len() == count, "snapshot concealed count does not match phase/melds");
        let mut visible = snap.hand.clone();
        visible.extend(snap.draw);
        visible.extend(snap.dora.iter().copied());
        for river in &snap.rivers { visible.extend(river.iter().map(|r| r.tile)); }
        for (actor, melds) in snap.melds.iter().enumerate() {
            for meld in melds {
                ensure!(meld.actor() == Some(actor as u8), "meld actor mismatch");
                let (tiles, target) = match *meld {
                    Event::Chi {pai, consumed, target, ..} => {
                        let mut ids = [pai.deaka().as_u8(), consumed[0].deaka().as_u8(), consumed[1].deaka().as_u8()];
                        ids.sort_unstable();
                        ensure!(ids[2] < 27 && ids[0]/9 == ids[2]/9 && ids[1] == ids[0]+1 && ids[2] == ids[0]+2 && (target+1)%4 == actor as u8, "invalid chi");
                        (vec![pai, consumed[0], consumed[1]], Some(target))
                    },
                    Event::Pon {pai, consumed, target, ..} => (vec![pai, consumed[0], consumed[1]], Some(target)),
                    Event::Daiminkan {pai, consumed, target, ..} => (vec![pai, consumed[0], consumed[1], consumed[2]], Some(target)),
                    Event::Kakan {pai, consumed, ..} => (vec![pai, consumed[0], consumed[1], consumed[2]], None),
                    Event::Ankan {consumed, ..} => (consumed.to_vec(), None),
                    _ => bail!("not a meld"),
                };
                if !matches!(meld, Event::Chi {..}) {
                    ensure!(tiles.iter().all(|t| t.deaka() == tiles[0].deaka()), "invalid pon/kan");
                }
                if let Some(t) = target { ensure!(t < 4 && t != actor as u8, "invalid meld source"); }
                visible.extend(tiles);
            }
        }
        let mut counts = [0; 34]; let mut red = [0; 3]; let mut normal = [0; 34];
        for tile in visible {
            ensure!(!tile.is_unknown(), "unknown snapshot tile");
            let i = tile.deaka().as_usize(); counts[i] += 1;
            if tile.is_aka() {red[tile.as_usize()-34] += 1;} else {normal[i] += 1;}
        }
        ensure!(counts.iter().all(|&c| c <= 4) && red.iter().all(|&c| c <= 1) && [4,13,22].iter().all(|&i| normal[i] <= 3), "physical tile conflict");

        let mut state = PlayerState::new(snap.player_id);
        state.bakaze = snap.bakaze; state.kyoku = snap.kyoku-1;
        state.honba = snap.honba; state.kyotaku = snap.kyotaku;
        state.oya = state.rel(snap.oya) as u8;
        state.jikaze = must_tile!(tu8!(E) + (4-state.oya)%4);
        state.is_all_last = snap.bakaze.as_u8() == tu8!(S) && snap.kyoku == 4;
        state.scores = snap.scores; state.scores.rotate_left(own); state.update_rank();
        state.tehai_len_div3 = (4-n) as u8;
        state.is_menzen = snap.melds[own].iter().all(|m| matches!(m, Event::Ankan {..}));
        ensure!(!snap.riichi[own] || state.is_menzen, "open hand cannot be in riichi");
        state.can_w_riichi = snap.rivers[own].is_empty() && snap.melds.iter().all(|m| m.is_empty());
        state.at_turn = snap.rivers[own].len() as u8;
        for &dora in &snap.dora { state.add_dora_indicator(dora)?; }
        for &tile in &snap.hand { state.witness_tile(tile)?; state.move_tile(tile, MoveType::Tsumo)?; }
        for (actor, melds) in snap.melds.iter().enumerate() {
            let rel = state.rel(actor as u8);
            for meld in melds {
                let (tiles, kind) = match *meld {
                    Event::Chi {pai, consumed, ..} => (vec![pai, consumed[0], consumed[1]], 0),
                    Event::Pon {pai, consumed, ..} => (vec![pai, consumed[0], consumed[1]], 1),
                    Event::Daiminkan {pai, consumed, ..} | Event::Kakan {pai, consumed, ..} => (vec![pai, consumed[0], consumed[1], consumed[2]], 2),
                    Event::Ankan {consumed, ..} => (consumed.to_vec(), 3),
                    _ => unreachable!(),
                };
                if kind == 3 {state.ankan_overview[rel].push(tiles[0].deaka());}
                else {state.fuuro_overview[rel].push(tiles.iter().copied().collect());}
                if kind >= 2 {state.kans_on_board += 1;}
                if rel == 0 {
                    let tid = tiles.iter().map(|t| t.deaka().as_u8()).min().unwrap();
                    match kind {0 => state.chis.push(tid), 1 => state.pons.push(tid), 2 => state.minkans.push(tid), _ => state.ankans.push(tid)}
                }
                for tile in tiles {state.witness_tile(tile)?; state.update_doras_owned(rel, tile);}
            }
        }
        ensure!(state.kans_on_board <= 4 && snap.dora.len() <= state.kans_on_board as usize+1, "dora/kan mismatch");
        state.pad_kawa_at_start();
        for (actor, river) in snap.rivers.iter().enumerate() {
            let rel = state.rel(actor as u8);
            let reaction_last = snap.phase == "reaction" && snap.target == Some(actor as u8);
            for (index, r) in river.iter().enumerate() {
                if reaction_last && index+1 == river.len() {continue;}
                state.witness_tile(r.tile)?;
                let sutehai = Sutehai {tile:r.tile, is_dora:state.dora_factor[r.tile.deaka().as_usize()] > 0, is_tedashi:r.tedashi, is_riichi:r.riichi};
                state.kawa[rel].push(Some(KawaItem {chi_pon:None, kan:Default::default(), sutehai}));
                state.kawa_overview[rel].push(r.tile);
                if r.tedashi {state.last_tedashis[rel] = Some(sutehai);}
                if r.riichi {state.riichi_sutehais[rel] = Some(sutehai);}
                if rel == 0 {state.discarded_tiles[r.tile.deaka().as_usize()] = true;}
            }
            state.riichi_declared[rel] = snap.riichi[actor] || snap.riichi_declared.map_or(false, |r| r[actor]);
            state.riichi_accepted[rel] = snap.riichi[actor];
        }
        state.tiles_left = snap.remaining;
        state.update_shanten();
        match snap.phase.as_str() {
            "draw" => {
                ensure!(snap.target.is_none() && snap.last_call.is_none(), "mixed snapshot phase");
                state.update_waits_and_furiten();
                state.tiles_left += 1;
                let tile = snap.draw.ok_or_else(|| anyhow::anyhow!("missing draw tile"))?;
                state.update(&Event::Tsumo {actor:snap.player_id, pai:tile})?;
                if state.riichi_declared[0] {state.last_cans.can_riichi = false;}
            },
            "reaction" => {
                ensure!(snap.draw.is_none() && snap.last_call.is_none(), "mixed snapshot phase");
                let target = snap.target.ok_or_else(|| anyhow::anyhow!("missing reaction source"))?;
                ensure!(target < 4 && target != snap.player_id, "invalid reaction source");
                let r = snap.rivers[target as usize].last().ok_or_else(|| anyhow::anyhow!("missing reaction discard"))?;
                ensure!(Some(r.tile) == snap.reaction_tile, "reaction tile not river's latest tile");
                state.update_waits_and_furiten();
                state.update(&Event::Dahai {actor:target, pai:r.tile, tsumogiri:!r.tedashi})?;
                // update(dahai) would mark a new declaration based on accepted
                // riichi, so keep the supplied visible declaration marker.
                let rel = state.rel(target);
                if r.riichi {
                    let sutehai = &mut state.kawa[rel].last_mut().unwrap().as_mut().unwrap().sutehai;
                    sutehai.is_riichi = true;
                    state.riichi_sutehais[rel] = Some(*sutehai);
                }
                state.to_mark_same_cycle_furiten = None;
            },
            "call_discard" => {
                ensure!(snap.draw.is_none() && !snap.riichi[own], "invalid call discard phase");
                let call = snap.last_call.ok_or_else(|| anyhow::anyhow!("missing last call for kuikae"))?;
                ensure!(snap.melds[own].contains(&call), "last call not in own visible melds");
                state.last_cans = ActionCandidate {can_discard:true, target_actor:snap.player_id, ..Default::default()};
                let pai = match call {
                    Event::Chi {pai, consumed, ..} => {
                        let min = consumed.iter().map(|t| t.deaka().as_usize()).min().unwrap();
                        let max = consumed.iter().map(|t| t.deaka().as_usize()).max().unwrap();
                        let tid = pai.deaka().as_usize();
                        if tid < min && max%9 < 8 {state.forbidden_tiles[max+1] = true;}
                        if tid > max && !min.is_multiple_of(9) {state.forbidden_tiles[min-1] = true;}
                        pai
                    },
                    Event::Pon {pai, ..} => pai,
                    _ => bail!("last call must be chi or pon"),
                };
                state.forbidden_tiles[pai.deaka().as_usize()] = true;
                state.update_shanten_discards();
            },
            _ => bail!("unsupported snapshot phase"),
        }
        Ok(state)
    }
}

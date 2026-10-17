use std::{env, fs, io::{self, Write}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: majan-convert TENHOU6_OR_MAJSOUL_EXPORT.json")?;
    let text = fs::read_to_string(path)?;
    let raw: serde_json::Value = serde_json::from_str(&text)?;
    // The Mahjong Soul export path documented by mjai-reviewer emits Tenhou6.
    // Verify the supported rule scope before entering the converter.
    let rule = raw.get("rule").ok_or("missing rule metadata")?;
    let disp = rule.get("disp").and_then(|s| s.as_str()).ok_or("missing rule.disp")?;
    if !disp.contains('南') && !disp.contains("South") {
        return Err("only four-player red hanchan exports are supported".into());
    }
    for key in ["aka51", "aka52", "aka53"] {
        if rule.get(key).and_then(|v| v.as_u64()).or_else(|| rule.get("aka").and_then(|v| v.as_u64())) != Some(1) {
            return Err("exactly one red five per suit is required".into());
        }
    }
    let log = convlog::tenhou::Log::from_json_str(&text)?;
    let events = convlog::tenhou_to_mjai(&log)?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    for event in events {
        serde_json::to_writer(&mut out, &event)?;
        writeln!(out)?;
    }
    Ok(())
}

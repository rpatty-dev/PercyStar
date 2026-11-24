use crate::audio_player::LoadedWavTrack;
use rusty_opus::OpusDecoder;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

// one minute up front and two hours of stereo at 48 khz at most per stream
const MAX_RESERVE_SAMPLES: usize = 48_000 * 2 * 60;
const MAX_STREAM_SAMPLES: usize = 48_000 * 2 * 7200;

#[derive(Debug, Clone)]
pub struct AwcDecodeResult {
    pub wavs: Vec<(u32, PathBuf)>,
    pub summary: String,
    pub details: Vec<String>,
}

pub struct AwcMemoryResult {
    pub tracks: Vec<LoadedWavTrack>,
    pub summary: String,
    pub details: Vec<String>,
}

fn fmt_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ")
}

fn fmt_ascii(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| if (0x20..=0x7E).contains(&b) { b as char } else { '.' })
        .collect()
}

fn push_hex_dump(out: &mut Vec<String>, label: &str, data: &[u8], base_off: usize, max_len: usize) {
    let slice = &data[..data.len().min(max_len)];
    out.push(format!("{label} ({} bytes shown of {}):", slice.len(), data.len()));
    for (row, chunk) in slice.chunks(16).enumerate() {
        let addr = base_off + row * 16;
        let hex = fmt_hex(chunk);
        out.push(format!("  {addr:04X}: {hex:<47}  |{}|", fmt_ascii(chunk)));
    }
}

#[inline]
fn ru16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn ru32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn ru64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5], b[o + 6], b[o + 7]])
}

// pure rust adat awc to wav decodr zero python or ffmpeg needed
#[cfg_attr(not(test), allow(dead_code))]
pub fn decode_awc_bytes_to_wavs(data: &[u8], out_dir: &Path, stem: &str) -> Vec<(u32, PathBuf)> {
    decode_awc_bytes_to_wavs_detailed(data, out_dir, stem).wavs
}

enum ParsedAdatHeader {
    Err { summary: String, details: Vec<String> },
    Ok {
        flags: u32,
        sc: usize,
        raw_hdr_sz: usize,
        data_chunks: Vec<(usize, usize, u32)>,
        skipped_chunks: Vec<String>,
    },
}

fn parse_adat_header(data: &[u8]) -> ParsedAdatHeader {
    if data.len() < 16 {
        return ParsedAdatHeader::Err {
            summary: format!("Payload too small ({} bytes, need >= 16 for ADAT header)", data.len()),
            details: vec![format!("First bytes (hex): {}", fmt_hex(&data[..data.len().min(16)]))],
        };
    }

    if &data[..4] != b"ADAT" {
        let ascii4 = fmt_ascii(&data[..4]);
        let hint = if &data[..4] == b"TADA" {
            " (big-endian 'TADA' ADAT header; byte-swapping not implemented)"
        } else if &data[..4] == b"RSC9" || &data[..4] == b"RBF0" {
            " (container is RSC9/RBF0, not raw ADAT)"
        } else if data[0] == 0x78 && matches!(data[1], 0x01 | 0x5E | 0x9C | 0xDA) {
            " (looks like zlib-compressed data)"
        } else if data.len() >= 4 && data[..4] == [0x28, 0xB5, 0x2F, 0xFD] {
            " (looks like zstd-compressed data)"
        } else {
            ""
        };
        let mut details = Vec::new();
        push_hex_dump(&mut details, "Payload head", data, 0, 128);
        return ParsedAdatHeader::Err {
            summary: format!("Not an ADAT header: got '{}' ({}) in {} bytes{}", ascii4, fmt_hex(&data[..4]), data.len(), hint),
            details,
        };
    }

    let flags = ru32(data, 4);
    let sc = ru32(data, 8) as usize;
    let raw_hdr_sz = ru32(data, 12) as usize;
    let hdr_sz = raw_hdr_sz.min(data.len());
    if sc == 0 || sc > 65536 || 16 + sc * 2 > data.len() {
        let mut details = Vec::new();
        push_hex_dump(&mut details, "ADAT head", data, 0, 128);
        return ParsedAdatHeader::Err {
            summary: format!("Invalid ADAT stream count sc={sc} (flags=0x{flags:08X}, hdr_sz={raw_hdr_sz}, payload={} bytes)", data.len()),
            details,
        };
    }

    struct CandidateScan {
        data_chunks: Vec<(usize, usize, u32)>,
        format_rates: Vec<u32>,
        total_entries: usize,
        tag_counts: BTreeMap<u8, usize>,
        sample_entries: Vec<String>,
        skipped_chunks: Vec<String>,
        bogus: usize,
    }

    let scan_forward = |start_off: usize| -> CandidateScan {
        let mut off = start_off;
        let mut data_chunks: Vec<(usize, usize, u32)> = Vec::new();
        let mut format_rates: Vec<u32> = Vec::new();
        let mut total_entries = 0usize;
        let mut tag_counts: BTreeMap<u8, usize> = BTreeMap::new();
        let mut skipped_chunks: Vec<String> = Vec::new();
        let mut sample_entries: Vec<String> = Vec::new();
        let mut bogus = 0usize;
        while off + 8 <= hdr_sz {
            let val = ru64(data, off);
            let tag = (val >> 56) as u8;
            let sz = ((val >> 28) & 0x0FFF_FFFF) as usize;
            let coff = (val & 0x0FFF_FFFF) as usize;
            if tag == 0 && coff == 0 {
                break;
            }
            total_entries += 1;
            *tag_counts.entry(tag).or_insert(0) += 1;
            if sample_entries.len() < 8 {
                sample_entries.push(format!("tag=0x{tag:02X} off=0x{coff:X} sz={sz}"));
            }
            if tag == 0x76 && sz >= 10 && coff + 10 <= data.len() {
                let sr = ru16(data, coff + 8) as u32;
                if (8_000..=192_000).contains(&sr) {
                    format_rates.push(sr);
                    if let Some(last) = data_chunks.last_mut() {
                        if last.2 == 0 {
                            last.2 = sr;
                        }
                    }
                }
            } else if tag == 0x55 {
                if sz == 0 {
                    bogus += 1;
                    skipped_chunks.push(format!("0x55 chunk at off=0x{coff:X} has sz=0"));
                } else if coff < 0x20 {
                    bogus += 1;
                    skipped_chunks.push(format!("0x55 chunk has invalid off=0x{coff:X} (< 0x20), sz={sz}"));
                } else if coff >= data.len() {
                    bogus += 1;
                    skipped_chunks.push(format!("0x55 chunk out of bounds: off=0x{coff:X} >= len={}", data.len()));
                } else if coff + sz > data.len() {
                    let avail = data.len() - coff;
                    skipped_chunks.push(format!("0x55 chunk truncated in RPF (off=0x{coff:X}, header_sz={sz}, clamped to {avail} bytes)"));
                    data_chunks.push((coff, avail, 0));
                } else {
                    data_chunks.push((coff, sz, 0));
                }
            } else if !matches!(tag, 0x48 | 0xA3 | 0xBD | 0xFA | 0xFF) || coff >= data.len() {
                bogus += 1;
            }
            off += 8;
        }
        CandidateScan { data_chunks, format_rates, total_entries, tag_counts, sample_entries, skipped_chunks, bogus }
    };

    let mut candidates: Vec<(bool, CandidateScan)> = Vec::new();
    let off_u32 = 16 + sc * 4;
    let off_u16 = 16 + sc * 2;
    let off_u16_aligned = (off_u16 + 7) & !7;
    if off_u32 + 8 <= hdr_sz {
        candidates.push((true, scan_forward(off_u32)));
    }
    if off_u16 != off_u32 && off_u16 + 8 <= hdr_sz {
        candidates.push((false, scan_forward(off_u16)));
    }
    if off_u16_aligned != off_u16 && off_u16_aligned != off_u32 && off_u16_aligned + 8 <= hdr_sz {
        candidates.push((false, scan_forward(off_u16_aligned)));
    }

    let mut best = candidates
        .into_iter()
        .max_by_key(|(is_u32, c)| {
            let clean = c.data_chunks.iter().filter(|&&(coff, _, _)| coff >= hdr_sz).count() as isize;
            let exact_sc = if c.data_chunks.len() == sc { 100isize } else { 0 };
            (exact_sc + clean - (c.bogus as isize) * 4, c.data_chunks.len(), *is_u32, c.format_rates.len())
        })
        .map(|(_, c)| c);

    let needs_backward_recovery = best.as_ref().map_or(true, |candidate| {
        !candidate.data_chunks.is_empty() && candidate.data_chunks.len() != sc
    });
    if hdr_sz >= 0x18 && needs_backward_recovery {
        let mut data_chunks = Vec::new();
        let mut format_rates = Vec::new();
        let mut pending_rate = 0u32;
        let mut back = hdr_sz;
        while back >= 0x18 {
            let entry_off = back - 8;
            let val = ru64(data, entry_off);
            let tag = (val >> 56) as u8;
            let sz = ((val >> 28) & 0x0FFF_FFFF) as usize;
            let coff = (val & 0x0FFF_FFFF) as usize;
            if tag == 0x55 {
                if sz > 0 && coff >= hdr_sz && coff < data.len() {
                    data_chunks.push((coff, sz.min(data.len() - coff), pending_rate));
                }
                pending_rate = 0;
            } else if tag == 0x76 {
                pending_rate = 0;
                if sz >= 10 && coff + 10 <= data.len() {
                    let sr = ru16(data, coff + 8) as u32;
                    if (8_000..=192_000).contains(&sr) {
                        format_rates.push(sr);
                        pending_rate = sr;
                    }
                }
            } else if !matches!(tag, 0x48 | 0x55 | 0x76 | 0xA3 | 0xBD | 0xFA | 0xFF) && coff > data.len() {
                break;
            }
            back -= 8;
        }
        data_chunks.reverse();
        let current_is_exact = best.as_ref().is_some_and(|candidate| candidate.data_chunks.len() == sc);
        let recovery_is_exact = data_chunks.len() == sc && !current_is_exact;
        let recovered = CandidateScan {
            data_chunks,
            // Recovered sample rates are attached directly to their 0x55
            // chunks above. Do not reuse a forward candidate's unrelated
            // rates for chunks that have no paired 0x76 record. Keep the
            // legacy rate list only for the old no-candidate fallback.
            format_rates: if recovery_is_exact { Vec::new() } else { format_rates },
            total_entries: 0,
            tag_counts: BTreeMap::new(),
            sample_entries: Vec::new(),
            skipped_chunks: Vec::new(),
            bogus: 0,
        };
        if recovery_is_exact || best.is_none() {
            best = Some(recovered);
        }
    }

    let Some(chosen) = best else {
        let mut details = vec![format!("flags=0x{flags:08X}, payload={} bytes", data.len())];
        push_hex_dump(&mut details, "ADAT header", data, 0, hdr_sz.max(64).min(512));
        return ParsedAdatHeader::Err {
            summary: format!("ADAT header too small for chunk table (table_off=0x{off_u32:X}, hdr_sz=0x{hdr_sz:X}, raw_hdr_sz=0x{raw_hdr_sz:X}, sc={sc})"),
            details,
        };
    };

    let mut data_chunks = chosen.data_chunks;
    let format_rates = chosen.format_rates;
    let total_entries = chosen.total_entries;
    let tag_counts = chosen.tag_counts;
    let sample_entries = chosen.sample_entries;
    let skipped_chunks = chosen.skipped_chunks;

    if data_chunks.is_empty() && hdr_sz >= 0x18 {
        let mut back = hdr_sz;
        while back >= 0x18 {
            let entry_off = back - 8;
            let val = ru64(data, entry_off);
            let tag = (val >> 56) as u8;
            let sz = ((val >> 28) & 0x0FFF_FFFF) as usize;
            let coff = (val & 0x0FFF_FFFF) as usize;
            if tag == 0x55 && sz > 0 && coff >= hdr_sz && coff < data.len() {
                data_chunks.push((coff, sz.min(data.len() - coff), 0));
            } else if !matches!(tag, 0x48 | 0x55 | 0x76 | 0xA3 | 0xBD | 0xFA | 0xFF) && coff > data.len() {
                break;
            }
            back -= 8;
        }
        data_chunks.reverse();
    }

    let default_sr = format_rates.first().copied().unwrap_or(48_000);
    for (idx, chunk) in data_chunks.iter_mut().enumerate() {
        if chunk.2 == 0 {
            chunk.2 = format_rates.get(idx).copied().unwrap_or(default_sr);
        }
    }

    if data_chunks.is_empty() {
        let tags_str = if tag_counts.is_empty() {
            "none".to_string()
        } else {
            tag_counts.iter().map(|(t, c)| format!("0x{t:02X}x{c}")).collect::<Vec<_>>().join(", ")
        };
        let stream_ids: Vec<String> = (0..sc.min(16))
            .filter(|&i| 16 + i * 4 + 4 <= data.len())
            .map(|i| format!("0x{:08X}", ru32(data, 16 + i * 4)))
            .collect();
        let mut details = vec![
            format!(
                "ADAT header: ver=0x{:04X}, flags=0x{:04X} (raw=0x{flags:08X}), streams(sc)={sc}, hdr_sz=0x{raw_hdr_sz:X} ({raw_hdr_sz}), payload={} bytes",
                flags & 0xFFFF,
                flags >> 16,
                data.len()
            ),
            format!("Stream table @ 0x10 (u32[{}]): [{}]", stream_ids.len(), stream_ids.join(", ")),
            format!(
                "Chunk table @ 0x{:X}..0x{hdr_sz:X} ({} bytes, rem_mod8={}): entries={total_entries}, seen tags: [{tags_str}]",
                16 + sc * 4,
                hdr_sz.saturating_sub(16 + sc * 4),
                hdr_sz.saturating_sub(16 + sc * 4) % 8
            ),
        ];
        if !sample_entries.is_empty() {
            details.push(format!("First entries: {}", sample_entries.join(" | ")));
        }
        details.extend(skipped_chunks);

        let mut alt_0x55 = Vec::new();
        for probe in 0x10..hdr_sz.saturating_sub(7) {
            let val = ru64(data, probe);
            let tag = (val >> 56) as u8;
            let sz = ((val >> 28) & 0x0FFF_FFFF) as usize;
            let coff = (val & 0x0FFF_FFFF) as usize;
            if tag == 0x55 && sz > 0 && coff >= hdr_sz && coff + sz <= data.len() {
                alt_0x55.push(format!("@0x{probe:02X}(off=0x{coff:X},sz={sz})"));
                if alt_0x55.len() >= 8 {
                    break;
                }
            }
        }
        if !alt_0x55.is_empty() {
            details.push(format!("Unaligned 0x55 candidates in header: {}", alt_0x55.join(", ")));
        }

        push_hex_dump(&mut details, "ADAT header hex", data, 0, hdr_sz.min(512));
        if hdr_sz < data.len() {
            push_hex_dump(&mut details, "Post-header bytes @ hdr_sz", &data[hdr_sz..], hdr_sz, 96);
        }

        return ParsedAdatHeader::Err {
            summary: format!("No valid 0x55 Opus data chunks found (seen tags: [{tags_str}])"),
            details,
        };
    }

    ParsedAdatHeader::Ok {
        flags,
        sc,
        raw_hdr_sz,
        data_chunks,
        skipped_chunks,
    }
}

// checks if 0x55 data chunk starts with 512kb superblock directory (0xffffffff + u32 block count per stream, 24 bytes each)
fn probe_superblock_streams(blob: &[u8]) -> Option<usize> {
    if blob.len() < 4096 {
        return None;
    }
    let mut n = 0usize;
    let mut total_blks = 0usize;
    while (n + 1) * 24 <= blob.len().min(2048) {
        if ru32(blob, n * 24) != 0xFFFF_FFFF {
            break;
        }
        let cnt = ru32(blob, n * 24 + 4) as usize;
        if cnt == 0 || cnt > 255 {
            break;
        }
        total_blks += cnt;
        n += 1;
    }
    if n >= 1 && total_blks <= 255 {
        Some(n)
    } else {
        None
    }
}

fn expand_superblock_jobs(data: &[u8], data_chunks: &[(usize, usize, u32)]) -> Vec<(usize, usize, u32, Option<(usize, usize)>)> {
    let mut jobs = Vec::with_capacity(data_chunks.len());
    for &(coff, csz, sr) in data_chunks {
        if let Some(n_sub) = probe_superblock_streams(&data[coff..coff + csz]) {
            for sub_idx in 0..n_sub {
                jobs.push((coff, csz, sr, Some((sub_idx, n_sub))));
            }
        } else {
            jobs.push((coff, csz, sr, None));
        }
    }
    jobs
}

pub fn decode_single_awc_chunk(
    data: &[u8],
    coff: usize,
    csz: usize,
    sub_stream: Option<(usize, usize)>,
) -> Result<(u16, Vec<i16>), String> {
    if coff >= data.len() || coff.saturating_add(csz) > data.len() {
        return Err(format!("Chunk out of bounds (off=0x{coff:X}, sz={csz}, len={})", data.len()));
    }
    let mut pcm_f32 = vec![0.0f32; 5760 * 2];
    let mut samples_i16 = Vec::new();
    let ch = decode_opus_chunk_into(0, data, coff, csz, sub_stream, &mut pcm_f32, &mut samples_i16)
        .map_err(|notes| notes.join("\n"))?;
    Ok((ch, samples_i16))
}

fn collect_2kb_packets<'a>(blob: &'a [u8], packets: &mut Vec<&'a [u8]>) {
    for blk in blob.chunks(2048) {
        let mut p = 0usize;
        while p + 2 <= blk.len() {
            let plen = ru16(blk, p) as usize;
            if plen == 0 || p + 2 + plen > blk.len() {
                break;
            }
            packets.push(&blk[p + 2..p + 2 + plen]);
            p += 2 + plen;
        }
    }
}

fn decode_opus_chunk_into(
    idx: usize,
    data: &[u8],
    coff: usize,
    csz: usize,
    sub_stream: Option<(usize, usize)>,
    pcm_f32: &mut Vec<f32>,
    samples_i16: &mut Vec<i16>,
) -> Result<u16, Vec<String>> {
    samples_i16.clear();
    let blob = &data[coff..coff + csz];
    let mut packets: Vec<&[u8]> = Vec::new();

    if let Some((sub_idx, n_sub)) = sub_stream {
        for sb in blob.chunks(0x80000) {
            if sb.len() < n_sub * 24 {
                break;
            }
            let mut counts = Vec::with_capacity(n_sub);
            let mut valid_dir = true;
            let mut total_blks = 0usize;
            for s in 0..n_sub {
                if ru32(sb, s * 24) != 0xFFFF_FFFF {
                    valid_dir = false;
                    break;
                }
                let c = ru32(sb, s * 24 + 4) as usize;
                total_blks += c;
                counts.push(c);
            }
            if !valid_dir || total_blks == 0 || total_blks > 256 {
                continue;
            }
            let dir_bytes = n_sub * 24 + total_blks * 4;
            let dir_blks = dir_bytes.div_ceil(2048).max(1);
            let start_blk: usize = dir_blks + counts[..sub_idx].iter().sum::<usize>();
            let end_blk = start_blk + counts[sub_idx];
            let byte_start = start_blk * 2048;
            let byte_end = (end_blk * 2048).min(sb.len());
            if byte_start < byte_end {
                collect_2kb_packets(&sb[byte_start..byte_end], &mut packets);
            }
        }
    } else {
        collect_2kb_packets(blob, &mut packets);
    }

    if packets.is_empty() {
        let first_u16 = if blob.len() >= 2 { format!("0x{:04X} ({})", ru16(blob, 0), ru16(blob, 0)) } else { "n/a".into() };
        let mut notes = vec![format!("Chunk #{idx} (off=0x{coff:X}, sz={csz}): 0 Opus packets found in 2KB blocks (first_u16={first_u16})")];
        push_hex_dump(&mut notes, &format!("Chunk #{idx} head"), blob, coff, 64);
        return Err(notes);
    }

    let toc = packets.iter().max_by_key(|p| p.len()).map(|p| p[0]).unwrap_or(packets[0][0]);
    let ch: usize = if (toc & 0x04) != 0 { 2 } else { 1 };
    let mut dec = OpusDecoder::new(48_000, ch)
        .map_err(|err| vec![format!("Chunk #{idx} (ch={ch}): OpusDecoder::new(48000, {ch}) failed: {err:?}")])?;
    if pcm_f32.len() < 5760 * ch {
        pcm_f32.resize(5760 * ch, 0.0);
    }
    // the packet count comes from the file so the reservation is only a bounded hint
    let need = packets.len().saturating_mul(960 * ch).min(MAX_RESERVE_SAMPLES);
    if samples_i16.capacity() < need {
        samples_i16.reserve(need - samples_i16.capacity());
    }
    let mut ok_pkts = 0usize;
    let mut err_pkts = 0usize;
    let mut first_err: Option<String> = None;

    for pkt in &packets {
        // a chunk of minimal packets would otherwise decode to tens of gigabytes
        if samples_i16.len() >= MAX_STREAM_SAMPLES {
            break;
        }
        match dec.decode(pkt, 5760, &mut pcm_f32[..5760 * ch]) {
            Ok(n_per_ch) => {
                ok_pkts += 1;
                for &s in &pcm_f32[..n_per_ch * ch] {
                    samples_i16.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
                }
            }
            Err(err) => {
                err_pkts += 1;
                if first_err.is_none() {
                    first_err = Some(format!("{err:?}"));
                }
            }
        }
    }
    if samples_i16.is_empty() {
        let mut notes = vec![format!(
            "Chunk #{idx} (off=0x{coff:X}, sz={csz}, ch={ch}, pkts={}): 0 PCM samples ({ok_pkts} ok, {err_pkts} err, toc=0x{:02X}, err={})",
            packets.len(),
            packets[0][0],
            first_err.as_deref().unwrap_or("none")
        )];
        push_hex_dump(&mut notes, &format!("Chunk #{idx} head"), blob, coff, 64);
        return Err(notes);
    }
    Ok(ch as u16)
}

pub fn decode_awc_bytes_in_memory<F>(data: &[u8], stem: &str, on_progress: F) -> AwcMemoryResult
where
    F: Fn(usize, usize) + Sync,
{
    let (flags, sc, raw_hdr_sz, data_chunks, skipped_chunks) = match parse_adat_header(data) {
        ParsedAdatHeader::Err { summary, details } => {
            return AwcMemoryResult { tracks: Vec::new(), summary, details };
        }
        ParsedAdatHeader::Ok { flags, sc, raw_hdr_sz, data_chunks, skipped_chunks } => {
            (flags, sc, raw_hdr_sz, data_chunks, skipped_chunks)
        }
    };

    let jobs = expand_superblock_jobs(data, &data_chunks);
    let total = jobs.len();
    let next_idx = AtomicUsize::new(0);
    let done_cnt = AtomicUsize::new(0);
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(total)
        .max(1);

    let results: Vec<(Vec<LoadedWavTrack>, Vec<String>)> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            handles.push(s.spawn(|| {
                let mut local_tracks = Vec::new();
                let mut local_notes = Vec::new();
                let mut pcm_f32 = vec![0.0f32; 5760 * 2];
                let mut samples_i16 = Vec::new();
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }
                    let (coff, csz, sample_rate, sub_stream) = jobs[idx];
                    match decode_opus_chunk_into(idx, data, coff, csz, sub_stream, &mut pcm_f32, &mut samples_i16) {
                        Ok(ch) => {
                            let name = format!("{stem}_s{idx:02}.wav");
                            local_tracks.push(LoadedWavTrack::from_pcm_summary(
                                idx as u32 + 1,
                                name,
                                sample_rate,
                                ch,
                                &samples_i16,
                                coff,
                                csz,
                                sub_stream,
                            ));
                        }
                        Err(notes) => {
                            if local_notes.len() < 20 {
                                local_notes.extend(notes);
                            }
                        }
                    }
                    let d = done_cnt.fetch_add(1, Ordering::Relaxed) + 1;
                    if d == total || d.is_multiple_of(8) {
                        on_progress(d, total);
                    }
                }
                (local_tracks, local_notes)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut tracks = Vec::new();
    let mut chunk_notes = Vec::new();
    for (lt, ln) in results {
        tracks.extend(lt);
        chunk_notes.extend(ln);
    }
    tracks.sort_by_key(|t| t.stream_idx);

    let summary = if tracks.is_empty() {
        format!("Found {} 0x55 chunk(s) (sc={sc}, flags=0x{flags:08X}), but 0 decoded to playable WAV", total)
    } else {
        format!("Decoded {} playable .wav stream(s) natively", tracks.len())
    };
    let mut details = Vec::new();
    if tracks.is_empty() {
        details.push(format!("ADAT header: flags=0x{flags:08X}, streams(sc)={sc}, hdr_sz=0x{raw_hdr_sz:X}, 0x55_chunks={total}"));
    }
    details.extend(skipped_chunks);
    details.extend(chunk_notes);

    AwcMemoryResult { tracks, summary, details }
}

pub fn decode_awc_bytes_to_wavs_detailed(data: &[u8], out_dir: &Path, stem: &str) -> AwcDecodeResult {
    let (flags, sc, raw_hdr_sz, data_chunks, skipped_chunks) = match parse_adat_header(data) {
        ParsedAdatHeader::Err { summary, details } => {
            return AwcDecodeResult { wavs: Vec::new(), summary, details };
        }
        ParsedAdatHeader::Ok { flags, sc, raw_hdr_sz, data_chunks, skipped_chunks } => {
            (flags, sc, raw_hdr_sz, data_chunks, skipped_chunks)
        }
    };

    let jobs = expand_superblock_jobs(data, &data_chunks);
    let _ = fs::create_dir_all(out_dir);
    let mut wavs = Vec::new();
    let mut pcm_f32 = vec![0.0f32; 5760 * 2];
    let mut samples_i16 = Vec::new();
    let mut chunk_notes: Vec<String> = Vec::new();

    for (idx, &(coff, csz, sample_rate, sub_stream)) in jobs.iter().enumerate() {
        match decode_opus_chunk_into(idx, data, coff, csz, sub_stream, &mut pcm_f32, &mut samples_i16) {
            Ok(ch) => {
                let wav_p = out_dir.join(format!("{stem}_s{idx:02}.wav"));
                match write_pcm_wav(&wav_p, sample_rate, ch, &samples_i16) {
                    Ok(()) => wavs.push((idx as u32 + 1, wav_p)),
                    Err(err) => chunk_notes.push(format!("Chunk #{idx}: WAV write error ({}): {err}", wav_p.display())),
                }
            }
            Err(notes) => chunk_notes.extend(notes),
        }
    }

    let summary = if wavs.is_empty() {
        format!("Found {} 0x55 chunk(s) (sc={sc}, flags=0x{flags:08X}), but 0 decoded to playable WAV", jobs.len())
    } else {
        format!("Decoded {} playable .wav stream(s) natively", wavs.len())
    };
    let mut details = Vec::new();
    if wavs.is_empty() {
        details.push(format!("ADAT header: flags=0x{flags:08X}, streams(sc)={sc}, hdr_sz=0x{raw_hdr_sz:X}, 0x55_chunks={}", jobs.len()));
    }
    details.extend(skipped_chunks);
    details.extend(chunk_notes);

    AwcDecodeResult { wavs, summary, details }
}

pub fn write_pcm_wav(path: &Path, sample_rate: u32, channels: u16, samples: &[i16]) -> std::io::Result<()> {
    let data_bytes = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * (channels as u32) * 2;
    let block_align = channels * 2;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 2) };
        w.write_all(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &s in samples {
            w.write_all(&s.to_le_bytes())?;
        }
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_opus::{Application, OpusEncoder};

    #[test]
    fn roundtrip_adat_opus_to_wav() {
        let mut enc = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
        let mut pkt = vec![0u8; 256];
        let pcm = vec![0.25f32; 960];
        let n = enc.encode(&pcm, 960, &mut pkt).unwrap();
        pkt.truncate(n);

        let hdr_sz = 0x30usize;
        let mut blk = vec![0u8; 2048];
        blk[0..2].copy_from_slice(&(n as u16).to_le_bytes());
        blk[2..2 + n].copy_from_slice(&pkt);

        let mut adat = vec![0u8; hdr_sz + blk.len()];
        adat[0..4].copy_from_slice(b"ADAT");
        adat[4..8].copy_from_slice(&1u32.to_le_bytes());
        adat[8..12].copy_from_slice(&1u32.to_le_bytes());
        adat[12..16].copy_from_slice(&(hdr_sz as u32).to_le_bytes());
        let entry: u64 = (0x55u64 << 56) | ((blk.len() as u64) << 28) | (hdr_sz as u64);
        adat[20..28].copy_from_slice(&entry.to_le_bytes());
        adat[hdr_sz..].copy_from_slice(&blk);

        let tmp = std::env::temp_dir().join("percystar_awc_unit_test");
        let _ = fs::remove_dir_all(&tmp);
        let wavs = decode_awc_bytes_to_wavs(&adat, &tmp, "unit");
        assert_eq!(wavs.len(), 1);
        let bytes = fs::read(&wavs[0].1).unwrap();
        assert!(bytes.starts_with(b"RIFF"));
        assert!(bytes.len() > 44 + 960);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn reports_clear_diagnostics_on_invalid_awc() {
        let tmp = std::env::temp_dir().join("percystar_awc_diag_test");

        let small = decode_awc_bytes_to_wavs_detailed(b"short", &tmp, "test");
        assert!(small.wavs.is_empty());
        assert!(small.summary.contains("Payload too small"));

        let tada = decode_awc_bytes_to_wavs_detailed(&[b'T', b'A', b'D', b'A', 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0x40], &tmp, "test");
        assert!(tada.wavs.is_empty());
        assert!(tada.summary.contains("TADA"));

        let mut no_opus = vec![0u8; 0x30];
        no_opus[0..4].copy_from_slice(b"ADAT");
        no_opus[8..12].copy_from_slice(&1u32.to_le_bytes());
        no_opus[12..16].copy_from_slice(&0x30u32.to_le_bytes());
        let entry_val = (0x48u64 << 56) | (16u64 << 28) | 0x30u64;
        no_opus[20..28].copy_from_slice(&entry_val.to_le_bytes());
        let res = decode_awc_bytes_to_wavs_detailed(&no_opus, &tmp, "test");
        assert!(res.wavs.is_empty());
        assert!(res.summary.contains("No valid 0x55 Opus data chunks"));
        assert!(res.details.iter().any(|d| d.contains("0x48")));
    }

    #[test]
    fn uses_0x76_sample_rate_for_opus_streams() {
        let mut enc = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
        let pcm = vec![0.0f32; 960];
        let mut pkt = vec![0u8; 256];
        let plen = enc.encode(&pcm, 960, &mut pkt).unwrap();
        pkt.truncate(plen);

        let mut blk = vec![0u8; 2048];
        blk[0..2].copy_from_slice(&(plen as u16).to_le_bytes());
        blk[2..2 + plen].copy_from_slice(&pkt);

        let hdr_sz = 0x38usize;
        let data_off = hdr_sz;
        let fmt_off = data_off + blk.len();
        let mut fmt_chunk = [0u8; 32];
        fmt_chunk[8..10].copy_from_slice(&32_000u16.to_le_bytes());

        let mut adat = vec![0u8; fmt_off + fmt_chunk.len()];
        adat[0..4].copy_from_slice(b"ADAT");
        adat[4..8].copy_from_slice(&0xFF02_0001u32.to_le_bytes());
        adat[8..12].copy_from_slice(&1u32.to_le_bytes());
        adat[12..16].copy_from_slice(&(hdr_sz as u32).to_le_bytes());
        let e55: u64 = (0x55u64 << 56) | ((blk.len() as u64) << 28) | (data_off as u64);
        let e76: u64 = (0x76u64 << 56) | ((fmt_chunk.len() as u64) << 28) | (fmt_off as u64);
        adat[20..28].copy_from_slice(&e55.to_le_bytes());
        adat[28..36].copy_from_slice(&e76.to_le_bytes());
        adat[data_off..data_off + blk.len()].copy_from_slice(&blk);
        adat[fmt_off..fmt_off + fmt_chunk.len()].copy_from_slice(&fmt_chunk);

        let res = decode_awc_bytes_in_memory(&adat, "dispatch", |_, _| {});
        assert_eq!(res.tracks.len(), 1);
        assert_eq!(res.tracks[0].sample_rate, 32_000);
    }

    #[test]
    fn adat_u16_stream_table_parses_all_streams() {
        let mut enc = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
        let pcm = vec![0.0f32; 960];
        let mut pkt = vec![0u8; 256];
        let plen = enc.encode(&pcm, 960, &mut pkt).unwrap();
        pkt.truncate(plen);

        let mut blk = vec![0u8; 512];
        blk[0..2].copy_from_slice(&(plen as u16).to_le_bytes());
        blk[2..2 + plen].copy_from_slice(&pkt);

        // Simulate 3-stream ADAT with a u16[3] stream table (16 + 3*2 = 22 bytes -> chunk table at 22)
        let sc = 3usize;
        let chunk_off = 16 + sc * 2;
        let hdr_sz = chunk_off + sc * 8;
        let mut adat = vec![0u8; hdr_sz + blk.len() * sc];
        adat[0..4].copy_from_slice(b"ADAT");
        adat[4..8].copy_from_slice(&0xFF02_0001u32.to_le_bytes());
        adat[8..12].copy_from_slice(&(sc as u32).to_le_bytes());
        adat[12..16].copy_from_slice(&(hdr_sz as u32).to_le_bytes());
        for i in 0..sc {
            let coff = hdr_sz + i * blk.len();
            let e55: u64 = (0x55u64 << 56) | ((blk.len() as u64) << 28) | (coff as u64);
            adat[chunk_off + i * 8..chunk_off + (i + 1) * 8].copy_from_slice(&e55.to_le_bytes());
            adat[coff..coff + blk.len()].copy_from_slice(&blk);
        }

        let res = decode_awc_bytes_in_memory(&adat, "hud_u16", |_, _| {});
        assert_eq!(res.tracks.len(), 3);
    }

    #[test]
    fn late_aligned_table_beats_eight_false_u32_candidates() {
        // Reproduce the selection failure shape seen in issue #48 without
        // bundling game audio: the u32 candidate starts at 0x7ec and sees
        // eight in-bounds 0x55-looking records inside the header. A synthetic
        // 503-entry table is later in the header, on the reverse walk's
        // 2-mod-8 alignment, and points beyond the header.
        let sc = 503usize;
        let hdr_sz = 0x518Ausize;
        let data_end = hdr_sz + sc * 32;
        let mut adat = vec![0u8; data_end];
        adat[0..4].copy_from_slice(b"ADAT");
        adat[4..8].copy_from_slice(&0xFF02_0001u32.to_le_bytes());
        adat[8..12].copy_from_slice(&(sc as u32).to_le_bytes());
        adat[12..16].copy_from_slice(&(hdr_sz as u32).to_le_bytes());

        for i in 0..8 {
            let false_off = 0x100 + i * 0x20;
            let entry = (0x55u64 << 56) | (8u64 << 28) | false_off as u64;
            let at = 16 + sc * 4 + i * 8;
            adat[at..at + 8].copy_from_slice(&entry.to_le_bytes());
        }

        // Two other 0x55-looking header records in the real sample point past
        // the container. They must not be counted as recovered streams.
        let outside_off = (data_end + 0x1000) as u64;
        for at in [0x5C2usize, 0x902] {
            let entry = (0x55u64 << 56) | (0x800u64 << 28) | outside_off;
            adat[at..at + 8].copy_from_slice(&entry.to_le_bytes());
        }

        // Model the late per-stream sequence: 0x55 data, 0x76 format,
        // 0xA3 metadata, with a 0x36 record in most groups. This reproduces the
        // mixed 24/32-byte groups and reverse walk's 2-mod-8 alignment.
        let table_start = 0x13C2usize;
        let mut at = table_start;
        for i in 0..sc {
            let data_off = hdr_sz + i * 16;
            let fmt_off = hdr_sz + sc * 16 + i * 16;
            let sample_rate = if i % 2 == 0 { 32_000u16 } else { 48_000u16 };

            let e55 = (0x55u64 << 56) | (8u64 << 28) | data_off as u64;
            let e76 = (0x76u64 << 56) | (10u64 << 28) | fmt_off as u64;
            let ea3 = (0xA3u64 << 56) | (8u64 << 28) | fmt_off as u64;
            adat[at..at + 8].copy_from_slice(&e55.to_le_bytes());
            adat[at + 8..at + 16].copy_from_slice(&e76.to_le_bytes());
            adat[at + 16..at + 24].copy_from_slice(&ea3.to_le_bytes());
            adat[fmt_off + 8..fmt_off + 10].copy_from_slice(&sample_rate.to_le_bytes());
            at += 24;
            if i < 468 {
                let e36 = (0x36u64 << 56) | (8u64 << 28) | fmt_off as u64;
                adat[at..at + 8].copy_from_slice(&e36.to_le_bytes());
                at += 8;
            }
        }
        assert_eq!(at, hdr_sz);

        let ParsedAdatHeader::Ok { data_chunks, sc: parsed_sc, .. } = parse_adat_header(&adat) else {
            panic!("synthetic ADAT header should parse");
        };
        assert_eq!(parsed_sc, 503);
        assert_eq!(data_chunks.len(), 503);
        for (i, chunk) in data_chunks.iter().enumerate() {
            assert_eq!(chunk.2, if i % 2 == 0 { 32_000 } else { 48_000 });
        }
    }

}

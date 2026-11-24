#!/usr/bin/env python3
"""
TTPF & Xbox Series X BCPack Forensic Analyzer & Recovery Tool.

Commands:
  analyze <file.ttpf> [--json <report.json>]
      Full structural analysis of RDS5 chunks, ETPF textures, mip-fallback recoverability,
      and BCPack (codec 1..7) header/entropy/slice correlation.

  dump-bcpack <file.ttpf> --out <dir> [--codec <1..7|all>] [--limit <N>]
      Dumps raw BCPack chunks (.bcpack.bin) and bcpack_manifest.json with linked textures.

  extract-mip-fallback <file.ttpf> --out <dir> [--limit <N>]
      Recovers lower mipmaps (mip1..mip6) from ZLIB chunks for textures whose mip0 sits in BCPack,
      writing valid DX10 .dds files + recovery_manifest.json.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import os
import struct
import sys
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DXGI_INFO: Dict[int, Tuple[str, bool, int]] = {
    # dxgi: (name, is_bc, bytes_per_block_or_pixel)
    28: ("R8G8B8A8_UNORM", False, 4),
    29: ("R8G8B8A8_UNORM_SRGB", False, 4),
    41: ("R32_FLOAT", False, 4),
    49: ("R8G8_UNORM", False, 2),
    54: ("R16_FLOAT", False, 2),
    56: ("R16_UNORM", False, 2),
    61: ("R8_UNORM", False, 1),
    65: ("A8_UNORM", False, 1),
    71: ("BC1_UNORM", True, 8),
    72: ("BC1_UNORM_SRGB", True, 8),
    74: ("BC2_UNORM", True, 16),
    75: ("BC2_UNORM_SRGB", True, 16),
    77: ("BC3_UNORM", True, 16),
    78: ("BC3_UNORM_SRGB", True, 16),
    80: ("BC4_UNORM", True, 8),
    81: ("BC4_SNORM", True, 8),
    83: ("BC5_UNORM", True, 16),
    84: ("BC5_SNORM", True, 16),
    87: ("B8G8R8A8_UNORM", False, 4),
    95: ("BC6H_UF16", True, 16),
    96: ("BC6H_SF16", True, 16),
    98: ("BC7_UNORM", True, 16),
    99: ("BC7_UNORM_SRGB", True, 16),
}

CODEC_NAMES = {
    0: "RAW",
    1: "BCPack_BC1",
    2: "BCPack_BC2",
    3: "BCPack_BC3",
    4: "BCPack_BC4",
    5: "BCPack_BC5",
    6: "BCPack_BC6H",
    7: "BCPack_BC7",
    8: "ZLIB",
}


@dataclass
class Chunk:
    index: int
    file_start: int
    file_end: int
    virt_start: int
    virt_end: int
    codec: int

    @property
    def comp_size(self) -> int:
        return self.file_end - self.file_start

    @property
    def virt_size(self) -> int:
        return self.virt_end - self.virt_start


@dataclass
class TextureDesc:
    index: int
    width: int
    height: int
    dxgi: int
    dxgi_name: str
    has_mips: bool
    virt_offset: int
    alloc_size: int
    flags: int
    hash_u64: int


def parse_ttpf(path: Path) -> Tuple[dict, List[Chunk], List[TextureDesc]]:
    with open(path, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24 or hdr[:4] != b"RDS5":
            raise ValueError(f"Not an RDS5 TTPF file: {path}")
        n_chunks = struct.unpack_from("<I", hdr, 4)[0]
        tot_uncomp = struct.unpack_from("<Q", hdr, 8)[0] & 0xFFFFFFFFFF
        hdr_pages = struct.unpack_from("<I", hdr, 16)[0]
        etpf_off = hdr_pages * 0x100
        tbl = f.read(n_chunks * 8)

        chunks: List[Chunk] = []
        prev_c, vstart, cur_codec = etpf_off, 0, 0
        for i in range(n_chunks):
            lo, hi = struct.unpack_from("<II", tbl, i * 8)
            cend = (lo & 0x03FFFFFF) * 0x100
            next_codec = (lo >> 26) & 0xF
            r = (lo >> 30) & 0x3
            vend = (hi & 0x000FFFFF) * 0x4000 + r * 0x1000
            if cend > prev_c and vend > vstart:
                chunks.append(Chunk(i, prev_c, cend, vstart, vend, cur_codec))
                vstart = vend
            prev_c = cend
            cur_codec = next_codec

        f.seek(etpf_off)
        ehdr = f.read(16)
        emagic, ever, tex_count, eflags = struct.unpack("<4sIII", ehdr)
        if emagic != b"ETPF":
            raise ValueError(f"Expected ETPF at 0x{etpf_off:X}, got {emagic!r}")
        raw_descs = f.read(tex_count * 16)
        raw_hashes = f.read(tex_count * 8)

    raw_voffs = [
        (struct.unpack_from("<I", raw_descs, i * 16 + 8)[0] & 0x1FFFFF) * 0x1000
        for i in range(tex_count)
    ]
    total_v = chunks[-1].virt_end if chunks else tot_uncomp

    textures: List[TextureDesc] = []
    for i in range(tex_count):
        u0, u1, u2, u3 = struct.unpack_from("<IIII", raw_descs, i * 16)
        w = (u0 & 0x3FFF) + 1
        h = ((u0 >> 14) & 0x3FFF) + 1
        dxgi = (u1 >> 3) & 0x7F
        has_mips = (u1 >> 24) > 0
        voff = raw_voffs[i]
        next_voff = raw_voffs[i + 1] if i + 1 < tex_count else total_v
        alloc_sz = max(0, next_voff - voff)
        h64 = struct.unpack_from("<Q", raw_hashes, i * 8)[0] if (i + 1) * 8 <= len(raw_hashes) else 0
        dxgi_name = DXGI_INFO.get(dxgi, (f"DXGI_{dxgi}", False, 0))[0]
        textures.append(TextureDesc(i, w, h, dxgi, dxgi_name, has_mips, voff, alloc_sz, u3, h64))

    meta = {
        "file": str(path),
        "file_size": path.stat().st_size,
        "chunk_count": len(chunks),
        "etpf_offset": etpf_off,
        "etpf_version": ever,
        "etpf_flags": eflags,
        "texture_count": tex_count,
        "total_virtual_size": tot_uncomp,
    }
    return meta, chunks, textures


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = collections.Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def is_zlib_head(b: bytes) -> bool:
    return len(b) >= 2 and b[0] == 0x78 and b[1] in (0x01, 0x5E, 0x9C, 0xDA)


def read_virtual_zlib_only(f, chunks: List[Chunk], vends: List[int], off: int, length: int) -> Optional[bytes]:
    end = off + length
    out = bytearray(length)
    covered = 0
    idx = bisect.bisect_right(vends, off)
    for c in chunks[idx:]:
        if c.virt_start >= end:
            break
        f.seek(c.file_start)
        raw = f.read(c.comp_size)
        if is_zlib_head(raw):
            try:
                dec = zlib.decompress(raw)
            except zlib.error:
                dobj = zlib.decompressobj()
                dec = dobj.decompress(raw)
        elif len(raw) == c.virt_size:
            dec = raw
        else:
            return None  # hit BCPack chunk
        s0 = max(c.virt_start, off)
        s1 = min(c.virt_start + len(dec), end)
        if s1 > s0:
            out[s0 - off : s1 - off] = dec[s0 - c.virt_start : s1 - c.virt_start]
            covered += s1 - s0
    return bytes(out) if covered == length else None


def padded_4kb_size(w: int, h: int, is_bc: bool, bpe: int) -> int:
    ew = max(1, (w + 3) // 4) if is_bc else max(1, w)
    eh = max(1, (h + 3) // 4) if is_bc else max(1, h)
    # 4KB tile alignment
    lin = ew * eh * bpe
    return ((lin + 4095) // 4096) * 4096


def build_dds_dx10_header(w: int, h: int, dxgi: int, is_bc: bool, bpe: int) -> bytes:
    ew = max(1, (w + 3) // 4) if is_bc else max(1, w)
    eh = max(1, (h + 3) // 4) if is_bc else max(1, h)
    pitch = ew * eh * bpe if is_bc else ew * bpe
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | (0x80000 if is_bc else 0x8)
    hdr = bytearray(128 + 20)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)
    struct.pack_into("<I", hdr, 8, flags)
    struct.pack_into("<I", hdr, 12, h)
    struct.pack_into("<I", hdr, 16, w)
    struct.pack_into("<I", hdr, 20, pitch)
    struct.pack_into("<I", hdr, 24, 1)  # depth
    struct.pack_into("<I", hdr, 28, 1)  # mipMapCount
    # pixelFormat at offset 76
    struct.pack_into("<I", hdr, 76, 32)
    struct.pack_into("<I", hdr, 80, 0x4)  # DDPF_FOURCC
    hdr[84:88] = b"DX10"
    struct.pack_into("<I", hdr, 108, 0x1000)  # DDSCAPS_TEXTURE
    # DX10 header at offset 128
    struct.pack_into("<IIIII", hdr, 128, dxgi, 3, 0, 1, 0)
    return bytes(hdr)


def cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.file)
    meta, chunks, textures = parse_ttpf(path)
    vends = [c.virt_end for c in chunks]

    by_codec: Dict[int, List[Chunk]] = collections.defaultdict(list)
    for c in chunks:
        by_codec[c.codec].append(c)

    codec_summary = {}
    for cod, lst in sorted(by_codec.items()):
        tot_c = sum(c.comp_size for c in lst)
        tot_v = sum(c.virt_size for c in lst)
        codec_summary[CODEC_NAMES.get(cod, f"Codec_{cod}")] = {
            "codec_id": cod,
            "chunks": len(lst),
            "compressed_bytes": tot_c,
            "virtual_bytes": tot_v,
            "ratio": round(tot_c / tot_v, 4) if tot_v else 0.0,
            "min_vlen": min(c.virt_size for c in lst),
            "max_vlen": max(c.virt_size for c in lst),
        }

    # Texture classification
    zlib_ok = 0
    mixed = 0
    bcpack_only = 0
    dxgi_bcpack = collections.Counter()
    for t in textures:
        idx0 = bisect.bisect_right(vends, t.virt_offset)
        hits = set()
        for c in chunks[idx0:]:
            if c.virt_start >= t.virt_offset + t.alloc_size:
                break
            hits.add(c.codec)
        if hits <= {0, 8}:
            zlib_ok += 1
        elif 8 in hits or 0 in hits:
            mixed += 1
            dxgi_bcpack[t.dxgi_name] += 1
        else:
            bcpack_only += 1
            dxgi_bcpack[t.dxgi_name] += 1

    # BCPack header pattern forensic analysis
    bcpack_forensics = {}
    with open(path, "rb") as f:
        for cod in sorted(k for k in by_codec if 1 <= k <= 7):
            lst = by_codec[cod]
            b0_counts = collections.Counter()
            b1_counts = collections.Counter()
            slice_corr = []
            sample_entropies = []
            for idx_s, c in enumerate(lst[:200]):
                f.seek(c.file_start)
                raw = f.read(min(c.comp_size, 64))
                if len(raw) >= 20:
                    b0_counts[f"0x{raw[0]:02X}"] += 1
                    b1_counts[f"0x{raw[1]:02X}"] += 1
                    if idx_s < 12:
                        f.seek(c.file_start)
                        full = f.read(c.comp_size).rstrip(b"\x00")
                        ent = round(shannon_entropy(full), 4)
                        sample_entropies.append(ent)
                        tile_div = 2048 if cod == 1 else 4096
                        slice_corr.append({
                            "chunk_index": c.index,
                            "vlen": c.virt_size,
                            "slices_256blk": c.virt_size // tile_div,
                            "nopad_csize": len(full),
                            "entropy": ent,
                            "byte0_1": raw[0:2].hex(),
                            "byte16_17": raw[16:18].hex(),
                            "byte16_dec": raw[16],
                            "byte17_dec": raw[17],
                            "head24": raw[:24].hex(),
                        })
            bcpack_forensics[CODEC_NAMES.get(cod, f"Codec_{cod}")] = {
                "sampled_chunks": min(len(lst), 200),
                "byte0_distribution": dict(b0_counts.most_common(10)),
                "byte1_distribution": dict(b1_counts.most_common(10)),
                "avg_entropy_sample": round(sum(sample_entropies) / len(sample_entropies), 4) if sample_entropies else 0.0,
                "slice_header_samples": slice_corr,
            }

    report = {
        "meta": meta,
        "codecs": codec_summary,
        "textures": {
            "total": len(textures),
            "zlib_or_raw_100pct": zlib_ok,
            "mixed_mip_recoverable": mixed,
            "bcpack_only_100pct": bcpack_only,
            "recoverable_total": zlib_ok + mixed,
            "recoverable_pct": round((zlib_ok + mixed) / len(textures) * 100, 2) if textures else 0.0,
            "bcpack_dxgi_breakdown": dict(dxgi_bcpack.most_common()),
        },
        "bcpack_forensics": bcpack_forensics,
    }

    print(f"=== TTPF Forensic Analysis: {path.name} ===")
    print(f"File size: {meta['file_size']:,} B | Chunks: {meta['chunk_count']:,} | Textures: {meta['texture_count']:,}")
    print("\n[Codecs]")
    for name, info in codec_summary.items():
        print(
            f"  {name:12s}: {info['chunks']:6d} chunks | "
            f"comp={info['compressed_bytes']/1e6:7.2f} MB | virt={info['virtual_bytes']/1e6:7.2f} MB | "
            f"ratio={info['ratio']:.3f}"
        )
    print("\n[Texture Recoverability]")
    tinfo = report["textures"]
    print(f"  100% ZLIB/RAW     : {tinfo['zlib_or_raw_100pct']:5d} ({tinfo['zlib_or_raw_100pct']/tinfo['total']*100:.2f}%)")
    print(f"  Mixed (Mip-Recov) : {tinfo['mixed_mip_recoverable']:5d} ({tinfo['mixed_mip_recoverable']/tinfo['total']*100:.2f}%)")
    print(f"  100% BCPack-Only  : {tinfo['bcpack_only_100pct']:5d} ({tinfo['bcpack_only_100pct']/tinfo['total']*100:.2f}%)")
    print(f"  TOTAL Decodable   : {tinfo['recoverable_total']:5d} ({tinfo['recoverable_pct']:.2f}%)")

    print("\n[BCPack Header Forensics]")
    for cname, finfo in bcpack_forensics.items():
        print(f"  {cname}: byte0={finfo['byte0_distribution']} byte1={finfo['byte1_distribution']} avg_ent={finfo['avg_entropy_sample']}")
        for s in finfo["slice_header_samples"][:4]:
            print(
                f"    chunk#{s['chunk_index']:5d} vlen={s['vlen']:7d} (256blk_slices={s['slices_256blk']:3d}) "
                f"b16={s['byte16_dec']:3d} b17={s['byte17_dec']:3d} head={s['head24']}"
            )

    if args.json:
        out_json = Path(args.json)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {out_json}")
    return 0


def cmd_dump_bcpack(args: argparse.Namespace) -> int:
    path = Path(args.file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, chunks, textures = parse_ttpf(path)

    target_codecs = set(range(1, 8)) if args.codec == "all" else {int(args.codec)}
    bc_chunks = [c for c in chunks if c.codec in target_codecs]
    if args.limit:
        bc_chunks = bc_chunks[: args.limit]

    # Map virtual ranges to textures
    tex_starts = [t.virt_offset for t in textures]
    manifest = []
    with open(path, "rb") as f:
        for c in bc_chunks:
            f.seek(c.file_start)
            raw = f.read(c.comp_size)
            nopad = raw.rstrip(b"\x00")
            fname = f"chunk_{c.index:05d}_codec{c.codec}_{CODEC_NAMES.get(c.codec, 'BC')}_v{c.virt_size}.bcpack.bin"
            (out_dir / fname).write_bytes(raw)
            # Find overlapping textures
            i0 = max(0, bisect.bisect_right(tex_starts, c.virt_start) - 1)
            linked = []
            for t in textures[i0:]:
                if t.virt_offset >= c.virt_end:
                    break
                if t.virt_offset + t.alloc_size > c.virt_start:
                    linked.append({
                        "tex_index": t.index,
                        "hash": f"0x{t.hash_u64:016X}",
                        "width": t.width,
                        "height": t.height,
                        "dxgi": t.dxgi,
                        "dxgi_name": t.dxgi_name,
                        "has_mips": t.has_mips,
                    })
            manifest.append({
                "file": fname,
                "chunk_index": c.index,
                "codec": c.codec,
                "codec_name": CODEC_NAMES.get(c.codec, f"Codec_{c.codec}"),
                "file_offset": c.file_start,
                "comp_size": c.comp_size,
                "nopad_size": len(nopad),
                "virt_offset": c.virt_start,
                "virt_size": c.virt_size,
                "header_hex": raw[:32].hex(),
                "linked_textures": linked,
            })

    (out_dir / "bcpack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dumped {len(manifest)} BCPack chunks + bcpack_manifest.json to {out_dir}")
    return 0


def cmd_extract_mip_fallback(args: argparse.Namespace) -> int:
    path = Path(args.file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, chunks, textures = parse_ttpf(path)
    vends = [c.virt_end for c in chunks]

    recovered = []
    skipped = 0
    with open(path, "rb") as f:
        for t in textures:
            if args.limit and len(recovered) >= args.limit:
                break
            if not t.has_mips or t.dxgi not in DXGI_INFO:
                continue
            idx0 = bisect.bisect_right(vends, t.virt_offset)
            hits = set()
            for c in chunks[idx0:]:
                if c.virt_start >= t.virt_offset + t.alloc_size:
                    break
                hits.add(c.codec)
            # Only target textures that actually hit BCPack AND also have ZLIB/RAW
            if not (any(1 <= h <= 7 for h in hits) and (8 in hits or 0 in hits)):
                continue
            fmt_name, is_bc, bpe = DXGI_INFO[t.dxgi]
            mip0_pad = padded_4kb_size(t.width, t.height, is_bc, bpe)
            cursor_end = t.virt_offset + t.alloc_size - mip0_pad
            found = False
            for mip in range(1, 7):
                mw = max(4, t.width >> mip)
                mh = max(4, t.height >> mip)
                m_pad = padded_4kb_size(mw, mh, is_bc, bpe)
                if cursor_end < t.virt_offset + m_pad:
                    break
                cursor_end -= m_pad
                raw = read_virtual_zlib_only(f, chunks, vends, cursor_end, m_pad)
                if raw is not None and not all(b == raw[0] for b in raw[:256]):
                    ew = max(1, (mw + 3) // 4) if is_bc else mw
                    eh = max(1, (mh + 3) // 4) if is_bc else mh
                    lin_sz = ew * eh * bpe
                    dds = build_dds_dx10_header(mw, mh, t.dxgi, is_bc, bpe) + raw[:lin_sz]
                    fname = f"tex_{t.index:05d}_{t.hash_u64:016X}_mip{mip}_{mw}x{mh}_{fmt_name}.dds"
                    (out_dir / fname).write_bytes(dds)
                    recovered.append({
                        "tex_index": t.index,
                        "hash": f"0x{t.hash_u64:016X}",
                        "orig_size": f"{t.width}x{t.height}",
                        "recovered_mip": mip,
                        "recovered_size": f"{mw}x{mh}",
                        "dxgi": fmt_name,
                        "file": fname,
                    })
                    found = True
                    break
            if not found:
                skipped += 1

    (out_dir / "recovery_manifest.json").write_text(json.dumps(recovered, indent=2), encoding="utf-8")
    print(f"Recovered {len(recovered)} BCPack-mip0 textures via ZLIB mip-fallback to {out_dir} (unrecoverable mixed: {skipped})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="TTPF & BCPack Forensic Analyzer & Recovery Tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Analyze RDS5 chunks, textures, and BCPack headers")
    p_an.add_argument("file", help="Path to .ttpf archive")
    p_an.add_argument("--json", help="Optional output path for JSON report")

    p_dp = sub.add_parser("dump-bcpack", help="Dump raw BCPack chunks + manifest")
    p_dp.add_argument("file", help="Path to .ttpf archive")
    p_dp.add_argument("--out", required=True, help="Output directory")
    p_dp.add_argument("--codec", default="all", help="Codec ID (1..7) or 'all'")
    p_dp.add_argument("--limit", type=int, default=None, help="Max chunks to dump")

    p_ex = sub.add_parser("extract-mip-fallback", help="Recover lower mipmaps for BCPack mip0 textures")
    p_ex.add_argument("file", help="Path to .ttpf archive")
    p_ex.add_argument("--out", required=True, help="Output directory")
    p_ex.add_argument("--limit", type=int, default=None, help="Max textures to recover")

    args = p.parse_args()
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "dump-bcpack":
        return cmd_dump_bcpack(args)
    if args.cmd == "extract-mip-fallback":
        return cmd_extract_mip_fallback(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

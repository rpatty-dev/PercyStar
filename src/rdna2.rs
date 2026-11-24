// Clean-room Xbox Series X (Scarlett / AMD RDNA2 GFX10.3) PS\x00\x09 PSO Cache parser,
// GFX10.3 ISA disassembler, HLSL SM 6.6 lifter, and SPIR-V 1.5 binary emitter.
// Based on AMD RDNA 2 ISA Reference Guide & Mesa RADV/ACO gfx1030 public encoding tables.

use std::collections::HashMap;
use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderStage {
    Compute,
    Pixel,
    Vertex,
    Geometry,
    Domain,
    Hull,
    MeshNgg,
    RayTracing,
    Unknown,
}

impl ShaderStage {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Compute => "CS",
            Self::Pixel => "PS",
            Self::Vertex => "VS",
            Self::Geometry => "GS",
            Self::Domain => "DS",
            Self::Hull => "HS",
            Self::MeshNgg => "NGG",
            Self::RayTracing => "RT",
            Self::Unknown => "GFX10",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResourceBinding {
    pub kind: &'static str, // "SRV", "CBV", "UAV", "SAMPLER", "ROOT_TABLE"
    pub kind_raw: u32,
    pub byte_offset: u32,
    pub slot_id: u32,
    pub dwords: u32,
    pub sgpr_flags: u32,
}

#[derive(Clone, Debug)]
pub struct Ps09ResourceName {
    pub declaration: String,
    pub register: String,
    pub name: Option<String>,
    pub hash: u32,
}

#[derive(Clone, Debug)]
pub struct Ps09ShaderEntry {
    pub index: u32,
    pub code_offset: u32,
    pub code_size: u32,
    pub stage: ShaderStage,
    pub num_threads: Option<[u16; 3]>,
    pub rsrc1: Option<u32>,
    pub rsrc2: Option<u32>,
    pub bindings: Vec<ResourceBinding>,
    pub resource_names: Vec<Ps09ResourceName>,
    pub pso_hash: u64,
    pub sga_hash: Option<u64>,
    pub has_rt: bool,
    pub has_ngg: bool,
    pub insn_count: u32,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Ps09Header {
    pub id: String,
    pub driver: String,
    pub graphics_pso_count: u32,
    pub compute_pso_count: u32,
    pub meta_count: u32,
    pub shader_count: u32,
    pub meta_size: u32,
    pub code_size: u32,
    pub off_meta: usize,
    pub off_code: usize,
}

#[derive(Clone, Debug)]
pub struct Ps09Cache {
    pub header: Ps09Header,
    pub shaders: Vec<Ps09ShaderEntry>,
    pub raw: Vec<u8>,
}

fn rd32(b: &[u8], off: usize) -> u32 {
    // offsets are derived from file values the sum must not wrap
    if off.checked_add(4).map_or(true, |end| end > b.len()) {
        return 0;
    }
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn rd64(b: &[u8], off: usize) -> u64 {
    if off.checked_add(8).map_or(true, |end| end > b.len()) {
        return 0;
    }
    u64::from_le_bytes([
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ])
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).trim().to_string()
}

fn parse_bindings(meta_slice: &[u8]) -> Vec<ResourceBinding> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 32 <= meta_slice.len() {
        let tag = rd32(meta_slice, i);
        if tag == 0x0000_0041 {
            let kind_raw = rd32(meta_slice, i + 4);
            let byte_offset = rd32(meta_slice, i + 8);
            let slot_id = rd32(meta_slice, i + 12);
            let dwords = rd32(meta_slice, i + 16);
            let sgpr_flags = rd32(meta_slice, i + 20);
            let kind = match kind_raw {
                0x0300 => "SRV",
                0x0302 => "CBV",
                0x0303 => "UAV",
                0x0401 => "ROOT_TABLE",
                0x0002 => "SAMPLER",
                _ => "DESC",
            };
            out.push(ResourceBinding {
                kind,
                kind_raw,
                byte_offset,
                slot_id,
                dwords,
                sgpr_flags,
            });
            i += 32;
        } else {
            i += 4;
        }
    }
    out
}

/// Detect exact code base: normally `off_meta + meta_size`, but gracefully handles partial
/// chunk dumps that dropped chunks before code start by checking shader[0..2] alignment.
fn locate_code_base(data: &[u8], off_c3: usize, c3: u32, expected: usize) -> usize {
    if expected.checked_add(0x1c).is_some_and(|end| end <= data.len()) && rd32(data, expected + 0x18) == 0xD580_0000 {
        return expected;
    }
    // Check aligned offsets near expected (e.g., -64KB if a chunk was missing in extracted dir)
    for delta in [0isize, -0x10000, 0x10000, -0x20000, 0x20000] {
        let Some(cand) = expected.checked_add_signed(delta) else { continue };
        if cand.checked_add(0x28).is_some_and(|end| end <= data.len())
            && (rd32(data, cand + 0x18) == 0xD580_0000 || rd32(data, cand + 0x24) == 0xBFA0_0003)
        {
            return cand;
        }
    }
    // Fallback: verify using shader[0] & shader[1] s_endpgm / s_code_end tails
    if c3 >= 2 {
        let o0 = rd32(data, off_c3) as usize;
        let s0 = rd32(data, off_c3 + 4) as usize;
        if s0 >= 0x40 && expected >= 0x20000 {
            let start = expected.saturating_sub(0x20000);
            let end = (expected + 0x20000).min(data.len().saturating_sub(o0 + s0));
            let mut p = start;
            while p < end {
                if rd32(data, p + 0x18) == 0xD580_0000 && rd32(data, p + 0x1c) == 0x03FE_0100 {
                    return p;
                }
                p += 4;
            }
        }
    }
    expected.min(data.len())
}

fn quick_classify_code(code: &[u8]) -> (ShaderStage, bool, bool, u32) {
    let mut has_mrt = false;
    let mut has_pos = false;
    let mut has_prim = false;
    let mut has_rt = false;
    let mut has_interp = false;
    let mut insns = 0u32;
    let body = skip_scarlett_prologue(code);
    let mut i = 0usize;
    while i + 4 <= body.len() {
        let w0 = rd32(body, i);
        if w0 == 0xBF81_0000 {
            insns += 1;
            break;
        }
        if w0 == 0xBF9F_0000 {
            break;
        }
        let enc6 = w0 >> 26;
        if enc6 == 0x3E {
            let tgt = (w0 >> 4) & 0x3F;
            if tgt <= 8 {
                has_mrt = true;
            } else if (12..=15).contains(&tgt) {
                has_pos = true;
            } else if tgt == 20 {
                has_prim = true;
            }
        } else if enc6 == 0x3C {
            let op = (w0 >> 18) & 0xFF;
            if op == 0x66 || op == 0x67 {
                has_rt = true;
            }
        } else if enc6 == 0x32 {
            has_interp = true;
        }
        let step = gfx103_insn_len(body, i);
        i += step;
        insns += 1;
    }
    let st = if has_rt {
        ShaderStage::RayTracing
    } else if has_prim {
        ShaderStage::MeshNgg
    } else if has_mrt || has_interp {
        ShaderStage::Pixel
    } else if has_pos {
        ShaderStage::Vertex
    } else {
        ShaderStage::Compute
    };
    (st, has_rt, has_prim, insns)
}

/// Xbox Scarlett D3D12.X compiler prepends a 36-byte (9-dword) identity hash prologue
/// ending with `d5800000 03fe0100 00000ac0` before `bfa00003` (`s_setprio 3`).
pub fn skip_scarlett_prologue(code: &[u8]) -> &[u8] {
    if code.len() >= 0x28 && rd32(code, 0x18) == 0xD580_0000 && rd32(code, 0x1c) == 0x03FE_0100 {
        &code[0x24..]
    } else {
        code
    }
}

impl Ps09Cache {
    pub fn parse(raw: Vec<u8>) -> Option<Self> {
        if raw.len() < 0x80 || &raw[0..4] != b"PS\x00\x09" {
            return None;
        }
        let id = cstr(&raw[4..0x24]);
        let driver = cstr(&raw[0x24..0x44]);
        let c0 = rd32(&raw, 0x50);
        let c1 = rd32(&raw, 0x54);
        let c2 = rd32(&raw, 0x58);
        let c3 = rd32(&raw, 0x5c);
        let meta_size = rd32(&raw, 0x60);
        let code_size = rd32(&raw, 0x64);

        let off_c0 = 0x80usize;
        let off_c1 = off_c0 + (c0 as usize) * 0x70;
        let off_c2 = off_c1 + (c1 as usize) * 0x18;
        let off_c3 = off_c2 + (c2 as usize) * 8;
        let off_meta = off_c3 + (c3 as usize) * 8;
        if off_meta > raw.len() {
            return None;
        }
        let expected_code = off_meta + meta_size as usize;
        let off_code = locate_code_base(&raw, off_c3, c3, expected_code);

        // Map shader_idx -> (stage_hint, num_threads, rsrc1, rsrc2, bindings_meta_idx, pso_hash)
        let mut sh_meta: HashMap<u32, (Option<ShaderStage>, Option<[u16; 3]>, Option<u32>, Option<u32>, Option<u32>, u64)> =
            HashMap::new();

        // Compute PSOs (24 bytes each): hash64, rsrc_meta, unk_meta, shader_idx, bind_meta
        for i in 0..c1 as usize {
            let p = off_c1 + i * 0x18;
            if p + 0x18 > raw.len() {
                break;
            }
            let hash = rd64(&raw, p);
            let m_rsrc = rd32(&raw, p + 8);
            let sh_idx = rd32(&raw, p + 16);
            let m_bind = rd32(&raw, p + 20);
            let (mut r1, mut r2, mut nt) = (None, None, None);
            if m_rsrc < c2 {
                let mo = rd32(&raw, off_c2 + m_rsrc as usize * 8) as usize;
                let ms = rd32(&raw, off_c2 + m_rsrc as usize * 8 + 4) as usize;
                if off_meta + mo + 20 <= raw.len() && ms >= 20 {
                    r1 = Some(rd32(&raw, off_meta + mo));
                    r2 = Some(rd32(&raw, off_meta + mo + 4));
                    let xy = rd32(&raw, off_meta + mo + 12);
                    let z = rd32(&raw, off_meta + mo + 16);
                    let tx = (xy & 0xFFFF) as u16;
                    let ty = ((xy >> 16) & 0xFFFF) as u16;
                    let tz = (z & 0xFFFF) as u16;
                    if tx > 0 && tx <= 1024 && ty > 0 && ty <= 1024 && tz > 0 && tz <= 1024 {
                        nt = Some([tx, ty, tz]);
                    }
                }
            }
            sh_meta.entry(sh_idx).or_insert((
                Some(ShaderStage::Compute),
                nt,
                r1,
                r2,
                Some(m_bind),
                hash,
            ));
        }

        // Graphics PSOs (112 bytes each): +0x38..+0x70 pairs (shader_idx, bind_meta_idx)
        for i in 0..c0 as usize {
            let p = off_c0 + i * 0x70;
            if p + 0x70 > raw.len() {
                break;
            }
            let hash = rd64(&raw, p);
            for slot in 0..7usize {
                let sh_idx = rd32(&raw, p + 0x38 + slot * 8);
                let m_bind = rd32(&raw, p + 0x3c + slot * 8);
                if sh_idx != u32::MAX && sh_idx < c3 {
                    sh_meta.entry(sh_idx).or_insert((
                        None,
                        None,
                        None,
                        None,
                        if m_bind < c2 { Some(m_bind) } else { None },
                        hash,
                    ));
                }
            }
        }

        let mut shaders = Vec::with_capacity(c3 as usize);
        for i in 0..c3 {
            let so = rd32(&raw, off_c3 + i as usize * 8);
            let ss = rd32(&raw, off_c3 + i as usize * 8 + 4);
            let abs_s = off_code + so as usize;
            let slice = if abs_s < raw.len() {
                &raw[abs_s..(abs_s + ss as usize).min(raw.len())]
            } else {
                &[]
            };
            let (auto_stage, has_rt, has_ngg, insn_count) = quick_classify_code(slice);
            let (hint, nt, r1, r2, m_bind, pso_hash) = sh_meta
                .get(&i)
                .cloned()
                .unwrap_or((None, None, None, None, None, 0));
            let bindings = if let Some(mb) = m_bind {
                if mb < c2 {
                    let mo = rd32(&raw, off_c2 + mb as usize * 8) as usize;
                    let ms = rd32(&raw, off_c2 + mb as usize * 8 + 4) as usize;
                    if off_meta + mo < raw.len() {
                        parse_bindings(&raw[off_meta + mo..(off_meta + mo + ms).min(raw.len())])
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            let stage = if has_rt {
                ShaderStage::RayTracing
            } else if has_ngg {
                ShaderStage::MeshNgg
            } else {
                hint.unwrap_or(auto_stage)
            };
            shaders.push(Ps09ShaderEntry {
                index: i,
                code_offset: so,
                code_size: ss,
                stage,
                num_threads: nt,
                rsrc1: r1,
                rsrc2: r2,
                bindings,
                resource_names: Vec::new(),
                pso_hash,
                sga_hash: None,
                has_rt,
                has_ngg,
                insn_count,
            });
        }

        Some(Self {
            header: Ps09Header {
                id,
                driver,
                graphics_pso_count: c0,
                compute_pso_count: c1,
                meta_count: c2,
                shader_count: c3,
                meta_size,
                code_size,
                off_meta,
                off_code,
            },
            shaders,
            raw,
        })
    }

    pub fn shader_bytes(&self, entry: &Ps09ShaderEntry) -> &[u8] {
        let st = self.header.off_code + entry.code_offset as usize;
        if st >= self.raw.len() {
            return &[];
        }
        let en = (st + entry.code_size as usize).min(self.raw.len());
        &self.raw[st..en]
    }
}

// =============================================================================
// AMD RDNA2 (GFX10.3) ISA DECODER & DISASSEMBLER
// =============================================================================

#[derive(Clone, Debug)]
pub struct DecodedInsn {
    pub offset: usize,
    pub len: usize,
    pub words: [u32; 4],
    pub nwords: usize,
    pub mnemonic: String,
    pub operands: String,
    pub kind: InsnKind,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum InsnKind {
    EndPgm,
    CodeEnd,
    WaitCnt(u16),
    AluF32 { op: &'static str, dst: u16, src0: String, src1: String },
    AluFma { dst: u16, src0: String, src1: String },
    AluUnary { op: &'static str, dst: u16, src0: String },
    Mov { dst: String, src: String },
    SmemLoad { op: &'static str, sdst: u16, sbase: u16, offset: u32, count: u8 },
    ImageOp { op: &'static str, vdata: u16, vaddr: u16, srsrc: u16, ssamp: u16, dmask: u8, is_rt: bool },
    Export { target: String, v: [u8; 4], en: u8, done: bool },
    Interp { op: &'static str, vdst: u16, attr: u8, chan: char },
    Branch { op: &'static str, rel: i16 },
    Other,
}

fn fmt_src(s: u32) -> String {
    match s {
        0..=105 => format!("s{s}"),
        106 => "vcc_lo".into(),
        107 => "vcc_hi".into(),
        124 => "m0".into(),
        125 => "null".into(),
        126 => "exec_lo".into(),
        127 => "exec_hi".into(),
        128..=192 => format!("{}", s - 128),
        193..=208 => format!("-{}", s - 192),
        240 => "0.5".into(),
        241 => "-0.5".into(),
        242 => "1.0".into(),
        243 => "-1.0".into(),
        244 => "2.0".into(),
        245 => "-2.0".into(),
        246 => "4.0".into(),
        247 => "-4.0".into(),
        248 => "0.15915494".into(),
        255 => "LIT".into(),
        256..=511 => format!("v{}", s - 256),
        _ => format!("src({s})"),
    }
}

fn fmt_lit_f32(w: u32) -> String {
    let f = f32::from_bits(w);
    if f.is_finite() && (f == 0.0 || (f.abs() >= 1e-5 && f.abs() <= 1e7)) {
        format!("{f:.6} /*0x{w:08x}*/")
    } else {
        format!("0x{w:08x}")
    }
}

pub fn gfx103_insn_len(buf: &[u8], i: usize) -> usize {
    if i + 4 > buf.len() {
        return buf.len().saturating_sub(i);
    }
    let w0 = rd32(buf, i);
    let enc9 = w0 >> 23;
    let enc6 = w0 >> 26;
    let enc7 = w0 >> 25;
    if enc9 == 0x17F {
        4 // SOPP
    } else if enc9 == 0x17D {
        if (w0 & 0xFF) == 255 { 8 } else { 4 } // SOP1
    } else if enc9 == 0x17E {
        if (w0 & 0xFF) == 255 || ((w0 >> 8) & 0xFF) == 255 { 8 } else { 4 } // SOPC
    } else if (w0 >> 28) == 0xB {
        let op = (w0 >> 23) & 0x1F;
        if op == 21 { 8 } else { 4 } // SOPK (s_setreg_imm32_b32 = 8B)
    } else if (w0 >> 30) == 2 {
        if (w0 & 0xFF) == 255 || ((w0 >> 8) & 0xFF) == 255 { 8 } else { 4 } // SOP2
    } else if enc6 == 0x3C {
        // MIMG: 8B + 4B * nsize (bits 2:1 of w0 in GFX10!)
        let nsize = ((w0 >> 1) & 0x3) as usize;
        8 + nsize * 4
    } else if matches!(enc6, 0x3D | 0x3E | 0x38 | 0x39 | 0x36 | 0x37 | 0x34 | 0x35) {
        // VOP3 / VOP3P can also carry a 32-bit literal if any src == 255 in GFX10!
        if matches!(enc6, 0x34 | 0x35) && i + 8 <= buf.len() {
            let w1 = rd32(buf, i + 4);
            let s0 = w1 & 0x1FF;
            let s1 = (w1 >> 9) & 0x1FF;
            let s2 = (w1 >> 18) & 0x1FF;
            if s0 == 255 || s1 == 255 || s2 == 255 {
                return 12;
            }
        }
        8
    } else if enc6 == 0x32 {
        4 // VINTRP
    } else if enc7 == 0x3F || enc7 == 0x3E {
        let src0 = w0 & 0x1FF;
        if src0 == 255 || src0 == 249 || src0 == 250 { 8 } else { 4 }
    } else if (w0 >> 31) == 0 {
        let op = (w0 >> 25) & 0x3F;
        let src0 = w0 & 0x1FF;
        if src0 == 255 || src0 == 249 || src0 == 250 || matches!(op, 44 | 45 | 55 | 56) {
            8
        } else {
            4
        }
    } else {
        4
    }
}

pub fn decode_gfx103(code: &[u8], max_insns: usize) -> Vec<DecodedInsn> {
    let body = skip_scarlett_prologue(code);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 4 <= body.len() && out.len() < max_insns {
        let len = gfx103_insn_len(body, i).min(body.len() - i);
        let nwords = (len / 4).clamp(1, 4);
        let mut words = [0u32; 4];
        for k in 0..nwords {
            words[k] = rd32(body, i + k * 4);
        }
        let w0 = words[0];
        let w1 = words[1];
        let enc9 = w0 >> 23;
        let enc6 = w0 >> 26;
        let enc7 = w0 >> 25;

        let (mn, ops, kind) = if enc9 == 0x17F {
            let op = (w0 >> 16) & 0x7F;
            let simm = (w0 & 0xFFFF) as u16;
            let rel = simm as i16;
            let mn = match op {
                0 => "s_nop",
                1 => "s_endpgm",
                2 => "s_branch",
                4 => "s_cbranch_scc0",
                5 => "s_cbranch_scc1",
                6 => "s_cbranch_vccz",
                7 => "s_cbranch_vccnz",
                8 => "s_cbranch_execz",
                9 => "s_cbranch_execnz",
                10 => "s_barrier",
                12 => "s_waitcnt",
                16 => "s_sendmsg",
                22 => "s_waitcnt_depctr",
                23 => "s_waitcnt_vscnt",
                24 => "s_waitcnt_vmcnt",
                25 => "s_waitcnt_expcnt",
                26 => "s_waitcnt_lgkmcnt",
                31 => "s_code_end",
                32 => "s_setprio",
                33 => "s_clause",
                _ => "sopp",
            };
            let ops = match op {
                1 | 10 | 31 => String::new(),
                2 | 4..=9 => format!("{:+} (0x{:04x})", rel as i32 * 4, (i as i32 + 4 + rel as i32 * 4).max(0)),
                12 => format!(
                    "vmcnt({}) expcnt({}) lgkmcnt({})",
                    (simm & 0xF) | ((simm >> 10) & 0x30),
                    (simm >> 4) & 0x7,
                    (simm >> 8) & 0x3F
                ),
                _ => format!("0x{simm:04x}"),
            };
            let kind = match op {
                1 => InsnKind::EndPgm,
                31 => InsnKind::CodeEnd,
                12 => InsnKind::WaitCnt(simm),
                2 | 4..=9 => InsnKind::Branch { op: mn, rel },
                _ => InsnKind::Other,
            };
            (mn.to_string(), ops, kind)
        } else if enc9 == 0x17D {
            let op = (w0 >> 8) & 0xFF;
            let sdst = (w0 >> 16) & 0x7F;
            let ssrc = w0 & 0xFF;
            let mn = match op {
                3 => "s_mov_b32",
                4 => "s_mov_b64",
                5 => "s_cmov_b32",
                7 => "s_not_b32",
                9 => "s_wqm_b32",
                10 => "s_wqm_b64",
                34 => "s_and_saveexec_b32",
                36 => "s_or_saveexec_b32",
                44 => "s_swappc_b64",
                46 => "s_setpc_b64",
                _ => "sop1",
            };
            let rhs = if ssrc == 255 { fmt_lit_f32(w1) } else { fmt_src(ssrc) };
            let ops = format!("{}, {rhs}", fmt_src(sdst));
            let kind = if op == 3 || op == 4 {
                InsnKind::Mov { dst: fmt_src(sdst), src: rhs }
            } else {
                InsnKind::Other
            };
            (mn.to_string(), ops, kind)
        } else if enc9 == 0x17E {
            let op = (w0 >> 16) & 0x7F;
            let s0 = w0 & 0xFF;
            let s1 = (w0 >> 8) & 0xFF;
            let mn = match op {
                0 => "s_cmp_eq_i32",
                1 => "s_cmp_lg_i32",
                2 => "s_cmp_gt_i32",
                3 => "s_cmp_ge_i32",
                4 => "s_cmp_lt_i32",
                5 => "s_cmp_le_i32",
                6 => "s_cmp_eq_u32",
                7 => "s_cmp_lg_u32",
                8 => "s_cmp_gt_u32",
                9 => "s_cmp_ge_u32",
                10 => "s_cmp_lt_u32",
                11 => "s_cmp_le_u32",
                _ => "sopc",
            };
            let a = if s0 == 255 { format!("0x{w1:08x}") } else { fmt_src(s0) };
            let b = if s1 == 255 { format!("0x{w1:08x}") } else { fmt_src(s1) };
            (mn.to_string(), format!("{a}, {b}"), InsnKind::Other)
        } else if (w0 >> 28) == 0xB {
            let op = (w0 >> 23) & 0x1F;
            let sdst = (w0 >> 16) & 0x7F;
            let simm = w0 & 0xFFFF;
            let mn = match op {
                0 => "s_movk_i32",
                15 => "s_addk_i32",
                16 => "s_mulk_i32",
                _ => "sopk",
            };
            let ops = format!("{}, 0x{simm:04x}", fmt_src(sdst));
            let kind = if op == 0 {
                InsnKind::Mov { dst: fmt_src(sdst), src: format!("{}", simm as i16) }
            } else {
                InsnKind::Other
            };
            (mn.to_string(), ops, kind)
        } else if (w0 >> 30) == 2 {
            let op = (w0 >> 23) & 0x7F;
            let sdst = (w0 >> 16) & 0x7F;
            let s0 = w0 & 0xFF;
            let s1 = (w0 >> 8) & 0xFF;
            let mn = match op {
                0 => "s_add_u32",
                1 => "s_sub_u32",
                2 => "s_add_i32",
                3 => "s_sub_i32",
                14 => "s_and_b32",
                15 => "s_and_b64",
                16 => "s_or_b32",
                18 => "s_xor_b32",
                28 => "s_lshl_b32",
                30 => "s_lshr_b32",
                32 => "s_ashr_i32",
                36 => "s_mul_i32",
                40 => "s_cselect_b32",
                46 => "s_lshl1_add_u32",
                47 => "s_lshl2_add_u32",
                48 => "s_lshl3_add_u32",
                49 => "s_lshl4_add_u32",
                _ => "sop2",
            };
            let a = if s0 == 255 { fmt_lit_f32(w1) } else { fmt_src(s0) };
            let b = if s1 == 255 { fmt_lit_f32(w1) } else { fmt_src(s1) };
            (mn.to_string(), format!("{}, {a}, {b}", fmt_src(sdst)), InsnKind::Other)
        } else if enc6 == 0x3D {
            let op = (w0 >> 18) & 0xFF;
            let sdata = ((w0 >> 6) & 0x7F) as u16;
            let sbase = ((w0 & 0x3F) * 2) as u16;
            let off = w1 & 0x1F_FFFF;
            let (mn, cnt) = match op {
                0 => ("s_load_dword", 1),
                1 => ("s_load_dwordx2", 2),
                2 => ("s_load_dwordx4", 4),
                3 => ("s_load_dwordx8", 8),
                4 => ("s_load_dwordx16", 16),
                8 => ("s_buffer_load_dword", 1),
                9 => ("s_buffer_load_dwordx2", 2),
                10 => ("s_buffer_load_dwordx4", 4),
                11 => ("s_buffer_load_dwordx8", 8),
                12 => ("s_buffer_load_dwordx16", 16),
                _ => ("smem_op", 1),
            };
            let ops = if cnt > 1 {
                format!("s[{}:{}], s[{}:{}], 0x{off:x}", sdata, sdata + cnt as u16 - 1, sbase, sbase + 1)
            } else {
                format!("s{sdata}, s[{}:{}], 0x{off:x}", sbase, sbase + 1)
            };
            (mn.to_string(), ops, InsnKind::SmemLoad { op: mn, sdst: sdata, sbase, offset: off, count: cnt })
        } else if enc6 == 0x3E {
            let tgt = (w0 >> 4) & 0x3F;
            let en = (w0 & 0xF) as u8;
            let compr = ((w0 >> 10) & 1) != 0;
            let done = ((w0 >> 11) & 1) != 0;
            let v = [
                (w1 & 0xFF) as u8,
                ((w1 >> 8) & 0xFF) as u8,
                ((w1 >> 16) & 0xFF) as u8,
                ((w1 >> 24) & 0xFF) as u8,
            ];
            let tnm = match tgt {
                0..=7 => format!("mrt{tgt}"),
                8 => "mrtz".into(),
                9 => "null".into(),
                12..=15 => format!("pos{}", tgt - 12),
                20 => "prim".into(),
                32..=63 => format!("param{}", tgt - 32),
                _ => format!("tgt{tgt}"),
            };
            let ops = format!(
                "{tnm} v{}, v{}, v{}, v{} en=0x{en:x}{}{}",
                v[0], v[1], v[2], v[3],
                if compr { " compr" } else { "" },
                if done { " done" } else { "" }
            );
            ("exp".into(), ops, InsnKind::Export { target: tnm, v, en, done })
        } else if enc6 == 0x3C {
            let op = (w0 >> 18) & 0xFF;
            let dmask = ((w0 >> 8) & 0xF) as u8;
            let vaddr = (w1 & 0xFF) as u16;
            let vdata = ((w1 >> 8) & 0xFF) as u16;
            let srsrc = (((w1 >> 16) & 0x1F) * 4) as u16;
            let ssamp = (((w1 >> 21) & 0x1F) * 4) as u16;
            let (mn, is_rt) = match op {
                0 => ("image_load", false),
                1 => ("image_load_mip", false),
                8 => ("image_store", false),
                9 => ("image_store_mip", false),
                14 => ("image_get_resinfo", false),
                32 => ("image_sample", false),
                33 => ("image_sample_cl", false),
                34 => ("image_sample_d", false),
                36 => ("image_sample_l", false),
                37 => ("image_sample_b", false),
                39 => ("image_sample_lz", false),
                40 => ("image_sample_c", false),
                44 => ("image_sample_c_l", false),
                47 => ("image_sample_c_lz", false),
                64..=79 => ("image_gather4", false),
                0x66 => ("image_bvh_intersect_ray", true),
                0x67 => ("image_bvh64_intersect_ray", true),
                _ => ("image_op", false),
            };
            let nsa = if nwords > 2 { format!(" nsa=[v{}..]", words[2] & 0xFF) } else { String::new() };
            let ops = format!(
                "v{vdata}, v{vaddr}{nsa}, s[{}:{}], s[{}:{}] dmask=0x{dmask:x}",
                srsrc, srsrc + 7, ssamp, ssamp + 3
            );
            (mn.to_string(), ops, InsnKind::ImageOp { op: mn, vdata, vaddr, srsrc, ssamp, dmask, is_rt })
        } else if enc6 == 0x32 {
            let op = (w0 >> 16) & 3;
            let vdst = ((w0 >> 18) & 0xFF) as u16;
            let attr = ((w0 >> 10) & 0x3F) as u8;
            let chan = ['x', 'y', 'z', 'w'][((w0 >> 8) & 3) as usize];
            let vsrc = w0 & 0xFF;
            let mn = match op {
                0 => "v_interp_p1_f32",
                1 => "v_interp_p2_f32",
                2 => "v_interp_mov_f32",
                _ => "v_interp",
            };
            (mn.to_string(), format!("v{vdst}, v{vsrc}, attr{attr}.{chan}"), InsnKind::Interp { op: mn, vdst, attr, chan })
        } else if enc7 == 0x3F {
            let op = (w0 >> 9) & 0xFF;
            let vdst = ((w0 >> 17) & 0xFF) as u16;
            let src0 = w0 & 0x1FF;
            let s0 = if src0 == 255 { fmt_lit_f32(w1) } else { fmt_src(src0) };
            let mn = match op {
                1 => "v_mov_b32",
                2 => "v_readfirstlane_b32",
                5 => "v_cvt_f32_i32",
                6 => "v_cvt_f32_u32",
                7 => "v_cvt_u32_f32",
                8 => "v_cvt_i32_f32",
                10 => "v_cvt_f16_f32",
                11 => "v_cvt_f32_f16",
                32 => "v_fract_f32",
                33 => "v_trunc_f32",
                34 => "v_ceil_f32",
                35 => "v_rndne_f32",
                36 => "v_floor_f32",
                37 => "v_exp_f32",
                39 => "v_log_f32",
                42 => "v_rcp_f32",
                46 => "v_rsq_f32",
                51 => "v_sqrt_f32",
                53 => "v_sin_f32",
                54 => "v_cos_f32",
                55 => "v_not_b32",
                _ => "vop1",
            };
            let kind = if op == 1 {
                InsnKind::Mov { dst: format!("v{vdst}"), src: s0.clone() }
            } else {
                InsnKind::AluUnary { op: mn, dst: vdst, src0: s0.clone() }
            };
            (mn.to_string(), format!("v{vdst}, {s0}"), kind)
        } else if enc7 == 0x3E {
            let op = (w0 >> 17) & 0xFF;
            let src0 = w0 & 0x1FF;
            let vsrc1 = (w0 >> 9) & 0xFF;
            let s0 = if src0 == 255 { fmt_lit_f32(w1) } else { fmt_src(src0) };
            (format!("v_cmp_op{op}"), format!("vcc_lo, {s0}, v{vsrc1}"), InsnKind::Other)
        } else if (w0 >> 31) == 0 {
            let op = (w0 >> 25) & 0x3F;
            let vdst = ((w0 >> 17) & 0xFF) as u16;
            let vsrc1 = (w0 >> 9) & 0xFF;
            let src0 = w0 & 0x1FF;
            let s0 = if src0 == 255 { fmt_lit_f32(w1) } else { fmt_src(src0) };
            let s1 = format!("v{vsrc1}");
            let mn = match op {
                0 => "v_cndmask_b32",
                1 => "v_dot2c_f32_f16",
                3 => "v_add_f32",
                4 => "v_sub_f32",
                5 => "v_subrev_f32",
                6 => "v_fmac_f32_legacy",
                7 => "v_mul_legacy_f32",
                8 => "v_mul_f32",
                9 => "v_mul_i32_i24",
                11 => "v_mul_u32_u24",
                15 => "v_min_f32",
                16 => "v_max_f32",
                17 => "v_min_i32",
                18 => "v_max_i32",
                19 => "v_min_u32",
                20 => "v_max_u32",
                22 => "v_lshrrev_b32",
                24 => "v_ashrrev_i32",
                26 => "v_lshlrev_b32",
                27 => "v_and_b32",
                28 => "v_or_b32",
                29 => "v_xor_b32",
                37 => "v_add_nc_u32",
                38 => "v_sub_nc_u32",
                39 => "v_subrev_nc_u32",
                43 => "v_fmac_f32",
                44 => "v_fmamk_f32",
                45 => "v_fmaak_f32",
                50 => "v_add_f16",
                53 => "v_mul_f16",
                59 => "v_fmac_f16",
                _ => "vop2",
            };
            let kind = match op {
                43 | 6 => InsnKind::AluFma { dst: vdst, src0: s0.clone(), src1: s1.clone() },
                3 | 4 | 5 | 8 | 15 | 16 | 27 | 28 | 29 | 37 | 38 => {
                    InsnKind::AluF32 { op: mn, dst: vdst, src0: s0.clone(), src1: s1.clone() }
                }
                _ => InsnKind::Other,
            };
            let ops = if matches!(op, 44 | 45) {
                format!("v{vdst}, {s0}, {s1}, {}", fmt_lit_f32(w1))
            } else {
                format!("v{vdst}, {s0}, {s1}")
            };
            (mn.to_string(), ops, kind)
        } else if matches!(enc6, 0x34..=0x39) {
            let fam = match enc6 {
                0x35 => "vop3",
                0x34 => "vop3p",
                0x38 => "buffer",
                0x39 => "tbuffer",
                0x36 => "ds",
                _ => "flat",
            };
            let op = (w0 >> 16) & 0x3FF;
            let vdst = w0 & 0xFF;
            let s0 = fmt_src(w1 & 0x1FF);
            let s1 = fmt_src((w1 >> 9) & 0x1FF);
            let s2 = fmt_src((w1 >> 18) & 0x1FF);
            let mn = match (enc6, op) {
                (0x35, 0x14B) => "v_fma_f32".into(),
                (0x35, 0x16B) => "v_med3_f32".into(),
                (0x35, 0x164) => "v_bfe_u32".into(),
                (0x35, 0x165) => "v_bfe_i32".into(),
                (0x35, 0x144) => "v_mad_u32_u24".into(),
                (0x35, 0x146) => "v_cubeid_f32".into(),
                (0x35, 0x2FF) => "v_lshl_add_u32".into(),
                (0x35, 0x258) => "v_and_or_b32".into(),
                _ => format!("{fam}_0x{op:03x}"),
            };
            (mn, format!("v{vdst}, {s0}, {s1}, {s2}"), InsnKind::Other)
        } else {
            (format!("unk_0x{enc6:02x}"), format!("0x{w0:08x}"), InsnKind::Other)
        };

        let stop = matches!(kind, InsnKind::EndPgm | InsnKind::CodeEnd);
        out.push(DecodedInsn {
            offset: i,
            len,
            words,
            nwords,
            mnemonic: mn,
            operands: ops,
            kind,
        });
        i += len;
        if stop {
            break;
        }
    }
    out
}

pub fn format_gfx103_disasm(entry: &Ps09ShaderEntry, code: &[u8]) -> String {
    let insns = decode_gfx103(code, 4000);
    let mut s = String::with_capacity(insns.len() * 48 + 512);
    let _ = writeln!(
        s,
        "; =============================================================================\n\
         ; PercyStar AMD RDNA2 (GFX10.3 / Xbox Scarlett) Disassembly\n\
         ; Shader #{idx} | Stage: {st} | PSO Hash: 0x{h:016X} | Size: {sz} bytes ({ic} insns)",
        idx = entry.index,
        st = entry.stage.tag(),
        h = entry.sga_hash.unwrap_or(entry.pso_hash),
        sz = entry.code_size,
        ic = insns.len()
    );
    if let Some([x, y, z]) = entry.num_threads {
        let _ = writeln!(s, "; Compute Threadgroup: [numthreads({x}, {y}, {z})]");
    }
    if let (Some(r1), Some(r2)) = (entry.rsrc1, entry.rsrc2) {
        let _ = writeln!(
            s,
            "; COMPUTE_PGM_RSRC1=0x{r1:08X}  COMPUTE_PGM_RSRC2=0x{r2:08X} (user_sgprs={})",
            (r2 >> 1) & 0x1F
        );
    }
    if !entry.bindings.is_empty() {
        let _ = writeln!(s, "; PS\\x00\\x09 0x41 USER_DATA Bindings ({}):", entry.bindings.len());
        for b in &entry.bindings {
            let _ = writeln!(
                s,
                ";   {:<10} table_off=0x{:03X}  slot=0x{:03X}  dwords={}  sgpr_flags=0x{:08X}",
                b.kind, b.byte_offset, b.slot_id, b.dwords, b.sgpr_flags
            );
        }
    }
    if !entry.resource_names.is_empty() {
        let resolved = entry
            .resource_names
            .iter()
            .filter(|resource| resource.name.is_some())
            .count();
        let _ = writeln!(
            s,
            "; SGD5 Bindings ({} named / {}):",
            resolved,
            entry.resource_names.len()
        );
        for resource in &entry.resource_names {
            match &resource.name {
                Some(name) => {
                    let _ = writeln!(
                        s,
                        ";   {:>10} {} {}",
                        resource.register, resource.declaration, name
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        ";   {:>10} {} hash=0x{:08x}",
                        resource.register, resource.declaration, resource.hash
                    );
                }
            }
        }
    }
    let _ = writeln!(s, "; =============================================================================\n");
    for ins in &insns {
        let mut hx = String::new();
        for k in 0..ins.nwords {
            let _ = write!(hx, "{:08x} ", ins.words[k]);
        }
        let _ = writeln!(s, "  +{:04x}: {:<28} {:<26} {}", ins.offset, hx, ins.mnemonic, ins.operands);
    }
    s
}

// =============================================================================
// GFX10.3 -> READABLE HLSL (SM 6.6) LIFTER
// =============================================================================

pub fn lift_to_hlsl(entry: &Ps09ShaderEntry, code: &[u8]) -> String {
    let insns = decode_gfx103(code, 4000);
    let mut s = String::with_capacity(insns.len() * 44 + 1024);
    let _ = writeln!(s, "// ============================================================================");
    let _ = writeln!(s, "// PercyStar Clean-Room GFX10.3 (RDNA2) -> HLSL SM 6.6 Lifted Shader");
    let hash_kind = if entry.sga_hash.is_some() { "SGA" } else { "PSO" };
    let _ = writeln!(s, "// Shader #{} | Stage: {} | {hash_kind} 0x{:016X}", entry.index, entry.stage.tag(), entry.sga_hash.unwrap_or(entry.pso_hash));
    if entry.has_rt {
        let _ = writeln!(s, "// Uses RDNA2 Hardware Raytracing (image_bvh64_intersect_ray -> RayQuery)");
    }
    if entry.has_ngg {
        let _ = writeln!(s, "// Uses RDNA2 NGG / Meshlet Primitive Exports (exp prim)");
    }
    let _ = writeln!(s, "// ============================================================================\n");

    // Emit reconstructed resource bindings from PS\x00\x09 0x41 records
    let mut cb_n = 0u32;
    let mut srv_n = 0u32;
    let mut uav_n = 0u32;
    let mut smp_n = 0u32;
    for b in &entry.bindings {
        match b.kind {
            "CBV" => {
                let _ = writeln!(s, "cbuffer CB_{cb_n} : register(b{cb_n}) {{ uint4 cb{cb_n}_data[64]; }}; // slot=0x{:X} off=0x{:X}", b.slot_id, b.byte_offset);
                cb_n += 1;
            }
            "SRV" => {
                let _ = writeln!(s, "Texture2D<float4> Tex_{srv_n} : register(t{srv_n}); // slot=0x{:X} off=0x{:X} dw={}", b.slot_id, b.byte_offset, b.dwords);
                srv_n += 1;
            }
            "UAV" => {
                let _ = writeln!(s, "RWTexture2D<float4> Out_{uav_n} : register(u{uav_n}); // slot=0x{:X} off=0x{:X}", b.slot_id, b.byte_offset);
                uav_n += 1;
            }
            "SAMPLER" => {
                let _ = writeln!(s, "SamplerState Samp_{smp_n} : register(s{smp_n}); // slot=0x{:X}", b.slot_id);
                smp_n += 1;
            }
            _ => {}
        }
    }
    if !entry.bindings.is_empty() {
        let _ = writeln!(s);
    }
    let named_resources: Vec<_> = entry
        .resource_names
        .iter()
        .filter(|resource| resource.name.is_some())
        .collect();
    if !named_resources.is_empty() {
        for resource in named_resources {
            let name = resource.name.as_deref().unwrap_or_default();
            let _ = writeln!(
                s,
                "{} {} : register({});",
                resource.declaration, name, resource.register
            );
        }
        let _ = writeln!(s);
    }

    match entry.stage {
        ShaderStage::Compute | ShaderStage::RayTracing => {
            let [x, y, z] = entry.num_threads.unwrap_or([8, 8, 1]);
            let _ = writeln!(s, "[numthreads({x}, {y}, {z})]");
            let _ = writeln!(s, "void CSMain(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)");
        }
        ShaderStage::Pixel => {
            let _ = writeln!(s, "void PSMain(float4 pos : SV_Position, out float4 mrt0 : SV_Target0)");
        }
        ShaderStage::Vertex => {
            let _ = writeln!(s, "void VSMain(uint vid : SV_VertexID, out float4 pos : SV_Position)");
        }
        ShaderStage::Geometry => {
            let _ = writeln!(s, "[maxvertexcount(3)]\nvoid GSMain(triangle float4 input[3] : SV_Position, inout TriangleStream<float4> output)");
        }
        ShaderStage::Domain => {
            let _ = writeln!(s, r#"[domain("tri")]\nvoid DSMain(float3 bary : SV_DomainLocation, out float4 pos : SV_Position)"#);
        }
        ShaderStage::Hull => {
            let _ = writeln!(s, r#"[domain("tri")][outputcontrolpoints(3)]\nvoid HSMain(InputPatch<float4, 3> input, uint id : SV_OutputControlPointID)"#);
        }
        ShaderStage::MeshNgg => {
            let _ = writeln!(s, "[outputtopology(\"triangle\")]\n[numthreads(64, 1, 1)]\nvoid MSMain(uint gtid : SV_GroupThreadID)");
        }
        ShaderStage::Unknown => {
            let _ = writeln!(s, "void main()");
        }
    }
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "    float v[64]; uint s[64]; // RDNA2 Wave32 VGPR / SGPR register file");

    for ins in &insns {
        match &ins.kind {
            InsnKind::EndPgm => {
                let _ = writeln!(s, "    return; // +0x{:04x} s_endpgm", ins.offset);
            }
            InsnKind::CodeEnd | InsnKind::WaitCnt(_) => {}
            InsnKind::Mov { dst, src } => {
                let _ = writeln!(s, "    {dst} = {src};");
            }
            InsnKind::AluF32 { op, dst, src0, src1 } => {
                let expr = match *op {
                    "v_add_f32" | "v_add_nc_u32" => format!("{src0} + {src1}"),
                    "v_sub_f32" | "v_sub_nc_u32" => format!("{src0} - {src1}"),
                    "v_subrev_f32" => format!("{src1} - {src0}"),
                    "v_mul_f32" => format!("{src0} * {src1}"),
                    "v_min_f32" => format!("min({src0}, {src1})"),
                    "v_max_f32" => format!("max({src0}, {src1})"),
                    "v_and_b32" => format!("({src0} & {src1})"),
                    "v_or_b32" => format!("({src0} | {src1})"),
                    "v_xor_b32" => format!("({src0} ^ {src1})"),
                    _ => format!("{op}({src0}, {src1})"),
                };
                let _ = writeln!(s, "    v{dst} = {expr};");
            }
            InsnKind::AluFma { dst, src0, src1 } => {
                let _ = writeln!(s, "    v{dst} = mad({src0}, {src1}, v{dst});");
            }
            InsnKind::AluUnary { op, dst, src0 } => {
                let fn_nm = match *op {
                    "v_rcp_f32" => "rcp",
                    "v_rsq_f32" => "rsqrt",
                    "v_sqrt_f32" => "sqrt",
                    "v_exp_f32" => "exp2",
                    "v_log_f32" => "log2",
                    "v_sin_f32" => "sin",
                    "v_cos_f32" => "cos",
                    "v_floor_f32" => "floor",
                    "v_ceil_f32" => "ceil",
                    "v_fract_f32" => "frac",
                    "v_trunc_f32" => "trunc",
                    "v_cvt_f32_u32" => "(float)(uint)",
                    "v_cvt_f32_i32" => "(float)(int)",
                    "v_cvt_u32_f32" => "(uint)",
                    _ => *op,
                };
                let _ = writeln!(s, "    v{dst} = {fn_nm}({src0});");
            }
            InsnKind::SmemLoad { op, sdst, sbase, offset, count } => {
                let _ = writeln!(
                    s,
                    "    // {op}: s[{sdst}..{}] = LoadCB(s[{sbase}:{}], 0x{offset:x});",
                    sdst + (*count as u16).saturating_sub(1),
                    sbase + 1
                );
            }
            InsnKind::ImageOp { op, vdata, vaddr, srsrc, ssamp, dmask, is_rt } => {
                if *is_rt {
                    let _ = writeln!(
                        s,
                        "    // [HW RAYTRACING] v[{vdata}..{}+3] = RayQuery::BVH64Intersect(t_bvh=s[{srsrc}..], ray=v{vaddr}); // {op}",
                        vdata
                    );
                } else if *op == "image_store" || *op == "image_store_mip" {
                    let _ = writeln!(
                        s,
                        "    Out_s{srsrc}[uint2(v{vaddr}, v{})] = float4(v{vdata}, v{}, v{}, v{}); // {op}",
                        vaddr + 1, vdata + 1, vdata + 2, vdata + 3
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "    v[{vdata}..{}] = Tex_s{srsrc}.SampleLevel(Samp_s{ssamp}, float2(v{vaddr}, v{}), 0); // {op} dmask=0x{dmask:x}",
                        vdata + 3, vaddr + 1
                    );
                }
            }
            InsnKind::Interp { vdst, attr, chan, .. } => {
                let _ = writeln!(s, "    v{vdst} = interp_attr{attr}.{chan};");
            }
            InsnKind::Export { target, v, done, .. } => {
                let _ = writeln!(
                    s,
                    "    {target} = float4(v{}, v{}, v{}, v{});{}",
                    v[0], v[1], v[2], v[3],
                    if *done { " // done" } else { "" }
                );
            }
            InsnKind::Branch { op, rel } => {
                let _ = writeln!(s, "    // {op} -> +0x{:04x}", (ins.offset as i32 + 4 + (*rel as i32) * 4).max(0));
            }
            InsnKind::Other => {
                let _ = writeln!(s, "    // +0x{:04x}: {} {}", ins.offset, ins.mnemonic, ins.operands);
            }
        }
    }
    let _ = writeln!(s, "}}");
    s
}

// =============================================================================
// GFX10.3 -> VALID BINARY SPIR-V 1.5 EMITTER (Vulkan / SPIRV-Cross Ready)
// =============================================================================

fn push_spv_op(words: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let wc = (1 + operands.len()) as u32;
    words.push((wc << 16) | (opcode as u32));
    words.extend_from_slice(operands);
}

fn push_spv_str(words: &mut Vec<u32>, opcode: u16, prefix: &[u32], text: &str, suffix: &[u32]) {
    let bytes = text.as_bytes();
    let str_words = (bytes.len() + 1).div_ceil(4);
    let wc = (1 + prefix.len() + str_words + suffix.len()) as u32;
    words.push((wc << 16) | (opcode as u32));
    words.extend_from_slice(prefix);
    for wi in 0..str_words {
        let mut w = 0u32;
        for b in 0..4 {
            let idx = wi * 4 + b;
            let ch = if idx < bytes.len() { bytes[idx] } else { 0 };
            w |= (ch as u32) << (b * 8);
        }
        words.push(w);
    }
    words.extend_from_slice(suffix);
}

/// Emits a valid SPIR-V 1.5 module (`0x07230203` magic, GLSL.std.450, ExecutionModel, SSA ALU ops)
/// lifting the GFX10.3 VGPR ALU/FMA/Export skeleton with full `OpName` debug metadata.
pub fn lift_to_spirv(entry: &Ps09ShaderEntry, code: &[u8]) -> Vec<u8> {
    let insns = decode_gfx103(code, 2000);
    let mut body: Vec<u32> = Vec::with_capacity(512);
    let mut next_id = 1u32;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let id_ext_glsl = alloc();
    let id_void = alloc();
    let id_fn_ty = alloc();
    let id_f32 = alloc();
    let id_const_0 = alloc();
    let id_const_1 = alloc();
    let id_main = alloc();
    let id_entry_lbl = alloc();

    // Pre-build SSA instructions inside main() so we know exact ID bound
    let mut fn_ops: Vec<u32> = Vec::new();
    let mut vreg_id: HashMap<u16, u32> = HashMap::new();
    let get_v = |r: u16, map: &mut HashMap<u16, u32>| -> u32 {
        *map.get(&r).unwrap_or(&id_const_0)
    };
    let resolve = |tok: &str, map: &mut HashMap<u16, u32>| -> u32 {
        if let Some(rest) = tok.strip_prefix('v') {
            if let Ok(r) = rest.parse::<u16>() {
                return *map.get(&r).unwrap_or(&id_const_0);
            }
        }
        if tok == "1.0" { id_const_1 } else { id_const_0 }
    };

    let mut lifted_alu = 0u32;
    for ins in &insns {
        match &ins.kind {
            InsnKind::AluF32 { op, dst, src0, src1 } => {
                let a = resolve(src0, &mut vreg_id);
                let b = resolve(src1, &mut vreg_id);
                let spv_op = match *op {
                    "v_add_f32" => Some(129u16), // OpFAdd
                    "v_sub_f32" => Some(131u16), // OpFSub
                    "v_subrev_f32" => Some(131u16),
                    "v_mul_f32" => Some(133u16), // OpFMul
                    _ => None,
                };
                if let Some(code_op) = spv_op {
                    let res = alloc();
                    if *op == "v_subrev_f32" {
                        push_spv_op(&mut fn_ops, code_op, &[id_f32, res, b, a]);
                    } else {
                        push_spv_op(&mut fn_ops, code_op, &[id_f32, res, a, b]);
                    }
                    vreg_id.insert(*dst, res);
                    lifted_alu += 1;
                }
            }
            InsnKind::AluFma { dst, src0, src1 } => {
                let a = resolve(src0, &mut vreg_id);
                let b = resolve(src1, &mut vreg_id);
                let c = get_v(*dst, &mut vreg_id);
                let prod = alloc();
                let sum = alloc();
                push_spv_op(&mut fn_ops, 133, &[id_f32, prod, a, b]); // OpFMul
                push_spv_op(&mut fn_ops, 129, &[id_f32, sum, prod, c]); // OpFAdd
                vreg_id.insert(*dst, sum);
                lifted_alu += 1;
            }
            InsnKind::AluUnary { op, dst, src0 } => {
                let a = resolve(src0, &mut vreg_id);
                if *op == "v_rcp_f32" {
                    let res = alloc();
                    push_spv_op(&mut fn_ops, 136, &[id_f32, res, id_const_1, a]); // OpFDiv(1.0, a)
                    vreg_id.insert(*dst, res);
                    lifted_alu += 1;
                } else if *op == "v_sqrt_f32" {
                    let res = alloc();
                    push_spv_op(&mut fn_ops, 12, &[id_f32, res, id_ext_glsl, 31, a]); // OpExtInst Sqrt
                    vreg_id.insert(*dst, res);
                    lifted_alu += 1;
                }
            }
            _ => {}
        }
    }

    // Header (5 words)
    body.push(0x0723_0203); // SPIR-V magic
    body.push(0x0001_0500); // Version 1.5
    body.push(0x5053_5452); // Generator "PSTR" (PercyStar)
    body.push(next_id);     // Bound
    body.push(0);           // Schema

    // Capabilities & Memory Model
    push_spv_op(&mut body, 17, &[1]); // OpCapability Shader
    push_spv_str(&mut body, 11, &[id_ext_glsl], "GLSL.std.450", &[]); // OpExtInstImport
    push_spv_op(&mut body, 14, &[0, 1]); // OpMemoryModel Logical GLSL450

    let exec_model = match entry.stage {
        ShaderStage::Pixel => 4u32,  // Fragment
        ShaderStage::Vertex => 0u32, // Vertex
        ShaderStage::Geometry => 3u32,
        ShaderStage::Domain => 2u32,
        ShaderStage::Hull => 1u32,
        _ => 5u32,                   // GLCompute
    };
    push_spv_str(&mut body, 15, &[exec_model, id_main], "main", &[]); // OpEntryPoint
    if exec_model == 5 {
        let [x, y, z] = entry.num_threads.unwrap_or([8, 8, 1]);
        push_spv_op(&mut body, 16, &[id_main, 17, x as u32, y as u32, z as u32]); // LocalSize
    } else if exec_model == 4 {
        push_spv_op(&mut body, 16, &[id_main, 7]); // OriginUpperLeft
    }

    // Debug metadata (OpName)
    let tag = format!("percystar_sh{}_{}_{}alu", entry.index, entry.stage.tag(), lifted_alu);
    push_spv_str(&mut body, 5, &[id_main], &tag, &[]);

    // Types & Constants
    push_spv_op(&mut body, 19, &[id_void]); // OpTypeVoid
    push_spv_op(&mut body, 33, &[id_fn_ty, id_void]); // OpTypeFunction
    push_spv_op(&mut body, 22, &[id_f32, 32]); // OpTypeFloat 32
    push_spv_op(&mut body, 43, &[id_f32, id_const_0, 0.0f32.to_bits()]); // OpConstant 0.0
    push_spv_op(&mut body, 43, &[id_f32, id_const_1, 1.0f32.to_bits()]); // OpConstant 1.0

    // Function
    push_spv_op(&mut body, 54, &[id_void, id_main, 0, id_fn_ty]); // OpFunction
    push_spv_op(&mut body, 248, &[id_entry_lbl]); // OpLabel
    body.extend_from_slice(&fn_ops);
    push_spv_op(&mut body, 253, &[]); // OpReturn
    push_spv_op(&mut body, 56, &[]);  // OpFunctionEnd

    let mut bytes = Vec::with_capacity(body.len() * 4);
    for w in body {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

pub fn format_spirv_abi_summary(entry: &Ps09ShaderEntry, code: &[u8]) -> String {
    let spv = lift_to_spirv(entry, code);
    let words = spv.len() / 4;
    let mut s = String::new();
    let _ = writeln!(s, "=== SPIR-V 1.5 & D3D12.X / RDNA2 SGPR ABI SUMMARY ===");
    let _ = writeln!(s, "Shader Index      : #{}", entry.index);
    let _ = writeln!(s, "Detected Stage    : {} (has_rt={}, has_ngg={})", entry.stage.tag(), entry.has_rt, entry.has_ngg);
    let hash_kind = if entry.sga_hash.is_some() { "SGA Hash" } else { "PSO Hash" };
    let _ = writeln!(s, "{hash_kind:<17}: 0x{:016X}", entry.sga_hash.unwrap_or(entry.pso_hash));
    let _ = writeln!(s, "Code Offset/Size  : 0x{:08X} / {} bytes ({} GFX10.3 insns)", entry.code_offset, entry.code_size, entry.insn_count);
    let _ = writeln!(s, "Lifted SPIR-V 1.5 : {} bytes ({} words, magic=0x07230203)", spv.len(), words);
    if let Some([x, y, z]) = entry.num_threads {
        let _ = writeln!(s, "Compute LocalSize : ({x}, {y}, {z})");
    }
    if let (Some(r1), Some(r2)) = (entry.rsrc1, entry.rsrc2) {
        let _ = writeln!(s, "COMPUTE_PGM_RSRC1 : 0x{r1:08X} (vgprs_blocks={}, sgprs_blocks={})", r1 & 0x3F, (r1 >> 6) & 0xF);
        let _ = writeln!(s, "COMPUTE_PGM_RSRC2 : 0x{r2:08X} (user_sgpr={}, tgid_x={}, tgid_y={}, tgid_z={})",
            (r2 >> 1) & 0x1F, (r2 >> 7) & 1, (r2 >> 8) & 1, (r2 >> 9) & 1);
    }
    if !entry.resource_names.is_empty() {
        let resolved = entry
            .resource_names
            .iter()
            .filter(|resource| resource.name.is_some())
            .count();
        let _ = writeln!(
            s,
            "\n--- SGD5 Resource Bindings ({} named / {}) ---",
            resolved,
            entry.resource_names.len()
        );
        for resource in &entry.resource_names {
            match &resource.name {
                Some(name) => {
                    let _ = writeln!(
                        s,
                        "  {:<10} {} {}",
                        resource.register, resource.declaration, name
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "  {:<10} {} hash=0x{:08x}",
                        resource.register, resource.declaration, resource.hash
                    );
                }
            }
        }
    }
    let _ = writeln!(s, "\n--- PS\\x00\\x09 0x41 USER_DATA Root-Signature Bindings ({}) ---", entry.bindings.len());
    if entry.bindings.is_empty() {
        let _ = writeln!(s, "  (No 0x41 records linked to this stage — uses inline root constants or bindless heap)");
    } else {
        for (i, b) in entry.bindings.iter().enumerate() {
            let _ = writeln!(
                s,
                "  [{i:02}] {:<10} raw=0x{:04X}  table_off=0x{:03X}  slot=0x{:03X}  dwords={}  sgpr_flags=0x{:08X}",
                b.kind, b.kind_raw, b.byte_offset, b.slot_id, b.dwords, b.sgpr_flags
            );
        }
    }
    let _ = writeln!(s, "\n--- Mesa RADV (gfx1030) / shadPS4 Replay Notes ---");
    let _ = writeln!(s, "  • ISA matches AMD GFX10.3 (RX 6000 / RDNA2) 1:1 — standard Wave32, 12B NSA MIMG, s_code_end padding.");
    let _ = writeln!(s, "  • Target Vulkan SPIR-V 1.5+ or DXIL SM 6.6+ (legacy DXBC SM 5.1 lacks Wave32, NGG Mesh, & BVH64 RT).");
    let _ = writeln!(s, "  • Click 'Export .spv' or 'Export .hlsl' above to save lifted shader.");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_base_search_survives_offsets_near_the_address_space_end() {
        // expected plus the probe deltas used to wrap and index far out of bounds
        let data = vec![0u8; 128];
        for expected in [0usize, 0xFFE4, 0x1_FFE4, usize::MAX - 8, usize::MAX] {
            let _ = locate_code_base(&data, 0, 0, expected);
        }
        assert_eq!(rd32(&data, usize::MAX - 2), 0);
        assert_eq!(rd64(&data, usize::MAX - 6), 0);
        assert_eq!(rd32(&[1, 0, 0, 0], 0), 1);
    }

    #[test]
    fn decodes_and_lifts_real_gfx103_tent_filter_snippet() {
        // Real GFX10.3 dwords observed in GTA VI shaders.rpf:
        // v_rcp_f32 v5, s17; v_add_f32 v3, 0.5, v3; v_mul_f32 v4, v4, v2;
        // image_sample_l (12B NSA); v_fmac_f32 v8, s11, v24; exp mrt0; s_endpgm
        let words: &[u32] = &[
            0x7e0a5411, // v_rcp_f32 v5, s17
            0x060606f0, // v_add_f32 v3, 0.5, v3
            0x10080504, // v_mul_f32 v4, v4, v2
            0xf0800f0a, 0x00461004, 0x00000003, // MIMG 12B NSA
            0x5610300b, // v_fmac_f32 v8, s11, v24
            0xf800080f, 0x03020100, // exp mrt0 v0,v1,v2,v3 done
            0xbf810000, // s_endpgm
        ];
        let mut bytes = Vec::new();
        for &w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let insns = decode_gfx103(&bytes, 64);
        assert_eq!(insns.len(), 7);
        assert_eq!(insns[0].mnemonic, "v_rcp_f32");
        assert_eq!(insns[1].mnemonic, "v_add_f32");
        assert_eq!(insns[2].mnemonic, "v_mul_f32");
        assert_eq!(insns[3].len, 12);
        assert_eq!(insns[4].mnemonic, "v_fmac_f32");
        assert_eq!(insns[5].mnemonic, "exp");
        assert_eq!(insns[6].mnemonic, "s_endpgm");

        let entry = Ps09ShaderEntry {
            index: 0,
            code_offset: 0,
            code_size: bytes.len() as u32,
            stage: ShaderStage::Pixel,
            num_threads: None,
            rsrc1: None,
            rsrc2: None,
            bindings: vec![],
            resource_names: vec![],
            pso_hash: 0xDEAD_BEEF,
            sga_hash: None,
            has_rt: false,
            has_ngg: false,
            insn_count: 7,
        };
        let hlsl = lift_to_hlsl(&entry, &bytes);
        assert!(hlsl.contains("rcp("));
        assert!(hlsl.contains("mad("));
        assert!(hlsl.contains("mrt0 = float4("));

        let spv = lift_to_spirv(&entry, &bytes);
        assert!(spv.len() >= 20 && spv.len().is_multiple_of(4));
        assert_eq!(u32::from_le_bytes([spv[0], spv[1], spv[2], spv[3]]), 0x0723_0203);
    }
}

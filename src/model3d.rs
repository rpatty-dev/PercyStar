use eframe::egui::{self, Color32, ColorImage, CornerRadius, RichText, TextureHandle, TextureOptions};
use flate2::{Decompress, FlushDecompress, Status};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ShadeMode {
    #[default]
    Textured,
    Solid,
    SolidWire,
    Wireframe,
}

pub struct Texture2D {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<Color32>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum MaterialPass {
    #[default]
    Opaque,
    Cutout,
    Alpha,
    Decal,
    Glass { texture_alpha: bool },
}

#[derive(Clone, Default)]
pub struct GeomMaterial {
    pub texture: Option<Arc<Texture2D>>,
    pub is_vehicle_paint: bool,
    pass: MaterialPass,
    auxiliary_pass: bool,
}

impl GeomMaterial {
    fn blends(&self, has_uvs: bool) -> bool {
        matches!(self.pass, MaterialPass::Glass { .. })
            || (has_uvs && self.texture.is_some() && matches!(self.pass, MaterialPass::Alpha | MaterialPass::Decal))
    }
}

const GLASS_PREVIEW_ALPHA: u8 = 64;

const DEFAULT_PAINT_COLOR: [f32; 3] = [0.78, 0.11, 0.14];

const PAINT_PRESETS: [(&str, [f32; 3]); 8] = [
    ("Red", [0.78, 0.11, 0.14]),
    ("Vice Pink", [0.92, 0.25, 0.58]),
    ("Blue", [0.10, 0.30, 0.76]),
    ("Yellow", [0.92, 0.74, 0.10]),
    ("Green", [0.09, 0.48, 0.28]),
    ("Silver", [0.76, 0.79, 0.84]),
    ("Black", [0.10, 0.11, 0.13]),
    ("White", [0.93, 0.94, 0.96]),
];

impl Texture2D {
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> Color32 {
        if self.w == 0 || self.h == 0 {
            return Color32::WHITE;
        }
        let x = ((u.rem_euclid(1.0) * (self.w as f32)) as usize).min(self.w - 1);
        let y = ((v.rem_euclid(1.0) * (self.h as f32)) as usize).min(self.h - 1);
        self.pixels[y * self.w + x]
    }
}

const DEFAULT_YAW: f32 = 2.45;
const DEFAULT_PITCH: f32 = 0.38;

pub struct Mesh3D {
    pub name: String,
    pub verts: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
    pub tri_geom: Vec<u16>,
    pub normals: Vec<[f32; 3]>,
    pub edges: Vec<(u32, u32)>,
    pub texture: Option<Texture2D>,
    pub geom_materials: Vec<GeomMaterial>,
    pub has_paint_geom: bool,
    pub paint_color: [f32; 3],
    pub tex_status: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub geom_count: usize,
    pub meshlet_count: usize,
    pub bone_count: usize,
    pub lod_label: String,
    // separate drawables in dictionary entries empty for single drawable meshes
    pub parts: Vec<MeshPart>,
    // none shows every part on grid some isolates a single part
    pub active_part: Option<usize>,
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub shade_mode: ShadeMode,
    pub auto_spin: bool,
    pub sky_bg: bool,
    // reused softwre rasterizer buffrs zero allocs per fram
    fb_w: usize,
    fb_h: usize,
    color_buf: Vec<Color32>,
    z_buf: Vec<f32>,
    proj_buf: Vec<[f32; 3]>,
    blend_order: Vec<(usize, f32, bool)>,
    tex: Option<TextureHandle>,
    pub dirty: bool,
    last_dragging: bool,
}

impl Mesh3D {
    pub fn from_parts(
        name: String,
        verts: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        tris: Vec<[u32; 3]>,
        tri_geom: Vec<u16>,
        edges: Vec<(u32, u32)>,
        texture: Option<Texture2D>,
        tex_status: String,
        geom_count: usize,
        meshlet_count: usize,
        lod_label: String,
    ) -> Self {
        let (mut min_v, mut max_v) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &verts {
            for k in 0..3 {
                if v[k] < min_v[k] { min_v[k] = v[k]; }
                if v[k] > max_v[k] { max_v[k] = v[k]; }
            }
        }
        let center = if verts.is_empty() {
            [0.0; 3]
        } else {
            [
                (min_v[0] + max_v[0]) * 0.5,
                (min_v[1] + max_v[1]) * 0.5,
                (min_v[2] + max_v[2]) * 0.5,
            ]
        };
        let mut r2 = 0.0f32;
        for v in &verts {
            let dx = v[0] - center[0];
            let dy = v[1] - center[1];
            let dz = v[2] - center[2];
            r2 = r2.max(dx * dx + dy * dy + dz * dz);
        }
        let radius = r2.sqrt().max(1e-3);

        // precomput unit face normals once so drag stays fast
        let mut normals = Vec::with_capacity(tris.len());
        for t in &tris {
            let a = verts[t[0] as usize];
            let b = verts[t[1] as usize];
            let c = verts[t[2] as usize];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let nx = ab[1] * ac[2] - ab[2] * ac[1];
            let ny = ab[2] * ac[0] - ab[0] * ac[2];
            let nz = ab[0] * ac[1] - ab[1] * ac[0];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len > 1e-12 {
                normals.push([nx / len, ny / len, nz / len]);
            } else {
                normals.push([0.0, 0.0, 1.0]);
            }
        }

        Self {
            name,
            verts,
            uvs,
            tris,
            tri_geom,
            normals,
            edges,
            texture,
            geom_materials: Vec::new(),
            has_paint_geom: false,
            paint_color: DEFAULT_PAINT_COLOR,
            tex_status,
            center,
            radius,
            geom_count,
            meshlet_count,
            bone_count: 0,
            lod_label,
            parts: Vec::new(),
            active_part: None,
            yaw: DEFAULT_YAW,
            pitch: DEFAULT_PITCH,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            shade_mode: ShadeMode::Textured,
            auto_spin: false,
            sky_bg: false,
            fb_w: 0,
            fb_h: 0,
            color_buf: Vec::new(),
            z_buf: Vec::new(),
            proj_buf: Vec::new(),
            blend_order: Vec::new(),
            tex: None,
            dirty: true,
            last_dragging: false,
        }
    }

    // triangle range currently shown either active part or everything
    pub fn visible_tris(&self) -> std::ops::Range<usize> {
        self.active_part
            .and_then(|i| self.parts.get(i))
            .map(|p| p.tris.start.min(self.tris.len())..p.tris.end.min(self.tris.len()))
            .unwrap_or(0..self.tris.len())
    }

    // isolates one part or all with none and reframes camera on it
    pub fn set_active_part(&mut self, part: Option<usize>) {
        self.active_part = part.filter(|&i| i < self.parts.len());
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for t in &self.tris[self.visible_tris()] {
            for &i in t {
                let v = self.verts[i as usize];
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        if lo[0] <= hi[0] {
            self.center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
            let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            self.radius = ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * 0.5).max(1e-3);
        }
        self.zoom = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
        self.dirty = true;
    }

    // writes visible parts as obj one group per drawable and recenters isolated part
    pub fn export_obj(&self, path: &Path) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "# PercyStar RAGE9 Model Export: {}", self.name)?;
        let groups: Vec<(&str, std::ops::Range<usize>)> = match self.active_part.and_then(|i| self.parts.get(i)) {
            Some(p) => vec![(p.name.as_str(), self.visible_tris())],
            None if !self.parts.is_empty() => self.parts.iter().map(|p| (p.name.as_str(), p.tris.clone())).collect(),
            None => vec![(self.name.as_str(), 0..self.tris.len())],
        };
        let shift = if self.active_part.is_some() { [self.center[0], self.center[1], 0.0] } else { [0.0; 3] };
        let has_uv = self.uvs.len() == self.verts.len() && !self.uvs.is_empty();
        let mut remap = vec![0u32; self.verts.len()];
        let mut next = 1u32;
        for (name, range) in groups {
            writeln!(w, "o {name}")?;
            let tris = &self.tris[range];
            for &i in tris.iter().flatten() {
                let i = i as usize;
                if remap[i] == 0 {
                    remap[i] = next;
                    next += 1;
                    let v = self.verts[i];
                    writeln!(w, "v {:.6} {:.6} {:.6}", v[0] - shift[0], v[1] - shift[1], v[2] - shift[2])?;
                    if has_uv {
                        writeln!(w, "vt {:.6} {:.6}", self.uvs[i][0], self.uvs[i][1])?;
                    }
                }
            }
            for t in tris {
                let (a, b, c) = (remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]);
                if has_uv {
                    writeln!(w, "f {a}/{a} {b}/{b} {c}/{c}")?;
                } else {
                    writeln!(w, "f {a} {b} {c}")?;
                }
            }
        }
        Ok(())
    }
}

pub fn parse_obj(path: &Path) -> Option<Mesh3D> {
    let txt = fs::read_to_string(path).ok()?;
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    let mut edges = Vec::new();
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                if let (Some(x), Some(y), Some(z)) = (
                    it.next().and_then(|s| s.parse::<f32>().ok()),
                    it.next().and_then(|s| s.parse::<f32>().ok()),
                    it.next().and_then(|s| s.parse::<f32>().ok()),
                ) {
                    verts.push([x, y, z]);
                }
            }
            Some("f") => {
                let idxs: Vec<u32> = it
                    .filter_map(|tok| tok.split('/').next()?.parse::<usize>().ok()?.checked_sub(1))
                    .filter(|&i| i < verts.len())
                    .map(|i| i as u32)
                    .collect();
                if idxs.len() >= 3 {
                    for k in 1..idxs.len() - 1 {
                        tris.push([idxs[0], idxs[k], idxs[k + 1]]);
                    }
                } else if idxs.len() == 2 {
                    edges.push((idxs[0], idxs[1]));
                }
            }
            _ => {}
        }
    }
    if verts.is_empty() { return None; }
    // flip external y up obj files into z up turntabl space
    if !txt.starts_with("# PercyStar") {
        for v in &mut verts {
            *v = [v[0], -v[2], v[1]];
        }
    }
    let tri_geom = vec![0u16; tris.len()];
    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    Some(Mesh3D::from_parts(name, verts, Vec::new(), tris, tri_geom, edges, None, "OBJ".into(), 1, 0, "OBJ".into()))
}

pub fn extract_0drpz_mesh(name: &str, data: &[u8]) -> Option<Mesh3D> {
    let mut verts = Vec::new();
    let mut off = 0x40usize;
    while off + 12 <= data.len() && verts.len() < 12000 {
        let x = rf32(data, off);
        let y = rf32(data, off + 4);
        let z = rf32(data, off + 8);
        if x.is_finite() && y.is_finite() && z.is_finite()
            && x.abs() < 250.0 && y.abs() < 250.0 && z.abs() < 250.0
            && (x.abs() > 1e-4 || y.abs() > 1e-4 || z.abs() > 1e-4)
        {
            verts.push([x, y, z]);
            off += 12;
        } else {
            off += 4;
        }
    }
    if verts.len() < 12 { return None; }
    Some(Mesh3D::from_parts(name.to_string(), verts, Vec::new(), Vec::new(), Vec::new(), Vec::new(), None, "Points".into(), 1, 0, "Points".into()))
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
#[inline]
fn rf32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
pub fn is_zlib_head(b: &[u8]) -> bool {
    b.len() >= 3 && b[0] == 0x78 && matches!(b[1], 0x01 | 0x5e | 0x9c | 0xda) && (b[2] & 0x06) != 0x06
}

#[inline]
fn vptr(q: u64) -> Option<usize> {
    (0x5000_0000..0x5800_0000).contains(&q).then(|| (q - 0x5000_0000) as usize)
}

#[inline]
fn pptr(q: u64) -> Option<usize> {
    (0x6000_0000..0x7000_0000).contains(&q).then(|| (q - 0x6000_0000) as usize)
}

// decomprss one zlib stream from memory slice returns false if incomplete
fn decompress_zlib_slice(src: &[u8], dec: &mut Decompress, out: &mut Vec<u8>) -> Option<usize> {
    dec.reset(true);
    let mut tmp = [0u8; 65536];
    let mut pos = 0usize;
    loop {
        let bi = dec.total_in();
        let bo = dec.total_out();
        let st = dec.decompress(&src[pos..], &mut tmp, FlushDecompress::None).ok()?;
        let cin = (dec.total_in() - bi) as usize;
        let cout = (dec.total_out() - bo) as usize;
        pos += cin;
        if cout > 0 {
            out.extend_from_slice(&tmp[..cout]);
        }
        if st == Status::StreamEnd {
            return Some(pos);
        }
        if cin == 0 && cout == 0 {
            return None;
        }
    }
}

fn decompress_zlib_at(f: &mut File, file_off: u64, max_out: usize) -> std::io::Result<(Vec<u8>, u64)> {
    f.seek(SeekFrom::Start(file_off))?;
    let mut dec = Decompress::new(true);
    let mut in_buf = [0u8; 65536];
    let mut out_chunk = [0u8; 65536];
    let mut out = Vec::with_capacity(65536);
    let mut total_consumed = 0u64;
    loop {
        let n = f.read(&mut in_buf)?;
        if n == 0 { break; }
        let mut pos = 0usize;
        while pos < n {
            let bi = dec.total_in();
            let bo = dec.total_out();
            let status = dec
                .decompress(&in_buf[pos..n], &mut out_chunk, FlushDecompress::None)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let cin = (dec.total_in() - bi) as usize;
            let cout = (dec.total_out() - bo) as usize;
            pos += cin;
            total_consumed += cin as u64;
            if cout > 0 {
                out.extend_from_slice(&out_chunk[..cout]);
                if out.len() > max_out {
                    return Ok((out, file_off + total_consumed));
                }
            }
            if status == Status::StreamEnd {
                return Ok((out, file_off + total_consumed));
            }
            if cin == 0 && cout == 0 {
                break;
            }
        }
    }
    Ok((out, file_off + total_consumed))
}

fn find_next_zlib(f: &mut File, start_off: u64) -> std::io::Result<Option<u64>> {
    f.seek(SeekFrom::Start(start_off))?;
    let mut buf = [0u8; 4096];
    let n = f.read(&mut buf)?;
    for i in 0..n.saturating_sub(3) {
        if is_zlib_head(&buf[i..n]) {
            return Ok(Some(start_off + i as u64));
        }
    }
    Ok(None)
}

// reads all virtual chnks from a rage9 resource stopping right after blockmap
fn read_virtual_from_file(f: &mut File, file_off: u64) -> Result<(Vec<u8>, u64, usize), String> {
    let (mut v, mut next_off) = decompress_zlib_at(f, file_off, 16 * 1024 * 1024)
        .map_err(|e| format!("Virtual chunk 0 error: {e}"))?;
    if v.len() < 0x80 {
        return Err("Virtual chunk too small".into());
    }
    // mid resource virtual chunks starting directly with grmgeometry are never a resource root
    if ru64(&v, 0x60) == 0x1601 {
        return Err("Mid-resource geometry chunk (not a RAGE9 resource root)".into());
    }
    let Some(bmap_off) = vptr(ru64(&v, 8)) else {
        return Err("Not a valid RAGE9 resource (missing BlockMap)".into());
    };
    let mut v_chunks_read = 1usize;
    while v.len() <= bmap_off + 10 && v_chunks_read < 32 {
        let Some(zoff) = find_next_zlib(f, next_off).map_err(|e| e.to_string())? else { break };
        let (chunk, nxt) = decompress_zlib_at(f, zoff, 16 * 1024 * 1024).map_err(|e| e.to_string())?;
        v.extend_from_slice(&chunk);
        next_off = nxt;
        v_chunks_read += 1;
    }
    if bmap_off + 10 > v.len() {
        return Err("Invalid RAGE9 BlockMap offset".into());
    }
    let v_cnt = v[bmap_off + 8] as usize;
    let p_cnt = v[bmap_off + 9] as usize;
    if !(1..=32).contains(&v_cnt) || !(1..=64).contains(&p_cnt) {
        return Err(format!("Invalid RAGE9 BlockMap counts (v={v_cnt}, p={p_cnt})"));
    }
    for _ in v_chunks_read..v_cnt {
        let Some(zoff) = find_next_zlib(f, next_off).map_err(|e| e.to_string())? else { break };
        let (chunk, nxt) = decompress_zlib_at(f, zoff, 16 * 1024 * 1024).map_err(|e| e.to_string())?;
        v.extend_from_slice(&chunk);
        next_off = nxt;
    }
    Ok((v, next_off, p_cnt))
}

// finds dr ft dd modle string in virtual buffr (also checks drawable +0xa8 name ptr!)
pub fn extract_rage9_model_name(v: &[u8]) -> Option<(String, &'static str)> {
    let mut i = 1usize;
    while i + 2 < v.len() {
        if v[i] == b'#' && v[i - 1] == b'.' {
            let ext = match (v[i + 1], v[i + 2]) {
                (b'd', b'r') => ".#dr",
                (b'f', b't') => ".#ft",
                (b'd', b'd') => ".#dd",
                _ => {
                    i += 1;
                    continue;
                }
            };
            let dot = i - 1;
            let mut start = dot;
            while start > 0 {
                let c = v[start - 1];
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' {
                    start -= 1;
                } else {
                    break;
                }
            }
            if dot - start >= 2 && dot - start <= 96 {
                let stem = String::from_utf8_lossy(&v[start..dot]);
                return Some((format!("{stem}{ext}"), ext));
            }
        }
        i += 1;
    }
    // Fallback: RAGE9 gtaDrawable name pointer at +0xa8 (catches 3,000+ models without .#dr suffix!)
    if v.len() >= 0xb0 {
        if let Some(no) = vptr(ru64(v, 0xa8)) {
            if no < v.len() {
                if let Some(rel) = v[no..].iter().position(|&b| b == 0) {
                    if (2..=96).contains(&rel) {
                        let s = &v[no..no + rel];
                        if s.iter().all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'#'))
                            && s.iter().any(|b| b.is_ascii_alphabetic())
                        {
                            let nm = String::from_utf8_lossy(s).trim_end_matches('.').to_string();
                            if nm.len() >= 2 {
                                let ext = if nm.ends_with(".#ft") { ".#ft" } else { ".#dr" };
                                let full = if nm.contains(".#") { nm } else { format!("{nm}.#dr") };
                                return Some((full, ext));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

// checks if virtual resource is an actual vehicle and not a small tuning part
pub fn is_vehicle_virtual(v: &[u8], name: &str) -> bool {
    let lo = name.to_ascii_lowercase();
    if !lo.ends_with(".#ft") || v.len() < 28_000 {
        return false;
    }
    for part in [
        "_dmg_", "_livery", "_roof_", "_roofext_", "_spoiler", "_exhaust",
        "_bumf_", "_bumr_", "_skirt_", "_bonnet", "_wing_", "_door_",
        "_mirror", "_lights_", "_intakes_", "_platemount", "_towbar",
        "_speakermount", "_deliverybag", "_antennaset", "_radar_", "_horn",
        "_outboard", "_inboard", "_mast", "_mount", "_turret", "_weap_",
        "_wheeliebar", "_induction_", "_whiplight", "_rack_", "_sign_mesh",
        "_armour_plate", "_fairing_", "_rescuehoist", "_footboards", "_callsign",
        "_shorad", "_badge", "_trim", "_overlay",
    ] {
        if lo.contains(part) {
            return false;
        }
    }
    [b"wheel_lf".as_slice(), b"door_dside_f", b"chassis", b"bodyshell", b"cockpit", b"rudder", b"rotor_main"]
        .iter()
        .any(|needle| v.windows(needle.len()).any(|w| w == *needle))
}

// zero seek in memory probe falls back to disk if stream crosses chunk boundry
pub fn probe_large_rage9_model_in_buf(buf: &[u8], rpf_path: &Path, abs_off: u64) -> Option<(String, bool, usize)> {
    let mut dec = Decompress::new(true);
    let mut v = Vec::with_capacity(131072);
    let mut pos = match decompress_zlib_slice(buf, &mut dec, &mut v) {
        Some(n) => n,
        None => return probe_large_rage9_model(rpf_path, abs_off),
    };
    if v.len() < 0x80 || ru64(&v, 0x60) == 0x1601 { return None; }
    let bmap_off = vptr(ru64(&v, 8))?;
    if bmap_off > 4 * 1024 * 1024 { return None; }

    let mut v_chunks_read = 1usize;
    while v.len() <= bmap_off + 10 && v_chunks_read < 32 {
        while pos + 3 < buf.len() && !is_zlib_head(&buf[pos..]) {
            pos += 1;
        }
        if pos + 3 >= buf.len() {
            return probe_large_rage9_model(rpf_path, abs_off);
        }
        match decompress_zlib_slice(&buf[pos..], &mut dec, &mut v) {
            Some(n) => {
                pos += n;
                v_chunks_read += 1;
            }
            None => return probe_large_rage9_model(rpf_path, abs_off),
        }
    }
    if bmap_off + 10 > v.len() { return None; }
    let v_cnt = v[bmap_off + 8] as usize;
    let p_cnt = v[bmap_off + 9] as usize;
    if !(1..=32).contains(&v_cnt) || !(1..=64).contains(&p_cnt) {
        return None;
    }
    while v_chunks_read < v_cnt {
        while pos + 3 < buf.len() && !is_zlib_head(&buf[pos..]) {
            pos += 1;
        }
        if pos + 3 >= buf.len() {
            return probe_large_rage9_model(rpf_path, abs_off);
        }
        match decompress_zlib_slice(&buf[pos..], &mut dec, &mut v) {
            Some(n) => {
                pos += n;
                v_chunks_read += 1;
            }
            None => return probe_large_rage9_model(rpf_path, abs_off),
        }
    }
    if !has_rage9_meshlets(&v) {
        return None;
    }
    let (name, _) = extract_rage9_model_name(&v)?;
    Some((name.clone(), is_vehicle_virtual(&v, &name), v.len()))
}

pub fn probe_large_rage9_model(rpf_path: &Path, file_off: u64) -> Option<(String, bool, usize)> {
    let mut f = File::open(rpf_path).ok()?;
    let (v, _, _) = read_virtual_from_file(&mut f, file_off).ok()?;
    if !has_rage9_meshlets(&v) {
        return None;
    }
    let (name, _) = extract_rage9_model_name(&v)?;
    Some((name.clone(), is_vehicle_virtual(&v, &name), v.len()))
}

pub fn has_rage9_meshlets(v: &[u8]) -> bool {
    let mut off = 0x28usize;
    while off + 0x10 <= v.len() {
        if ru64(v, off - 8) == 0x1601
            && pptr(ru64(v, off)).is_some()
            && resolve_meshlet_buffer(v, ru64(v, off - 0x28)).is_some()
        {
            return true;
        }
        off += 8;
    }
    false
}

// shared meshlet link resolver so lod0 and fallback scanner stay dry
#[inline]
fn resolve_meshlet_buffer(v: &[u8], link_ptr: u64) -> Option<(usize, usize)> {
    let lo = vptr(link_ptr)?;
    if lo + 8 > v.len() { return None; }
    let mo = vptr(ru64(v, lo))?;
    if mo + 0x20 > v.len() { return None; }
    let mb_poff = pptr(ru64(v, mo + 0x18))?;
    let mb_words = ru32(v, mo + 0x08) as usize;
    Some((mb_poff, mb_words * 4))
}

// one meshlet geometry vb phys offset vb bytes mb phys offset mb bytes
type GeomRef = (usize, usize, usize, usize);

// lod0 high detail meshlet geometries of drawable at base
fn drawable_lod0_geoms(v: &[u8], base: usize) -> Vec<GeomRef> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    if base + 0x80 > v.len() { return out; }
    let Some(lod0_off) = vptr(ru64(v, base + 0x50)) else { return out };
    if lod0_off + 0x10 > v.len() { return out; }
    let Some(models_off) = vptr(ru64(v, lod0_off)) else { return out };
    let model_cnt = ru16(v, lod0_off + 8) as usize;
    if model_cnt == 0 || model_cnt > 128 || models_off + model_cnt * 8 > v.len() { return out; }

    for mi in 0..model_cnt {
        let Some(mp) = vptr(ru64(v, models_off + mi * 8)) else { continue };
        if mp + 0x18 > v.len() { continue; }
        let Some(garr) = vptr(ru64(v, mp + 0x08)) else { continue };
        let gcnt = ru16(v, mp + 0x10) as usize;
        if gcnt == 0 || gcnt > 256 || garr + gcnt * 8 > v.len() { continue; }

        for gi in 0..gcnt {
            let Some(gp) = vptr(ru64(v, garr + gi * 8)) else { continue };
            if gp + 0x70 > v.len() || ru64(v, gp + 0x60) != 0x1601 { continue; }
            let Some(vb_poff) = pptr(ru64(v, gp + 0x68)) else { continue };
            let Some((mb_poff, mb_bytes)) = resolve_meshlet_buffer(v, ru64(v, gp + 0x40)) else { continue };
            let vb_bytes = (ru32(v, gp + 0x58) as usize) * 4;
            if seen.insert((vb_poff, mb_poff)) {
                out.push((vb_poff, vb_bytes, mb_poff, mb_bytes));
            }
        }
    }
    out
}

fn collect_all_geoms(v: &[u8]) -> Vec<GeomRef> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut off = 0x28usize;
    while off + 0x10 <= v.len() {
        if ru64(v, off - 8) == 0x1601 {
            if let Some(vb_poff) = pptr(ru64(v, off)) {
                if let Some((mb_poff, mb_bytes)) = resolve_meshlet_buffer(v, ru64(v, off - 0x28)) {
                    let vb_bytes = (ru32(v, off - 16) as usize) * 4;
                    if seen.insert((vb_poff, mb_poff)) {
                        out.push((vb_poff, vb_bytes, mb_poff, mb_bytes));
                    }
                }
            }
        }
        off += 8;
    }
    out
}

fn try_extract_embedded_texture(v: &[u8], p: &[u8]) -> (Option<Texture2D>, String) {
    let mut bases = vec![0usize];
    if v.len() >= 0x20 {
        if let Some(fo) = vptr(ru64(v, 0x18)) {
            bases.push(fo);
        }
    }
    for base in bases {
        if base + 0x18 > v.len() { continue; }
        let Some(sg) = vptr(ru64(v, base + 0x10)) else { continue };
        if sg + 0x10 > v.len() { continue; }
        let Some(txd) = vptr(ru64(v, sg + 0x08)) else { continue };
        // Use our verified Xbox Scarlett SW_4KB_D + reverse-mip + pure-Rust BC7/BC3/BC1 decoder!
        let decoded = crate::tex::extract_txd_textures(v, p, txd);
        if !decoded.is_empty() {
            // Prefer diffuse/albedo texture over normal/spec/blank if multiple embedded
            let best_i = decoded
                .iter()
                .position(|t| {
                    let lo = t.name.to_ascii_lowercase();
                    !lo.contains("_n") && !lo.contains("_nm") && !lo.contains("_s") && !lo.contains("blank") && t.w >= 32
                })
                .unwrap_or(0);
            let total_d = decoded.len();
            let best = decoded.into_iter().nth(best_i).unwrap();
            let desc = format!("Embedded {} ({}x{} {}, {}/{})", best.name, best.w, best.h, best.fmt_name, 1, total_d);
            return (Some(crate::tex::to_model_tex(best)), desc);
        }
    }
    (None, "External .#td (Material Shading)".into())
}

// maps each lod0 vb and mb offset of drawable at base to its shader index and render mask
fn drawable_lod0_shader_map(v: &[u8], base: usize) -> HashMap<(usize, usize), (u16, u8)> {
    let mut out = HashMap::new();
    if base + 0x80 > v.len() { return out; }
    let Some(lod0_off) = vptr(ru64(v, base + 0x50)) else { return out };
    if lod0_off + 0x10 > v.len() { return out; }
    let Some(models_off) = vptr(ru64(v, lod0_off)) else { return out };
    let model_cnt = ru16(v, lod0_off + 8) as usize;
    if model_cnt == 0 || model_cnt > 128 || models_off + model_cnt * 8 > v.len() { return out; }

    for mi in 0..model_cnt {
        let Some(mp) = vptr(ru64(v, models_off + mi * 8)) else { continue };
        if mp + 0x28 > v.len() { continue; }
        let Some(garr) = vptr(ru64(v, mp + 0x08)) else { continue };
        let gcnt = ru16(v, mp + 0x10) as usize;
        if gcnt == 0 || gcnt > 256 || garr + gcnt * 8 > v.len() { continue; }
        let smap_off = vptr(ru64(v, mp + 0x20)).filter(|&so| so + gcnt * 2 <= v.len());
        let render_mask = v.get(mp + 0x34).copied().unwrap_or(0xff);

        for gi in 0..gcnt {
            let Some(gp) = vptr(ru64(v, garr + gi * 8)) else { continue };
            if gp + 0x70 > v.len() || ru64(v, gp + 0x60) != 0x1601 { continue; }
            let Some(vb_poff) = pptr(ru64(v, gp + 0x68)) else { continue };
            let Some((mb_poff, _)) = resolve_meshlet_buffer(v, ru64(v, gp + 0x40)) else { continue };
            let sidx = smap_off.map(|so| ru16(v, so + gi * 2)).unwrap_or(gi as u16);
            let binding = out.entry((vb_poff, mb_poff)).or_insert((sidx, render_mask));
            if binding.1 & 1 == 0 && render_mask & 1 != 0 {
                *binding = (sidx, render_mask);
            }
        }
    }
    out
}

// resolves external ttpf diffuse textures and vehicle paint shader flags for lod0 geoms
fn resolve_drawable_materials(
    v: &[u8],
    base: usize,
    geoms: &[GeomRef],
    ttpf_idx: Option<&HashMap<u64, crate::tex::TtpfHashHit>>,
    is_vehicle: bool,
    seen_hashes: &mut HashSet<u64>,
    decode_budget: &mut usize,
) -> Vec<GeomMaterial> {
    const DIFFUSE_TEX: u32 = 0xc8e8_c282; // diffuseTex
    const TINT_PALETTE_TEX: u32 = 0x2c8d_7182; // tintPaletteTex
    const ALT0_TEX: u32 = 0x3463_94ea; // vehicle body livery detail slot 0
    const EMISSIVE_TEX: u32 = 0x8e0d_63ee; // emissiveTex
    const DIRT_DIFFUSE_TEX: u32 = 0x7485_dafb; // dirtDiffuseTex
    const SHARED_LUT_PARAM: u32 = 0x19df_25d7; // shared 128x256 vehicle lookup texture
    const GLASS_SHADER: u32 = 0xe3ba_3129;

    let mut mats = vec![GeomMaterial::default(); geoms.len()];
    if base + 0x18 > v.len() { return mats; }
    let Some(sg) = vptr(ru64(v, base + 0x10)) else { return mats };
    if sg + 0x20 > v.len() { return mats; }
    let Some(sp) = vptr(ru64(v, sg + 0x10)) else { return mats };
    let scnt = ru16(v, sg + 0x18) as usize;
    if scnt == 0 || scnt > 512 || sp + scnt * 8 > v.len() { return mats; }
    // verify rage9 atem meta shader table tag
    let Some(s0) = vptr(ru64(v, sp)) else { return mats };
    if s0 + 8 > v.len() || (ru64(v, s0) >> 32) != 0x6d65_7461 { return mats; }

    let smap = drawable_lod0_shader_map(v, base);
    let has_main_pass = smap.values().any(|&(_, mask)| mask & 1 != 0);
    let mut shader_cache: HashMap<u16, GeomMaterial> = HashMap::new();

    for (gi, &(vb_poff, _, mb_poff, _)) in geoms.iter().enumerate() {
        let (sidx, mask) = smap.get(&(vb_poff, mb_poff)).copied().unwrap_or((gi as u16, 0xff));
        if has_main_pass && mask & 1 == 0 {
            mats[gi].auxiliary_pass = true;
            continue;
        }
        if let Some(cached) = shader_cache.get(&sidx) {
            mats[gi] = cached.clone();
            continue;
        }
        let mut mat = GeomMaterial::default();
        if (sidx as usize) < scnt {
            if let Some(saddr) = vptr(ru64(v, sp + (sidx as usize) * 8)) {
                if saddr + 0x28 <= v.len() {
                    mat.pass = match v.get(saddr + 0x3c) {
                        Some(2) if ru32(v, saddr) == GLASS_SHADER => MaterialPass::Glass { texture_alpha: false },
                        Some(2) => MaterialPass::Alpha,
                        Some(4) => MaterialPass::Decal,
                        Some(8) => MaterialPass::Cutout,
                        _ => MaterialPass::Opaque,
                    };
                    if let (Some(p2), Some(p4)) = (vptr(ru64(v, saddr + 0x10)), vptr(ru64(v, saddr + 0x20))) {
                        if p4 + 8 <= v.len() {
                            let tcnt = (v[p4 + 1] as usize).min(48);
                            let mut has_tint = false;
                            let mut diffuse_cand: Option<(u64, crate::tex::TtpfHashHit)> = None;
                            let mut alt0_cand: Option<(u64, crate::tex::TtpfHashHit)> = None;
                            let mut emissive_cand: Option<(u64, crate::tex::TtpfHashHit)> = None;
                            let mut fallback_srgb: Option<(u64, crate::tex::TtpfHashHit)> = None;
                            let uses_opacity = !matches!(mat.pass, MaterialPass::Opaque);

                            for k in 0..tcnt {
                                let ent = p4 + 8 + k * 8;
                                if ent + 8 > v.len() { break; }
                                let param_hash = ru32(v, ent);
                                let slot = (ru32(v, ent + 4) / 4) as usize;
                                if param_hash == TINT_PALETTE_TEX {
                                    has_tint = true;
                                }
                                let Some(idx_map) = ttpf_idx else { continue };
                                if p2 + slot * 8 + 8 > v.len() { continue; }
                                let Some(tp) = vptr(ru64(v, p2 + slot * 8)) else { continue };
                                if tp + 0x40 > v.len() || ru64(v, tp + 0x10) != 0x980 { continue; }
                                let Some(ref_off) = vptr(ru64(v, tp + 0x30)) else { continue };
                                if ref_off + 16 > v.len() { continue; }
                                let h64 = ru64(v, ref_off + 8);
                                let Some(hit) = idx_map.get(&h64) else { continue };
                                if hit.is_normal_format() { continue; }
                                if hit.dxgi == 65 && !(uses_opacity && param_hash == DIFFUSE_TEX) { continue; }

                                if param_hash == DIFFUSE_TEX && hit.w >= 4 && hit.h >= 4 {
                                    diffuse_cand = Some((h64, hit.clone()));
                                } else if param_hash == ALT0_TEX && hit.w >= 16 && hit.h >= 16 {
                                    alt0_cand = Some((h64, hit.clone()));
                                } else if param_hash == EMISSIVE_TEX && hit.w >= 16 && hit.h >= 16 {
                                    emissive_cand = Some((h64, hit.clone()));
                                } else if hit.is_srgb()
                                    && hit.w >= 16
                                    && hit.h >= 16
                                    && param_hash != DIRT_DIFFUSE_TEX
                                    && param_hash != SHARED_LUT_PARAM
                                    && fallback_srgb.is_none()
                                {
                                    fallback_srgb = Some((h64, hit.clone()));
                                }
                            }

                            let chosen = if let Some(c) = diffuse_cand {
                                Some(c)
                            } else if has_tint && is_vehicle {
                                alt0_cand
                            } else {
                                alt0_cand.or(emissive_cand).or(fallback_srgb)
                            };

                            let had_tex_ref = chosen.is_some();
                            if let Some((h64, hit)) = chosen {
                                let is_new = seen_hashes.insert(h64);
                                if !is_new || *decode_budget > 0 {
                                    if let Some(tex) = crate::tex::decode_ttpf_texture_cached(&hit, h64) {
                                        mat.texture = Some(tex);
                                        if is_new && *decode_budget > 0 {
                                            *decode_budget -= 1;
                                        }
                                    }
                                }
                            }
                            if matches!(mat.pass, MaterialPass::Glass { .. }) {
                                if mat.texture.as_ref().is_some_and(|t| t.w <= 4 && t.h <= 4 && t.pixels.iter().all(|p| p.a() == 0)) {
                                    mat.texture = None;
                                }
                                mat.pass = MaterialPass::Glass {
                                    texture_alpha: mat.texture.as_ref().is_some_and(|t| t.pixels.iter().any(|p| p.a() < 255)),
                                };
                            }
                            mat.is_vehicle_paint = is_vehicle && has_tint && !had_tex_ref && mat.pass == MaterialPass::Opaque;
                        }
                    }
                }
            }
        }
        shader_cache.insert(sidx, mat.clone());
        mats[gi] = mat;
    }
    mats
}

// one drawable inside a decoded resource plain drawables have single part dictionaries have one per entry
#[derive(Clone)]
pub struct MeshPart {
    pub name: String,
    pub geoms: usize,
    // triangle range in mesh3d tris
    pub tris: std::ops::Range<usize>,
}

// detects rage9 drawable dictionary root and returns name and drawable offset per entry
fn drawable_dictionary_entries(v: &[u8]) -> Vec<(String, usize)> {
    if v.len() < 0x40 { return Vec::new(); }
    let (Some(hashes), Some(arr)) = (vptr(ru64(v, 0x20)), vptr(ru64(v, 0x30))) else { return Vec::new() };
    let cnt = ru16(v, 0x38) as usize;
    if !(2..=4096).contains(&cnt) || ru16(v, 0x28) as usize != cnt || hashes + cnt * 4 > v.len() || arr + cnt * 8 > v.len() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let Some(d) = vptr(ru64(v, arr + i * 8)) else { return Vec::new() };
        if d + 0xb0 > v.len() || vptr(ru64(v, d + 0x50)).is_none() { return Vec::new(); }
        let name = vptr(ru64(v, d + 0xa8))
            .and_then(|n| {
                let len = v.get(n..)?.iter().take(128).position(|&b| b == 0)?;
                let s = &v[n..n + len];
                (len >= 2 && s.iter().all(|b| b.is_ascii_graphic())).then(|| String::from_utf8_lossy(s).into_owned())
            })
            .unwrap_or_else(|| format!("0x{:08x}", ru32(v, hashes + i * 4)));
        out.push((name, d));
    }
    out
}

#[derive(Default)]
struct MeshBuild {
    verts: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    tris: Vec<[u32; 3]>,
    tri_geom: Vec<u16>,
    meshlets: usize,
}

// decodes one meshlet geometry from physical buffer and appends it to mesh build
fn append_meshlet_geom(p: &[u8], (vb_off, vb_size, mb_off, _mb_size): GeomRef, geom_id: u16, b: &mut MeshBuild) {
    if mb_off + 0x10 > p.len() || vb_off + 0x80 > p.len() {
        return;
    }
    let num_meshlets = (ru32(p, mb_off) & 0xFFFF) as usize;
    let tot_tris = ru32(p, mb_off + 4) as usize;
    let tot_vrefs = ru32(p, mb_off + 8) as usize;
    if num_meshlets == 0 || tot_tris == 0 || tot_vrefs == 0 {
        return;
    }
    let tri_base = mb_off + 0x0c + num_meshlets * 16;
    let remap_base = tri_base + tot_tris * 4;
    if remap_base + tot_vrefs * 2 > p.len() {
        return;
    }

    let mut hdrs = [0u32; 8];
    for (k, h) in hdrs.iter_mut().enumerate() {
        if vb_off + k * 4 + 4 <= p.len() {
            *h = ru32(p, vb_off + k * 4);
        }
    }
    let stream0_off = ((hdrs[0] >> 8) << 1) as usize;
    let mut stream0_end = vb_size;
    let mut uv_stream_off: Option<usize> = None;
    for &h in &hdrs[1..] {
        if h == 0 { continue; }
        let so = ((h >> 8) << 1) as usize;
        let st = h & 0xff;
        if so > stream0_off && so < stream0_end {
            stream0_end = so;
        }
        if st == 0x28 && uv_stream_off.is_none() {
            uv_stream_off = Some(so);
        }
    }
    if vb_off + stream0_off >= p.len() {
        return;
    }
    let max_vb_verts = ((stream0_end.saturating_sub(stream0_off)) / 8).min((p.len() - (vb_off + stream0_off)) / 8);
    if max_vb_verts == 0 {
        return;
    }

    let ox = rf32(p, vb_off + 0x60);
    let oy = rf32(p, vb_off + 0x64);
    let oz = rf32(p, vb_off + 0x68);
    let sx = rf32(p, vb_off + 0x70);
    let sy = rf32(p, vb_off + 0x74);
    let sz = rf32(p, vb_off + 0x78);
    // +0x6c picks which world axis sits in the 16 bit slot 0 is zyx 1 is zxy 2 is xyz
    let slot_axis: [usize; 3] = match ru32(p, vb_off + 0x6c) {
        0 => [2, 1, 0],
        1 => [2, 0, 1],
        _ => [0, 1, 2],
    };

    let base_v = b.verts.len() as u32;
    let mut max_v_used = 0usize;
    let start_tri = b.tris.len();
    b.tris.reserve(tot_tris);

    for mi in 0..num_meshlets {
        let m_off = mb_off + 0x0c + mi * 16;
        let v_off = ru16(p, m_off + 4) as usize;
        let t_off = ru16(p, m_off + 6) as usize;
        let t_end = if mi + 1 < num_meshlets { ru16(p, mb_off + 0x0c + (mi + 1) * 16 + 6) as usize } else { tot_tris };
        for ti in t_off..t_end.min(tot_tris) {
            let w = ru32(p, tri_base + ti * 4);
            let r0 = v_off + (w & 0x7F) as usize;
            let r1 = v_off + ((w >> 7) & 0x7F) as usize;
            let r2 = v_off + ((w >> 14) & 0x7F) as usize;
            if r0 < tot_vrefs && r1 < tot_vrefs && r2 < tot_vrefs {
                let i0 = ru16(p, remap_base + r0 * 2) as usize;
                let i1 = ru16(p, remap_base + r1 * 2) as usize;
                let i2 = ru16(p, remap_base + r2 * 2) as usize;
                if i0 < max_vb_verts && i1 < max_vb_verts && i2 < max_vb_verts && i0 != i1 && i1 != i2 && i0 != i2 {
                    b.tris.push([base_v + i0 as u32, base_v + i1 as u32, base_v + i2 as u32]);
                    max_v_used = max_v_used.max(i0).max(i1).max(i2);
                }
            }
        }
    }

    let added_tris = b.tris.len() - start_tri;
    if added_tris == 0 {
        return;
    }
    let v_cnt = (max_v_used + 1).min(max_vb_verts);
    let vbase = vb_off + stream0_off;
    b.verts.reserve(v_cnt);
    b.uvs.reserve(v_cnt);
    for i in 0..v_cnt {
        let o = vbase + i * 8;
        // packed rage9 24 bit xy and 16 bit z across 8 bytes
        let ix = (p[o + 1] as u32) | ((p[o + 2] as u32) << 8) | ((p[o + 3] as u32) << 16);
        let iy = (p[o] as u32) | ((p[o + 4] as u32) << 8) | ((p[o + 5] as u32) << 16);
        let iz = (p[o + 6] as u32) | ((p[o + 7] as u32) << 8);
        let slots = [ox + (ix as f32) * sx, oy + (iy as f32) * sy, oz + (iz as f32) * sz];
        let mut pos = [0.0f32; 3];
        for (slot, &axis) in slot_axis.iter().enumerate() {
            pos[axis] = slots[slot];
        }
        b.verts.push(pos);
        let uv = uv_stream_off
            .map(|uvo| vb_off + uvo + i * 4)
            .filter(|&u_pos| u_pos + 4 <= p.len())
            .map(|u_pos| [ru16(p, u_pos) as f32 / 8192.0, ru16(p, u_pos + 2) as f32 / 8192.0])
            .unwrap_or([0.0, 0.0]);
        b.uvs.push(uv);
    }
    b.tri_geom.resize(b.tri_geom.len() + added_tris, geom_id);
    b.meshlets += num_meshlets;
}

// lays dictionary parts out on a square xy grid so independent drawables sit side by side
fn layout_parts_in_grid(verts: &mut [[f32; 3]], part_verts: &[std::ops::Range<usize>]) {
    let n = part_verts.len();
    if n < 2 { return; }
    let bounds: Vec<([f32; 3], [f32; 3])> = part_verts
        .iter()
        .map(|r| {
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &verts[r.clone()] {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            (lo, hi)
        })
        .collect();
    let cell = bounds
        .iter()
        .filter(|(lo, hi)| lo[0] <= hi[0])
        .map(|(lo, hi)| (hi[0] - lo[0]).max(hi[1] - lo[1]))
        .fold(1.0f32, f32::max)
        * 1.15;
    let cols = (n as f32).sqrt().ceil() as usize;
    for (i, (r, (lo, hi))) in part_verts.iter().zip(&bounds).enumerate() {
        if lo[0] > hi[0] { continue; }
        let tx = (i % cols) as f32 * cell - (lo[0] + hi[0]) * 0.5;
        let ty = -((i / cols) as f32) * cell - (lo[1] + hi[1]) * 0.5;
        for v in &mut verts[r.clone()] {
            v[0] += tx;
            v[1] += ty;
        }
    }
}

// reads rage9 crSkeletonData bone count from drawable+0x18 -> skel+0x6c
fn extract_rage9_bone_count(v: &[u8]) -> usize {
    let mut bases = vec![0usize];
    if v.len() >= 0x20 {
        if let Some(fo) = vptr(ru64(v, 0x18)) {
            if fo + 0x20 <= v.len() { bases.push(fo); }
        }
    }
    for base in bases {
        if base + 0x20 > v.len() { continue; }
        let Some(so) = vptr(ru64(v, base + 0x18)) else { continue };
        if so + 0x70 <= v.len() {
            let cnt = ru16(v, so + 0x6c) as usize;
            if (1..=1024).contains(&cnt) && vptr(ru64(v, so + 0x28)).is_some() {
                return cnt;
            }
        }
    }
    0
}

pub fn decode_rage9_model_from_rpf(rpf_path: &Path, file_off: u64, fallback_name: &str) -> Result<Mesh3D, String> {
    let mut f = File::open(rpf_path).map_err(|e| format!("Open error: {e}"))?;
    let (v, mut next_off, p_cnt) = read_virtual_from_file(&mut f, file_off)?;

    let mut p = Vec::new();
    for _ in 0..p_cnt {
        let Some(zoff) = find_next_zlib(&mut f, next_off).map_err(|e| e.to_string())? else { break };
        let (chunk, nxt) = decompress_zlib_at(&mut f, zoff, 32 * 1024 * 1024).map_err(|e| e.to_string())?;
        p.extend_from_slice(&chunk);
        next_off = nxt;
    }
    if p.len() < 0x90 {
        return Err("Physical GPU buffer missing or empty".into());
    }

    let model_name = extract_rage9_model_name(&v)
        .map(|(n, _)| n)
        .unwrap_or_else(|| fallback_name.to_string());

    // each group becomes one meshpart dictionary entries or single drawable
    let dict = drawable_dictionary_entries(&v);
    // part name drawable offset for material lookup lod0 geometries
    type Group = (String, Option<usize>, Vec<GeomRef>);
    let (groups, lod_label): (Vec<Group>, String) = if !dict.is_empty() {
        let g: Vec<_> = dict
            .iter()
            .map(|(nm, base)| (nm.clone(), Some(*base), drawable_lod0_geoms(&v, *base)))
            .filter(|(_, _, gs)| !gs.is_empty())
            .collect();
        let label = format!("LOD0 · dictionary of {}", dict.len());
        (g, label)
    } else {
        let mut bases = vec![0usize];
        if v.len() >= 0x20 {
            if let Some(fo) = vptr(ru64(&v, 0x18)) {
                if fo + 0x80 <= v.len() { bases.push(fo); }
            }
        }
        let lod0_with_base = bases
            .into_iter()
            .map(|b| (b, drawable_lod0_geoms(&v, b)))
            .find(|(_, gs)| !gs.is_empty());
        if let Some((b_off, lod0)) = lod0_with_base {
            (vec![(model_name.clone(), Some(b_off), lod0)], "LOD0".into())
        } else {
            (vec![(model_name.clone(), None, collect_all_geoms(&v))], "All".into())
        }
    };
    let geom_count: usize = groups.iter().map(|(_, _, g)| g.len()).sum();
    if geom_count == 0 {
        return Err("No RAGE9 meshlet geometries found in resource".into());
    }

    let (texture, mut tex_status) = try_extract_embedded_texture(&v, &p);
    let ttpf_idx = if texture.is_none() {
        crate::tex::ttpf_hash_index_for_rpf(rpf_path)
    } else {
        None
    };

    let is_vehicle = is_vehicle_virtual(&v, &model_name);
    let mut b = MeshBuild::default();
    let mut parts = Vec::with_capacity(groups.len());
    let mut part_verts = Vec::with_capacity(groups.len());
    let mut geom_materials: Vec<GeomMaterial> = Vec::with_capacity(geom_count);
    let mut seen_hashes: HashSet<u64> = HashSet::new();
    let mut decode_budget = 384usize;
    let mut gi_idx = 0u16;
    for (name, base_opt, geoms) in &groups {
        let mats = if texture.is_none() {
            if let Some(base) = *base_opt {
                resolve_drawable_materials(&v, base, geoms, ttpf_idx.as_deref(), is_vehicle, &mut seen_hashes, &mut decode_budget)
            } else {
                vec![GeomMaterial::default(); geoms.len()]
            }
        } else {
            vec![GeomMaterial::default(); geoms.len()]
        };
        let (t0, v0) = (b.tris.len(), b.verts.len());
        for (&g, mat) in geoms.iter().zip(mats) {
            append_meshlet_geom(&p, g, gi_idx, &mut b);
            geom_materials.push(mat);
            gi_idx = gi_idx.wrapping_add(1);
        }
        if b.tris.len() > t0 {
            parts.push(MeshPart { name: name.clone(), geoms: geoms.len(), tris: t0..b.tris.len() });
            part_verts.push(v0..b.verts.len());
        }
    }
    if b.verts.is_empty() {
        return Err("Decoded 0 vertices from meshlets".into());
    }
    layout_parts_in_grid(&mut b.verts, &part_verts);

    let resolved_tex_geoms = geom_materials.iter().filter(|m| m.texture.is_some()).count();
    let paint_geoms = geom_materials.iter().filter(|m| m.is_vehicle_paint).count();
    let preview_geoms = geom_materials.iter().filter(|m| !m.auxiliary_pass).count().max(1);
    if texture.is_none() {
        if resolved_tex_geoms > 0 && paint_geoms > 0 {
            tex_status = format!("TTPF: {resolved_tex_geoms}/{preview_geoms} mats + Car Paint");
        } else if resolved_tex_geoms > 0 {
            tex_status = format!("TTPF: {resolved_tex_geoms}/{preview_geoms} mats");
        } else if paint_geoms > 0 {
            tex_status = format!("Car Paint ({paint_geoms}/{preview_geoms} geoms)");
        }
    }

    let mut mesh = Mesh3D::from_parts(
        model_name,
        b.verts,
        b.uvs,
        b.tris,
        b.tri_geom,
        Vec::new(),
        texture,
        tex_status,
        geom_count,
        b.meshlets,
        lod_label,
    );
    mesh.geom_materials = geom_materials;
    mesh.has_paint_geom = paint_geoms > 0;
    mesh.bone_count = extract_rage9_bone_count(&v);
    if parts.len() > 1 {
        mesh.parts = parts;
    }
    Ok(mesh)
}

#[inline]
fn draw_line_z(
    color_buf: &mut [Color32],
    z_buf: &mut [f32],
    w: usize,
    h: usize,
    p0: [f32; 3],
    p1: [f32; 3],
    col: Color32,
    z_bias: f32,
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let steps = dx.abs().max(dy.abs());
    if steps < 0.5 || steps > 4000.0 {
        return;
    }
    let n = steps.ceil() as i32;
    let inv = 1.0 / (n as f32);
    let sx = dx * inv;
    let sy = dy * inv;
    let sz = (p1[2] - p0[2]) * inv;
    let mut x = p0[0];
    let mut y = p0[1];
    let mut z = p0[2] + z_bias;
    for _ in 0..=n {
        let xi = x as i32;
        let yi = y as i32;
        if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
            let idx = (yi as usize) * w + (xi as usize);
            if z >= z_buf[idx] {
                z_buf[idx] = z;
                color_buf[idx] = col;
            }
        }
        x += sx;
        y += sy;
        z += sz;
    }
}

#[inline]
fn blend_preview_pixel(dst: &mut Color32, src: Color32, alpha: u8) {
    let a = alpha as u16;
    let mix = |s: u8, d: u8| ((s as u16 * a + d as u16 * (255 - a) + 127) / 255) as u8;
    *dst = Color32::from_rgb(mix(src.r(), dst.r()), mix(src.g(), dst.g()), mix(src.b(), dst.b()));
}

fn rasterize_mesh(mesh: &mut Mesh3D, w: usize, h: usize) {
    let npix = w * h;
    if mesh.fb_w != w || mesh.fb_h != h {
        mesh.fb_w = w;
        mesh.fb_h = h;
        mesh.color_buf.resize(npix, Color32::from_rgb(13, 14, 18));
        mesh.z_buf.resize(npix, f32::NEG_INFINITY);
    }

    for y in 0..h {
        let t = (y as f32) / (h.max(1) as f32);
        let c = if mesh.sky_bg {
            if t < 0.56 {
                let s = t / 0.56;
                Color32::from_rgb(
                    (78.0 + s * 108.0) as u8,
                    (134.0 + s * 84.0) as u8,
                    (202.0 + s * 40.0) as u8,
                )
            } else {
                let s = (t - 0.56) / 0.44;
                Color32::from_rgb(
                    (52.0 - s * 22.0) as u8,
                    (78.0 - s * 28.0) as u8,
                    (102.0 - s * 30.0) as u8,
                )
            }
        } else {
            Color32::from_rgb(
                (11.0 + t * 9.0) as u8,
                (12.0 + t * 10.0) as u8,
                (17.0 + t * 14.0) as u8,
            )
        };
        mesh.color_buf[y * w..(y + 1) * w].fill(c);
    }
    mesh.z_buf.fill(f32::NEG_INFINITY);

    let (sy, cy) = mesh.yaw.sin_cos();
    let (sp, cp) = mesh.pitch.sin_cos();
    let min_dim = (w.min(h) as f32) * 0.44;
    let scale = (min_dim / mesh.radius) * mesh.zoom;
    let cx = (w as f32) * 0.5 + mesh.pan_x * (w as f32);
    let cy_s = (h as f32) * 0.5 + mesh.pan_y * (h as f32);
    let ctr = mesh.center;
    let persp_k = 1.0 / (mesh.radius * 4.5);

    mesh.proj_buf.resize(mesh.verts.len(), [0.0; 3]);
    for (dst, v) in mesh.proj_buf.iter_mut().zip(&mesh.verts) {
        let x = v[0] - ctr[0];
        let y = v[1] - ctr[1];
        let z = v[2] - ctr[2];
        let x1 = x * cy - y * sy;
        let y1 = x * sy + y * cy;
        let y2 = -y1 * cp + z * sp;
        let z2 = z * cp + y1 * sp;
        let persp = 1.0 / (1.0 - (y2 * persp_k).clamp(-0.65, 0.65));
        *dst = [cx + x1 * scale * persp, cy_s - z2 * scale * persp, y2];
    }

    let wf = w as f32;
    let hf = h as f32;

    if !mesh.tris.is_empty() {
        let do_textured = mesh.shade_mode == ShadeMode::Textured;
        let do_solid = matches!(mesh.shade_mode, ShadeMode::Textured | ShadeMode::Solid | ShadeMode::SolidWire);
        let do_wire = matches!(mesh.shade_mode, ShadeMode::Wireframe | ShadeMode::SolidWire);
        let has_uvs = mesh.uvs.len() == mesh.verts.len() && !mesh.uvs.is_empty();

        const MAT_PALETTE: [[f32; 3]; 12] = [
            [0.82, 0.84, 0.88], [0.24, 0.26, 0.30], [0.18, 0.24, 0.32], [0.72, 0.75, 0.80],
            [0.34, 0.36, 0.40], [0.86, 0.24, 0.24], [0.48, 0.50, 0.56], [0.28, 0.24, 0.22],
            [0.78, 0.72, 0.62], [0.38, 0.44, 0.52], [0.64, 0.66, 0.70], [0.20, 0.21, 0.24],
        ];
        let l1 = [0.42f32, 0.65f32, 0.63f32];
        let l2 = [-0.60f32, 0.25f32, -0.45f32];

        if do_solid {
            let vis = mesh.visible_tris();
            let has_preview_texture = do_textured && has_uvs
                && mesh.geom_materials.iter().any(|m| m.pass == MaterialPass::Cutout && m.texture.is_none())
                && mesh.tri_geom.get(vis.clone()).is_some_and(|ids| {
                    ids.iter().any(|&g| mesh.geom_materials.get(g as usize).is_some_and(|m| m.texture.is_some()))
                });
            let use_blending = do_textured && mesh.texture.is_none()
                && mesh.geom_materials.iter().any(|m| m.blends(has_uvs));
            let blend_material = |ti: usize| {
                if !use_blending { return None; }
                let g = mesh.tri_geom.get(ti).copied().unwrap_or(0) as usize;
                mesh.geom_materials.get(g).filter(|m| m.blends(has_uvs) && !m.auxiliary_pass)
            };
            mesh.blend_order.clear();
            if use_blending {
                for ti in vis.clone() {
                    if let Some(mat) = blend_material(ti) {
                        let [a, b, c] = mesh.tris[ti].map(|i| mesh.proj_buf[i as usize][2]);
                        mesh.blend_order.push((ti, a + b + c, mat.pass == MaterialPass::Decal));
                    }
                }
                mesh.blend_order.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.total_cmp(&b.1)));
            }
            let solid_tris = vis.filter(|&ti| blend_material(ti).is_none());
            for ti in solid_tris.chain(mesh.blend_order.iter().map(|&(ti, _, _)| ti)) {
                let t = mesh.tris[ti];
                let g = mesh.tri_geom.get(ti).copied().unwrap_or(0) as usize;
                let gmat = mesh.geom_materials.get(g);
                let active_tex: Option<&Texture2D> = if do_textured && has_uvs {
                    mesh.texture.as_ref().or_else(|| gmat.and_then(|m| m.texture.as_deref()))
                } else {
                    None
                };
                if do_textured && gmat.is_some_and(|m| m.auxiliary_pass
                    || (m.pass == MaterialPass::Cutout && active_tex.is_none() && has_preview_texture))
                {
                    continue;
                }
                let blend = blend_material(ti).is_some();
                let pass = if do_textured && mesh.texture.is_some() {
                    MaterialPass::Cutout
                } else if do_textured {
                    gmat.map_or(MaterialPass::Opaque, |m| m.pass)
                } else {
                    MaterialPass::Opaque
                };
                let glass = matches!(pass, MaterialPass::Glass { .. });
                let depth_bias = if blend && pass == MaterialPass::Decal { mesh.radius * 0.0001 } else { 0.0 };

                let i0 = t[0] as usize;
                let i1 = t[1] as usize;
                let i2 = t[2] as usize;
                let p0 = mesh.proj_buf[i0];
                let p1 = mesh.proj_buf[i1];
                let p2 = mesh.proj_buf[i2];

                let min_x = p0[0].min(p1[0]).min(p2[0]);
                let max_x = p0[0].max(p1[0]).max(p2[0]);
                let min_y = p0[1].min(p1[1]).min(p2[1]);
                let max_y = p0[1].max(p1[1]).max(p2[1]);
                if max_x < 0.0 || min_x >= wf || max_y < 0.0 || min_y >= hf {
                    continue;
                }

                let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
                let abs_area = area.abs();
                if abs_area < 0.15 {
                    continue;
                }

                let n = mesh.normals[ti];
                let nx1 = n[0] * cy - n[1] * sy;
                let ny1 = n[0] * sy + n[1] * cy;
                let ny2 = -ny1 * cp + n[2] * sp;
                let nz2 = n[2] * cp + ny1 * sp;

                let ndl1 = (nx1 * l1[0] + ny2 * l1[1] + nz2 * l1[2]).abs();
                let ndl2 = (nx1 * l2[0] + ny2 * l2[1] + nz2 * l2[2]).max(0.0);
                let spec = ndl1.powi(10) * 0.35;
                let rim = (1.0 - ny2.abs()).powi(2) * 0.22;

                let shade_r = (0.18 + ndl1 * 0.72 + spec + rim * 0.45).min(1.15);
                let shade_g = (0.18 + ndl1 * 0.72 + ndl2 * 0.18 + spec + rim * 0.40).min(1.15);
                let shade_b = (0.20 + ndl1 * 0.70 + ndl2 * 0.28 + spec + rim * 0.55).min(1.15);

                let is_paint = do_textured
                    && active_tex.is_none()
                    && gmat.map(|m| m.is_vehicle_paint).unwrap_or(false);
                let base_albedo = if is_paint {
                    mesh.paint_color
                } else if glass {
                    [0.62, 0.71, 0.78]
                } else if do_textured && active_tex.is_none() {
                    MAT_PALETTE[g % MAT_PALETTE.len()]
                } else {
                    [0.86, 0.88, 0.92]
                };
                let clearcoat = if is_paint { ndl1.powi(24) * 0.55 + rim * 0.14 } else { 0.0 };
                let flat_col = Color32::from_rgb(
                    ((base_albedo[0] * shade_r + clearcoat).min(1.0) * 255.0) as u8,
                    ((base_albedo[1] * shade_g + clearcoat).min(1.0) * 255.0) as u8,
                    ((base_albedo[2] * shade_b + clearcoat).min(1.0) * 255.0) as u8,
                );

                let ix0 = (min_x.floor() as i32).max(0) as usize;
                let ix1 = (max_x.ceil() as i32).min(w as i32 - 1) as usize;
                let iy0 = (min_y.floor() as i32).max(0) as usize;
                let iy1 = (max_y.ceil() as i32).min(h as i32 - 1) as usize;

                // flip sign so edge funcs stay positive inside triangls
                let sign = if area > 0.0 { -1.0 } else { 1.0 };
                let inv_abs = 1.0 / abs_area;

                let e01_dx = (p1[1] - p0[1]) * sign;
                let e01_dy = (p0[0] - p1[0]) * sign;
                let e12_dx = (p2[1] - p1[1]) * sign;
                let e12_dy = (p1[0] - p2[0]) * sign;
                let e20_dx = (p0[1] - p2[1]) * sign;
                let e20_dy = (p2[0] - p0[0]) * sign;

                let px_start = ix0 as f32 + 0.5;
                let py_start = iy0 as f32 + 0.5;

                let mut w0_row = (px_start - p1[0]) * e12_dx + (py_start - p1[1]) * e12_dy;
                let mut w1_row = (px_start - p2[0]) * e20_dx + (py_start - p2[1]) * e20_dy;
                let mut w2_row = (px_start - p0[0]) * e01_dx + (py_start - p0[1]) * e01_dy;
                let span_len = ix1 - ix0 + 1;

                if let Some(tex) = active_tex {
                    let uv0 = mesh.uvs[i0];
                    let uv1 = mesh.uvs[i1];
                    let uv2 = mesh.uvs[i2];
                    for y in iy0..=iy1 {
                        let row_start = y * w + ix0;
                        let z_row = &mut mesh.z_buf[row_start..row_start + span_len];
                        let c_row = &mut mesh.color_buf[row_start..row_start + span_len];
                        let (mut w0, mut w1, mut w2) = (w0_row, w1_row, w2_row);
                        for x in 0..span_len {
                            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                                let (b0, b1, b2) = (w0 * inv_abs, w1 * inv_abs, w2 * inv_abs);
                                let z = b0 * p0[2] + b1 * p1[2] + b2 * p2[2];
                                if z + depth_bias > z_row[x] {
                                    let tc = tex.sample(
                                        b0 * uv0[0] + b1 * uv1[0] + b2 * uv2[0],
                                        b0 * uv0[1] + b1 * uv1[1] + b2 * uv2[1],
                                    );
                                    let alpha = match pass {
                                        MaterialPass::Opaque => 255,
                                        MaterialPass::Cutout => if tc.a() >= 128 { 255 } else { 0 },
                                        MaterialPass::Glass { texture_alpha: false } => GLASS_PREVIEW_ALPHA,
                                        _ => tc.a(),
                                    };
                                    if alpha != 0 {
                                        let [r, g, b, _] = if pass != MaterialPass::Opaque && tc.a() != 255 {
                                            tc.to_srgba_unmultiplied()
                                        } else {
                                            tc.to_array()
                                        };
                                        let src = Color32::from_rgb(
                                            ((r as f32 * shade_r).min(255.0)) as u8,
                                            ((g as f32 * shade_g).min(255.0)) as u8,
                                            ((b as f32 * shade_b).min(255.0)) as u8,
                                        );
                                        if blend {
                                            blend_preview_pixel(&mut c_row[x], src, alpha);
                                        } else {
                                            z_row[x] = z;
                                            c_row[x] = src;
                                        }
                                    }
                                }
                            }
                            w0 += e12_dx;
                            w1 += e20_dx;
                            w2 += e01_dx;
                        }
                        w0_row += e12_dy;
                        w1_row += e20_dy;
                        w2_row += e01_dy;
                    }
                    continue;
                }

                for y in iy0..=iy1 {
                    let row_start = y * w + ix0;
                    let z_row = &mut mesh.z_buf[row_start..row_start + span_len];
                    let c_row = &mut mesh.color_buf[row_start..row_start + span_len];
                    let (mut w0, mut w1, mut w2) = (w0_row, w1_row, w2_row);
                    for x in 0..span_len {
                        if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                            let z = (w0 * p0[2] + w1 * p1[2] + w2 * p2[2]) * inv_abs;
                            if z > z_row[x] {
                                if glass {
                                    blend_preview_pixel(&mut c_row[x], flat_col, GLASS_PREVIEW_ALPHA);
                                } else {
                                    z_row[x] = z;
                                    c_row[x] = flat_col;
                                }
                            }
                        }
                        w0 += e12_dx;
                        w1 += e20_dx;
                        w2 += e01_dx;
                    }
                    w0_row += e12_dy;
                    w1_row += e20_dy;
                    w2_row += e01_dy;
                }
            }
        }

        if do_wire {
            let wire_col = if do_solid { Color32::from_rgb(252, 175, 23) } else { Color32::from_rgb(72, 210, 255) };
            let z_bias = if do_solid { mesh.radius * 0.004 } else { 0.0 };
            let vis = mesh.visible_tris();
            let step = (vis.len() / 25_000).max(1);
            for t in mesh.tris[vis].iter().step_by(step) {
                let p0 = mesh.proj_buf[t[0] as usize];
                let p1 = mesh.proj_buf[t[1] as usize];
                let p2 = mesh.proj_buf[t[2] as usize];
                draw_line_z(&mut mesh.color_buf, &mut mesh.z_buf, w, h, p0, p1, wire_col, z_bias);
                draw_line_z(&mut mesh.color_buf, &mut mesh.z_buf, w, h, p1, p2, wire_col, z_bias);
                draw_line_z(&mut mesh.color_buf, &mut mesh.z_buf, w, h, p2, p0, wire_col, z_bias);
            }
        }
    } else if !mesh.edges.is_empty() {
        let col = Color32::from_rgb(72, 210, 255);
        for &(a, b) in mesh.edges.iter().take(60_000) {
            draw_line_z(&mut mesh.color_buf, &mut mesh.z_buf, w, h, mesh.proj_buf[a as usize], mesh.proj_buf[b as usize], col, 0.0);
        }
    } else {
        let col = Color32::from_rgb(252, 175, 23);
        for p0 in &mesh.proj_buf {
            let xi = p0[0] as i32;
            let yi = p0[1] as i32;
            if xi >= 1 && yi >= 1 && (xi as usize) + 1 < w && (yi as usize) + 1 < h {
                let idx = (yi as usize) * w + (xi as usize);
                if p0[2] > mesh.z_buf[idx] {
                    mesh.z_buf[idx] = p0[2];
                    mesh.color_buf[idx] = col;
                }
            }
        }
    }
}

// drawable picker for multi part meshes drawable dictionaries noop otherwise
pub fn part_picker(ui: &mut egui::Ui, mesh: &mut Mesh3D) {
    if mesh.parts.len() < 2 {
        return;
    }
    let label = |m: &Mesh3D, i: Option<usize>| match i.and_then(|i| m.parts.get(i)) {
        Some(p) => format!("{} ({} geoms, {} tris)", p.name, p.geoms, p.tris.len()),
        None => format!("All {} drawables (grid)", m.parts.len()),
    };
    let mut sel = mesh.active_part;
    egui::ComboBox::from_id_salt("mesh_part_picker")
        .selected_text(label(mesh, sel))
        .width(260.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut sel, None, label(mesh, None));
            for i in 0..mesh.parts.len() {
                ui.selectable_value(&mut sel, Some(i), label(mesh, Some(i)));
            }
        })
        .response
        .on_hover_text("Drawable dictionary: independent models, each built around its own origin. Shown side by side; pick one to isolate it.");
    if sel != mesh.active_part {
        mesh.set_active_part(sel);
    }
}

pub fn render_mesh3d_viewport(ui: &mut egui::Ui, mesh: &mut Mesh3D) {
    ui.horizontal_wrapped(|ui| {
        let bone_s = if mesh.bone_count > 0 { format!("  ·  {} bones", mesh.bone_count) } else { String::new() };
        ui.label(
            RichText::new(format!(
                "{}  ·  {} verts  ·  {} tris  ·  {} geoms / {} meshlets ({}){}",
                mesh.name, mesh.verts.len(), mesh.tris.len(), mesh.geom_count, mesh.meshlet_count, mesh.lod_label, bone_s
            ))
            .strong()
            .color(Color32::from_rgb(252, 175, 23)),
        );
        part_picker(ui, mesh);
        ui.add_space(6.0);
        for (mode, label) in [
            (ShadeMode::Textured, "Textured/Mat"),
            (ShadeMode::Solid, "Clay Solid"),
            (ShadeMode::SolidWire, "Solid+Wire"),
            (ShadeMode::Wireframe, "Wire"),
        ] {
            if ui.selectable_label(mesh.shade_mode == mode, label).clicked() {
                mesh.shade_mode = mode;
                mesh.dirty = true;
            }
        }
        if ui.selectable_label(mesh.auto_spin, "Spin").clicked() {
            mesh.auto_spin = !mesh.auto_spin;
            mesh.dirty = true;
        }
        if ui.button("Load Texture...").clicked() {
            if let Some(tp) = rfd::FileDialog::new().add_filter("Textures", &["png", "bmp"]).pick_file() {
                if let Ok(img) = image::open(&tp) {
                    let rgba = img.to_rgba8();
                    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
                    let pixels: Vec<Color32> = rgba
                        .chunks_exact(4)
                        .map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
                        .collect();
                    mesh.texture = Some(Texture2D { w, h, pixels });
                    mesh.tex_status = format!("Loaded {} ({w}x{h})", tp.file_name().unwrap_or_default().to_string_lossy());
                    mesh.shade_mode = ShadeMode::Textured;
                    mesh.dirty = true;
                }
            }
        }
        if ui.button("Reset").clicked() {
            mesh.yaw = DEFAULT_YAW;
            mesh.pitch = DEFAULT_PITCH;
            mesh.zoom = 1.0;
            mesh.pan_x = 0.0;
            mesh.pan_y = 0.0;
            mesh.dirty = true;
        }
        if ui.button("Export .OBJ").clicked() {
            let safe_name = mesh.name.replace('#', "");
            if let Some(p) = rfd::FileDialog::new().set_file_name(format!("{safe_name}.obj")).save_file() {
                let _ = mesh.export_obj(&p);
            }
        }
        if mesh.has_paint_geom {
            ui.add_space(4.0);
            ui.label(RichText::new("Paint:").size(11.0).color(Color32::from_rgb(175, 185, 205)));
            for &(label, rgb) in &PAINT_PRESETS {
                let col = Color32::from_rgb(
                    (rgb[0] * 255.0) as u8,
                    (rgb[1] * 255.0) as u8,
                    (rgb[2] * 255.0) as u8,
                );
                let active = (mesh.paint_color[0] - rgb[0]).abs() < 0.02
                    && (mesh.paint_color[1] - rgb[1]).abs() < 0.02
                    && (mesh.paint_color[2] - rgb[2]).abs() < 0.02;
                let mut btn = egui::Button::new("  ").fill(col).min_size(egui::vec2(16.0, 16.0));
                if active {
                    btn = btn.stroke(egui::Stroke::new(2.0_f32, Color32::WHITE));
                }
                if ui.add(btn).on_hover_text(label).clicked() {
                    mesh.paint_color = rgb;
                    mesh.shade_mode = ShadeMode::Textured;
                    mesh.dirty = true;
                }
            }
            if ui.color_edit_button_rgb(&mut mesh.paint_color).changed() {
                mesh.shade_mode = ShadeMode::Textured;
                mesh.dirty = true;
            }
        }
        ui.label(RichText::new(&mesh.tex_status).size(11.0).color(Color32::from_rgb(130, 145, 168)));
    });
    ui.add_space(4.0);

    let avail = ui.available_size();
    let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

    let mut dragging = false;
    if resp.dragged_by(egui::PointerButton::Primary) {
        let d = resp.drag_delta();
        mesh.yaw += d.x * 0.01;
        mesh.pitch = (mesh.pitch + d.y * 0.01).clamp(-1.5, 1.5);
        mesh.dirty = true;
        dragging = true;
    } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
        let d = resp.drag_delta();
        mesh.pan_x += d.x / rect.width().max(1.0);
        mesh.pan_y += d.y / rect.height().max(1.0);
        mesh.dirty = true;
        dragging = true;
    }
    if resp.double_clicked() {
        mesh.yaw = DEFAULT_YAW;
        mesh.pitch = DEFAULT_PITCH;
        mesh.zoom = 1.0;
        mesh.pan_x = 0.0;
        mesh.pan_y = 0.0;
        mesh.dirty = true;
    }
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            mesh.zoom = (mesh.zoom * (1.0 + scroll * 0.0022)).clamp(0.15, 25.0);
            mesh.dirty = true;
        }
        // codewalker-style wasd + q/e camera navigation
        let (fw, rt, up) = ui.input(|i| {
            let f = (i.key_down(egui::Key::W) as i32 - i.key_down(egui::Key::S) as i32) as f32;
            let r = (i.key_down(egui::Key::D) as i32 - i.key_down(egui::Key::A) as i32) as f32;
            let u = (i.key_down(egui::Key::E) as i32 - i.key_down(egui::Key::Q) as i32) as f32;
            (f, r, u)
        });
        if fw != 0.0 || rt != 0.0 || up != 0.0 {
            mesh.zoom = (mesh.zoom * (1.0 + fw * 0.03)).clamp(0.15, 25.0);
            mesh.pan_x -= rt * 0.015;
            mesh.pan_y += up * 0.015;
            mesh.dirty = true;
            ui.ctx().request_repaint();
        }
    }
    if mesh.auto_spin && !dragging {
        mesh.yaw += 0.014;
        mesh.dirty = true;
        ui.ctx().request_repaint();
    }
    if mesh.last_dragging && !dragging {
        mesh.dirty = true;
    }
    mesh.last_dragging = dragging;

    let heavy = mesh.visible_tris().len() > 25_000;
    let ppp = ui.ctx().pixels_per_point();
    let res_scale = ppp * if dragging && heavy { 0.55 } else { 1.0 };
    let target_w = ((rect.width() * res_scale) as usize).clamp(160, 1920);
    let target_h = ((rect.height() * res_scale) as usize).clamp(120, 1200);

    if mesh.dirty || mesh.fb_w != target_w || mesh.fb_h != target_h || mesh.tex.is_none() {
        rasterize_mesh(mesh, target_w, target_h);
        let ci = ColorImage { size: [target_w, target_h], pixels: mesh.color_buf.clone() };
        if let Some(tex) = &mut mesh.tex {
            tex.set(ci, TextureOptions::LINEAR);
        } else {
            mesh.tex = Some(ui.ctx().load_texture("percystar_3d_fb", ci, TextureOptions::LINEAR));
        }
        mesh.dirty = false;
    }

    let p = ui.painter_at(rect);
    p.rect_filled(rect, CornerRadius::same(6), Color32::from_rgb(11, 12, 17));
    if let Some(tex) = &mesh.tex {
        p.image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
    }
    p.text(
        egui::pos2(rect.min.x + 10.0, rect.max.y - 16.0),
        egui::Align2::LEFT_BOTTOM,
        "Left-drag: Orbit  ·  Right-drag: Pan  ·  Scroll: Zoom  ·  Double-click: Reset",
        egui::FontId::proportional(11.0),
        Color32::from_rgba_unmultiplied(180, 188, 204, 150),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upright_and_not_mirrored() {
        let verts = vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]];
        let mut m = Mesh3D::from_parts("t".into(), verts, vec![], vec![], vec![], vec![], None, String::new(), 1, 0, String::new());
        m.yaw = 0.0;
        m.pitch = 0.4;
        rasterize_mesh(&mut m, 200, 200);
        let p = &m.proj_buf;
        assert!(p[1][1] < p[0][1], "+Z must be screen-up");
        assert!(p[2][0] > 100.0, "+X must be screen-right (no mirror)");
        assert!(p[3][1] < p[4][1], "positive pitch views from above");
        assert!(p[4][2] > p[3][2], "-Y side is closer to camera");
    }

    #[test]
    fn cutout_holes_do_not_occlude_geometry() {
        let verts = vec![
            [-1.0, -0.1, -1.0], [1.0, -0.1, -1.0], [0.0, -0.1, 1.0],
            [-1.0, 0.1, -1.0], [1.0, 0.1, -1.0], [0.0, 0.1, 1.0],
        ];
        let mut m = Mesh3D::from_parts(
            "cutout".into(), verts, vec![[0.0; 2]; 6], vec![[0, 1, 2], [3, 4, 5]],
            vec![0, 1], vec![], None, String::new(), 2, 0, String::new(),
        );
        m.yaw = 0.0;
        m.pitch = 0.0;
        m.geom_materials = vec![
            GeomMaterial {
                texture: Some(Arc::new(Texture2D { w: 1, h: 1, pixels: vec![Color32::TRANSPARENT] })),
                pass: MaterialPass::Cutout,
                ..Default::default()
            },
            GeomMaterial {
                texture: Some(Arc::new(Texture2D { w: 1, h: 1, pixels: vec![Color32::GREEN] })),
                ..Default::default()
            },
        ];
        let center = 50 * 100 + 50;
        rasterize_mesh(&mut m, 100, 100);
        assert!(m.color_buf[center].g() > 0, "back triangle shows through hole");
        assert!(m.z_buf[center] < 0.0, "hole must not write depth");
        Arc::get_mut(m.geom_materials[0].texture.as_mut().unwrap()).unwrap().pixels[0] = Color32::RED;
        rasterize_mesh(&mut m, 100, 100);
        assert!(m.color_buf[center].r() > 0 && m.z_buf[center] > 0.0, "wire writes depth");
    }

    #[test]
    fn blended_layers_respect_depth_and_decal_order() {
        let verts = [-0.2, 0.0, 0.0, -0.1, 0.2].into_iter()
            .flat_map(|y| [[-1.0, y, -1.0], [1.0, y, -1.0], [0.0, y, 1.0]])
            .collect();
        let mut m = Mesh3D::from_parts(
            "layers".into(), verts, vec![[0.0; 2]; 15], (0..5).map(|i| [i * 3, i * 3 + 1, i * 3 + 2]).collect(),
            (0..5).collect(), vec![], None, String::new(), 5, 0, String::new(),
        );
        m.yaw = 0.0;
        m.pitch = 0.0;
        let glass = MaterialPass::Glass { texture_alpha: true };
        m.geom_materials = [
            ([255, 0, 0, 128], glass), ([0, 0, 255, 255], MaterialPass::Opaque),
            ([0, 255, 0, 128], MaterialPass::Decal), ([0, 255, 0, 128], glass),
            ([255, 255, 255, 255], glass),
        ].into_iter().map(|([r, g, b, a], pass)| GeomMaterial {
            texture: Some(Arc::new(Texture2D { w: 1, h: 1, pixels: vec![Color32::from_rgba_unmultiplied(r, g, b, a)] })),
            pass,
            ..Default::default()
        }).collect();
        let center = 50 * 100 + 50;
        rasterize_mesh(&mut m, 100, 100);
        let composited = m.color_buf[center];
        assert_eq!(m.z_buf[center], 0.0, "only opaque layer writes depth");
        assert!(composited.r() > composited.g() && composited.b() > 0, "near red over green with blue visible");
        m.geom_materials[4].texture = None;
        m.geom_materials[4].pass = MaterialPass::Opaque;
        rasterize_mesh(&mut m, 100, 100);
        assert_eq!(m.color_buf[center], composited, "hidden layer must not affect pixel");
        m.geom_materials[2].texture = None;
        m.geom_materials[2].auxiliary_pass = true;
        rasterize_mesh(&mut m, 100, 100);
        assert!(m.color_buf[center].g() < composited.g(), "coplanar decal blends over base");
    }

    #[test]
    fn model_name_and_vehicle_filter() {
        let mut buf = vec![0u8; 30_000];
        buf[100..116].copy_from_slice(b"test_vehicle.#ft");
        buf[500..508].copy_from_slice(b"wheel_lf");
        let (nm, ext) = extract_rage9_model_name(&buf).unwrap();
        assert_eq!(nm, "test_vehicle.#ft");
        assert_eq!(ext, ".#ft");
        assert!(is_vehicle_virtual(&buf, &nm));
        assert!(!is_vehicle_virtual(&buf, "test_vehicle_spoiler_01.#ft"));
    }
}

#[cfg(test)]
mod dictionary_decode_tests {
    use super::*;

    // regression for bk_02_road_str_slod drawable dictionary splitting and +0x6c axis order
    #[test]
    fn brickell_slod_dictionary_splits_and_stands_upright() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        if !Path::new(&rpf).exists() {
            return;
        }
        let m = decode_rage9_model_from_rpf(Path::new(&rpf), 8_679_358_480, "x").expect("decode dictionary");
        assert_eq!(m.parts.len(), 8, "expected 8 drawables");
        assert_eq!(m.geom_count, 317);
        let tower = m.parts.iter().find(|p| p.name == "bk_04_slod.#dr").expect("bk_04_slod part");
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for t in &m.tris[tower.tris.clone()] {
            for &i in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(m.verts[i as usize][k]);
                    hi[k] = hi[k].max(m.verts[i as usize][k]);
                }
            }
        }
        // drawable bbox x 224m y 197m z 208m
        let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        assert!((ext[0] - 224.2).abs() < 2.0 && (ext[1] - 197.1).abs() < 2.0 && (ext[2] - 207.7).abs() < 2.0, "extents {ext:?}");
    }

    // external ttpf material resolution on vehicle paint and world slod
    #[test]
    fn ttpf_vehicle_and_world_materials() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        if !Path::new(&rpf).exists() {
            return;
        }
        let cases = [
            ("cheetah2", 74_051_276_816u64),
            ("cheetah2_hi", 74_053_521_424u64),
            ("brickell_slod", 8_679_358_480u64),
        ];
        let out_dir = std::env::var("PERCYSTAR_RENDER_DIR").ok();
        if let Some(ref d) = out_dir {
            let _ = fs::create_dir_all(d);
        }
        for (tag, off) in cases {
            let mut m = decode_rage9_model_from_rpf(Path::new(&rpf), off, tag).expect("decode model");
            let tex_mats = m.geom_materials.iter().filter(|gm| gm.texture.is_some()).count();
            let paint_mats = m.geom_materials.iter().filter(|gm| gm.is_vehicle_paint).count();
            eprintln!("[{tag}] status='{}' geoms={} tex_mats={} paint_mats={}", m.tex_status, m.geom_count, tex_mats, paint_mats);
            if tag.starts_with("cheetah2") {
                assert!(tex_mats > 0, "{tag} should resolve external TTPF textures");
                assert!(paint_mats > 0 && m.has_paint_geom, "{tag} should detect vehicle paint geoms");
            } else {
                assert!(tex_mats > 0, "{tag} should resolve external TTPF textures");
            }
            if let Some(ref d) = out_dir {
                rasterize_mesh(&mut m, 800, 600);
                let mut rgba = Vec::with_capacity(800 * 600 * 4);
                for c in &m.color_buf {
                    rgba.extend_from_slice(&c.to_srgba_unmultiplied());
                }
                image::save_buffer(Path::new(d).join(format!("{tag}.png")), &rgba, 800, 600, image::ColorType::Rgba8)
                    .expect("save render");
            }
        }
    }
}

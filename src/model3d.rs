use eframe::egui::{self, Color32, ColorImage, CornerRadius, RichText, TextureHandle, TextureOptions};
use flate2::{Decompress, FlushDecompress, Status};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, OnceLock};

mod export;
mod raster;
mod shading;
mod texture;
mod vertex_stream;

pub use shading::NormalSource;
pub use texture::Texture2D;
pub use raster::rasterize_mesh;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ShadeMode {
    #[default]
    Textured,
    Solid,
    SolidWire,
    Wireframe,
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
    // surface color texture currently applied first enabled entry of color_layers
    pub texture: Option<Arc<Texture2D>>,
    pub is_vehicle_paint: bool,
    pass: MaterialPass,
    auxiliary_pass: bool,
    // tinted opaque vehicle body material shown as car paint while no color texture is applied
    paint_capable: bool,
    // indices into mesh tex_layers color candidates in priority order
    color_layers: Vec<u16>,
    normal_layers: Vec<u16>,
    overlay_layers: Vec<u16>,
    // normal map and multiply overlays currently applied
    normal: Option<Arc<Texture2D>>,
    overlays: Vec<Arc<Texture2D>>,
}

// how an applied texture layer contributes to the preview
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexLayerKind {
    // surface color
    Color,
    // tangent space normal map that perturbs the lighting
    Normal,
    // dirt dust and mask maps multiplied over the surface
    Overlay,
}

// one pooled texture bound by the materials of a mesh that can be switched on or off in the preview
#[derive(Clone)]
pub struct TexLayer {
    pub hash64: u64,
    pub label: String,
    pub kind: TexLayerKind,
    pub w: usize,
    pub h: usize,
    // pool file and descriptor index
    pub source: String,
    // number of drawn geometries whose material binds it
    pub mats: usize,
    pub enabled: bool,
    // part of the default look
    pub default_on: bool,
    hit: crate::tex::TtpfHashHit,
    tex: Option<Arc<Texture2D>>,
    failed: bool,
}

#[derive(Clone, Copy)]
pub struct ElsPoint {
    pub position: [f32; 3],
    pub bank: bool,
    pub slot: u8,
}

// Decoded from the RAGE 9 Sirens PSO preset (id 3) in data.rpf. As in GTA V,
// each light consumes one bit per beat, most-significant bit first. GTA VI keeps
// the 32-step sequencers but stores the carcols-style data in hashed PSO fields.
const RAGE9_SIREN_BPM: f64 = 500.0;
const RAGE9_SIREN_MULTIPLES: u64 = 3;
const RAGE9_SIREN_SEQUENCERS: [u32; 20] = [
    0xAAAA_AAAA, 0xAA00_AA80, 0x5500_5500, 0x5555_5555,
    0xAA5A_A954, 0x5555_554A, 0x5555_554A, 0xAA5A_A954,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

fn rage9_siren_lit(slot: u8, phase: u64) -> bool {
    let Some(&sequencer) = slot.checked_sub(1).and_then(|i| RAGE9_SIREN_SEQUENCERS.get(i as usize)) else {
        return false;
    };
    let subdivisions = RAGE9_SIREN_MULTIPLES * 2;
    let beat = (phase / subdivisions) % 32;
    let pulse = phase % subdivisions;
    sequencer & (1 << (31 - beat)) != 0 && pulse.is_multiple_of(2)
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

const DEFAULT_YAW: f32 = 2.45;
const DEFAULT_PITCH: f32 = 0.38;
// render time budget of a frame while the camera moves
const MOVING_FRAME_MS: f32 = 11.0;

#[derive(Clone)]
pub struct Mesh3D {
    pub name: String,
    pub verts: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
    pub tri_geom: Vec<u16>,
    // unit geometric normal of every triangle
    pub normals: Vec<[f32; 3]>,
    // how shading normals are rebuilt see corner_normals
    pub normal_source: NormalSource,
    // faceted preview that shows the real triangles instead of smooth shading
    pub flat_shading: bool,
    pub edges: Vec<(u32, u32)>,
    pub texture: Option<Texture2D>,
    pub geom_materials: Vec<GeomMaterial>,
    // pooled textures bound by the materials toggled from the viewer
    pub tex_layers: Vec<TexLayer>,
    pub has_paint_geom: bool,
    pub paint_color: [f32; 3],
    pub has_els: bool,
    pub els_enabled: bool,
    pub els_flash: bool,
    pub els_points: Vec<ElsPoint>,
    vehicle_attach_points: Vec<(String, [f32; 3])>,
    els_flash_phase: u64,
    pub tex_status: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub geom_count: usize,
    pub meshlet_count: usize,
    pub bone_count: usize,
    pub lod_label: String,
    // decoded lod level and geometry count per level all zero when the resource has no lod table
    pub lod: usize,
    pub lod_geoms: [usize; MAX_LODS],
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
    id_buf: Vec<u32>,
    proj_buf: Vec<[f32; 4]>,
    fixed_buf: Vec<[i32; 2]>,
    bins: Vec<Vec<u32>>,
    blend_bins: Vec<Vec<u32>>,
    blend_order: Vec<(usize, f32, bool)>,
    corner_normals: OnceLock<Vec<[[f32; 3]; 3]>>,
    tex: Option<TextureHandle>,
    pub dirty: bool,
    // viewport size and quality of the frame on screen
    view_size: (usize, usize),
    view_refined: bool,
    // cost of the last frame in milliseconds per megasample steers the resolution while the camera moves
    frame_cost: f32,
}

impl Mesh3D {
    // rebinds every material to its enabled texture layers decoding newly enabled ones on demand
    pub fn apply_tex_layers(&mut self) {
        let mut budget = 384usize;
        for l in &mut self.tex_layers {
            if l.enabled && l.tex.is_none() && !l.failed && budget > 0 {
                budget -= 1;
                l.tex = crate::tex::decode_ttpf_texture_cached(&l.hit, l.hash64);
                l.failed = l.tex.is_none();
            }
        }
        let layers = &self.tex_layers;
        let live = |ids: &[u16]| -> Vec<Arc<Texture2D>> {
            ids.iter()
                .filter_map(|&i| layers.get(i as usize))
                .filter(|l| l.enabled)
                .filter_map(|l| l.tex.clone())
                .collect()
        };
        for m in &mut self.geom_materials {
            // highest priority ticked color layer wins even when it could not be decoded
            m.texture = m.color_layers.iter().filter_map(|&i| layers.get(i as usize)).find(|l| l.enabled).and_then(|l| l.tex.clone());
            m.normal = live(&m.normal_layers).into_iter().next();
            m.overlays = live(&m.overlay_layers);
            if matches!(m.pass, MaterialPass::Glass { .. }) {
                if m.texture.as_ref().is_some_and(|t| t.w <= 4 && t.h <= 4 && t.pixels.iter().all(|p| p.a() == 0)) {
                    m.texture = None;
                }
                m.pass = MaterialPass::Glass {
                    texture_alpha: m.texture.as_ref().is_some_and(|t| t.pixels.iter().any(|p| p.a() < 255)),
                };
            }
            // an enabled color texture that failed to decode keeps the neutral material look instead of paint
            let color_bound = m.color_layers.iter().any(|&i| layers.get(i as usize).is_some_and(|l| l.enabled));
            m.is_vehicle_paint = m.paint_capable && !color_bound;
        }
        self.has_paint_geom = self.geom_materials.iter().any(|m| m.paint_capable);
        self.refresh_tex_status();
        self.dirty = true;
    }

    // switches layers on or off by predicate and reapplies them
    pub fn set_tex_layers(&mut self, on: impl Fn(&TexLayer) -> bool) {
        for l in &mut self.tex_layers {
            l.enabled = on(l);
        }
        self.apply_tex_layers();
    }

    fn refresh_tex_status(&mut self) {
        if self.geom_materials.is_empty() {
            return;
        }
        if let Some(at) = self.tex_status.find(" + ELS ") {
            self.tex_status.truncate(at);
        }
        if self.texture.is_none() {
            let textured = self.geom_materials.iter().filter(|m| m.texture.is_some()).count();
            let paint = self.geom_materials.iter().filter(|m| m.is_vehicle_paint).count();
            let drawn = self.geom_materials.iter().filter(|m| !m.auxiliary_pass).count().max(1);
            if textured > 0 && paint > 0 {
                self.tex_status = format!("TTPF: {textured}/{drawn} mats + Car Paint");
            } else if textured > 0 {
                self.tex_status = format!("TTPF: {textured}/{drawn} mats");
            } else if paint > 0 {
                self.tex_status = format!("Car Paint ({paint}/{drawn} geoms)");
            } else if !self.tex_layers.is_empty() {
                self.tex_status = "TTPF: no color textures applied".into();
            }
        }
        if self.has_els {
            self.tex_status.push_str(&format!(" + ELS {} emitters", self.els_points.len()));
        }
    }

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
            normal_source: NormalSource::Crease,
            flat_shading: false,
            edges,
            texture,
            geom_materials: Vec::new(),
            has_paint_geom: false,
            tex_layers: Vec::new(),
            paint_color: DEFAULT_PAINT_COLOR,
            has_els: false,
            els_enabled: false,
            els_flash: true,
            els_points: Vec::new(),
            vehicle_attach_points: Vec::new(),
            els_flash_phase: 0,
            tex_status,
            center,
            radius,
            geom_count,
            meshlet_count,
            bone_count: 0,
            lod_label,
            lod: 0,
            lod_geoms: [0; MAX_LODS],
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
            id_buf: Vec::new(),
            proj_buf: Vec::new(),
            fixed_buf: Vec::new(),
            bins: Vec::new(),
            blend_bins: Vec::new(),
            blend_order: Vec::new(),
            corner_normals: OnceLock::new(),
            tex: None,
            dirty: true,
            view_size: (0, 0),
            view_refined: false,
            frame_cost: 0.0,
        }
    }

    // unit shading normal of every triangle corner rebuilt from topology on first use
    pub fn corner_normals(&self) -> &[[[f32; 3]; 3]] {
        self.corner_normals.get_or_init(|| shading::corner_normals(&self.verts, &self.tris, &self.normals, self.normal_source))
    }

    // call after replacing vertices or triangles so shading normals are rebuilt
    pub fn invalidate_normals(&mut self) {
        self.corner_normals = OnceLock::new();
        self.dirty = true;
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

    pub fn render_preview_image(&mut self, w: usize, h: usize, path: &Path) -> Result<(), String> {
        rasterize_mesh(self, w, h);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for c in &self.color_buf {
            rgba.extend_from_slice(&c.to_srgba_unmultiplied());
        }
        image::save_buffer(path, &rgba, w as u32, h as u32, image::ColorType::Rgba8)
            .map_err(|e| format!("Failed to save preview image {}: {e}", path.display()))
    }

    // Adds an independently stored vehicle option at the same model-space
    // origin. RAGE 9 tuning parts (bumpers, light bars, liveries, wheels, etc.)
    // are separate fragment resources authored against the parent vehicle.
    pub fn append_vehicle_part(&mut self, mut part: Mesh3D) {
        let offset = if part.name.to_ascii_lowercase().starts_with("cmn_siren_roofbar1_") {
            self.vehicle_attach_points
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("siren_roof_1"))
                .map(|(_, position)| *position)
                .unwrap_or([0.0; 3])
        } else {
            [0.0; 3]
        };
        if offset != [0.0; 3] {
            for vertex in &mut part.verts {
                for axis in 0..3 {
                    vertex[axis] += offset[axis];
                }
            }
            for light in &mut part.els_points {
                for axis in 0..3 {
                    light.position[axis] += offset[axis];
                }
            }
            for (_, point) in &mut part.vehicle_attach_points {
                for axis in 0..3 {
                    point[axis] += offset[axis];
                }
            }
        }
        let vertex_base = self.verts.len() as u32;
        let triangle_base = self.tris.len();
        let geom_base = self.geom_materials.len() as u16;
        let layer_base = self.tex_layers.len() as u16;
        if self.parts.is_empty() && triangle_base > 0 {
            self.parts.push(MeshPart {
                name: self.name.clone(),
                geoms: self.geom_count,
                tris: 0..triangle_base,
            });
        }

        self.verts.append(&mut part.verts);
        self.uvs.append(&mut part.uvs);
        self.normals.append(&mut part.normals);
        self.tris.extend(part.tris.drain(..).map(|t| {
            [t[0] + vertex_base, t[1] + vertex_base, t[2] + vertex_base]
        }));
        self.tri_geom.extend(part.tri_geom.drain(..).map(|g| g.saturating_add(geom_base)));
        self.edges.extend(part.edges.drain(..).map(|(a, b)| (a + vertex_base, b + vertex_base)));

        for material in &mut part.geom_materials {
            for layer in material
                .color_layers
                .iter_mut()
                .chain(&mut material.normal_layers)
                .chain(&mut material.overlay_layers)
            {
                *layer = layer.saturating_add(layer_base);
            }
        }
        self.geom_materials.append(&mut part.geom_materials);
        self.tex_layers.append(&mut part.tex_layers);
        self.els_points.append(&mut part.els_points);
        self.vehicle_attach_points.append(&mut part.vehicle_attach_points);

        let triangle_end = self.tris.len();
        if triangle_end > triangle_base {
            self.parts.push(MeshPart {
                name: part.name,
                geoms: part.geom_count,
                tris: triangle_base..triangle_end,
            });
        }
        self.geom_count += part.geom_count;
        self.meshlet_count += part.meshlet_count;
        self.bone_count = self.bone_count.max(part.bone_count);
        self.has_els |= part.has_els;
        self.active_part = None;
        // merged vertices and triangles invalidate the cached shading normals
        self.invalidate_normals();
        self.set_active_part(None);
        self.apply_tex_layers();
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
pub fn read_virtual_from_file(f: &mut File, file_off: u64) -> Result<(Vec<u8>, u64, usize), String> {
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
    // Meta-only 0x7a resources legitimately use p_cnt = 0 (no graphics pages).
    if !(1..=32).contains(&v_cnt) || p_cnt > 64 {
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

// drawable lod slots high med low vlow at +0x50 +0x58 +0x60 +0x68
pub const MAX_LODS: usize = 4;
pub const LOD_NAMES: [&str; MAX_LODS] = ["LOD0", "LOD1", "LOD2", "LOD3"];

// meshlet geometries of one lod level of drawable at base lod 0 is high detail
fn drawable_lod_geoms(v: &[u8], base: usize, lod: usize) -> Vec<GeomRef> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    if lod >= MAX_LODS || base + 0x80 > v.len() { return out; }
    let Some(lod0_off) = vptr(ru64(v, base + 0x50 + lod * 8)) else { return out };
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

// maps each vb and mb offset of one lod of drawable at base to its shader index and render mask
fn drawable_lod_shader_map(v: &[u8], base: usize, lod: usize) -> HashMap<(usize, usize), (u16, u8)> {
    let mut out = HashMap::new();
    if lod >= MAX_LODS || base + 0x80 > v.len() { return out; }
    let Some(lod0_off) = vptr(ru64(v, base + 0x50 + lod * 8)) else { return out };
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

// texture parameters of the shader at saddr as param name hash and bound 64 bit texture hash none when the slot is empty
fn shader_texture_params(v: &[u8], saddr: usize) -> Vec<(u32, Option<u64>)> {
    let mut out = Vec::new();
    if saddr + 0x28 > v.len() { return out; }
    let (Some(p2), Some(p4)) = (vptr(ru64(v, saddr + 0x10)), vptr(ru64(v, saddr + 0x20))) else { return out };
    if p4 + 8 > v.len() { return out; }
    let tcnt = (v[p4 + 1] as usize).min(48);
    for k in 0..tcnt {
        let ent = p4 + 8 + k * 8;
        if ent + 8 > v.len() { break; }
        let param_hash = ru32(v, ent);
        let slot = (ru32(v, ent + 4) / 4) as usize;
        let tex_hash = (|| {
            if p2 + slot * 8 + 8 > v.len() { return None; }
            let tp = vptr(ru64(v, p2 + slot * 8))?;
            if tp + 0x40 > v.len() || ru64(v, tp + 0x10) != 0x980 { return None; }
            let ref_off = vptr(ru64(v, tp + 0x30))?;
            if ref_off + 16 > v.len() { return None; }
            Some(ru64(v, ref_off + 8))
        })();
        out.push((param_hash, tex_hash));
    }
    out
}

// shader addresses of the shader group of drawable at base
fn drawable_shader_addrs(v: &[u8], base: usize) -> Vec<usize> {
    if base + 0x18 > v.len() { return Vec::new(); }
    let Some(sg) = vptr(ru64(v, base + 0x10)) else { return Vec::new() };
    if sg + 0x20 > v.len() { return Vec::new(); }
    let Some(sp) = vptr(ru64(v, sg + 0x10)) else { return Vec::new() };
    let scnt = ru16(v, sg + 0x18) as usize;
    if scnt == 0 || scnt > 512 || sp + scnt * 8 > v.len() { return Vec::new(); }
    // verify rage9 atem meta shader table tag
    let Some(s0) = vptr(ru64(v, sp)) else { return Vec::new() };
    if s0 + 8 > v.len() || (ru64(v, s0) >> 32) != 0x6d65_7461 { return Vec::new(); }
    (0..scnt).filter_map(|i| vptr(ru64(v, sp + i * 8))).collect()
}

// known shader texture parameter names by joaat hash
pub fn texture_param_name(param_hash: u32) -> Option<&'static str> {
    Some(match param_hash {
        0xc8e8_c282 => "diffuseTex",
        0x2c8d_7182 => "tintPaletteTex",
        0x8e0d_63ee => "emissiveTex",
        0x7485_dafb => "dirtDiffuseTex",
        0xcc15_5f4b => "dirtNoiseTex",
        0xfb03_ae36 => "DustTexture",
        _ => return None,
    })
}

// one external texture a model binds in its shaders
#[derive(Clone, Debug)]
pub struct ModelTexRef {
    pub hash64: u64,
    // shader parameters this texture is bound to
    pub params: Vec<u32>,
    // number of shaders that bind it
    pub shaders: usize,
    // location in the ttpf pools none when no pool holds this hash
    pub hit: Option<crate::tex::TtpfHashHit>,
}

impl ModelTexRef {
    // short role label from the parameter name or the pixel format
    pub fn role(&self) -> String {
        if let Some(n) = self.params.iter().find_map(|&p| texture_param_name(p)) {
            return n.to_string();
        }
        let kind = match &self.hit {
            Some(h) if h.is_normal_format() => "normal",
            Some(h) if h.is_srgb() => "color",
            Some(_) => "linear",
            None => "param",
        };
        format!("{kind}_{:08x}", self.params.first().copied().unwrap_or(0))
    }
}

// every external texture bound by the shaders of the model at file_off in first use order
pub fn list_model_texture_refs(rpf_path: &Path, file_off: u64) -> Result<Vec<ModelTexRef>, String> {
    let mut f = File::open(rpf_path).map_err(|e| format!("Open error: {e}"))?;
    let (v, _, _) = read_virtual_from_file(&mut f, file_off)?;
    let mut bases: Vec<usize> = drawable_dictionary_entries(&v).into_iter().map(|(_, b)| b).collect();
    if bases.is_empty() {
        bases.push(0);
        if v.len() >= 0x20 {
            if let Some(fo) = vptr(ru64(&v, 0x18)) {
                bases.push(fo);
            }
        }
    }
    let idx = crate::tex::ttpf_hash_index_for_rpf(rpf_path);
    let mut out: Vec<ModelTexRef> = Vec::new();
    let mut pos: HashMap<u64, usize> = HashMap::new();
    let mut seen_shaders = HashSet::new();
    for base in bases {
        for saddr in drawable_shader_addrs(&v, base) {
            if !seen_shaders.insert(saddr) { continue; }
            let mut in_shader = HashSet::new();
            for (param, tex_hash) in shader_texture_params(&v, saddr) {
                let Some(h64) = tex_hash else { continue };
                let i = *pos.entry(h64).or_insert_with(|| {
                    out.push(ModelTexRef {
                        hash64: h64,
                        params: Vec::new(),
                        shaders: 0,
                        hit: idx.as_ref().and_then(|m| m.get(&h64).cloned()),
                    });
                    out.len() - 1
                });
                if !out[i].params.contains(&param) { out[i].params.push(param); }
                if in_shader.insert(h64) { out[i].shaders += 1; }
            }
        }
    }
    Ok(out)
}

// resolves external ttpf diffuse textures and vehicle paint shader flags for lod0 geoms
fn resolve_drawable_materials(
    v: &[u8],
    base: usize,
    lod: usize,
    geoms: &[GeomRef],
    ttpf_idx: Option<&HashMap<u64, crate::tex::TtpfHashHit>>,
    is_vehicle: bool,
    layers: &mut Vec<TexLayer>,
    layer_pos: &mut HashMap<u64, u16>,
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

    let smap = drawable_lod_shader_map(v, base, lod);
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
                    let mut has_tint = false;
                    let mut diffuse_cand: Option<u16> = None;
                    let mut alt0_cand: Option<u16> = None;
                    let mut emissive_cand: Option<u16> = None;
                    let mut fallback_srgb: Option<u16> = None;
                    let uses_opacity = !matches!(mat.pass, MaterialPass::Opaque);
                    let mut layer_of = |h64: u64, hit: &crate::tex::TtpfHashHit, param: u32, kind: TexLayerKind| -> Option<u16> {
                        if let Some(&i) = layer_pos.get(&h64) {
                            return Some(i);
                        }
                        let i = u16::try_from(layers.len()).ok()?;
                        let label = match (texture_param_name(param), kind) {
                            (Some(n), _) => n.to_string(),
                            (None, _) if param == ALT0_TEX => "liveryDetail".to_string(),
                            (None, TexLayerKind::Color) => format!("color_{param:08x}"),
                            (None, TexLayerKind::Normal) => format!("normal_{param:08x}"),
                            (None, TexLayerKind::Overlay) => format!("overlay_{param:08x}"),
                        };
                        layers.push(TexLayer {
                            hash64: h64,
                            label,
                            kind,
                            w: hit.w,
                            h: hit.h,
                            source: format!("{} #{}", hit.path.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default(), hit.tex_idx),
                            mats: 0,
                            enabled: false,
                            default_on: false,
                            hit: hit.clone(),
                            tex: None,
                            failed: false,
                        });
                        layer_pos.insert(h64, i);
                        Some(i)
                    };

                    for (param_hash, tex_hash) in shader_texture_params(v, saddr) {
                        if param_hash == TINT_PALETTE_TEX {
                            has_tint = true;
                        }
                        let Some(idx_map) = ttpf_idx else { continue };
                        let Some(h64) = tex_hash else { continue };
                        let Some(hit) = idx_map.get(&h64) else { continue };
                        let big = hit.w >= 16 && hit.h >= 16;
                        if hit.is_normal_format() {
                            if big {
                                if let Some(i) = layer_of(h64, hit, param_hash, TexLayerKind::Normal) {
                                    if !mat.normal_layers.contains(&i) { mat.normal_layers.push(i); }
                                }
                            }
                            continue;
                        }
                        if hit.dxgi == 65 && !(uses_opacity && param_hash == DIFFUSE_TEX) { continue; }

                        if param_hash == DIFFUSE_TEX && hit.w >= 4 && hit.h >= 4 {
                            diffuse_cand = layer_of(h64, hit, param_hash, TexLayerKind::Color);
                        } else if param_hash == ALT0_TEX && big {
                            alt0_cand = layer_of(h64, hit, param_hash, TexLayerKind::Color);
                        } else if param_hash == EMISSIVE_TEX && big {
                            emissive_cand = layer_of(h64, hit, param_hash, TexLayerKind::Color);
                        } else if hit.is_srgb()
                            && big
                            && param_hash != DIRT_DIFFUSE_TEX
                            && param_hash != SHARED_LUT_PARAM
                            && fallback_srgb.is_none()
                        {
                            fallback_srgb = layer_of(h64, hit, param_hash, TexLayerKind::Color);
                        } else if big && param_hash != SHARED_LUT_PARAM {
                            if let Some(i) = layer_of(h64, hit, param_hash, TexLayerKind::Overlay) {
                                if !mat.overlay_layers.contains(&i) { mat.overlay_layers.push(i); }
                            }
                        }
                    }

                    // color candidates in priority order tinted vehicle bodies only take diffuse or livery detail
                    let tinted_body = has_tint && is_vehicle;
                    let extra = if tinted_body { [None, None] } else { [emissive_cand, fallback_srgb] };
                    for i in [diffuse_cand, alt0_cand].into_iter().chain(extra).flatten() {
                        if !mat.color_layers.contains(&i) { mat.color_layers.push(i); }
                    }
                    // a texture already used as color or normal is not also multiplied over the same material
                    mat.overlay_layers.retain(|i| !mat.color_layers.contains(i) && !mat.normal_layers.contains(i));
                    mat.paint_capable = tinted_body && mat.pass == MaterialPass::Opaque;
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
    if mb_off + 0x10 > p.len() {
        return;
    }
    let Some(vertex_end) = vb_off.checked_add(vb_size) else { return };
    let Some(vertex_bytes) = p.get(vb_off..vertex_end) else { return };
    let Ok(vertices) = vertex_stream::decode(vertex_bytes) else { return };
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

    let max_vb_verts = vertices.positions.len();

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
    b.verts.extend_from_slice(&vertices.positions[..v_cnt]);
    if let Some(uvs) = &vertices.uvs {
        b.uvs.extend_from_slice(&uvs[..v_cnt]);
    } else {
        b.uvs.resize(b.uvs.len() + v_cnt, [0.0; 2]);
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

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_rotate(q: [f32; 4], p: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let uv = [
        qv[1] * p[2] - qv[2] * p[1],
        qv[2] * p[0] - qv[0] * p[2],
        qv[0] * p[1] - qv[1] * p[0],
    ];
    let uuv = [
        qv[1] * uv[2] - qv[2] * uv[1],
        qv[2] * uv[0] - qv[0] * uv[2],
        qv[0] * uv[1] - qv[1] * uv[0],
    ];
    [
        p[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        p[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        p[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    ]
}

fn rage9_siren_slot(name: &str) -> Option<u16> {
    let lower = name.to_ascii_lowercase();
    let siren_at = lower.rfind("siren")?;
    lower[siren_at + "siren".len()..].parse::<u16>().ok()
}

// RAGE 9 vehicle ELS emitters are skeleton bones whose final token is siren<number>,
// for example siren1, ext_front_siren2, or ext_rear_1_roof_siren1.
// They are independent from emissiveTex, which is commonly the dashboard gauge atlas.
fn extract_rage9_skeleton_points(v: &[u8]) -> (Vec<ElsPoint>, Vec<(String, [f32; 3])>) {
    let mut bases = vec![0usize];
    if v.len() >= 0x20 {
        if let Some(fo) = vptr(ru64(v, 0x18)) {
            if fo + 0x20 <= v.len() { bases.push(fo); }
        }
    }
    for base in bases {
        let Some(so) = (base + 0x20 <= v.len()).then(|| vptr(ru64(v, base + 0x18))).flatten() else { continue };
        if so + 0x70 > v.len() { continue; }
        let count = ru16(v, so + 0x6c) as usize;
        let Some(bones) = vptr(ru64(v, so + 0x28)) else { continue };
        if !(1..=1024).contains(&count) || bones + count * 0x50 > v.len() { continue; }

        let mut world_rot = vec![[0.0, 0.0, 0.0, 1.0]; count];
        let mut world_pos = vec![[0.0; 3]; count];
        let mut world_scale = vec![[1.0; 3]; count];
        let mut out = Vec::new();
        let mut attach_points = Vec::new();
        for i in 0..count {
            let b = bones + i * 0x50;
            let mut rot = [rf32(v, b), rf32(v, b + 4), rf32(v, b + 8), rf32(v, b + 12)];
            let len = (rot.iter().map(|x| x * x).sum::<f32>()).sqrt();
            if len > 1e-6 {
                for x in &mut rot { *x /= len; }
            } else {
                rot = [0.0, 0.0, 0.0, 1.0];
            }
            let pos = [rf32(v, b + 0x10), rf32(v, b + 0x14), rf32(v, b + 0x18)];
            let scale = [rf32(v, b + 0x20), rf32(v, b + 0x24), rf32(v, b + 0x28)];
            let parent = ru16(v, b + 0x46) as usize;
            if parent < i {
                let scaled = [
                    pos[0] * world_scale[parent][0],
                    pos[1] * world_scale[parent][1],
                    pos[2] * world_scale[parent][2],
                ];
                let offset = quat_rotate(world_rot[parent], scaled);
                world_pos[i] = [
                    world_pos[parent][0] + offset[0],
                    world_pos[parent][1] + offset[1],
                    world_pos[parent][2] + offset[2],
                ];
                world_rot[i] = quat_mul(world_rot[parent], rot);
                world_scale[i] = [
                    world_scale[parent][0] * scale[0],
                    world_scale[parent][1] * scale[1],
                    world_scale[parent][2] * scale[2],
                ];
            } else {
                world_pos[i] = pos;
                world_rot[i] = rot;
                world_scale[i] = scale;
            }

            let Some(name_off) = vptr(ru64(v, b + 0x30)) else { continue };
            let end = v[name_off..].iter().position(|&c| c == 0).map(|n| name_off + n).unwrap_or(name_off);
            let Ok(name) = std::str::from_utf8(&v[name_off..end]) else { continue };
            attach_points.push((name.to_string(), world_pos[i]));
            let Some(number) = rage9_siren_slot(name) else { continue };
            out.push(ElsPoint {
                position: world_pos[i],
                bank: number % 2 == 0,
                slot: number.min(u8::MAX as u16) as u8,
            });
        }
        if !out.is_empty() || !attach_points.is_empty() {
            return (out, attach_points);
        }
    }
    (Vec::new(), Vec::new())
}

pub fn decode_rage9_model_from_rpf(rpf_path: &Path, file_off: u64, fallback_name: &str) -> Result<Mesh3D, String> {
    decode_rage9_model_from_rpf_lod(rpf_path, file_off, fallback_name, 0)
}

pub fn decode_rage9_model_from_rpf_opt(rpf_path: &Path, file_off: u64, fallback_name: &str, layout_grid: bool) -> Result<Mesh3D, String> {
    decode_rage9_model_from_rpf_lod_opt(rpf_path, file_off, fallback_name, 0, layout_grid)
}

// decodes one lod level 0 high to 3 vlow falls back to the nearest populated level
pub fn decode_rage9_model_from_rpf_lod(rpf_path: &Path, file_off: u64, fallback_name: &str, want_lod: usize) -> Result<Mesh3D, String> {
    decode_rage9_model_from_rpf_lod_opt(rpf_path, file_off, fallback_name, want_lod, true)
}

pub fn decode_rage9_model_from_rpf_lod_opt(rpf_path: &Path, file_off: u64, fallback_name: &str, want_lod: usize, layout_grid: bool) -> Result<Mesh3D, String> {
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
    // drawable bases that carry lod tables
    let lod_bases: Vec<(String, usize)> = if !dict.is_empty() {
        dict.iter().map(|(nm, base)| (nm.clone(), *base)).collect()
    } else {
        let mut bases = vec![0usize];
        if v.len() >= 0x20 {
            if let Some(fo) = vptr(ru64(&v, 0x18)) {
                if fo + 0x80 <= v.len() { bases.push(fo); }
            }
        }
        bases
            .into_iter()
            .find(|&b| (0..MAX_LODS).any(|l| !drawable_lod_geoms(&v, b, l).is_empty()))
            .map(|b| vec![(model_name.clone(), b)])
            .unwrap_or_default()
    };
    let mut lod_geoms = [0usize; MAX_LODS];
    for (l, slot) in lod_geoms.iter_mut().enumerate() {
        *slot = lod_bases.iter().map(|(_, b)| drawable_lod_geoms(&v, *b, l).len()).sum();
    }
    // requested level or the nearest populated one preferring higher detail
    let want_lod = want_lod.min(MAX_LODS - 1);
    let lod = (0..=want_lod)
        .rev()
        .chain(want_lod + 1..MAX_LODS)
        .find(|&l| lod_geoms[l] > 0)
        .unwrap_or(0);
    let (groups, lod_label): (Vec<Group>, String) = if lod_geoms[lod] > 0 {
        let g: Vec<_> = lod_bases
            .iter()
            .map(|(nm, base)| (nm.clone(), Some(*base), drawable_lod_geoms(&v, *base, lod)))
            .filter(|(_, _, gs)| !gs.is_empty())
            .collect();
        let label = if dict.is_empty() {
            LOD_NAMES[lod].to_string()
        } else {
            format!("{} · dictionary of {}", LOD_NAMES[lod], dict.len())
        };
        (g, label)
    } else {
        (vec![(model_name.clone(), None, collect_all_geoms(&v))], "All".into())
    };
    let geom_count: usize = groups.iter().map(|(_, _, g)| g.len()).sum();
    if geom_count == 0 {
        return Err("No RAGE9 meshlet geometries found in resource".into());
    }

    let (texture, tex_status) = try_extract_embedded_texture(&v, &p);
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
    let mut tex_layers: Vec<TexLayer> = Vec::new();
    let mut layer_pos: HashMap<u64, u16> = HashMap::new();
    let mut gi_idx = 0u16;
    for (name, base_opt, geoms) in &groups {
        let mats = if texture.is_none() {
            if let Some(base) = *base_opt {
                resolve_drawable_materials(&v, base, lod, geoms, ttpf_idx.as_deref(), is_vehicle, &mut tex_layers, &mut layer_pos)
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
    if layout_grid {
        layout_parts_in_grid(&mut b.verts, &part_verts);
    }

    // Optional vehicle fragments are often too small to pass the whole-vehicle
    // classifier but can still carry their own siren emitter bones.
    let (els_points, vehicle_attach_points) = extract_rage9_skeleton_points(&v);

    // default look applies the first color candidate of every drawn material
    for m in geom_materials.iter().filter(|m| !m.auxiliary_pass) {
        for &i in m.color_layers.iter().chain(&m.normal_layers).chain(&m.overlay_layers) {
            tex_layers[i as usize].mats += 1;
        }
        if let Some(&first) = m.color_layers.first() {
            tex_layers[first as usize].default_on = true;
        }
    }
    for l in &mut tex_layers {
        l.enabled = l.default_on;
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
    mesh.normal_source = NormalSource::SharedVertices;
    mesh.geom_materials = geom_materials;
    mesh.tex_layers = tex_layers;
    mesh.has_els = !els_points.is_empty();
    mesh.els_points = els_points;
    mesh.vehicle_attach_points = vehicle_attach_points;
    mesh.apply_tex_layers();
    mesh.bone_count = extract_rage9_bone_count(&v);
    mesh.lod = lod;
    mesh.lod_geoms = lod_geoms;
    if parts.len() > 1 {
        mesh.parts = parts;
    }
    Ok(mesh)
}

// dropdown that switches individual material textures of the mesh on and off
pub fn tex_layer_menu(ui: &mut egui::Ui, mesh: &mut Mesh3D) {
    if mesh.tex_layers.is_empty() || mesh.texture.is_some() {
        return;
    }
    let on = mesh.tex_layers.iter().filter(|l| l.enabled).count();
    let mut changed = false;
    let mut preset: Option<fn(&TexLayer) -> bool> = None;
    ui.menu_button(format!("Applied textures {on}/{} ⏷", mesh.tex_layers.len()), |ui| {
        ui.set_min_width(430.0);
        ui.horizontal(|ui| {
            if ui.button("Default").on_hover_text("One color texture per material").clicked() { preset = Some(|l| l.default_on); }
            if ui.button("Color + normals").clicked() { preset = Some(|l| l.default_on || l.kind == TexLayerKind::Normal); }
            if ui.button("None").clicked() { preset = Some(|_| false); }
        });
        ui.label(
            RichText::new("Color: first ticked one per material is drawn. Normal: bends the lighting. Overlay: multiplied on top.")
                .size(10.5)
                .color(Color32::from_rgb(150, 160, 180)),
        );
        ui.separator();
        egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
            for (kind, title) in [
                (TexLayerKind::Color, "Color"),
                (TexLayerKind::Normal, "Normal maps"),
                (TexLayerKind::Overlay, "Overlays (dirt, dust, masks)"),
            ] {
                if !mesh.tex_layers.iter().any(|l| l.kind == kind) {
                    continue;
                }
                ui.label(RichText::new(title).strong().size(11.5).color(Color32::from_rgb(255, 168, 28)));
                for l in mesh.tex_layers.iter_mut().filter(|l| l.kind == kind) {
                    let text = format!("{}  ·  {}x{}  ·  {} mat{}", l.label, l.w, l.h, l.mats, if l.mats == 1 { "" } else { "s" });
                    let tip = format!("{}\nhash {:016x}{}", l.source, l.hash64, if l.failed { "\ncould not be decoded" } else { "" });
                    if ui.checkbox(&mut l.enabled, text).on_hover_text(tip).changed() {
                        changed = true;
                    }
                }
                ui.add_space(4.0);
            }
        });
    });
    if let Some(f) = preset {
        mesh.shade_mode = ShadeMode::Textured;
        mesh.set_tex_layers(f);
    } else if changed {
        mesh.shade_mode = ShadeMode::Textured;
        mesh.apply_tex_layers();
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
        if !mesh.tris.is_empty()
            && ui
                .selectable_label(!mesh.flat_shading, "Smooth")
                .on_hover_text("Smooth shading normals rebuilt from the mesh topology. Switch off to see the real facets.")
                .clicked()
        {
            mesh.flat_shading = !mesh.flat_shading;
            mesh.dirty = true;
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
                    // model textures keep straight alpha
                    let pixels: Vec<Color32> = rgba
                        .as_chunks::<4>().0.iter()
                        .map(|c| Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3]))
                        .collect();
                    mesh.texture = Some(Texture2D::new(w, h, pixels));
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
        if ui.button("Export .OBJ").on_hover_text("Wavefront OBJ with smooth normals, a material library and the applied textures as PNG").clicked() {
            let safe_name = mesh.name.replace('#', "");
            if let Some(p) = rfd::FileDialog::new().set_file_name(format!("{safe_name}.obj")).save_file() {
                if let Err(e) = mesh.export_obj_with_materials(&p) {
                    mesh.tex_status = format!("OBJ export failed: {e}");
                }
            }
        }
        if !mesh.tris.is_empty()
            && ui.button("Export .GLB").on_hover_text("Binary glTF 2.0 with normals, materials and embedded textures, ready for Blender or any engine").clicked()
        {
            let safe_name = mesh.name.replace('#', "");
            if let Some(p) = rfd::FileDialog::new().add_filter("glTF binary", &["glb"]).set_file_name(format!("{safe_name}.glb")).save_file() {
                if let Err(e) = mesh.export_glb(&p) {
                    mesh.tex_status = format!("GLB export failed: {e}");
                }
            }
        }
        tex_layer_menu(ui, mesh);
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
        if mesh.has_els {
            ui.add_space(4.0);
            let label = if mesh.els_enabled { "ELS On" } else { "ELS Off" };
            if ui
                .selectable_label(mesh.els_enabled, label)
                .on_hover_text(format!("Preview {} RAGE 9 siren-bone emitters.", mesh.els_points.len()))
                .clicked()
            {
                mesh.els_enabled = !mesh.els_enabled;
                mesh.shade_mode = ShadeMode::Textured;
                mesh.dirty = true;
            }
            if ui
                .add_enabled(mesh.els_enabled, egui::Checkbox::new(&mut mesh.els_flash, "Flash"))
                .on_hover_text("Use the game-authored 32-step siren sequencers at 500 BPM.")
                .changed()
            {
                mesh.dirty = true;
            }
        }
        ui.label(RichText::new(&mesh.tex_status).size(11.0).color(Color32::from_rgb(130, 145, 168)));
    });
    ui.add_space(4.0);

    let avail = ui.available_size();
    let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

    // moving frames trade resolution for speed the resting frame is supersampled
    let mut moving = false;
    if resp.dragged_by(egui::PointerButton::Primary) {
        let d = resp.drag_delta();
        mesh.yaw += d.x * 0.01;
        mesh.pitch = (mesh.pitch + d.y * 0.01).clamp(-1.5, 1.5);
        mesh.dirty = true;
        moving = true;
    } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
        let d = resp.drag_delta();
        mesh.pan_x += d.x / rect.width().max(1.0);
        mesh.pan_y += d.y / rect.height().max(1.0);
        mesh.dirty = true;
        moving = true;
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
            moving = true;
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
            moving = true;
            ui.ctx().request_repaint();
        }
    }
    if mesh.auto_spin && !moving {
        mesh.yaw += 0.014;
        mesh.dirty = true;
        moving = true;
        ui.ctx().request_repaint();
    }
    if mesh.els_enabled && mesh.els_flash {
        let steps_per_second = RAGE9_SIREN_BPM / 60.0 * (RAGE9_SIREN_MULTIPLES * 2) as f64;
        let phase = (ui.input(|i| i.time) * steps_per_second) as u64;
        if phase != mesh.els_flash_phase {
            mesh.els_flash_phase = phase;
            mesh.dirty = true;
            // flashing redraws about 50 times a second so it renders at the adaptive moving resolution
            moving = true;
        }
        ui.ctx().request_repaint();
    } else if mesh.els_flash_phase != 0 {
        mesh.els_flash_phase = 0;
        mesh.dirty = true;
    }

    let ppp = ui.ctx().pixels_per_point();
    let target_w = ((rect.width() * ppp) as usize).clamp(160, 1920);
    let target_h = ((rect.height() * ppp) as usize).clamp(120, 1200);
    let megasamples = (target_w * target_h) as f32 / 1.0e6;
    let scale = if moving {
        // largest scale whose predicted cost fits a 60 hz frame
        if mesh.frame_cost > 0.0 { (MOVING_FRAME_MS / (mesh.frame_cost * megasamples)).sqrt().clamp(0.5, 1.5) } else { 1.0 }
    } else {
        raster::idle_supersampling(target_w, target_h)
    };

    if mesh.dirty || mesh.view_size != (target_w, target_h) || mesh.tex.is_none() || (!moving && !mesh.view_refined) {
        let started = std::time::Instant::now();
        let (w, h, pixels) = raster::render_image(mesh, target_w, target_h, scale);
        let ms = started.elapsed().as_secs_f32() * 1000.0;
        let cost = ms / (megasamples * scale * scale).max(1e-3);
        mesh.frame_cost = if mesh.frame_cost > 0.0 { mesh.frame_cost * 0.6 + cost * 0.4 } else { cost };
        let ci = ColorImage { size: [w, h], pixels };
        if let Some(tex) = &mut mesh.tex {
            tex.set(ci, TextureOptions::LINEAR);
        } else {
            mesh.tex = Some(ui.ctx().load_texture("percystar_3d_fb", ci, TextureOptions::LINEAR));
        }
        mesh.dirty = false;
        mesh.view_size = (target_w, target_h);
        mesh.view_refined = !moving;
        if moving {
            // make sure a resting frame follows once input stops
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(120));
        }
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
    fn rage9_siren_uses_game_sequencers_and_sub_flashes() {
        assert_eq!(rage9_siren_slot("siren1"), Some(1));
        assert_eq!(rage9_siren_slot("ext_front_siren2"), Some(2));
        assert_eq!(rage9_siren_slot("siren_bumper_r_bar4"), None);
        assert_eq!(rage9_siren_slot("siren_roof_round_1"), None);
        assert_eq!(rage9_siren_slot("siren_roof"), None);
        // siren1 starts with 1, siren4 with the inverse 0.
        assert!(rage9_siren_lit(1, 0));
        assert!(!rage9_siren_lit(4, 0));
        // Three flashes are emitted inside an active beat.
        assert!(!rage9_siren_lit(1, 1));
        assert!(rage9_siren_lit(1, 2));
        assert!(rage9_siren_lit(1, 4));
        // The next bit reverses the pair; unused and invalid slots stay dark.
        assert!(!rage9_siren_lit(1, 6));
        assert!(rage9_siren_lit(4, 6));
        assert!(!rage9_siren_lit(9, 0));
        assert!(!rage9_siren_lit(0, 0));
        assert!(!rage9_siren_lit(21, 0));
    }

    #[test]
    fn vehicle_parts_merge_in_parent_model_space() {
        let mut base = Mesh3D::from_parts(
            "car_hi".into(),
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0; 2]; 3],
            vec![[0, 1, 2]],
            vec![0],
            vec![],
            None,
            String::new(),
            1,
            1,
            "High".into(),
        );
        let mut lightbar = Mesh3D::from_parts(
            "car_lightbar_hi".into(),
            vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
            vec![[0.0; 2]; 3],
            vec![[0, 1, 2]],
            vec![0],
            vec![],
            None,
            String::new(),
            1,
            1,
            "High".into(),
        );
        lightbar.has_els = true;
        lightbar.els_points.push(ElsPoint { position: [0.5, 0.5, 2.0], bank: false, slot: 1 });
        base.append_vehicle_part(lightbar);
        assert_eq!(base.verts.len(), 6);
        assert_eq!(base.tris, vec![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(base.parts.len(), 2);
        assert_eq!(base.els_points.len(), 1);
        assert!(base.has_els);
    }

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
                texture: Some(Arc::new(Texture2D::new(1, 1, vec![Color32::TRANSPARENT]))),
                pass: MaterialPass::Cutout,
                ..Default::default()
            },
            GeomMaterial {
                texture: Some(Arc::new(Texture2D::new(1, 1, vec![Color32::GREEN]))),
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
            texture: Some(Arc::new(Texture2D::new(1, 1, vec![Color32::from_rgba_premultiplied(r, g, b, a)]))),
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
mod showcase_tests {
    use super::*;

    fn tex(w: usize, h: usize, px: Vec<Color32>) -> Texture2D {
        Texture2D::new(w, h, px)
    }

    // appends a parametric surface as an indexed grid the way a gpu vertex buffer stores it
    // positions uvs triangles and the geometry id of every triangle
    type Soup = (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[u32; 3]>, Vec<u16>);

    fn grid(m: &mut Soup, nu: u32, nv: u32, geom: u16, f: impl Fn(f32, f32) -> ([f32; 3], [f32; 2])) {
        let base = m.0.len() as u32;
        for j in 0..=nv {
            for i in 0..=nu {
                let (p, uv) = f(i as f32 / nu as f32, j as f32 / nv as f32);
                m.0.push(p);
                m.1.push(uv);
            }
        }
        for j in 0..nv {
            for i in 0..nu {
                let a = base + j * (nu + 1) + i;
                let b = a + nu + 1;
                m.2.push([a, a + 1, b]);
                m.2.push([a + 1, b + 1, b]);
                m.3.extend([geom, geom]);
            }
        }
    }

    // sphere cylinder torus box ground and a glass pane with painted textured cutout and glass materials
    pub(super) fn showcase() -> Mesh3D {
        use std::f32::consts::{PI, TAU};
        let mut m = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // geom 0 painted sphere
        grid(&mut m, 28, 14, 0, |u, v| {
            let (th, ph) = (u * TAU, v * PI);
            ([-1.6 + ph.sin() * th.cos() * 0.8, ph.sin() * th.sin() * 0.8, 0.8 + ph.cos() * 0.8], [u, v])
        });
        // geom 1 textured torus
        grid(&mut m, 40, 16, 1, |u, v| {
            let (a, b) = (u * TAU, v * TAU);
            let r = 0.75 + 0.28 * b.cos();
            ([1.5 + r * a.cos(), r * a.sin(), 0.5 + 0.28 * b.sin()], [u * 6.0, v * 2.0])
        });
        // geom 2 low poly cylinder with a duplicated uv seam
        grid(&mut m, 10, 1, 2, |u, v| {
            let a = u * TAU;
            ([0.45 * a.cos(), 1.7 + 0.45 * a.sin(), v * 1.4], [u * 3.0, v])
        });
        // geom 3 box with split vertices hard edges
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                grid(&mut m, 1, 1, 3, |u, v| {
                    let mut p = [0.0f32; 3];
                    p[axis] = sign * 0.5;
                    p[(axis + 1) % 3] = (u - 0.5) * if sign > 0.0 { 1.0 } else { -1.0 };
                    p[(axis + 2) % 3] = v - 0.5;
                    ([p[0], p[1] - 1.9, p[2] + 0.5], [u, v])
                });
            }
        }
        // geom 4 ground with a fine texture that aliases when minified
        grid(&mut m, 1, 1, 4, |u, v| ([(u - 0.5) * 6.4, (v - 0.5) * 5.6 - 0.1, 0.0], [u * 24.0, v * 24.0]));
        // geom 5 cutout fence
        grid(&mut m, 1, 1, 5, |u, v| ([-2.9 + u * 2.4, -1.2, v * 1.3], [u * 6.0, v * 3.0]));
        // geom 6 glass pane in front of the torus
        grid(&mut m, 1, 1, 6, |u, v| ([0.6 + u * 1.8, -1.35, 0.1 + v * 1.2], [u, v]));
        let n_geoms = 7;
        let mut mesh = Mesh3D::from_parts("showcase".into(), m.0, m.1, m.2, m.3, vec![], None, String::new(), n_geoms, 0, String::new());
        mesh.normal_source = NormalSource::SharedVertices;
        let checker = |n: usize, cell: usize, a: [u8; 3], b: [u8; 3]| {
            tex(n, n, (0..n * n).map(|i| if ((i % n) / cell + (i / n) / cell).is_multiple_of(2) { Color32::from_rgb(a[0], a[1], a[2]) } else { Color32::from_rgb(b[0], b[1], b[2]) }).collect())
        };
        // round holes punched into an opaque sheet transparent texels are black like bc1
        let fence = tex(64, 64, (0..64 * 64).map(|i| {
            let (x, y) = ((i % 64) as f32 - 31.5, (i / 64) as f32 - 31.5);
            if x * x + y * y < 22.0 * 22.0 { Color32::from_rgba_premultiplied(0, 0, 0, 0) } else { Color32::from_rgb(214, 196, 150) }
        }).collect());
        let mut mats = vec![GeomMaterial::default(); n_geoms];
        mats[0].paint_capable = true;
        mats[0].is_vehicle_paint = true;
        mats[1].texture = Some(Arc::new(checker(256, 16, [228, 120, 40], [250, 236, 210])));
        mats[2].texture = Some(Arc::new(checker(64, 8, [70, 130, 200], [225, 232, 240])));
        mats[4].texture = Some(Arc::new(checker(256, 2, [56, 60, 68], [176, 180, 188])));
        mats[5].texture = Some(Arc::new(fence));
        mats[5].pass = MaterialPass::Cutout;
        mats[6].pass = MaterialPass::Glass { texture_alpha: false };
        mesh.geom_materials = mats;
        mesh.has_paint_geom = true;
        mesh.pitch = 0.42;
        mesh.zoom = 1.45;
        mesh
    }

    fn save(dir: &str, name: &str, w: usize, h: usize, px: &[Color32]) {
        let _ = fs::create_dir_all(dir);
        let rgba: Vec<u8> = px.iter().flat_map(|c| c.to_array()).collect();
        image::save_buffer(Path::new(dir).join(name), &rgba, w as u32, h as u32, image::ColorType::Rgba8).expect("save png");
    }

    // set PERCYSTAR_RENDER_DIR to also write the frames as png
    #[test]
    fn showcase_renders_smooth_filtered_and_antialiased() {
        let out = std::env::var("PERCYSTAR_RENDER_DIR").ok();
        let mut m = showcase();
        let (w, h) = (960usize, 600usize);
        rasterize_mesh(&mut m, w, h);
        let plain = m.color_buf.clone();
        let started = std::time::Instant::now();
        let (iw, ih, hq) = raster::render_image(&mut m, w, h, raster::idle_supersampling(w, h));
        eprintln!("supersampled frame {:.1} ms", started.elapsed().as_secs_f32() * 1000.0);
        assert_eq!((iw, ih), (w, h));
        m.flat_shading = true;
        rasterize_mesh(&mut m, w, h);
        let flat = m.color_buf.clone();
        m.flat_shading = false;
        if let Some(d) = &out {
            save(d, "showcase_smooth.png", w, h, &plain);
            save(d, "showcase_supersampled.png", w, h, &hq);
            save(d, "showcase_flat.png", w, h, &flat);
            for (mode, name) in [(ShadeMode::Solid, "showcase_clay.png"), (ShadeMode::SolidWire, "showcase_wire.png")] {
                m.shade_mode = mode;
                let (_, _, px) = raster::render_image(&mut m, w, h, 2.0);
                save(d, name, w, h, &px);
            }
            m.shade_mode = ShadeMode::Textured;
        }
        // every frame is fully covered no sample is left unshaded inside the ground quad
        assert!(plain.iter().chain(&hq).chain(&flat).all(|c| c.a() == 255));
        // smooth shading removes the facet steps so neighbouring pixels differ far less on the painted sphere
        let roughness = |px: &[Color32]| {
            let mut sum = 0u64;
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let c = px[y * w + x];
                    // second difference isolates steps from gradients
                    let (l, r) = (px[y * w + x - 1], px[y * w + x + 1]);
                    sum += (2 * c.g() as i32 - l.g() as i32 - r.g() as i32).unsigned_abs() as u64;
                }
            }
            sum
        };
        let (rs, rf, rh) = (roughness(&plain), roughness(&flat), roughness(&hq));
        eprintln!("roughness flat {rf} smooth {rs} supersampled {rh}");
        assert!(rs < rf, "smooth shading must be smoother than facets");
        assert!(rh < rs, "supersampling must reduce pixel scale aliasing");
        // the same frame renders identically on every run regardless of thread scheduling
        rasterize_mesh(&mut m, w, h);
        assert!(m.color_buf == plain, "rendering must be deterministic");
    }


    // cargo test --release viewer_frame_times -- --ignored --nocapture
    #[test]
    #[ignore]
    fn viewer_frame_times() {
        let mut m = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // one million triangle textured sphere
        grid(&mut m, 1000, 500, 0, |u, v| {
            let (th, ph) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
            ([ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos()], [u * 8.0, v * 4.0])
        });
        let mut dense = Mesh3D::from_parts("dense".into(), m.0, m.1, m.2, m.3, vec![], None, String::new(), 1, 0, String::new());
        dense.normal_source = NormalSource::SharedVertices;
        let px = (0..256 * 256).map(|i| if ((i % 256) / 16 + (i / 256) / 16) % 2 == 0 { Color32::from_rgb(230, 120, 40) } else { Color32::WHITE }).collect();
        dense.geom_materials = vec![GeomMaterial { texture: Some(Arc::new(tex(256, 256, px))), ..Default::default() }];
        let started = std::time::Instant::now();
        let _ = dense.corner_normals();
        eprintln!("normals for {} triangles {:.0} ms", dense.tris.len(), started.elapsed().as_secs_f32() * 1000.0);
        for (name, mesh) in [("showcase", &mut showcase()), ("1M tris", &mut dense)] {
            for (w, h) in [(1280usize, 720usize), (1920, 1200)] {
                for scale in [1.0, raster::idle_supersampling(w, h)] {
                    let _ = raster::render_image(mesh, w, h, scale);
                    let started = std::time::Instant::now();
                    for _ in 0..3 {
                        mesh.yaw += 0.05;
                        let _ = raster::render_image(mesh, w, h, scale);
                    }
                    eprintln!("{name} {w}x{h} scale {scale:.2}: {:.1} ms", started.elapsed().as_secs_f32() * 1000.0 / 3.0);
                }
            }
        }
    }

    // drives the real viewport code through egui without a window
    #[test]
    fn viewport_renders_moving_frames_and_refines_at_rest() {
        let ctx = egui::Context::default();
        let mut mesh = showcase();
        let frame = |mesh: &mut Mesh3D, events: Vec<egui::Event>| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 640.0))),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| render_mesh3d_viewport(ui, mesh));
            });
        };
        frame(&mut mesh, vec![]);
        assert!(mesh.tex.is_some() && mesh.view_refined && !mesh.dirty, "first frame is a resting frame");
        let (w, h) = mesh.view_size;
        assert!(w >= 160 && h >= 120 && mesh.frame_cost > 0.0);
        // resting frames are supersampled the texture on screen has the viewport size
        assert_eq!(mesh.tex.as_ref().unwrap().size(), [w, h]);
        assert!(mesh.fb_w > w && mesh.fb_h > h, "resting frame renders more samples than pixels");

        // nothing changed nothing is rendered again
        let before = mesh.frame_cost;
        frame(&mut mesh, vec![]);
        assert_eq!(mesh.frame_cost, before);

        // spinning renders moving frames at an adaptive resolution
        mesh.auto_spin = true;
        let yaw = mesh.yaw;
        frame(&mut mesh, vec![]);
        assert!(mesh.yaw != yaw && !mesh.view_refined);
        let moving = (mesh.fb_w, mesh.fb_h);
        assert!(moving.0 * 2 >= w && moving.0 * 2 <= w * 3, "moving scale stays between one half and one and a half: {moving:?} for {w}x{h}");

        // once motion stops the next frame is refined again
        mesh.auto_spin = false;
        frame(&mut mesh, vec![]);
        assert!(mesh.view_refined && mesh.fb_w > w);

        // dragging orbits the camera
        let at = egui::pos2(450.0, 400.0);
        let press = egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() };
        frame(&mut mesh, vec![egui::Event::PointerMoved(at), press]);
        let yaw = mesh.yaw;
        frame(&mut mesh, vec![egui::Event::PointerMoved(at + egui::vec2(60.0, 0.0))]);
        frame(&mut mesh, vec![egui::Event::PointerMoved(at + egui::vec2(120.0, 10.0))]);
        assert!(mesh.yaw > yaw, "drag must orbit: {} -> {}", yaw, mesh.yaw);
        assert!(!mesh.view_refined, "a dragged frame is a moving frame");

        // switching shading modes marks the frame dirty and renders without panicking in every mode
        for mode in [ShadeMode::Solid, ShadeMode::SolidWire, ShadeMode::Wireframe, ShadeMode::Textured] {
            mesh.shade_mode = mode;
            mesh.flat_shading = mode == ShadeMode::Solid;
            mesh.dirty = true;
            frame(&mut mesh, vec![]);
        }
    }

    // siren emitters glow over the frame flash with the sequencer and hide behind nearer geometry
    #[test]
    fn siren_glow_flashes_and_respects_depth() {
        let mut m = showcase();
        m.yaw = 0.0;
        m.pitch = 0.0;
        // one emitter in the open above the ground and one inside the opaque box
        m.els_points = vec![
            ElsPoint { position: [0.0, 0.0, 2.2], bank: false, slot: 1 },
            ElsPoint { position: [0.0, -1.9, 0.5], bank: true, slot: 4 },
        ];
        m.has_els = true;
        rasterize_mesh(&mut m, 400, 300);
        let off = m.color_buf.clone();
        m.els_enabled = true;
        m.els_flash = false;
        rasterize_mesh(&mut m, 400, 300);
        let on = m.color_buf.clone();
        let red_gain: i64 = on.iter().zip(&off).map(|(a, b)| a.r() as i64 - b.r() as i64).sum();
        let blue_gain: i64 = on.iter().zip(&off).map(|(a, b)| a.b() as i64 - b.b() as i64).sum();
        assert!(red_gain > 2000, "the visible red emitter glows: {red_gain}");
        assert!(blue_gain < red_gain / 4, "the blue emitter inside the box stays hidden: {blue_gain}");
        // flashing at a phase where slot 1 is dark draws nothing
        m.els_flash = true;
        m.els_flash_phase = 1;
        rasterize_mesh(&mut m, 400, 300);
        assert!(m.color_buf == off, "a dark sequencer step adds no glow");
    }

    // shared edges are drawn exactly once so glass does not double blend along its diagonal
    #[test]
    fn shared_edges_are_watertight_and_never_double_blended() {
        for (yaw, pitch) in [(2.45, 0.38), (0.3, 0.1), (1.1, 0.9), (4.0, -0.2), (5.5, 0.6)] {
            // every covered sample of a quad must belong to exactly one of its two triangles
            let mut solo = Mesh3D::from_parts(
                "quad".into(), vec![[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.3, 1.0], [-1.0, 0.3, 1.0]], vec![[0.0; 2]; 4],
                vec![[0, 1, 2], [0, 2, 3]], vec![0, 0], vec![], None, String::new(), 1, 0, String::new(),
            );
            solo.yaw = yaw;
            solo.pitch = pitch;
            // exact check on coverage using the id buffer of an opaque version
            rasterize_mesh(&mut solo, 300, 200);
            let covered = solo.id_buf.iter().filter(|&&i| i != u32::MAX).count();
            let mut a = solo.clone();
            a.tris.truncate(1);
            a.tri_geom.truncate(1);
            a.normals.truncate(1);
            a.invalidate_normals();
            rasterize_mesh(&mut a, 300, 200);
            let mut b = solo.clone();
            b.tris.remove(0);
            b.tri_geom.remove(0);
            b.normals.remove(0);
            b.invalidate_normals();
            rasterize_mesh(&mut b, 300, 200);
            let (ca, cb) = (a.id_buf.iter().filter(|&&i| i != u32::MAX).count(), b.id_buf.iter().filter(|&&i| i != u32::MAX).count());
            assert_eq!(ca + cb, covered, "yaw {yaw} pitch {pitch}: the two triangles of a quad must tile it without gap or overlap");
            let overlap = a.id_buf.iter().zip(&b.id_buf).filter(|(x, y)| **x != u32::MAX && **y != u32::MAX).count();
            assert_eq!(overlap, 0, "yaw {yaw} pitch {pitch}: shared edge samples drawn twice");
        }
    }

    // sub pixel triangles must still cover the surface instead of leaving background holes
    #[test]
    fn dense_meshes_have_no_pinholes() {
        let mut m = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        grid(&mut m, 700, 700, 0, |u, v| ([(u - 0.5) * 2.0, 0.0, (v - 0.5) * 2.0], [u, v]));
        let mut mesh = Mesh3D::from_parts("dense".into(), m.0, m.1, m.2, m.3, vec![], None, String::new(), 1, 0, String::new());
        mesh.yaw = 0.0;
        mesh.pitch = 0.0;
        mesh.zoom = 0.5;
        rasterize_mesh(&mut mesh, 400, 400);
        let ids = &mesh.id_buf;
        let inside: Vec<usize> = (0..400 * 400).filter(|&i| ids[i] != u32::MAX).collect();
        let (x0, x1) = (inside.iter().map(|i| i % 400).min().unwrap(), inside.iter().map(|i| i % 400).max().unwrap());
        let (y0, y1) = (inside.iter().map(|i| i / 400).min().unwrap(), inside.iter().map(|i| i / 400).max().unwrap());
        // 980 thousand triangles on a square of about 124 pixels the square must be solid
        assert!((x1 - x0) > 100 && (y1 - y0) > 100);
        let holes = (y0 + 2..y1 - 1).flat_map(|y| (x0 + 2..x1 - 1).map(move |x| y * 400 + x)).filter(|&i| ids[i] == u32::MAX).count();
        assert_eq!(holes, 0, "background shows through a dense mesh");
    }

    // a texture mapped onto a receding plane must follow perspective not screen space
    #[test]
    fn texture_mapping_is_perspective_correct() {
        // one big quad from near to far split along the diagonal the two halves of the texture are black and white
        let verts = vec![[-1.0, -3.0, 0.0], [1.0, -3.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 3.0, 0.0]];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut m = Mesh3D::from_parts("plane".into(), verts, uvs, vec![[0, 1, 2], [0, 2, 3]], vec![0, 0], vec![], None, String::new(), 1, 0, String::new());
        m.yaw = 0.0;
        m.pitch = 0.5;
        // v below one half is red above is blue
        let t = tex(1, 64, (0..64).map(|i| if i < 32 { Color32::from_rgb(255, 0, 0) } else { Color32::from_rgb(0, 0, 255) }).collect());
        m.geom_materials = vec![GeomMaterial { texture: Some(Arc::new(t)), ..Default::default() }];
        rasterize_mesh(&mut m, 400, 400);
        // world midpoint of the quad y 0 projects to this screen row the color change must sit there
        let mid_row = (m.proj_buf[0][1] + m.proj_buf[2][1]) * 0.5;
        let true_mid = {
            // project the world space center exactly like the renderer
            let mut probe = Mesh3D::from_parts("p".into(), vec![[0.0, 0.0, 0.0], [-1.0, -3.0, 0.0], [1.0, 3.0, 0.0]], vec![], vec![], vec![], vec![], None, String::new(), 1, 0, String::new());
            probe.center = m.center;
            probe.radius = m.radius;
            probe.yaw = 0.0;
            probe.pitch = 0.5;
            rasterize_mesh(&mut probe, 400, 400);
            probe.proj_buf[0][1]
        };
        assert!((mid_row - true_mid).abs() > 6.0, "the test needs visible foreshortening");
        let x = 200;
        let switch = (1..399).find(|&y| {
            let (a, b) = (m.color_buf[(y - 1) * 400 + x], m.color_buf[y * 400 + x]);
            a.b() > a.r() && b.r() > b.b()
        }).expect("blue far half above red near half") as f32;
        assert!((switch - true_mid).abs() <= 1.5, "texture seam at row {switch} but the surface midpoint is at {true_mid} affine would give {mid_row}");
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

    // props keep several lod levels in one file each level decodes and detail never increases
    #[test]
    fn in_file_lod_levels_decode() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        let Some(root) = Path::new(&rpf).parent() else { return };
        let pack = root.join("pack").join("shared_minigame000").join("shared_minigame000.rpf");
        if !pack.exists() {
            return;
        }
        for (tag, off, want) in [
            ("td_resi_wall_windows_b_3", 798_629_904u64, [6usize, 6, 6, 5]),
            ("a_td_resdoor_double_l", 800_751_632u64, [8, 8, 8, 5]),
        ] {
            let base = decode_rage9_model_from_rpf(&pack, off, tag).expect("decode lod0");
            assert_eq!(base.lod, 0);
            assert_eq!(base.lod_geoms, want, "{tag} lod table");
            let mut prev_tris = base.tris.len();
            for l in 1..MAX_LODS {
                let m = decode_rage9_model_from_rpf_lod(&pack, off, tag, l).expect("decode lod");
                eprintln!("[{tag}] {} geoms={} tris={}", m.lod_label, m.geom_count, m.tris.len());
                assert_eq!((m.lod, m.geom_count), (l, want[l]));
                assert!(!m.tris.is_empty() && m.tris.len() <= prev_tris, "{tag} lod{l} gained triangles");
                prev_tris = m.tris.len();
            }
            assert!(prev_tris < base.tris.len(), "{tag} lowest lod should be lighter than lod0");
        }
    }

    // vehicles split detail across _hi _mid and base files and the mid file adds a second level
    #[test]
    fn vehicle_detail_files_decode() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        if !Path::new(&rpf).exists() {
            return;
        }
        let mut prev_tris = usize::MAX;
        for (tag, off, want) in [
            ("cheetah2_hi", 74_053_521_424u64, [35usize, 0, 0, 0]),
            ("cheetah2_mid", 74_058_629_136u64, [34, 30, 0, 0]),
            ("cheetah2", 74_051_276_816u64, [3, 0, 0, 0]),
        ] {
            let m = decode_rage9_model_from_rpf(Path::new(&rpf), off, tag).expect("decode vehicle");
            assert_eq!(m.lod_geoms, want, "{tag} lod table");
            assert!(m.tris.len() < prev_tris, "{tag} should be lighter than the file above");
            prev_tris = m.tris.len();
            // a missing level falls back to the nearest populated one
            let last = want.iter().rposition(|&n| n > 0).unwrap();
            let low = decode_rage9_model_from_rpf_lod(Path::new(&rpf), off, tag, 3).expect("fallback");
            eprintln!("[{tag}] lod0 tris={} lowest {} tris={}", m.tris.len(), low.lod_label, low.tris.len());
            assert_eq!(low.lod, last);
            if last > 0 {
                assert!(low.tris.len() < m.tris.len(), "{tag} lower level should be lighter");
                prev_tris = low.tris.len();
            } else {
                assert_eq!(low.tris.len(), m.tris.len());
            }
        }
    }

    // texture layers start as one color texture per material and can be switched individually
    #[test]
    fn vehicle_texture_layers_toggle() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        if !Path::new(&rpf).exists() {
            return;
        }
        let out_dir = std::env::var("PERCYSTAR_RENDER_DIR").ok();
        let mut m = decode_rage9_model_from_rpf(Path::new(&rpf), 74_053_521_424, "cheetah2_hi").expect("decode");
        let count = |m: &Mesh3D, k: TexLayerKind| m.tex_layers.iter().filter(|l| l.kind == k).count();
        eprintln!(
            "layers color={} normal={} overlay={} default_on={}",
            count(&m, TexLayerKind::Color),
            count(&m, TexLayerKind::Normal),
            count(&m, TexLayerKind::Overlay),
            m.tex_layers.iter().filter(|l| l.default_on).count()
        );
        assert!(count(&m, TexLayerKind::Color) > 0 && count(&m, TexLayerKind::Normal) > 0 && count(&m, TexLayerKind::Overlay) > 0);
        assert!(m.tex_layers.iter().all(|l| l.enabled == l.default_on && l.mats > 0));
        let textured = |m: &Mesh3D| m.geom_materials.iter().filter(|g| g.texture.is_some()).count();
        let default_textured = textured(&m);
        // textures that cannot be decoded come back as a flat swatch so the exact count depends on the decoder
        assert!(default_textured >= 19, "default look keeps at least the 19 textured materials, got {default_textured}");
        assert!(m.geom_materials.iter().all(|g| g.normal.is_none() && g.overlays.is_empty()));
        let shot = |m: &mut Mesh3D, tag: &str| {
            rasterize_mesh(m, 800, 600);
            if let Some(d) = &out_dir {
                let _ = fs::create_dir_all(d);
                let rgba: Vec<u8> = m.color_buf.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
                image::save_buffer(Path::new(d).join(format!("layers_{tag}.png")), &rgba, 800, 600, image::ColorType::Rgba8).expect("save");
            }
            m.color_buf.clone()
        };
        let base = shot(&mut m, "default");

        m.set_tex_layers(|l| l.default_on || l.kind == TexLayerKind::Normal);
        assert!(m.geom_materials.iter().any(|g| g.normal.is_some()));
        assert_eq!(textured(&m), default_textured);
        let with_normals = shot(&mut m, "normals");
        assert!(with_normals != base, "normal maps should change the render");

        m.set_tex_layers(|l| l.default_on || l.label == "DustTexture");
        assert!(m.geom_materials.iter().any(|g| !g.overlays.is_empty()));
        assert!(shot(&mut m, "dust") != base, "an overlay should change the render");

        m.set_tex_layers(|_| false);
        assert_eq!(textured(&m), 0);
        assert!(m.geom_materials.iter().all(|g| g.normal.is_none() && g.overlays.is_empty()));
        assert!(m.has_paint_geom && m.geom_materials.iter().any(|g| g.is_vehicle_paint));
        shot(&mut m, "none");

        // one layer off only unbinds the materials that drew it
        m.set_tex_layers(|l| l.default_on);
        let draws = |m: &Mesh3D, li: usize| {
            let Some(t) = m.tex_layers[li].tex.as_ref() else { return 0 };
            m.geom_materials.iter().filter(|g| g.texture.as_ref().is_some_and(|x| Arc::ptr_eq(x, t))).count()
        };
        let first = (0..m.tex_layers.len()).find(|&li| draws(&m, li) > 0).expect("a drawn layer");
        m.tex_layers[first].enabled = false;
        m.apply_tex_layers();
        assert_eq!(draws(&m, first), 0, "a switched off layer is no longer drawn");
        m.set_tex_layers(|l| l.default_on);
        assert_eq!(textured(&m), default_textured);
        assert!(shot(&mut m, "default_again") == base, "returning to the default set reproduces the default render");
    }

    // vehicles bind their textures by hash into the ttpf pools
    #[test]
    fn vehicle_texture_refs_resolve() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        if !Path::new(&rpf).exists() {
            return;
        }
        for (tag, off) in [("cheetah2_hi", 74_053_521_424u64), ("cheetah2_mid", 74_058_629_136u64), ("cheetah2", 74_051_276_816u64)] {
            let refs = list_model_texture_refs(Path::new(&rpf), off).expect("list refs");
            let hits: Vec<_> = refs.iter().filter_map(|r| r.hit.as_ref().map(|h| (r, h))).collect();
            let bytes: usize = hits.iter().map(|(_, h)| h.w * h.h * 4).sum();
            eprintln!("[{tag}] unique={} resolved={} rgba_mb={}", refs.len(), hits.len(), bytes >> 20);
            if std::env::var("PERCYSTAR_TEX_DUMP").is_ok() {
                for r in &refs {
                    match &r.hit {
                        Some(h) => eprintln!("   {:<24} {:016x} {}x{} dxgi{} {} #{} x{}", r.role(), r.hash64, h.w, h.h, h.dxgi, h.path.file_name().unwrap().to_string_lossy(), h.tex_idx, r.shaders),
                        None => eprintln!("   {:<24} {:016x} unresolved x{}", r.role(), r.hash64, r.shaders),
                    }
                }
            }
            assert!(hits.len() >= 3, "{tag} should resolve textures");
            assert!(refs.iter().any(|r| r.role() == "diffuseTex" && r.hit.is_some()), "{tag} should resolve a diffuse texture");
            assert!(hits.iter().all(|(r, _)| r.shaders > 0 && !r.params.is_empty()));
            let (first, hit) = hits[0];
            let tex = crate::tex::decode_ttpf_texture(&hit.path, hit.tex_idx).expect("decode first ref");
            assert_eq!((tex.w, tex.h), (hit.w, hit.h), "{tag} {:016x}", first.hash64);
        }
    }

    #[test]
    fn vehicle_option_fragments_compose() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        if !Path::new(&rpf).exists() {
            return;
        }
        let mut car = decode_rage9_model_from_rpf(Path::new(&rpf), 73_817_919_504, "stanierg2_hi")
            .expect("decode base vehicle");
        let option = decode_rage9_model_from_rpf(Path::new(&rpf), 73_831_055_376, "stanierg2_siren_roofbar1_mount_01_hi")
            .expect("decode separate roofbar mount");
        let (verts, tris) = (car.verts.len(), car.tris.len());
        car.append_vehicle_part(option);
        assert!(car.verts.len() > verts && car.tris.len() > tris);
        assert!(car.parts.iter().any(|p| p.name.contains("siren_roofbar")));
        assert!(!car.has_els, "a mount point is not itself an emergency light");

        let roofbar = decode_rage9_model_from_rpf(Path::new(&rpf), 74_963_226_640, "cmn_siren_roofbar1_02_hi")
            .expect("decode shared roofbar");
        car.append_vehicle_part(roofbar);
        assert!(car.has_els && car.els_points.len() >= 12);
        assert!(car.els_points.iter().all(|p| p.position[2] > 0.7), "roofbar emitters attach at the vehicle mount");
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
            ("ambulance_hi", 71_757_271_056u64),
            ("firetruk2_hi", 74_492_911_632u64),
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
            } else if tag == "ambulance_hi" {
                assert_eq!(m.els_points.len(), 37, "ambulance should expose its siren1..37 bones");
            } else if tag == "firetruk2_hi" {
                assert!(!m.els_points.is_empty(), "firetruck should expose prefixed siren bones");
            } else {
                assert!(tex_mats > 0, "{tag} should resolve external TTPF textures");
            }
            if let Some(ref d) = out_dir {
                if m.has_els {
                    m.els_enabled = true;
                    m.els_flash = false;
                }
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

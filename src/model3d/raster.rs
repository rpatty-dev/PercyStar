// software renderer of the model viewer
// a visibility pass picks the front triangle of every sample so each one is shaded exactly once
// then sorted translucent triangles are blended on top rows are split into bands rendered in parallel
use super::texture::{encode_srgb, linear_to_srgb_table, srgb_to_linear_table};
use super::{MaterialPass, Mesh3D, ShadeMode, Texture2D, GLASS_PREVIEW_ALPHA};
use eframe::egui::Color32;
use std::sync::{Arc, Mutex};

// sub pixel steps per pixel vertices snap to this grid so shared edges are watertight
const SUB: i64 = 256;
const NO_TRI: u32 = u32::MAX;
// screen coordinates are clamped so the 64 bit edge functions cannot overflow
const MAX_COORD: f32 = 1.0e6;
// supersampled frames stay under this many samples
const MAX_SAMPLES: f32 = 6.0e6;

const MAT_PALETTE: [[f32; 3]; 12] = [
    [0.82, 0.84, 0.88], [0.24, 0.26, 0.30], [0.18, 0.24, 0.32], [0.72, 0.75, 0.80],
    [0.34, 0.36, 0.40], [0.86, 0.24, 0.24], [0.48, 0.50, 0.56], [0.28, 0.24, 0.22],
    [0.78, 0.72, 0.62], [0.38, 0.44, 0.52], [0.64, 0.66, 0.70], [0.20, 0.21, 0.24],
];
const CLAY: [f32; 3] = [0.86, 0.88, 0.92];
const GLASS_TINT: [f32; 3] = [0.62, 0.71, 0.78];

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
fn unit(a: [f32; 3]) -> Option<[f32; 3]> {
    let l2 = dot(a, a);
    (l2 > 1e-20 && l2.is_finite()).then(|| {
        let inv = l2.sqrt().recip();
        [a[0] * inv, a[1] * inv, a[2] * inv]
    })
}

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Surface {
    Matte,
    Paint,
    Glass,
}

// everything the renderer needs to know about one geometry resolved once per frame
#[derive(Clone, Copy)]
struct GeomDraw<'a> {
    hidden: bool,
    blend: bool,
    pass: MaterialPass,
    depth_bias: f32,
    tex: Option<&'a Texture2D>,
    normal_tex: Option<&'a Texture2D>,
    overlays: &'a [Arc<Texture2D>],
    // linear light
    albedo: [f32; 3],
    surface: Surface,
}

#[derive(Clone, Copy)]
struct Flags {
    textured: bool,
    has_uvs: bool,
    has_preview_texture: bool,
    use_blending: bool,
}

fn geom_draw<'a>(mesh: &'a Mesh3D, g: usize, f: Flags) -> GeomDraw<'a> {
    let gmat = mesh.geom_materials.get(g);
    let tex: Option<&Texture2D> = if f.textured && f.has_uvs {
        mesh.texture.as_ref().or_else(|| gmat.and_then(|m| m.texture.as_deref()))
    } else {
        None
    };
    let hidden = f.textured
        && gmat.is_some_and(|m| m.auxiliary_pass || (m.pass == MaterialPass::Cutout && tex.is_none() && f.has_preview_texture));
    // normal map and multiply overlays of the material only in textured mode without a manual texture
    let layered = f.textured && f.has_uvs && mesh.texture.is_none();
    let normal_tex = if layered { gmat.and_then(|m| m.normal.as_deref()) } else { None };
    let overlays: &[Arc<Texture2D>] = match gmat {
        Some(m) if layered => &m.overlays,
        _ => &[],
    };
    let blend = f.use_blending && gmat.is_some_and(|m| m.blends(f.has_uvs) && !m.auxiliary_pass);
    let pass = if f.textured && mesh.texture.is_some() {
        MaterialPass::Cutout
    } else if f.textured {
        gmat.map_or(MaterialPass::Opaque, |m| m.pass)
    } else {
        MaterialPass::Opaque
    };
    let glass = matches!(pass, MaterialPass::Glass { .. });
    let is_paint = f.textured && tex.is_none() && gmat.is_some_and(|m| m.is_vehicle_paint);
    let srgb = if is_paint {
        mesh.paint_color
    } else if glass {
        GLASS_TINT
    } else if f.textured && tex.is_none() {
        MAT_PALETTE[g % MAT_PALETTE.len()]
    } else {
        CLAY
    };
    GeomDraw {
        hidden,
        blend,
        pass,
        depth_bias: if blend && pass == MaterialPass::Decal { mesh.radius * 0.0001 } else { 0.0 },
        tex,
        normal_tex,
        overlays,
        albedo: srgb.map(srgb_to_linear),
        surface: if is_paint { Surface::Paint } else if glass { Surface::Glass } else { Surface::Matte },
    }
}

// screen space triangle wound so its doubled area is positive
struct TriGeo {
    idx: [usize; 3],
    // corner slots of the source triangle in the same order as idx
    corner: [usize; 3],
    x: [i64; 3],
    y: [i64; 3],
    area2: i64,
}

impl TriGeo {
    // edge function opposite vertex i as per pixel step in x step in y and value at the sample center of pixel 0 0
    #[inline]
    fn edge(&self, i: usize) -> (i64, i64, i64, i64) {
        let (a, b) = ((i + 1) % 3, (i + 2) % 3);
        let (dx, dy) = (self.x[b] - self.x[a], self.y[b] - self.y[a]);
        let at_origin = (SUB / 2 - self.x[a]) * dy - (SUB / 2 - self.y[a]) * dx;
        // top left rule opposite directions of a shared edge get opposite answers so exactly one triangle owns it
        let bias = if dy > 0 || (dy == 0 && dx < 0) { 0 } else { -1 };
        (dy * SUB, -dx * SUB, at_origin, bias)
    }
}

// per triangle shading inputs built once for every run of samples
struct TriShade<'a> {
    geo: TriGeo,
    // perspective weights divided by the doubled area
    k: [f32; 3],
    z: [f32; 3],
    n: [[f32; 3]; 3],
    face_n: [f32; 3],
    uv: [[f32; 2]; 3],
    draw: GeomDraw<'a>,
    // half log2 of the uv area mapped onto one sample
    uv_density: f32,
    tangent: Option<([f32; 3], f32)>,
}

struct Lights {
    key: [f32; 3],
    fill: [f32; 3],
    half: [f32; 3],
    up: [f32; 3],
}

struct Scene<'a> {
    w: usize,
    verts: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    tris: &'a [[u32; 3]],
    tri_geom: &'a [u16],
    face_normals: &'a [[f32; 3]],
    // empty when the mesh is shown faceted
    corner_normals: &'a [[[f32; 3]; 3]],
    proj: &'a [[f32; 4]],
    fixed: &'a [[i32; 2]],
    draws: &'a [GeomDraw<'a>],
    fallback: &'a [GeomDraw<'a>],
    has_uvs: bool,
    // yaw and pitch sines and cosines
    rot: [f32; 4],
    lights: Lights,
    to_linear: &'static [f32; 256],
    to_srgb: &'static [u8; 4097],
}

impl<'a> Scene<'a> {
    #[inline]
    fn draw(&self, ti: usize) -> &GeomDraw<'a> {
        let g = self.tri_geom.get(ti).copied().unwrap_or(0) as usize;
        self.draws.get(g).unwrap_or(&self.fallback[g % self.fallback.len()])
    }

    // world vector into the lighting frame x right y toward the viewer z up
    #[inline]
    fn to_view(&self, n: [f32; 3]) -> [f32; 3] {
        let [sy, cy, sp, cp] = self.rot;
        let ny1 = n[0] * sy + n[1] * cy;
        [n[0] * cy - n[1] * sy, -ny1 * cp + n[2] * sp, n[2] * cp + ny1 * sp]
    }

    fn geo(&self, ti: usize) -> Option<TriGeo> {
        let t = self.tris[ti];
        let mut idx = [t[0] as usize, t[1] as usize, t[2] as usize];
        let mut corner = [0usize, 1, 2];
        let p = |i: usize| (self.fixed[i][0] as i64, self.fixed[i][1] as i64);
        let (a, b, c) = (p(idx[0]), p(idx[1]), p(idx[2]));
        let mut area2 = (c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0);
        if area2 == 0 {
            return None;
        }
        if area2 < 0 {
            idx.swap(1, 2);
            corner.swap(1, 2);
            area2 = -area2;
        }
        let (a, b, c) = (p(idx[0]), p(idx[1]), p(idx[2]));
        Some(TriGeo { idx, corner, x: [a.0, b.0, c.0], y: [a.1, b.1, c.1], area2 })
    }

    fn shade_setup(&self, ti: usize, geo: TriGeo) -> TriShade<'a> {
        let draw = *self.draw(ti);
        let inv_area = 1.0 / geo.area2 as f32;
        let pv = geo.idx.map(|i| self.proj[i]);
        let face_n = self.to_view(self.face_normals.get(ti).copied().unwrap_or([0.0, 0.0, 1.0]));
        let n = match self.corner_normals.get(ti) {
            Some(cn) => geo.corner.map(|c| self.to_view(cn[c])),
            None => [face_n; 3],
        };
        let uv = if self.has_uvs { geo.idx.map(|i| self.uvs[i]) } else { [[0.0; 2]; 3] };
        let (du1, dv1, du2, dv2) = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1], uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]);
        let det = du1 * dv2 - du2 * dv1;
        let uv_density = 0.5 * (det.abs() * (SUB * SUB) as f32 * inv_area).log2();
        // tangent of the triangle in the lighting frame with the handedness of its uv mapping
        let tangent = draw.normal_tex.and_then(|_| {
            if det.abs() < 1e-12 {
                return None;
            }
            let q = geo.idx.map(|i| self.verts[i]);
            let e1: [f32; 3] = std::array::from_fn(|k| q[1][k] - q[0][k]);
            let e2: [f32; 3] = std::array::from_fn(|k| q[2][k] - q[0][k]);
            let t = self.to_view(unit(std::array::from_fn(|k| (e1[k] * dv2 - e2[k] * dv1) / det))?);
            let b = self.to_view(unit(std::array::from_fn(|k| (e2[k] * du1 - e1[k] * du2) / det))?);
            Some((t, if dot(cross(face_n, t), b) < 0.0 { -1.0 } else { 1.0 }))
        });
        TriShade {
            k: [pv[0][3] * inv_area, pv[1][3] * inv_area, pv[2][3] * inv_area],
            z: [pv[0][2], pv[1][2], pv[2][2]],
            geo,
            n,
            face_n,
            uv,
            draw,
            uv_density,
            tangent,
        }
    }

    // direct and ambient light on a surface with the given linear albedo returns display referred linear color
    #[inline]
    fn light(&self, n: [f32; 3], albedo: [f32; 3], surface: Surface) -> [f32; 3] {
        const SKY: [f32; 3] = [0.34, 0.38, 0.44];
        const GROUND: [f32; 3] = [0.17, 0.155, 0.14];
        const KEY: [f32; 3] = [0.90, 0.865, 0.81];
        const FILL: [f32; 3] = [0.11, 0.136, 0.18];
        let l = &self.lights;
        let ndv = n[1].clamp(0.0, 1.0);
        let key = dot(n, l.key).max(0.0);
        let fill = dot(n, l.fill).max(0.0);
        let hemi = dot(n, l.up) * 0.5 + 0.5;
        let ndh = dot(n, l.half).max(0.0);
        let grazing = 1.0 - ndv;
        let rim = grazing * grazing * grazing * 0.10;
        let (gloss, coat) = match surface {
            Surface::Matte => (ndh.powi(32) * 0.06, 0.0),
            Surface::Paint => (ndh.powi(120) * 0.85 + ndh.powi(10) * 0.10, 0.30),
            Surface::Glass => (ndh.powi(90) * 0.9, 0.45),
        };
        // clearcoat mirrors a simple sky gradient weighted by fresnel
        let mirror = if coat > 0.0 {
            let fresnel = 0.04 + 0.96 * grazing.powi(5);
            let r_up = (2.0 * ndv * dot(n, l.up) - l.up[1]).clamp(-1.0, 1.0);
            let t = (r_up * 0.5 + 0.5).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            let w = (fresnel + 0.06) * coat * 2.0;
            [(0.10 + 0.50 * t) * w, (0.09 + 0.62 * t) * w, (0.08 + 0.80 * t) * w]
        } else {
            [0.0; 3]
        };
        std::array::from_fn(|c| {
            let ambient = GROUND[c] + (SKY[c] - GROUND[c]) * hemi;
            let x = albedo[c] * (ambient + KEY[c] * key + FILL[c] * fill + SKY[c] * rim) + gloss + mirror[c];
            // soft shoulder keeps highlights from clipping while mid tones stay untouched
            if x <= 0.75 { x } else { 0.75 + 0.25 * (1.0 - (-(x - 0.75) * 4.0).exp()) }
        })
    }

    // shades one sample from its edge function values returns linear color alpha in 0 to 255 and depth
    #[inline]
    fn shade(&self, t: &TriShade<'a>, e: [f32; 3]) -> ([f32; 3], f32, f32) {
        let q = [e[0] * t.k[0], e[1] * t.k[1], e[2] * t.k[2]];
        let inv = 1.0 / (q[0] + q[1] + q[2]);
        let p = [q[0] * inv, q[1] * inv, q[2] * inv];
        let z = p[0] * t.z[0] + p[1] * t.z[1] + p[2] * t.z[2];
        let d = &t.draw;
        let mut n = unit(std::array::from_fn(|c| p[0] * t.n[0][c] + p[1] * t.n[1][c] + p[2] * t.n[2][c])).unwrap_or(t.face_n);
        let mut albedo = d.albedo;
        let mut alpha = if d.surface == Surface::Glass { GLASS_PREVIEW_ALPHA as f32 } else { 255.0 };
        if d.tex.is_some() || d.normal_tex.is_some() || !d.overlays.is_empty() {
            let u = p[0] * t.uv[0][0] + p[1] * t.uv[1][0] + p[2] * t.uv[2][0];
            let v = p[0] * t.uv[0][1] + p[1] * t.uv[1][1] + p[2] * t.uv[2][1];
            let lin = |c: f32| self.to_linear[(c + 0.5) as usize & 255];
            if let Some(tex) = d.tex {
                let uses_alpha = d.pass != MaterialPass::Opaque;
                let tc = tex.sample_filtered(u, v, tex.lod_for(t.uv_density), uses_alpha);
                albedo = [lin(tc[0]), lin(tc[1]), lin(tc[2])];
                alpha = match d.pass {
                    MaterialPass::Glass { texture_alpha: false } => GLASS_PREVIEW_ALPHA as f32,
                    MaterialPass::Opaque => 255.0,
                    MaterialPass::Cutout => if tc[3] >= 127.5 { 255.0 } else { 0.0 },
                    _ => tc[3],
                };
            }
            for ov in d.overlays {
                let oc = ov.sample_filtered(u, v, ov.lod_for(t.uv_density), false);
                albedo = [albedo[0] * lin(oc[0]), albedo[1] * lin(oc[1]), albedo[2] * lin(oc[2])];
            }
            if let (Some(nt), Some((tan, hand))) = (d.normal_tex, t.tangent) {
                let nc = nt.sample_filtered(u, v, nt.lod_for(t.uv_density), false);
                let (tx, ty) = (nc[0] / 127.5 - 1.0, nc[1] / 127.5 - 1.0);
                let tz = (1.0 - tx * tx - ty * ty).max(0.0025).sqrt();
                // re orthogonalize the triangle tangent against the interpolated normal
                let nt_dot = dot(n, tan);
                if let Some(tv) = unit(std::array::from_fn(|c| tan[c] - n[c] * nt_dot)) {
                    let bv = cross(n, tv);
                    n = unit(std::array::from_fn(|c| tv[c] * tx + bv[c] * ty * hand + n[c] * tz)).unwrap_or(n);
                }
            }
        }
        // two sided lighting
        if n[1] < 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        (self.light(n, albedo, d.surface), alpha, z)
    }

    #[inline]
    fn encode(&self, c: [f32; 3]) -> Color32 {
        Color32::from_rgb(encode_srgb(self.to_srgb, c[0]), encode_srgb(self.to_srgb, c[1]), encode_srgb(self.to_srgb, c[2]))
    }

    // calls f with buffer index and the three edge values for every sample of the band the triangle covers
    #[inline(always)]
    fn scan<F: FnMut(usize, [f32; 3])>(&self, geo: &TriGeo, y0: usize, rows: usize, mut f: F) {
        let (min_x, max_x) = (geo.x[0].min(geo.x[1]).min(geo.x[2]), geo.x[0].max(geo.x[1]).max(geo.x[2]));
        let (min_y, max_y) = (geo.y[0].min(geo.y[1]).min(geo.y[2]), geo.y[0].max(geo.y[1]).max(geo.y[2]));
        // first and last pixel whose center lies inside the bounding box
        let ix0 = (min_x + SUB / 2 - 1).div_euclid(SUB).max(0);
        let ix1 = (max_x - SUB / 2).div_euclid(SUB).min(self.w as i64 - 1);
        let iy0 = (min_y + SUB / 2 - 1).div_euclid(SUB).max(y0 as i64);
        let iy1 = (max_y - SUB / 2).div_euclid(SUB).min((y0 + rows) as i64 - 1);
        if ix0 > ix1 || iy0 > iy1 {
            return;
        }
        let (e0, e1, e2) = (geo.edge(0), geo.edge(1), geo.edge(2));
        let mut row = [e0.2 + e0.0 * ix0 + e0.1 * iy0, e1.2 + e1.0 * ix0 + e1.1 * iy0, e2.2 + e2.0 * ix0 + e2.1 * iy0];
        for y in iy0..=iy1 {
            let base = (y as usize - y0) * self.w;
            let mut e = row;
            let mut entered = false;
            for x in ix0..=ix1 {
                if ((e[0] + e0.3) | (e[1] + e1.3) | (e[2] + e2.3)) >= 0 {
                    entered = true;
                    f(base + x as usize, [e[0] as f32, e[1] as f32, e[2] as f32]);
                } else if entered {
                    break;
                }
                e = [e[0] + e0.0, e[1] + e1.0, e[2] + e2.0];
            }
            row = [row[0] + e0.1, row[1] + e1.1, row[2] + e2.1];
        }
    }

    // depth and triangle id of the nearest opaque surface cutout texels below half coverage stay open
    fn visibility(&self, ti: u32, y0: usize, rows: usize, zbuf: &mut [f32], ids: &mut [u32]) {
        let Some(geo) = self.geo(ti as usize) else { return };
        let d = self.draw(ti as usize);
        let pv = geo.idx.map(|i| self.proj[i]);
        let inv_area = 1.0 / geo.area2 as f32;
        let k = [pv[0][3] * inv_area, pv[1][3] * inv_area, pv[2][3] * inv_area];
        let cutout = if d.pass == MaterialPass::Cutout { d.tex } else { None };
        let setup = cutout.map(|tex| {
            let uv = geo.idx.map(|i| self.uvs[i]);
            let det = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
            (tex, uv, tex.lod_for(0.5 * (det.abs() * (SUB * SUB) as f32 * inv_area).log2()))
        });
        self.scan(&geo, y0, rows, |i, e| {
            let q = [e[0] * k[0], e[1] * k[1], e[2] * k[2]];
            let inv = 1.0 / (q[0] + q[1] + q[2]);
            let z = (q[0] * pv[0][2] + q[1] * pv[1][2] + q[2] * pv[2][2]) * inv;
            if z > zbuf[i] {
                if let Some((tex, uv, lod)) = &setup {
                    let u = (q[0] * uv[0][0] + q[1] * uv[1][0] + q[2] * uv[2][0]) * inv;
                    let v = (q[0] * uv[0][1] + q[1] * uv[1][1] + q[2] * uv[2][1]) * inv;
                    if tex.sample_filtered(u, v, *lod, false)[3] < 127.5 {
                        return;
                    }
                }
                zbuf[i] = z;
                ids[i] = ti;
            }
        });
    }

    fn blend_tri(&self, ti: u32, y0: usize, rows: usize, color: &mut [Color32], zbuf: &[f32]) {
        let Some(geo) = self.geo(ti as usize) else { return };
        let t = self.shade_setup(ti as usize, geo);
        let bias = t.draw.depth_bias;
        self.scan(&t.geo, y0, rows, |i, e| {
            let (src, alpha, z) = self.shade(&t, e);
            if z + bias > zbuf[i] && alpha > 0.0 {
                let a = (alpha / 255.0).min(1.0);
                let dst = color[i];
                let mix = |s: f32, d: u8| s * a + self.to_linear[d as usize] * (1.0 - a);
                color[i] = self.encode([mix(src[0], dst.r()), mix(src[1], dst.g()), mix(src[2], dst.b())]);
            }
        });
    }

    // renders the rows of one band starting at screen row y0
    fn band(&self, y0: usize, color: &mut [Color32], zbuf: &mut [f32], ids: &mut [u32], opaque: &[u32], blended: &[u32]) {
        let rows = color.len() / self.w;
        for &ti in opaque {
            self.visibility(ti, y0, rows, zbuf, ids);
        }
        let mut cached: Option<(u32, TriShade)> = None;
        for row in 0..rows {
            let py = ((y0 + row) as i64) * SUB + SUB / 2;
            for x in 0..self.w {
                let i = row * self.w + x;
                let id = ids[i];
                if id == NO_TRI {
                    continue;
                }
                if cached.as_ref().map(|c| c.0) != Some(id) {
                    cached = self.geo(id as usize).map(|g| (id, self.shade_setup(id as usize, g)));
                }
                let Some((_, t)) = &cached else { continue };
                let px = (x as i64) * SUB + SUB / 2;
                let e: [f32; 3] = std::array::from_fn(|k| {
                    let (a, b) = ((k + 1) % 3, (k + 2) % 3);
                    ((px - t.geo.x[a]) * (t.geo.y[b] - t.geo.y[a]) - (py - t.geo.y[a]) * (t.geo.x[b] - t.geo.x[a])) as f32
                });
                color[i] = self.encode(self.shade(t, e).0);
            }
        }
        for &ti in blended {
            self.blend_tri(ti, y0, rows, color, zbuf);
        }
    }
}

fn worker_count() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(16)
}

fn fill_background(color: &mut [Color32], w: usize, h: usize, sky: bool) {
    for y in 0..h {
        let t = (y as f32) / (h.max(1) as f32);
        let c = if sky {
            if t < 0.56 {
                let s = t / 0.56;
                Color32::from_rgb((78.0 + s * 108.0) as u8, (134.0 + s * 84.0) as u8, (202.0 + s * 40.0) as u8)
            } else {
                let s = (t - 0.56) / 0.44;
                Color32::from_rgb((52.0 - s * 22.0) as u8, (78.0 - s * 28.0) as u8, (102.0 - s * 30.0) as u8)
            }
        } else {
            Color32::from_rgb((11.0 + t * 9.0) as u8, (12.0 + t * 10.0) as u8, (17.0 + t * 14.0) as u8)
        };
        color[y * w..(y + 1) * w].fill(c);
    }
}

// depth tested line thick pixels wide
#[allow(clippy::too_many_arguments)]
fn draw_line_z(color_buf: &mut [Color32], z_buf: &mut [f32], w: usize, h: usize, p0: [f32; 4], p1: [f32; 4], col: Color32, z_bias: f32, thick: usize) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let steps = dx.abs().max(dy.abs());
    if !(0.5..=8000.0).contains(&steps) {
        return;
    }
    let n = steps.ceil() as i32;
    let inv = 1.0 / (n as f32);
    let (sx, sy, sz) = (dx * inv, dy * inv, (p1[2] - p0[2]) * inv);
    // widen across the minor axis
    let (ox, oy) = if dx.abs() >= dy.abs() { (0i32, 1i32) } else { (1, 0) };
    let (mut x, mut y, mut z) = (p0[0], p0[1], p0[2] + z_bias);
    for _ in 0..=n {
        for k in 0..thick as i32 {
            let xi = x as i32 + ox * k;
            let yi = y as i32 + oy * k;
            if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                let idx = (yi as usize) * w + (xi as usize);
                if z >= z_buf[idx] {
                    z_buf[idx] = z;
                    color_buf[idx] = col;
                }
            }
        }
        x += sx;
        y += sy;
        z += sz;
    }
}

// renders the mesh into its color and depth buffers at w by h samples
pub fn rasterize_mesh(mesh: &mut Mesh3D, w: usize, h: usize) {
    rasterize(mesh, w, h, 1);
}

fn rasterize(mesh: &mut Mesh3D, w: usize, h: usize, line_px: usize) {
    if w == 0 || h == 0 {
        return;
    }
    let npix = w * h;
    let mut color = std::mem::take(&mut mesh.color_buf);
    let mut zbuf = std::mem::take(&mut mesh.z_buf);
    let mut ids = std::mem::take(&mut mesh.id_buf);
    let mut proj = std::mem::take(&mut mesh.proj_buf);
    let mut fixed = std::mem::take(&mut mesh.fixed_buf);
    let mut bins = std::mem::take(&mut mesh.bins);
    let mut blend_bins = std::mem::take(&mut mesh.blend_bins);
    let mut blend_order = std::mem::take(&mut mesh.blend_order);
    mesh.fb_w = w;
    mesh.fb_h = h;
    color.resize(npix, Color32::BLACK);
    zbuf.clear();
    zbuf.resize(npix, f32::NEG_INFINITY);
    ids.clear();
    ids.resize(npix, NO_TRI);
    fill_background(&mut color, w, h, mesh.sky_bg);

    let (sy, cy) = mesh.yaw.sin_cos();
    let (sp, cp) = mesh.pitch.sin_cos();
    let min_dim = (w.min(h) as f32) * 0.44;
    let scale = (min_dim / mesh.radius) * mesh.zoom;
    let cx = (w as f32) * 0.5 + mesh.pan_x * (w as f32);
    let cy_s = (h as f32) * 0.5 + mesh.pan_y * (h as f32);
    let ctr = mesh.center;
    let persp_k = 1.0 / (mesh.radius * 4.5);

    // screen x y view depth and the perspective factor that all attributes are interpolated with
    proj.clear();
    proj.extend(mesh.verts.iter().map(|v| {
        let (x, y, z) = (v[0] - ctr[0], v[1] - ctr[1], v[2] - ctr[2]);
        let x1 = x * cy - y * sy;
        let y1 = x * sy + y * cy;
        let y2 = -y1 * cp + z * sp;
        let z2 = z * cp + y1 * sp;
        let persp = 1.0 / (1.0 - (y2 * persp_k).clamp(-0.65, 0.65));
        [cx + x1 * scale * persp, cy_s - z2 * scale * persp, y2, persp]
    }));
    fixed.clear();
    fixed.extend(proj.iter().map(|p| {
        // the saturating cast maps nan to zero
        let snap = |v: f32| (v.clamp(-MAX_COORD, MAX_COORD) * SUB as f32).round() as i32;
        [snap(p[0]), snap(p[1])]
    }));

    if !mesh.tris.is_empty() {
        let textured = mesh.shade_mode == ShadeMode::Textured;
        let do_solid = matches!(mesh.shade_mode, ShadeMode::Textured | ShadeMode::Solid | ShadeMode::SolidWire);
        let do_wire = matches!(mesh.shade_mode, ShadeMode::Wireframe | ShadeMode::SolidWire);
        let has_uvs = mesh.uvs.len() == mesh.verts.len() && !mesh.uvs.is_empty();
        let vis = mesh.visible_tris();

        if do_solid {
            let flags = Flags {
                textured,
                has_uvs,
                has_preview_texture: textured
                    && has_uvs
                    && mesh.geom_materials.iter().any(|m| m.pass == MaterialPass::Cutout && m.texture.is_none())
                    && mesh.tri_geom.get(vis.clone()).is_some_and(|ids| {
                        ids.iter().any(|&g| mesh.geom_materials.get(g as usize).is_some_and(|m| m.texture.is_some()))
                    }),
                use_blending: textured && mesh.texture.is_none() && mesh.geom_materials.iter().any(|m| m.blends(has_uvs)),
            };
            let draws: Vec<GeomDraw> = (0..mesh.geom_materials.len()).map(|g| geom_draw(mesh, g, flags)).collect();
            let fallback: Vec<GeomDraw> =
                (0..MAT_PALETTE.len()).map(|g| geom_draw(mesh, g + MAT_PALETTE.len() * (mesh.geom_materials.len() / MAT_PALETTE.len() + 1), flags)).collect();
            let key = unit([0.42, 0.65, 0.63]).unwrap();
            let scene = Scene {
                w,
                verts: &mesh.verts,
                uvs: &mesh.uvs,
                tris: &mesh.tris,
                tri_geom: &mesh.tri_geom,
                face_normals: &mesh.normals,
                corner_normals: if mesh.flat_shading { &[] } else { mesh.corner_normals() },
                proj: &proj,
                fixed: &fixed,
                draws: &draws,
                fallback: &fallback,
                has_uvs,
                rot: [sy, cy, sp, cp],
                lights: Lights {
                    key,
                    fill: unit([-0.60, 0.25, -0.45]).unwrap(),
                    half: unit([key[0], key[1] + 1.0, key[2]]).unwrap(),
                    up: [0.0, sp, cp],
                },
                to_linear: srgb_to_linear_table(),
                to_srgb: linear_to_srgb_table(),
            };

            // far to near order of the translucent triangles decals first
            blend_order.clear();
            if flags.use_blending {
                for ti in vis.clone() {
                    let d = scene.draw(ti);
                    if d.blend && !d.hidden {
                        let [a, b, c] = mesh.tris[ti].map(|i| proj[i as usize][2]);
                        blend_order.push((ti, a + b + c, d.pass == MaterialPass::Decal));
                    }
                }
                blend_order.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.total_cmp(&b.1)));
            }

            let small = vis.len() < 2000 && npix < 150_000;
            let threads = if small { 1 } else { worker_count() };
            let band_count = if threads == 1 { 1 } else { (threads * 4).min(h.div_ceil(8)).max(1) };
            let band_rows = h.div_ceil(band_count);
            let band_count = h.div_ceil(band_rows);
            for b in [&mut bins, &mut blend_bins] {
                b.iter_mut().for_each(Vec::clear);
                b.resize_with(band_count, Vec::new);
            }
            let w_fixed = (w as i64) * SUB;
            let bin = |ti: usize, into: &mut [Vec<u32>]| {
                let [a, b, c] = mesh.tris[ti].map(|i| fixed[i as usize]);
                let (min_y, max_y) = (a[1].min(b[1]).min(c[1]) as i64, a[1].max(b[1]).max(c[1]) as i64);
                let (min_x, max_x) = (a[0].min(b[0]).min(c[0]) as i64, a[0].max(b[0]).max(c[0]) as i64);
                if max_x < 0 || min_x >= w_fixed {
                    return;
                }
                let y_lo = (min_y + SUB / 2 - 1).div_euclid(SUB).max(0);
                let y_hi = (max_y - SUB / 2).div_euclid(SUB).min(h as i64 - 1);
                if y_lo > y_hi {
                    return;
                }
                for band in &mut into[y_lo as usize / band_rows..=y_hi as usize / band_rows] {
                    band.push(ti as u32);
                }
            };
            for ti in vis.clone() {
                let d = scene.draw(ti);
                if !d.hidden && !d.blend {
                    bin(ti, &mut bins);
                }
            }
            for &(ti, _, _) in &blend_order {
                bin(ti, &mut blend_bins);
            }

            let stride = band_rows * w;
            let jobs = Mutex::new(color.chunks_mut(stride).zip(zbuf.chunks_mut(stride)).zip(ids.chunks_mut(stride)).enumerate());
            let work = || loop {
                let job = jobs.lock().unwrap().next();
                let Some((bi, ((c, z), id))) = job else { break };
                scene.band(bi * band_rows, c, z, id, &bins[bi], &blend_bins[bi]);
            };
            if threads == 1 {
                work();
            } else {
                std::thread::scope(|s| {
                    for _ in 1..threads.min(band_count) {
                        s.spawn(work);
                    }
                    work();
                });
            }
        }

        if do_wire {
            let wire_col = if do_solid { Color32::from_rgb(252, 175, 23) } else { Color32::from_rgb(72, 210, 255) };
            let z_bias = if do_solid { mesh.radius * 0.004 } else { 0.0 };
            let step = (vis.len() / 25_000).max(1);
            for t in mesh.tris[vis].iter().step_by(step) {
                let [p0, p1, p2] = t.map(|i| proj[i as usize]);
                draw_line_z(&mut color, &mut zbuf, w, h, p0, p1, wire_col, z_bias, line_px);
                draw_line_z(&mut color, &mut zbuf, w, h, p1, p2, wire_col, z_bias, line_px);
                draw_line_z(&mut color, &mut zbuf, w, h, p2, p0, wire_col, z_bias, line_px);
            }
        }
    } else if !mesh.edges.is_empty() {
        let col = Color32::from_rgb(72, 210, 255);
        for &(a, b) in mesh.edges.iter().take(60_000) {
            draw_line_z(&mut color, &mut zbuf, w, h, proj[a as usize], proj[b as usize], col, 0.0, line_px);
        }
    } else {
        let col = Color32::from_rgb(252, 175, 23);
        for p0 in &proj {
            let (xi, yi) = (p0[0] as i32, p0[1] as i32);
            if xi < 1 || yi < 1 {
                continue;
            }
            for (dx, dy) in (0..line_px).flat_map(|dy| (0..line_px).map(move |dx| (dx, dy))) {
                let (x, y) = (xi as usize + dx, yi as usize + dy);
                if x + 1 < w && y + 1 < h && p0[2] > zbuf[y * w + x] {
                    zbuf[y * w + x] = p0[2];
                    color[y * w + x] = col;
                }
            }
        }
    }

    draw_siren_glow(mesh, &mut color, &zbuf, w, h, &proj_light(mesh, w, h));

    mesh.color_buf = color;
    mesh.z_buf = zbuf;
    mesh.id_buf = ids;
    mesh.proj_buf = proj;
    mesh.fixed_buf = fixed;
    mesh.bins = bins;
    mesh.blend_bins = blend_bins;
    mesh.blend_order = blend_order;
}

// projects a model space point exactly like the vertices returns screen x y and view depth
fn proj_light(mesh: &Mesh3D, w: usize, h: usize) -> impl Fn([f32; 3]) -> [f32; 3] {
    let (sy, cy) = mesh.yaw.sin_cos();
    let (sp, cp) = mesh.pitch.sin_cos();
    let scale = ((w.min(h) as f32) * 0.44 / mesh.radius) * mesh.zoom;
    let cx = (w as f32) * 0.5 + mesh.pan_x * (w as f32);
    let cy_s = (h as f32) * 0.5 + mesh.pan_y * (h as f32);
    let ctr = mesh.center;
    let persp_k = 1.0 / (mesh.radius * 4.5);
    move |v: [f32; 3]| {
        let (x, y, z) = (v[0] - ctr[0], v[1] - ctr[1], v[2] - ctr[2]);
        let x1 = x * cy - y * sy;
        let y1 = x * sy + y * cy;
        let y2 = -y1 * cp + z * sp;
        let z2 = z * cp + y1 * sp;
        let persp = 1.0 / (1.0 - (y2 * persp_k).clamp(-0.65, 0.65));
        [cx + x1 * scale * persp, cy_s - z2 * scale * persp, y2]
    }
}

// additive glow of the lit siren emitters drawn over the shaded frame and hidden behind nearer geometry
fn draw_siren_glow(mesh: &Mesh3D, color: &mut [Color32], zbuf: &[f32], w: usize, h: usize, project: &impl Fn([f32; 3]) -> [f32; 3]) {
    if !mesh.els_enabled || mesh.els_points.is_empty() || w == 0 || h == 0 {
        return;
    }
    let (wf, hf) = (w as f32, h as f32);
    let glow_radius = ((w.min(h) as f32) * 0.014).clamp(5.0, 18.0 * (w.min(h) as f32 / 600.0).max(1.0));
    let core_radius = (glow_radius * 0.28).max(2.0);
    for light in &mesh.els_points {
        if mesh.els_flash && !super::rage9_siren_lit(light.slot, mesh.els_flash_phase) {
            continue;
        }
        let [sx, sy, depth] = project(light.position);
        if !(sx.is_finite() && sy.is_finite()) || sx < -glow_radius || sy < -glow_radius || sx >= wf + glow_radius || sy >= hf + glow_radius {
            continue;
        }
        let center = sy.clamp(0.0, hf - 1.0) as usize * w + sx.clamp(0.0, wf - 1.0) as usize;
        if zbuf[center].is_finite() && depth < zbuf[center] - mesh.radius * 0.08 {
            continue;
        }
        let rgb: [f32; 3] = if light.bank { [45.0, 115.0, 255.0] } else { [255.0, 35.0, 28.0] };
        let x0 = (sx - glow_radius).floor().max(0.0) as usize;
        let x1 = (sx + glow_radius).ceil().min(wf - 1.0) as usize;
        let y0 = (sy - glow_radius).floor().max(0.0) as usize;
        let y1 = (sy + glow_radius).ceil().min(hf - 1.0) as usize;
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (dx, dy) = (px as f32 + 0.5 - sx, py as f32 + 0.5 - sy);
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > glow_radius {
                    continue;
                }
                let strength = if distance <= core_radius {
                    1.0
                } else {
                    let t = 1.0 - (distance - core_radius) / (glow_radius - core_radius);
                    t * t * 0.42
                };
                let d = &mut color[py * w + px];
                *d = Color32::from_rgb(
                    (d.r() as f32 + rgb[0] * strength).min(255.0) as u8,
                    (d.g() as f32 + rgb[1] * strength).min(255.0) as u8,
                    (d.b() as f32 + rgb[2] * strength).min(255.0) as u8,
                );
            }
        }
    }
}

// source span and weights of every destination sample along one axis for ratios up to two
fn axis_taps(src: usize, dst: usize) -> Vec<(usize, [f32; 3])> {
    let ratio = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            let (lo, hi) = (i as f64 * ratio, (i + 1) as f64 * ratio);
            let start = lo.floor() as usize;
            let weights = std::array::from_fn(|j| {
                let (a, b) = ((start + j) as f64, (start + j + 1) as f64);
                ((hi.min(b) - lo.max(a)).max(0.0) / ratio) as f32
            });
            (start, weights)
        })
        .collect()
}

// area filtered reduction in linear light
fn downsample(src: &[Color32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<Color32> {
    let (to_linear, to_srgb) = (srgb_to_linear_table(), linear_to_srgb_table());
    let (tx, ty) = (axis_taps(sw, dw), axis_taps(sh, dh));
    let mut out = vec![Color32::BLACK; dw * dh];
    let threads = worker_count();
    let rows_per = dh.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (chunk_i, chunk) in out.chunks_mut(rows_per * dw).enumerate() {
            let (tx, ty) = (&tx, &ty);
            s.spawn(move || {
                for (row, line) in chunk.chunks_mut(dw).enumerate() {
                    let (y0, wy) = ty[chunk_i * rows_per + row];
                    for (x, px) in line.iter_mut().enumerate() {
                        let (x0, wx) = tx[x];
                        let mut acc = [0.0f32; 3];
                        for (j, &wyj) in wy.iter().enumerate() {
                            if wyj <= 0.0 {
                                continue;
                            }
                            let line = (y0 + j).min(sh - 1) * sw;
                            for (i, &wxi) in wx.iter().enumerate() {
                                if wxi <= 0.0 {
                                    continue;
                                }
                                let c = src[line + (x0 + i).min(sw - 1)];
                                let wt = wxi * wyj;
                                acc[0] += to_linear[c.r() as usize] * wt;
                                acc[1] += to_linear[c.g() as usize] * wt;
                                acc[2] += to_linear[c.b() as usize] * wt;
                            }
                        }
                        *px = Color32::from_rgb(encode_srgb(to_srgb, acc[0]), encode_srgb(to_srgb, acc[1]), encode_srgb(to_srgb, acc[2]));
                    }
                }
            });
        }
    });
    out
}

// supersampling factor that keeps a w by h viewport inside the sample budget
pub(super) fn idle_supersampling(w: usize, h: usize) -> f32 {
    (MAX_SAMPLES / (w * h).max(1) as f32).sqrt().clamp(1.0, 2.0)
}

// renders a viewport image scale above one supersamples and filters down below one renders fewer samples
// returns the image size and pixels
pub(super) fn render_image(mesh: &mut Mesh3D, out_w: usize, out_h: usize, scale: f32) -> (usize, usize, Vec<Color32>) {
    let scale = if scale.is_finite() { scale.clamp(0.25, 2.0) } else { 1.0 };
    let iw = ((out_w as f32 * scale).round() as usize).max(1);
    let ih = ((out_h as f32 * scale).round() as usize).max(1);
    rasterize(mesh, iw, ih, if scale > 1.0 { scale.ceil() as usize } else { 1 });
    if iw > out_w && ih > out_h {
        (out_w, out_h, downsample(&mesh.color_buf, iw, ih, out_w, out_h))
    } else {
        (iw, ih, mesh.color_buf.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_taps_cover_every_source_sample_once() {
        for (src, dst) in [(200, 100), (150, 100), (161, 100), (100, 100), (7, 5)] {
            let taps = axis_taps(src, dst);
            let mut cover = vec![0.0f32; src + 3];
            for (start, w) in &taps {
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-4, "weights of one sample sum to one");
                for (j, wj) in w.iter().enumerate() {
                    cover[start + j] += wj * src as f32 / dst as f32;
                }
            }
            assert!(cover[..src].iter().all(|c| (c - 1.0).abs() < 1e-3), "{src}->{dst} {cover:?}");
            assert!(cover[src..].iter().all(|c| c.abs() < 1e-3));
        }
    }

    #[test]
    fn downsample_averages_in_linear_light() {
        // black and white columns average to srgb 188 not 128
        let src: Vec<Color32> = (0..16 * 16).map(|i| if i % 2 == 0 { Color32::WHITE } else { Color32::BLACK }).collect();
        let out = downsample(&src, 16, 16, 8, 8);
        assert!(out.iter().all(|c| (c.r() as i32 - 188).abs() <= 1 && c.r() == c.g() && c.g() == c.b()), "{:?}", out[0]);
        let flat = downsample(&vec![Color32::from_rgb(40, 90, 200); 15 * 9], 15, 9, 10, 6);
        assert!(flat.iter().all(|c| *c == Color32::from_rgb(40, 90, 200)));
    }
}

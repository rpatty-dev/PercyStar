use crate::model3d::{Mesh3D, ShadeMode, Texture2D};
use eframe::egui::{self, Color32, ColorImage, CornerRadius, TextureHandle, TextureOptions};
use flate2::{Decompress, FlushDecompress};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct WorldMapData {
    pub w: usize,
    pub h: usize,
    pub bmin: [f32; 3],
    pub bmax: [f32; 3],
    pub heights: Vec<u8>,
    pub tex: Option<TextureHandle>,
}

pub struct WorldMapState {
    pub data: Option<WorldMapData>,
    pub status: String,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub show_nav_overlay: bool,
    pub show_3d_terrain: bool,
    pub terrain_mesh: Option<Mesh3D>,
    pub loading: bool,
    pending_mesh: Arc<Mutex<Option<Result<Mesh3D, String>>>>,
    gpu_launch: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
    gpu_starting: bool,
}

impl Default for WorldMapState {
    fn default() -> Self {
        Self {
            data: None,
            status: "Click 'Load Map' or open common.rpf".into(),
            zoom: 1.0,
            pan: [0.0, 0.0],
            show_nav_overlay: true,
            show_3d_terrain: false,
            terrain_mesh: None,
            loading: false,
            pending_mesh: Arc::new(Mutex::new(None)),
            gpu_launch: None,
            gpu_starting: false,
        }
    }
}

fn rd16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}

fn rdf32(d: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

// decode raw mhiu bytes into 2048x2048 world elevation data
pub fn parse_mhiu_data(raw: &[u8]) -> Option<WorldMapData> {
    if raw.len() < 44 || &raw[..4] != b"MHIU" {
        return None;
    }
    let w = rd16(raw, 6) as usize;
    let h = rd16(raw, 8) as usize;
    if w == 0 || h == 0 || w > 4096 || h > 4096 || raw.len() < 44 + h * 8 {
        return None;
    }
    let bmin = [rdf32(raw, 12), rdf32(raw, 16), rdf32(raw, 20)];
    let bmax = [rdf32(raw, 24), rdf32(raw, 28), rdf32(raw, 32)];
    let rows = &raw[44..44 + h * 8];
    let pix = &raw[44 + h * 8..];
    let mut heights = vec![23u8; w * h];
    for y in 0..h {
        let r = &rows[y * 8..y * 8 + 8];
        let sx = rd16(r, 0) as usize;
        let cnt = rd16(r, 2) as usize;
        let off = i32::from_le_bytes([r[4], r[5], r[6], r[7]]) as isize;
        if cnt > 0 && sx + cnt <= w {
            let src = off + sx as isize;
            if src >= 0 && (src as usize) + cnt <= pix.len() {
                heights[y * w + sx..y * w + sx + cnt].copy_from_slice(&pix[src as usize..src as usize + cnt]);
            }
        }
    }
    Some(WorldMapData {
        w,
        h,
        bmin,
        bmax,
        heights,
        tex: None,
    })
}

impl WorldMapData {
    pub fn elevation_m(&self, px: usize, py: usize) -> f32 {
        let v = self.heights.get(py * self.w + px).copied().unwrap_or(23) as f32;
        self.bmin[2] + (v / 255.0) * (self.bmax[2] - self.bmin[2])
    }

    pub fn build_color_image(&self) -> ColorImage {
        let mut pixels = Vec::with_capacity(self.w * self.h);
        for &v in &self.heights {
            let c = if v <= 23 {
                Color32::from_rgb(16, 44, 92)
            } else if v <= 30 {
                let t = (v - 23) * 14;
                Color32::from_rgb(45 + t, 115 + t, 72)
            } else if v <= 85 {
                let t = ((v - 30) as u16 * 120 / 55) as u8;
                Color32::from_rgb(80 + t, 145 + t / 2, 70)
            } else {
                let t = ((v - 85) as u16 * 100 / 170) as u8;
                Color32::from_rgb(155 + t, 145 + t, 130 + t)
            };
            pixels.push(c);
        }
        ColorImage {
            size: [self.w, self.h],
            pixels,
        }
    }

    // The flat map uses the same vertical image orientation as the GPU minimap.
    fn map_uv_for_world(&self, x: f32, y: f32) -> egui::Pos2 {
        egui::pos2(
            (x - self.bmin[0]) / (self.bmax[0] - self.bmin[0]).max(1.0),
            (y - self.bmin[1]) / (self.bmax[1] - self.bmin[1]).max(1.0),
        )
    }

    fn world_for_map_uv(&self, uv: egui::Pos2) -> [f32; 2] {
        [
            self.bmin[0] + uv.x * (self.bmax[0] - self.bmin[0]).max(1.0),
            self.bmin[1] + uv.y * (self.bmax[1] - self.bmin[1]).max(1.0),
        ]
    }

    fn height_pixel_for_map_uv(&self, uv: egui::Pos2) -> (usize, usize) {
        let u = uv.x.clamp(0.0, 1.0);
        let texture_v = uv.y.clamp(0.0, 1.0);
        (
            ((u * self.w as f32) as usize).min(self.w - 1),
            ((texture_v * self.h as f32) as usize).min(self.h - 1),
        )
    }

    // Build terrain in game units, matching true world elevation.
    pub fn build_world_terrain_mesh(&self, step: usize) -> Mesh3D {
        let mut m = self.build_terrain_mesh_with_tex(step, 2048, 1.0);
        m.name = "mhiu_world_terrain".into();
        m
    }

    pub fn build_terrain_mesh(&self, step: usize) -> Mesh3D {
        // Uniform 1:100 scaling keeps the embedded preview's slopes true to life.
        self.build_terrain_mesh_with_tex(step.max(4), 512, 0.01)
    }

    fn build_terrain_mesh_with_tex(&self, step: usize, max_tex: usize, scale: f32) -> Mesh3D {
        let step = step.max(2);
        let gw = self.w.div_ceil(step);
        let gh = self.h.div_ceil(step);
        let mut verts = Vec::with_capacity(gw * gh);
        let mut uvs = Vec::with_capacity(gw * gh);
        for gy in 0..gh {
            let py = (gy * step).min(self.h - 1);
            let wy = self.bmin[1] + (py as f32 / self.h as f32) * (self.bmax[1] - self.bmin[1]);
            for gx in 0..gw {
                let px = (gx * step).min(self.w - 1);
                let wx = self.bmin[0] + (px as f32 / self.w as f32) * (self.bmax[0] - self.bmin[0]);
                let wz = self.elevation_m(px, py).max(0.0);
                verts.push([wx * scale, wy * scale, wz * scale]);
                uvs.push([px as f32 / self.w as f32, py as f32 / self.h as f32]);
            }
        }
        let mut tris = Vec::with_capacity((gw - 1) * (gh - 1) * 2);
        for gy in 0..gh.saturating_sub(1) {
            for gx in 0..gw.saturating_sub(1) {
                let i0 = (gy * gw + gx) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + gw as u32;
                let i3 = i2 + 1;
                tris.push([i0, i2, i1]);
                tris.push([i1, i2, i3]);
            }
        }
        // bake hillshaded elevation texture for crisp 3d terrain
        let tw = self.w.min(max_tex);
        let th = self.h.min(max_tex);
        let mut tpix = Vec::with_capacity(tw * th);
        for ty in 0..th {
            let py = (ty * self.h / th).min(self.h - 1);
            let py1 = (py + 1).min(self.h - 1);
            for tx in 0..tw {
                let px = (tx * self.w / tw).min(self.w - 1);
                let px1 = (px + 1).min(self.w - 1);
                let v = self.heights[py * self.w + px];
                let vx = self.heights[py * self.w + px1] as f32;
                let vy = self.heights[py1 * self.w + px] as f32;
                let shade = (0.85 + ((v as f32 - vx) + (v as f32 - vy)) * 0.04).clamp(0.45, 1.25);
                let (r, g, b) = if v <= 23 {
                    (18.0, 52.0, 108.0)
                } else if v <= 26 {
                    (182.0, 170.0, 128.0)
                } else if v <= 55 {
                    let t = (v - 26) as f32 / 29.0;
                    (58.0 + t * 52.0, 128.0 + t * 24.0, 62.0)
                } else if v <= 110 {
                    let t = (v - 55) as f32 / 55.0;
                    (110.0 + t * 48.0, 152.0 - t * 22.0, 62.0 + t * 18.0)
                } else {
                    let t = ((v - 110) as f32 / 145.0).min(1.0);
                    (158.0 + t * 75.0, 130.0 + t * 90.0, 80.0 + t * 130.0)
                };
                tpix.push(Color32::from_rgb(
                    (r * shade).min(255.0) as u8,
                    (g * shade).min(255.0) as u8,
                    (b * shade).min(255.0) as u8,
                ));
            }
        }
        let tex = Some(Texture2D::new(tw, th, tpix));

        let tri_geom = vec![0u16; tris.len()];
        let mut m = Mesh3D::from_parts(
            format!("world_terrain_{gw}x{gh}"),
            verts,
            uvs,
            tris,
            tri_geom,
            Vec::new(),
            tex,
            format!("MHIU 3D Terrain ({}x{} -> {gw}x{gh})", self.w, self.h),
            1,
            1,
            "World Terrain".into(),
        );
        m.shade_mode = ShadeMode::Textured;
        m.sky_bg = true;
        m
    }
}

fn map_texture_uv_rect() -> egui::Rect {
    egui::Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0))
}

fn nav_tile_rect(map: &WorldMapData, map_rect: egui::Rect, wx: f32, wy: f32) -> egui::Rect {
    let uv = map.map_uv_for_world(wx, wy);
    let size = egui::vec2(
        (128.0 / (map.bmax[0] - map.bmin[0]).max(1.0) * map_rect.width()).max(2.0),
        (128.0 / (map.bmax[1] - map.bmin[1]).max(1.0) * map_rect.height()).max(2.0),
    );
    egui::Rect::from_min_size(
        map_rect.min + egui::vec2(uv.x * map_rect.width(), uv.y * map_rect.height()),
        size,
    )
}

// extract first mhiu heightmap directly from common.rpf
pub fn load_mhiu_from_rpf(path: &Path) -> Result<WorldMapData, String> {
    let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut i = 0x40usize;
    let mut out = vec![0u8; 65536];
    while i + 16 < data.len() {
        let b0 = data[i];
        let b1 = data[i + 1];
        if b0 == 0x78 && matches!(b1, 0x01 | 0x5e | 0x9c | 0xda) && ((b0 as u16) << 8 | b1 as u16).is_multiple_of(31) {
            let mut dec = Decompress::new(true);
            let _ = dec.decompress(&data[i..data.len().min(i + 262_144)], &mut out, FlushDecompress::None);
            let olen = dec.total_out() as usize;
            if olen >= 44 && &out[..4] == b"MHIU" {
                let mut stitched = Vec::with_capacity(1_300_000);
                let mut cur = i;
                while cur + 16 < data.len() {
                    let cb0 = data[cur];
                    let cb1 = data[cur + 1];
                    if cb0 != 0x78 || !matches!(cb1, 0x01 | 0x5e | 0x9c | 0xda) || !((cb0 as u16) << 8 | cb1 as u16).is_multiple_of(31) {
                        break;
                    }
                    let mut d = Decompress::new(true);
                    let _ = d.decompress(&data[cur..data.len().min(cur + 262_144)], &mut out, FlushDecompress::None);
                    let clen = d.total_in() as usize;
                    let colen = d.total_out() as usize;
                    if clen == 0 || colen == 0 {
                        break;
                    }
                    stitched.extend_from_slice(&out[..colen]);
                    cur = (cur + clen + 15) & !15;
                    if colen != 65536 {
                        break;
                    }
                }
                return parse_mhiu_data(&stitched).ok_or_else(|| "Invalid MHIU stream".into());
            }
        }
        i += 16;
    }
    Err("No MHIU heightmap found in archive".into())
}

// parse navmesh grid coords from filename nav_{gx}_{gy}_(wx,wy).ynv
pub fn parse_nav_coords(name: &str) -> Option<(f32, f32)> {
    let s = name.strip_prefix("nav_")?;
    let p = s.find("_(")?;
    let rest = &s[p + 2..];
    let end = rest.find(')')?;
    let (xs, ys) = rest[..end].split_once(',')?;
    Some((xs.parse().ok()?, ys.parse().ok()?))
}

// render interactive 2d/3d world map viewer tab
pub fn render_world_map_ui(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    wm: &mut WorldMapState,
    nav_tiles: &[(usize, f32, f32)],
    common_rpf: Option<&Path>,
    levels_rpf: Option<&Path>,
) -> Option<usize> {
    while let Some(rx) = &wm.gpu_launch {
        match rx.try_recv() {
            Ok(result) => {
                wm.gpu_starting = false;
                wm.status = match result {
                    Ok(()) => "GPU World opened in a separate window".into(),
                    Err(error) => format!("GPU World failed: {error}"),
                };
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                wm.gpu_starting = false;
                wm.gpu_launch = None;
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
        }
    }
    if let Some(res) = wm.pending_mesh.lock().unwrap().take() {
        wm.loading = false;
        match res {
            Ok(mut m) => {
                m.sky_bg = true;
                wm.status = format!("Loaded {} ({} parts, {} geoms) — WASD/Q/E + Drag to fly", m.name, m.parts.len(), m.geom_count);
                wm.terrain_mesh = Some(m);
                wm.show_3d_terrain = true;
            }
            Err(e) => wm.status = format!("SLOD load failed: {e}"),
        }
    }
    if wm.loading {
        ctx.request_repaint();
    }
    let mut clicked_nav = None;
    ui.horizontal(|ui| {
        if ui.button("Load World Map").clicked() {
            if let Some(p) = common_rpf {
                match load_mhiu_from_rpf(p) {
                    Ok(d) => {
                        wm.status = format!("Loaded {}x{} MHIU ({:.0}..{:.0}, {:.0}..{:.0})", d.w, d.h, d.bmin[0], d.bmax[0], d.bmin[1], d.bmax[1]);
                        wm.data = Some(d);
                        wm.terrain_mesh = None;
                    }
                    Err(e) => wm.status = e,
                }
            } else {
                wm.status = "Open a game folder or common.rpf first".into();
            }
        }
        let btn_lbl = if wm.loading { "Loading City Blocks..." } else { "3D City Blocks (Brickell SLOD)" };
        if ui.add_enabled(!wm.loading, egui::Button::new(btn_lbl)).clicked() {
            if let Some(lp) = levels_rpf {
                let lp = lp.to_path_buf();
                let slot = Arc::clone(&wm.pending_mesh);
                let c = ctx.clone();
                wm.loading = true;
                wm.status = "Loading Brickell SLOD (317 geoms + TTPF textures)...".into();
                std::thread::spawn(move || {
                    let res = crate::model3d::decode_rage9_model_from_rpf(&lp, 8_679_358_480, "brickell_slod");
                    *slot.lock().unwrap() = Some(res);
                    c.request_repaint();
                });
            } else {
                wm.status = "Open game folder (needs levels.rpf)".into();
            }
        }
        ui.checkbox(&mut wm.show_nav_overlay, format!("NavMesh Overlay ({})", nav_tiles.len()));
        if wm.data.is_some() || wm.terrain_mesh.is_some() {
            if ui.selectable_label(!wm.show_3d_terrain, "2D Map").clicked() {
                wm.show_3d_terrain = false;
            }
            if ui.selectable_label(wm.show_3d_terrain, "3D Terrain").clicked() {
                wm.show_3d_terrain = true;
                if let Some(d) = &wm.data {
                    if wm.terrain_mesh.as_ref().map(|m| m.name.contains("brickell")).unwrap_or(true) {
                        wm.terrain_mesh = Some(d.build_terrain_mesh(8));
                    }
                }
            }
            if ui.button("Reset View").clicked() {
                wm.zoom = 1.0;
                wm.pan = [0.0, 0.0];
            }
        }
        if ui.add_enabled(!wm.gpu_starting, egui::Button::new("Launch 3D GPU World (wgpu)")).on_hover_text("Open DX12/Vulkan GPU 3D World with PBR, Sun/Time-of-Day (F6), Minimap & Raytracing (R)").clicked() {
            match crate::world_gpu::launch_world_gpu_process(common_rpf, levels_rpf, ctx.clone()) {
                Ok(rx) => {
                    wm.gpu_launch = Some(rx);
                    wm.gpu_starting = true;
                    wm.status = "Starting GPU World...".into();
                }
                Err(e) => wm.status = e,
            }
        }
        if ui.add_enabled(!wm.gpu_starting, egui::Button::new(egui::RichText::new("RF World View").strong())).on_hover_text("RageForge-inspired 3D World View with Time-of-Day presets, Water toggle (V), 700m–16km draw radius & PercyStar PBR/RT shaders").clicked() {
            match crate::world_gpu::launch_world_gpu_process(common_rpf, levels_rpf, ctx.clone()) {
                Ok(rx) => {
                    wm.gpu_launch = Some(rx);
                    wm.gpu_starting = true;
                    wm.status = "Starting RF World View...".into();
                }
                Err(e) => wm.status = e,
            }
        }
        ui.label(egui::RichText::new(&wm.status).size(11.5).color(Color32::from_rgb(175, 195, 220)));
    });
    ui.separator();

    if wm.show_3d_terrain {
        if let Some(mesh) = wm.terrain_mesh.as_mut() {
            crate::model3d::render_mesh3d_viewport(ui, mesh);
        }
        return None;
    }

    let Some(map) = wm.data.as_mut() else {
        ui.centered_and_justified(|ui| {
            ui.label("No MHIU world heightmap loaded yet. Click 'Load World Map' above.");
        });
        return None;
    };

    if map.tex.is_none() {
        let ci = map.build_color_image();
        map.tex = Some(ctx.load_texture("percystar_world_mhiu", ci, TextureOptions::LINEAR));
    }

    let avail = ui.available_size();
    let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
    if resp.dragged() {
        let d = resp.drag_delta();
        wm.pan[0] += d.x;
        wm.pan[1] += d.y;
    }
    if resp.double_clicked() {
        wm.zoom = 1.0;
        wm.pan = [0.0, 0.0];
    }
    if resp.hovered() {
        let s = ui.input(|i| i.smooth_scroll_delta.y);
        if s != 0.0 {
            wm.zoom = (wm.zoom * (1.0 + s * 0.0025)).clamp(0.4, 24.0);
        }
    }

    let base_sz = rect.width().min(rect.height()) * 0.94 * wm.zoom;
    let center = rect.center() + egui::vec2(wm.pan[0], wm.pan[1]);
    let map_rect = egui::Rect::from_center_size(center, egui::vec2(base_sz, base_sz));

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(6), Color32::from_rgb(10, 14, 24));
    if let Some(tex) = &map.tex {
        painter.image(
            tex.id(),
            map_rect,
            map_texture_uv_rect(),
            Color32::WHITE,
        );
    }

    // draw navmesh tile grid overlay
    if wm.show_nav_overlay && !nav_tiles.is_empty() {
        let col = Color32::from_rgba_unmultiplied(255, 175, 45, 110);
        for &(idx, wx, wy) in nav_tiles {
            let r = nav_tile_rect(map, map_rect, wx, wy);
            if rect.intersects(r) {
                painter.rect_filled(r, 0.0, col);
                if resp.clicked() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        if r.contains(pos) {
                            clicked_nav = Some(idx);
                        }
                    }
                }
            }
        }
    }

    // live world coordinates hud
    if let Some(pos) = resp.hover_pos() {
        if map_rect.contains(pos) {
            let u = ((pos.x - map_rect.min.x) / map_rect.width()).clamp(0.0, 1.0);
            let v = ((pos.y - map_rect.min.y) / map_rect.height()).clamp(0.0, 1.0);
            let uv = egui::pos2(u, v);
            let [wx, wy] = map.world_for_map_uv(uv);
            let (px, py) = map.height_pixel_for_map_uv(uv);
            let wz = map.elevation_m(px, py);
            painter.text(
                egui::pos2(rect.min.x + 12.0, rect.max.y - 14.0),
                egui::Align2::LEFT_BOTTOM,
                format!("World X: {wx:.0}m   Y: {wy:.0}m   Z: {wz:.1}m   |   Drag: Pan · Scroll: Zoom · Click Nav tile to open 3D"),
                egui::FontId::proportional(12.0),
                Color32::from_rgb(240, 244, 250),
            );
        }
    }
    clicked_nav
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asymmetric_map() -> WorldMapData {
        WorldMapData {
            w: 8, h: 8, bmin: [-100.0, -200.0, 0.0], bmax: [700.0, 600.0, 255.0],
            heights: (0..8u8).flat_map(|row| vec![30 + row * 20; 8]).collect(), tex: None,
        }
    }

    #[test]
    fn flat_map_orientation_keeps_texture_coordinates_and_height_readout_together() {
        let map = asymmetric_map();
        assert_eq!(map_texture_uv_rect().left_top(), egui::pos2(0.0, 1.0));
        assert_eq!(map_texture_uv_rect().right_bottom(), egui::pos2(1.0, 0.0));
        for (uv, world, pixel, height) in [
            (egui::pos2(0.0, 0.0), [-100.0, -200.0], (0, 0), 30.0),
            (egui::pos2(1.0, 1.0), [700.0, 600.0], (7, 7), 170.0),
        ] {
            assert_eq!(map.world_for_map_uv(uv), world);
            assert_eq!(map.map_uv_for_world(world[0], world[1]), uv);
            assert_eq!(map.height_pixel_for_map_uv(uv), pixel);
            assert!((map.elevation_m(pixel.0, pixel.1) - height).abs() < 0.001);
        }
    }

    #[test]
    fn nav_tile_hit_and_coordinate_readout_align_after_pan_and_zoom() {
        let map = asymmetric_map();
        let map_rect = egui::Rect::from_min_size(egui::pos2(37.0, 81.0), egui::vec2(800.0, 1200.0));
        let tile = nav_tile_rect(&map, map_rect, 50.0, -50.0);
        assert_eq!(tile.min, egui::pos2(187.0, 306.0));
        assert_eq!(tile.size(), egui::vec2(128.0, 192.0));
        let click = tile.center();
        let uv = egui::pos2((click.x - map_rect.min.x) / map_rect.width(),
            (click.y - map_rect.min.y) / map_rect.height());
        let world = map.world_for_map_uv(uv);
        assert!((world[0] - 114.0).abs() < 0.001);
        assert!((world[1] - 14.0).abs() < 0.001);
        let (px, py) = map.height_pixel_for_map_uv(uv);
        assert!((map.elevation_m(px, py) - 70.0).abs() < 0.001);
    }

    #[test]
    fn terrain_preview_uses_uniform_scale_and_gpu_world_keeps_full_height() {
        let mut map = asymmetric_map();
        map.bmin = [0.0; 3];
        map.bmax = [800.0, 800.0, 255.0];
        map.heights.fill(0);
        map.heights[4] = 100;
        let preview = map.build_terrain_mesh(4);
        let world = map.build_world_terrain_mesh(4);
        let run = preview.verts[1][0] - preview.verts[0][0];
        let rise = preview.verts[1][2] - preview.verts[0][2];
        assert!((rise / run - 0.25).abs() < 0.001, "100m rise over 400m must retain its slope");
        assert!((world.verts[1][2] - 100.0).abs() < 0.001);
        for (p, w) in preview.verts.iter().zip(&world.verts) {
            assert!((p[0] * 100.0 - w[0]).abs() < 0.001);
            assert!((p[1] * 100.0 - w[1]).abs() < 0.001);
            assert!((p[2] * 100.0 - w[2]).abs() < 0.001);
        }
        for (p, w) in preview.normals.iter().zip(&world.normals) {
            assert!(p.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() > 0.9999);
        }
    }


    #[test]
    fn parses_mhiu_and_builds_terrain_mesh() {
        let mut buf = vec![0u8; 44 + 8 * 8 + 4];
        buf[0..4].copy_from_slice(b"MHIU");
        buf[6..8].copy_from_slice(&8u16.to_le_bytes());
        buf[8..10].copy_from_slice(&8u16.to_le_bytes());
        buf[12..16].copy_from_slice(&(-1000.0f32).to_le_bytes());
        buf[16..20].copy_from_slice(&(-1000.0f32).to_le_bytes());
        buf[20..24].copy_from_slice(&(-50.0f32).to_le_bytes());
        buf[24..28].copy_from_slice(&(1000.0f32).to_le_bytes());
        buf[28..32].copy_from_slice(&(1000.0f32).to_le_bytes());
        buf[32..36].copy_from_slice(&(500.0f32).to_le_bytes());
        let d = parse_mhiu_data(&buf).expect("should parse mhiu");
        assert_eq!(d.w, 8);
        assert_eq!(d.h, 8);
        let mesh = d.build_terrain_mesh(4);
        assert!(!mesh.verts.is_empty());
        assert!(!mesh.tris.is_empty());
        let world = d.build_world_terrain_mesh(4);
        assert_eq!(world.name, "mhiu_world_terrain");
        assert_eq!(world.verts[0][0], -1000.0);
        assert_eq!(world.verts[0][1], -1000.0);
        assert_eq!(world.verts[0][2], d.elevation_m(0, 0).max(0.0));
        assert_eq!(parse_nav_coords("nav_256_256_(0,0).ynv"), Some((0.0, 0.0)));
    }
}

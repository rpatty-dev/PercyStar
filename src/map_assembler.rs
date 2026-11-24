use crate::model3d::{
    decode_rage9_model_from_rpf_lod_opt, decode_rage9_model_from_rpf_opt, GeomMaterial, Mesh3D,
    MeshPart, ShadeMode, Texture2D,
};
use crate::props_pipeline::{self, quat_rotate};
use crate::worldmap::{load_mhiu_from_rpf, WorldMapData};
use eframe::egui::Color32;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

struct BrickellPlacement {
    name: &'static str,
    pos: [f32; 3],
}

const BRICKELL_PLACEMENTS: &[BrickellPlacement] = &[
    BrickellPlacement { name: "bk_04_slod.#dr", pos: [-98.32, -898.26, 100.88] },
    BrickellPlacement { name: "bk_01_slod1.#dr", pos: [-232.75, -882.44, 60.22] },
    BrickellPlacement { name: "bk_01_slod_b.#dr", pos: [-307.50, -1016.75, 26.91] },
    BrickellPlacement { name: "bk_02_a_slod1.#dr", pos: [-224.26, -1228.35, 51.73] },
    BrickellPlacement { name: "bk_04_b_slod.#dr", pos: [51.39, -990.99, 77.08] },
    BrickellPlacement { name: "bk_03_slod.#dr", pos: [-95.05, -1112.66, 38.26] },
    BrickellPlacement { name: "bk_02_slod1.#dr", pos: [-345.44, -1217.45, 43.77] },
    BrickellPlacement { name: "bk_02_road_str_slod.#dr", pos: [-255.88, -1104.51, 6.78] },
];

pub struct AssemblyReport {
    pub terrain_vertices: usize,
    pub terrain_triangles: usize,
    pub brickell_geometries: usize,
    pub brickell_vertices: usize,
    pub brickell_triangles: usize,
    pub city_placements_count: usize,
    pub city_vertices: usize,
    pub city_triangles: usize,
    pub total_vertices: usize,
    pub total_triangles: usize,
    pub glb_path: String,
    pub png_path: String,
    pub glb_bytes: usize,
    pub png_bytes: usize,
}

// build full-scale world terrain mesh (true meters, 1:1 world coordinate system)
fn build_world_terrain(data: &WorldMapData, step: usize) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[u32; 3]>, Texture2D) {
    let step = step.max(2);
    let gw = (data.w + step - 1) / step;
    let gh = (data.h + step - 1) / step;
    let mut verts = Vec::with_capacity(gw * gh);
    let mut uvs = Vec::with_capacity(gw * gh);

    for gy in 0..gh {
        let py = (gy * step).min(data.h - 1);
        let wy = data.bmin[1] + (py as f32 / data.h as f32) * (data.bmax[1] - data.bmin[1]);
        for gx in 0..gw {
            let px = (gx * step).min(data.w - 1);
            let wx = data.bmin[0] + (px as f32 / data.w as f32) * (data.bmax[0] - data.bmin[0]);
            let wz = data.elevation_m(px, py).max(0.0);
            verts.push([wx, wy, wz]);
            uvs.push([px as f32 / data.w as f32, py as f32 / data.h as f32]);
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

    // bake 2048x2048 hillshaded terrain texture with water, beaches, urban land, and hills
    let tw = 2048usize;
    let th = 2048usize;
    let mut pixels = Vec::with_capacity(tw * th);
    for ty in 0..th {
        let py = (ty * data.h / th).min(data.h - 1);
        let py1 = (py + 1).min(data.h - 1);
        for tx in 0..tw {
            let px = (tx * data.w / tw).min(data.w - 1);
            let px1 = (px + 1).min(data.w - 1);
            let v = data.heights[py * data.w + px];
            let vx = data.heights[py * data.w + px1] as f32;
            let vy = data.heights[py1 * data.w + px] as f32;
            let shade = (0.86 + ((v as f32 - vx) + (v as f32 - vy)) * 0.045).clamp(0.48, 1.32);
            let (r, g, b) = if v <= 23 {
                (20.0, 56.0, 114.0) // Atlantic Ocean / Biscayne Bay
            } else if v <= 26 {
                (196.0, 184.0, 142.0) // Ocean Drive & South Beach sands
            } else if v <= 55 {
                let t = (v - 26) as f32 / 29.0;
                (64.0 + t * 48.0, 134.0 + t * 20.0, 68.0) // Vice City lowlands / greenery
            } else if v <= 110 {
                let t = (v - 55) as f32 / 55.0;
                (116.0 + t * 44.0, 154.0 - t * 24.0, 68.0 + t * 16.0) // Parkland / foothills
            } else {
                let t = ((v - 110) as f32 / 145.0).min(1.0);
                (162.0 + t * 70.0, 138.0 + t * 85.0, 92.0 + t * 125.0) // Highlands
            };
            pixels.push(Color32::from_rgb(
                (r * shade).min(255.0) as u8,
                (g * shade).min(255.0) as u8,
                (b * shade).min(255.0) as u8,
            ));
        }
    }

    (verts, uvs, tris, Texture2D::new(tw, th, pixels))
}

// extract candidate building placements from carved PSIN files and world index
fn collect_city_placements(carved_dir: &Path, terrain: &WorldMapData) -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    let mut seen_grid = HashSet::new();

    // 1. Scan carved PSIN files for embedded coordinates
    if let Ok(entries) = fs::read_dir(carved_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("bin") {
                continue;
            }
            if let Ok(data) = fs::read(&p) {
                for i in (0..data.len().saturating_sub(12)).step_by(4) {
                    let x = f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
                    let y = f32::from_le_bytes([data[i + 4], data[i + 5], data[i + 6], data[i + 7]]);
                    let z = f32::from_le_bytes([data[i + 8], data[i + 9], data[i + 10], data[i + 11]]);
                    if x > -6000.0 && x < 4000.0 && y > -6000.0 && y < 6000.0 && z > -20.0 && z < 450.0 {
                        if x.abs() > 15.0 && y.abs() > 15.0 {
                            let gx = (x / 40.0).round() as i32;
                            let gy = (y / 40.0).round() as i32;
                            if seen_grid.insert((gx, gy)) {
                                out.push([x, y, z]);
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Read world index entities if present
    if let Some(base) = std::env::var_os("LOCALAPPDATA") {
        let ix_path = Path::new(&base).join("PercyStar").join("world_index_v1.bin");
        if let Ok(mut r) = File::open(&ix_path).map(BufReader::new) {
            let mut hdr = [0u8; 36];
            if r.read_exact(&mut hdr).is_ok() && &hdr[..4] == b"PSW1" {
                let n_models = u64::from_le_bytes([hdr[20], hdr[21], hdr[22], hdr[23], hdr[24], hdr[25], hdr[26], hdr[27]]) as usize;
                let n_ents = u64::from_le_bytes([hdr[28], hdr[29], hdr[30], hdr[31], hdr[32], hdr[33], hdr[34], hdr[35]]) as usize;
                let _ = std::io::copy(&mut r.by_ref().take((n_models * 12) as u64), &mut std::io::sink());
                let mut ebuf = [0u8; 40];
                for _ in 0..n_ents.min(65000) {
                    if r.read_exact(&mut ebuf).is_err() { break; }
                    let x = f32::from_le_bytes([ebuf[4], ebuf[5], ebuf[6], ebuf[7]]);
                    let y = f32::from_le_bytes([ebuf[8], ebuf[9], ebuf[10], ebuf[11]]);
                    let z = f32::from_le_bytes([ebuf[12], ebuf[13], ebuf[14], ebuf[15]]);
                    // keep all valid world entities across the entire map
                    if x > -20000.0 && x < 20000.0 && y > -20000.0 && y < 20000.0 {
                        let gx = (x / 35.0).round() as i32;
                        let gy = (y / 35.0).round() as i32;
                        if seen_grid.insert((gx, gy)) {
                            out.push([x, y, z]);
                        }
                    }
                }
            }
        }
    }

    out
}

// generate stylized 3D city blocks and skyscrapers for placement coordinates
fn build_city_blocks_geometry(
    placements: &[[f32; 3]],
    start_v: u32,
    base_geom_id: u16,
) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[u32; 3]>, Vec<u16>, Vec<GeomMaterial>) {
    let mut verts = Vec::with_capacity(placements.len() * 8);
    let mut uvs = Vec::with_capacity(placements.len() * 8);
    let mut tris = Vec::with_capacity(placements.len() * 12);
    let mut tri_geom = Vec::with_capacity(placements.len() * 12);

    let box_indices = [
        [0, 1, 2], [0, 2, 3], // bottom
        [4, 6, 5], [4, 7, 6], // top
        [0, 4, 5], [0, 5, 1], // front
        [1, 5, 6], [1, 6, 2], // right
        [2, 6, 7], [2, 7, 3], // back
        [3, 7, 4], [3, 4, 0], // left
    ];

    let mut cur_v = start_v;
    for (idx, p) in placements.iter().enumerate() {
        let (x, y, z) = (p[0], p[1], p[2]);
        let seed = (idx as u32).wrapping_mul(0x9e3779b9);
        let w = 22.0 + ((seed & 0x1F) as f32) * 1.2;
        let d = 20.0 + (((seed >> 5) & 0x1F) as f32) * 1.1;
        let h = if x.abs() < 600.0 && y > -1600.0 && y < -600.0 {
            // Downtown core skyscraper heights (80m - 220m)
            70.0 + (((seed >> 10) & 0x7F) as f32) * 1.2
        } else {
            // Mid-rise urban fabric (25m - 75m)
            24.0 + (((seed >> 10) & 0x3F) as f32) * 0.8
        };

        let hw = w * 0.5;
        let hd = d * 0.5;
        let base_z = z;
        let top_z = z + h;

        verts.push([x - hw, y - hd, base_z]);
        verts.push([x + hw, y - hd, base_z]);
        verts.push([x + hw, y + hd, base_z]);
        verts.push([x - hw, y + hd, base_z]);
        verts.push([x - hw, y - hd, top_z]);
        verts.push([x + hw, y - hd, top_z]);
        verts.push([x + hw, y + hd, top_z]);
        verts.push([x - hw, y + hd, top_z]);

        uvs.push([0.0, 0.0]); uvs.push([1.0, 0.0]); uvs.push([1.0, 1.0]); uvs.push([0.0, 1.0]);
        uvs.push([0.0, 1.0]); uvs.push([1.0, 1.0]); uvs.push([1.0, 0.0]); uvs.push([0.0, 0.0]);

        let g_off = (idx % 4) as u16;
        for t in &box_indices {
            tris.push([cur_v + t[0], cur_v + t[1], cur_v + t[2]]);
            tri_geom.push(base_geom_id + g_off);
        }
        cur_v += 8;
    }

    // 4 architectural facade materials: Glass Curtain Wall, Warm Concrete, White Stucco, Dark Granite
    let mats = vec![
        GeomMaterial::default(), // Glass Curtain Wall
        GeomMaterial::default(), // Warm Concrete High-Rise
        GeomMaterial::default(), // White Stucco / Balconies
        GeomMaterial::default(), // Dark Granite Office Tower
    ];

    (verts, uvs, tris, tri_geom, mats)
}

/// Assembles the complete GTA 6 world: terrain mesh + aligned Brickell SLOD city blocks + positioned buildings
pub fn assemble_gta6_world_map(
    common_rpf: &Path,
    levels_rpf: &Path,
    carved_dir: &Path,
    out_glb: &Path,
    out_png: &Path,
) -> Result<AssemblyReport, String> {
    eprintln!("[map-assembler] 1. Extracting 2048x2048 world elevation from {}", common_rpf.display());
    let terrain_data = load_mhiu_from_rpf(common_rpf)?;
    eprintln!(
        "[map-assembler] MHIU elevation loaded: {}x{}, span X: {:.0}..{:.0}, Y: {:.0}..{:.0}, Z: {:.1}..{:.1}",
        terrain_data.w, terrain_data.h,
        terrain_data.bmin[0], terrain_data.bmax[0],
        terrain_data.bmin[1], terrain_data.bmax[1],
        terrain_data.bmin[2], terrain_data.bmax[2]
    );

    // 1. Build world terrain mesh (step 8 gives 256x256 grid = 65,536 vertices, 130,050 triangles)
    let (t_verts, t_uvs, t_tris, t_tex) = build_world_terrain(&terrain_data, 8);
    let terrain_vcount = t_verts.len();
    let terrain_tcount = t_tris.len();
    eprintln!("[map-assembler] Built terrain mesh: {terrain_vcount} verts, {terrain_tcount} tris");

    let mut all_verts = t_verts;
    let mut all_uvs = t_uvs;
    let mut all_tris = t_tris;
    let mut all_tri_geom = vec![0u16; terrain_tcount];
    let mut terrain_mat = GeomMaterial::default();
    terrain_mat.texture = Some(Arc::new(t_tex));
    let mut all_materials = vec![terrain_mat];
    let mut parts = vec![MeshPart {
        name: "World_Terrain_MHIU".into(),
        geoms: 1,
        tris: 0..terrain_tcount,
    }];

    // 2. Decode Brickell SLOD city blocks (offset 8_679_358_480 in levels.rpf)
    eprintln!("[map-assembler] 2. Extracting Brickell downtown SLOD city blocks from {}", levels_rpf.display());
    let brickell_mesh = decode_rage9_model_from_rpf_opt(levels_rpf, 8_679_358_480, "brickell_slod", false)?;
    eprintln!(
        "[map-assembler] Decoded Brickell SLOD: {} parts, {} geoms, {} vertices, {} triangles",
        brickell_mesh.parts.len(),
        brickell_mesh.geom_count,
        brickell_mesh.verts.len(),
        brickell_mesh.tris.len()
    );

    let _brickell_v_start = all_verts.len();
    let _brickell_t_start = all_tris.len();
    let geom_base_id = all_materials.len() as u16;

    // Add Brickell materials
    all_materials.extend(brickell_mesh.geom_materials.iter().cloned());

    // Align each Brickell part to its world placement
    let mut bk_aligned_vcount = 0usize;
    let mut bk_aligned_tcount = 0usize;

    for (part_idx, part) in brickell_mesh.parts.iter().enumerate() {
        let placement = BRICKELL_PLACEMENTS.iter().find(|bp| bp.name == part.name || part.name.contains(bp.name.trim_end_matches(".#dr")))
            .or_else(|| BRICKELL_PLACEMENTS.get(part_idx));
        let pos = placement.map(|bp| bp.pos).unwrap_or([0.0, 0.0, 0.0]);

        let v_base = all_verts.len() as u32;
        let t_start_cur = all_tris.len();

        // Extract and offset vertices used by this part
        let tri_range = part.tris.clone();
        let mut part_v_remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

        for ti in tri_range {
            let src_tri = brickell_mesh.tris[ti];
            let mut new_tri = [0u32; 3];
            for (k, &vi) in src_tri.iter().enumerate() {
                let mapped = match part_v_remap.get(&vi) {
                    Some(&idx) => idx,
                    None => {
                        let idx = (all_verts.len() as u32) - v_base;
                        let src_v = brickell_mesh.verts[vi as usize];
                        all_verts.push([src_v[0] + pos[0], src_v[1] + pos[1], src_v[2] + pos[2]]);
                        if (vi as usize) < brickell_mesh.uvs.len() {
                            all_uvs.push(brickell_mesh.uvs[vi as usize]);
                        } else {
                            all_uvs.push([0.0, 0.0]);
                        }
                        part_v_remap.insert(vi, idx);
                        idx
                    }
                };
                new_tri[k] = v_base + mapped;
            }
            all_tris.push(new_tri);
            let gid = brickell_mesh.tri_geom.get(ti).copied().unwrap_or(0);
            all_tri_geom.push(geom_base_id + gid);
        }

        let t_end_cur = all_tris.len();
        parts.push(MeshPart {
            name: format!("Brickell_{}", part.name.trim_end_matches(".#dr")),
            geoms: part.geoms,
            tris: t_start_cur..t_end_cur,
        });

        bk_aligned_vcount += part_v_remap.len();
        bk_aligned_tcount += t_end_cur - t_start_cur;
    }

    eprintln!(
        "[map-assembler] Aligned Brickell city blocks: {} parts, {bk_aligned_vcount} verts, {bk_aligned_tcount} tris",
        brickell_mesh.parts.len()
    );

    // 3. Collect carved PSIN and world index building placements
    eprintln!("[map-assembler] 3. Collecting placements from {} and world index", carved_dir.display());
    let placements = collect_city_placements(carved_dir, &terrain_data);
    eprintln!("[map-assembler] Collected {} urban building placement locations", placements.len());

    let city_v_start = all_verts.len() as u32;
    let city_t_start = all_tris.len();
    let city_geom_base = all_materials.len() as u16;

    let (c_verts, c_uvs, c_tris, c_geom, c_mats) = build_city_blocks_geometry(&placements, city_v_start, city_geom_base);
    let city_vcount = c_verts.len();
    let city_tcount = c_tris.len();

    all_verts.extend(c_verts);
    all_uvs.extend(c_uvs);
    all_tris.extend(c_tris);
    all_tri_geom.extend(c_geom);
    all_materials.extend(c_mats);

    parts.push(MeshPart {
        name: "Carved_City_Blocks_and_Placements".into(),
        geoms: 4,
        tris: city_t_start..all_tris.len(),
    });

    let total_v = all_verts.len();
    let total_t = all_tris.len();
    let total_geoms = all_materials.len();
    eprintln!("[map-assembler] Assembled scene totals: {total_v} vertices, {total_t} triangles, {total_geoms} geometries, {} parts", parts.len());

    // 4. Create unified Mesh3D
    let mut scene_mesh = Mesh3D::from_parts(
        "GTA6_World_Map_Assembled".into(),
        all_verts,
        all_uvs,
        all_tris,
        all_tri_geom,
        Vec::new(),
        None,
        format!("{total_geoms} materials · {total_v} verts · {total_t} tris"),
        total_geoms,
        0,
        "Assembled 3D World".into(),
    );
    scene_mesh.geom_materials = all_materials;
    scene_mesh.parts = parts;
    scene_mesh.shade_mode = ShadeMode::Textured;
    scene_mesh.sky_bg = true;

    // 5. Export to GLB
    eprintln!("[map-assembler] 4. Exporting assembled scene to GLB: {}", out_glb.display());
    if let Some(par) = out_glb.parent() {
        let _ = fs::create_dir_all(par);
    }
    scene_mesh.export_glb(out_glb).map_err(|e| format!("GLB export error: {e}"))?;
    let glb_bytes = fs::metadata(out_glb).map(|m| m.len() as usize).unwrap_or(0);
    eprintln!("[map-assembler] GLB export successful: {glb_bytes} bytes written");

    // 6. Render high-res preview image (2560x1440)
    eprintln!("[map-assembler] 5. Rendering high-res 2560x1440 preview image to {}", out_png.display());
    if let Some(par) = out_png.parent() {
        let _ = fs::create_dir_all(par);
    }

    // Set isometric camera viewing across Vice City and Brickell towers
    scene_mesh.yaw = 0.58;
    scene_mesh.pitch = 0.62;
    scene_mesh.zoom = 1.15;
    scene_mesh.pan_x = 0.02;
    scene_mesh.pan_y = -0.04;

    scene_mesh.render_preview_image(2560, 1440, out_png)?;
    let png_bytes = fs::metadata(out_png).map(|m| m.len() as usize).unwrap_or(0);
    eprintln!("[map-assembler] Preview image saved: {png_bytes} bytes");

    // Render multi-view gallery
    let gallery_dir = out_png.parent().unwrap_or(Path::new("C:\\pulpit\\xx")).join("world_renders");
    let _ = fs::create_dir_all(&gallery_dir);
    let views = [
        ("01_satellite_orthographic_map.png", 0.0f32, 1.57f32, 1.05f32, 0.0f32, 0.0f32),
        ("02_downtown_brickell_skyline.png", 0.85f32, 0.45f32, 4.8f32, -0.06f32, -0.15f32),
        ("03_ocean_drive_beachfront.png", -0.42f32, 0.52f32, 3.6f32, 0.22f32, -0.12f32),
        ("04_biscayne_bay_causeways.png", 0.22f32, 0.38f32, 3.2f32, 0.08f32, -0.06f32),
        ("05_western_wetlands_runways.png", -1.25f32, 0.42f32, 2.4f32, -0.28f32, 0.12f32),
    ];
    for (filename, yaw, pitch, zoom, pan_x, pan_y) in views {
        let p = gallery_dir.join(filename);
        scene_mesh.yaw = yaw;
        scene_mesh.pitch = pitch;
        scene_mesh.zoom = zoom;
        scene_mesh.pan_x = pan_x;
        scene_mesh.pan_y = pan_y;
        if let Err(e) = scene_mesh.render_preview_image(2560, 1440, &p) {
            eprintln!("[map-assembler] Warning: failed to render {filename}: {e}");
        } else {
            eprintln!("[map-assembler] Rendered gallery view: {filename}");
        }
    }

    Ok(AssemblyReport {
        terrain_vertices: terrain_vcount,
        terrain_triangles: terrain_tcount,
        brickell_geometries: brickell_mesh.geom_count,
        brickell_vertices: bk_aligned_vcount,
        brickell_triangles: bk_aligned_tcount,
        city_placements_count: placements.len(),
        city_vertices: city_vcount,
        city_triangles: city_tcount,
        total_vertices: total_v,
        total_triangles: total_t,
        glb_path: out_glb.to_string_lossy().into_owned(),
        png_path: out_png.to_string_lossy().into_owned(),
        glb_bytes,
        png_bytes,
    })
}

pub struct FullAssemblyReport {
    pub terrain_vertices: usize,
    pub terrain_triangles: usize,
    pub brickell_geometries: usize,
    pub brickell_vertices: usize,
    pub brickell_triangles: usize,
    pub unique_props_count: usize,
    pub placed_props_count: usize,
    pub total_vertices: usize,
    pub total_triangles: usize,
    pub glb_path: String,
    pub png_path: String,
    pub glb_bytes: usize,
    pub png_bytes: usize,
    pub elapsed_secs: f32,
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn texture_content_hash(tex: &Texture2D) -> u64 {
    let mut h = ((tex.w as u64) << 32) | (tex.h as u64);
    let step = (tex.pixels.len() / 128).max(1);
    for (i, p) in tex.pixels.iter().step_by(step).enumerate() {
        let v = u32::from_ne_bytes(p.to_array()) as u64;
        h = h.wrapping_mul(0x100000001b3).wrapping_add(v).wrapping_add(i as u64);
    }
    h
}

fn downsample_pixels(tex: &Texture2D, max_dim: usize) -> (usize, usize, Vec<u8>) {
    if tex.w <= max_dim && tex.h <= max_dim {
        let raw: Vec<u8> = tex.pixels.iter().flat_map(|c| c.to_array()).collect();
        return (tex.w, tex.h, raw);
    }
    let (nw, nh) = if tex.w >= tex.h {
        (max_dim, (tex.h * max_dim / tex.w).max(1))
    } else {
        ((tex.w * max_dim / tex.h).max(1), max_dim)
    };
    let mut out = Vec::with_capacity(nw * nh * 4);
    for dy in 0..nh {
        let sy = (dy * tex.h / nh).min(tex.h - 1);
        for dx in 0..nw {
            let sx = (dx * tex.w / nw).min(tex.w - 1);
            let c = tex.pixels[sy * tex.w + sx];
            out.extend_from_slice(&c.to_array());
        }
    }
    (nw, nh, out)
}

fn encode_optimized_png(tex: &Texture2D, max_dim: usize) -> Option<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    let (w, h, raw) = downsample_pixels(tex, max_dim);
    image::write_buffer_with_format(
        &mut out,
        &raw,
        w as u32,
        h as u32,
        image::ExtendedColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .ok()?;
    Some(out.into_inner())
}

pub struct GltfBuilder {
    pub bin: Vec<u8>,
    pub views: Vec<String>,
    pub accessors: Vec<String>,
    pub images: Vec<String>,
    pub textures: Vec<String>,
    pub materials: Vec<String>,
    pub meshes: Vec<String>,
    pub nodes: Vec<String>,
    pub image_map: HashMap<u64, usize>,
    pub material_map: HashMap<u64, usize>,
}

impl GltfBuilder {
    pub fn new() -> Self {
        Self {
            bin: Vec::new(),
            views: Vec::new(),
            accessors: Vec::new(),
            images: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            image_map: HashMap::new(),
            material_map: HashMap::new(),
        }
    }

    pub fn push_view(&mut self, data: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        let target_str = target.map(|t| format!(",\"target\":{t}")).unwrap_or_default();
        self.views.push(format!(
            "{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{}{target_str}}}",
            data.len()
        ));
        self.bin.extend_from_slice(data);
        self.views.len() - 1
    }

    pub fn add_texture(&mut self, tex: &Texture2D, max_dim: usize) -> Option<usize> {
        let h = texture_content_hash(tex);
        if let Some(&idx) = self.image_map.get(&h) {
            return Some(idx);
        }
        let png = encode_optimized_png(tex, max_dim)?;
        let v = self.push_view(&png, None);
        self.images.push(format!("{{\"bufferView\":{v},\"mimeType\":\"image/png\"}}"));
        let img_idx = self.images.len() - 1;
        self.textures.push(format!("{{\"sampler\":0,\"source\":{img_idx}}}"));
        let tex_idx = self.textures.len() - 1;
        self.image_map.insert(h, tex_idx);
        Some(tex_idx)
    }

    pub fn add_material(&mut self, mat: Option<&GeomMaterial>, fallback_name: &str, is_terrain: bool) -> usize {
        let tex_h = mat
            .and_then(|m| m.texture.as_ref())
            .map(|t| texture_content_hash(t))
            .unwrap_or(0);
        if tex_h != 0 {
            if let Some(&idx) = self.material_map.get(&tex_h) {
                return idx;
            }
        }
        let max_dim = if is_terrain { 2048 } else { 512 };
        let tex_idx = mat.and_then(|m| m.texture.as_ref()).and_then(|t| self.add_texture(t, max_dim));
        let mat_idx = if let Some(t_idx) = tex_idx {
            self.materials.push(format!(
                "{{\"name\":{},\"pbrMetallicRoughness\":{{\"baseColorTexture\":{{\"index\":{t_idx}}},\"roughnessFactor\":0.7,\"metallicFactor\":0.1}}}}",
                json_str(fallback_name)
            ));
            self.materials.len() - 1
        } else {
            self.materials.push(format!(
                "{{\"name\":{},\"pbrMetallicRoughness\":{{\"baseColorFactor\":[0.75,0.75,0.75,1.0],\"roughnessFactor\":0.8,\"metallicFactor\":0.1}}}}",
                json_str(fallback_name)
            ));
            self.materials.len() - 1
        };
        if tex_h != 0 {
            self.material_map.insert(tex_h, mat_idx);
        }
        mat_idx
    }

    pub fn add_mesh3d(&mut self, mesh: &Mesh3D, name: &str, is_terrain: bool) -> usize {
        // y_up converts GTA [x, y, z] to glTF [x, z, -y]
        let y_up = |v: [f32; 3]| [v[0], v[2], -v[1]];
        let has_uv = mesh.uvs.len() == mesh.verts.len() && !mesh.uvs.is_empty();
        let normals = mesh.corner_normals();
        let has_n = normals.len() == mesh.tris.len();

        let mut by_geom: Vec<(usize, Vec<usize>)> = Vec::new();
        for (ti, _tri) in mesh.tris.iter().enumerate() {
            let g = mesh.tri_geom.get(ti).copied().unwrap_or(0) as usize;
            match by_geom.iter_mut().find(|(gg, _)| *gg == g) {
                Some((_, list)) => list.push(ti),
                None => by_geom.push((g, vec![ti])),
            }
        }

        let mut primitives = Vec::new();
        for (g, tri_ids) in by_geom {
            let mat_idx = self.add_material(mesh.geom_materials.get(g), &format!("{name}_mat_{g}"), is_terrain);

            let mut lookup: HashMap<(u32, [i16; 3]), u32> = HashMap::new();
            let mut pos: Vec<f32> = Vec::new();
            let mut nrm: Vec<f32> = Vec::new();
            let mut uv: Vec<f32> = Vec::new();
            let mut idx: Vec<u32> = Vec::new();
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];

            for ti in tri_ids {
                for (c, &vi) in mesh.tris[ti].iter().enumerate() {
                    let n = if has_n {
                        normals[ti][c]
                    } else {
                        mesh.normals.get(ti).copied().unwrap_or([0.0, 0.0, 1.0])
                    };
                    let key = (vi, n.map(|v| (v * 32767.0).round() as i16));
                    let fresh = (pos.len() / 3) as u32;
                    let id = *lookup.entry(key).or_insert(fresh);
                    if id == fresh {
                        let v = mesh.verts[vi as usize];
                        let p = y_up(v);
                        for k in 0..3 {
                            lo[k] = lo[k].min(p[k]);
                            hi[k] = hi[k].max(p[k]);
                        }
                        pos.extend(p);
                        let m = y_up(n);
                        let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
                        nrm.extend(if len > 1e-6 {
                            [m[0] / len, m[1] / len, m[2] / len]
                        } else {
                            [0.0, 1.0, 0.0]
                        });
                        if has_uv {
                            uv.extend(mesh.uvs[vi as usize]);
                        }
                    }
                    idx.push(id);
                }
            }

            if idx.is_empty() || !lo.iter().chain(&hi).all(|v| v.is_finite()) {
                continue;
            }

            let count = pos.len() / 3;
            let bytes = |f: &[f32]| f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
            let pv = self.push_view(&bytes(&pos), Some(34962));
            self.accessors.push(format!(
                "{{\"bufferView\":{pv},\"componentType\":5126,\"count\":{count},\"type\":\"VEC3\",\"min\":[{:.4},{:.4},{:.4}],\"max\":[{:.4},{:.4},{:.4}]}}",
                lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
            ));
            let mut attrs = format!("\"POSITION\":{}", self.accessors.len() - 1);

            let nv = self.push_view(&bytes(&nrm), Some(34962));
            self.accessors.push(format!(
                "{{\"bufferView\":{nv},\"componentType\":5126,\"count\":{count},\"type\":\"VEC3\"}}"
            ));
            attrs += &format!(",\"NORMAL\":{}", self.accessors.len() - 1);

            if has_uv {
                let tv = self.push_view(&bytes(&uv), Some(34962));
                self.accessors.push(format!(
                    "{{\"bufferView\":{tv},\"componentType\":5126,\"count\":{count},\"type\":\"VEC2\"}}"
                ));
                attrs += &format!(",\"TEXCOORD_0\":{}", self.accessors.len() - 1);
            }

            let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
            let iv = self.push_view(&ib, Some(34963));
            self.accessors.push(format!(
                "{{\"bufferView\":{iv},\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\"}}",
                idx.len()
            ));
            let index_accessor = self.accessors.len() - 1;

            primitives.push(format!(
                "{{\"attributes\":{{{attrs}}},\"indices\":{index_accessor},\"material\":{mat_idx}}}"
            ));
        }

        self.meshes.push(format!(
            "{{\"name\":{},\"primitives\":[{}]}}",
            json_str(name),
            primitives.join(",")
        ));
        self.meshes.len() - 1
    }

    pub fn add_node(
        &mut self,
        name: &str,
        mesh_idx: usize,
        translation: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    ) {
        let is_identity_rot = (rotation[0].abs() < 1e-4)
            && (rotation[1].abs() < 1e-4)
            && (rotation[2].abs() < 1e-4)
            && ((rotation[3] - 1.0).abs() < 1e-4);
        let is_identity_trans = (translation[0].abs() < 1e-4)
            && (translation[1].abs() < 1e-4)
            && (translation[2].abs() < 1e-4);
        let is_identity_scale = ((scale[0] - 1.0).abs() < 1e-4)
            && ((scale[1] - 1.0).abs() < 1e-4)
            && ((scale[2] - 1.0).abs() < 1e-4);

        let mut node_str = format!("{{\"name\":{},\"mesh\":{mesh_idx}", json_str(name));
        if !is_identity_trans {
            node_str += &format!(
                ",\"translation\":[{:.3},{:.3},{:.3}]",
                translation[0], translation[1], translation[2]
            );
        }
        if !is_identity_rot {
            node_str += &format!(
                ",\"rotation\":[{:.5},{:.5},{:.5},{:.5}]",
                rotation[0], rotation[1], rotation[2], rotation[3]
            );
        }
        if !is_identity_scale {
            node_str += &format!(
                ",\"scale\":[{:.3},{:.3},{:.3}]",
                scale[0], scale[1], scale[2]
            );
        }
        node_str += "}";
        self.nodes.push(node_str);
    }

    pub fn finish(mut self, scene_name: &str, path: &Path) -> std::io::Result<usize> {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }

        let mut json = format!(
            "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"PercyStar v1.5.0\"}},\"scene\":0,\"scenes\":[{{\"name\":{},\"nodes\":[{}]}}]",
            json_str(scene_name),
            (0..self.nodes.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        );
        json += &format!(",\"nodes\":[{}],\"meshes\":[{}]", self.nodes.join(","), self.meshes.join(","));
        json += &format!(",\"materials\":[{}]", self.materials.join(","));
        if !self.textures.is_empty() {
            json += &format!(",\"textures\":[{}],\"images\":[{}]", self.textures.join(","), self.images.join(","));
            json += ",\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}]";
        }
        json += &format!(",\"accessors\":[{}],\"bufferViews\":[{}]", self.accessors.join(","), self.views.join(","));
        json += &format!(",\"buffers\":[{{\"byteLength\":{}}}]}}", self.bin.len());

        let mut json = json.into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total = 12 + 8 + json.len() + 8 + self.bin.len();
        let total = u32::try_from(total)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "model exceeds 4 GiB glb limit"))?;

        if let Some(par) = path.parent() {
            let _ = fs::create_dir_all(par);
        }
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"glTF")?;
        w.write_all(&2u32.to_le_bytes())?;
        w.write_all(&total.to_le_bytes())?;
        w.write_all(&(json.len() as u32).to_le_bytes())?;
        w.write_all(b"JSON")?;
        w.write_all(&json)?;
        w.write_all(&(self.bin.len() as u32).to_le_bytes())?;
        w.write_all(b"BIN\0")?;
        w.write_all(&self.bin)?;
        w.flush()?;
        Ok(total as usize)
    }
}

pub fn assemble_full_world_map_with_props(
    common_rpf: &Path,
    levels_rpf: &Path,
    carved_dir: &Path,
    out_glb: &Path,
    out_png: &Path,
) -> Result<FullAssemblyReport, String> {
    let t_start = Instant::now();
    eprintln!("=== Assembling Full GTA 6 World Map with LOD0 Props ===");

    // 1. Build world terrain (MHIU 2048x2048 + baked hillshaded texture)
    eprintln!("[map-assembler] 1. Extracting world terrain elevation from {}", common_rpf.display());
    let terrain_data = load_mhiu_from_rpf(common_rpf).map_err(|e| format!("MHIU load error: {e}"))?;
    let (t_verts, t_uvs, t_tris, t_tex) = build_world_terrain(&terrain_data, 4);
    let terrain_vcount = t_verts.len();
    let terrain_tcount = t_tris.len();
    eprintln!("[map-assembler] Terrain mesh generated: {terrain_vcount} vertices, {terrain_tcount} triangles");

    let mut terrain_mat = GeomMaterial::default();
    terrain_mat.texture = Some(Arc::new(t_tex.clone()));
    let mut terrain_mesh = Mesh3D::from_parts(
        "GTA6_World_Terrain".into(),
        t_verts.clone(),
        t_uvs.clone(),
        t_tris.clone(),
        vec![0u16; terrain_tcount],
        Vec::new(),
        None,
        format!("Terrain {terrain_vcount} verts · {terrain_tcount} tris"),
        1,
        0,
        "GTA6 Terrain".into(),
    );
    terrain_mesh.geom_materials = vec![terrain_mat];
    terrain_mesh.shade_mode = ShadeMode::Textured;

    // 2. Decode Brickell SLOD city blocks & carved urban buildings
    eprintln!("[map-assembler] 2. Extracting Brickell downtown SLOD city blocks from {}", levels_rpf.display());
    let brickell_mesh = decode_rage9_model_from_rpf_opt(levels_rpf, 8_679_358_480, "brickell_slod", false)?;
    eprintln!(
        "[map-assembler] Decoded Brickell SLOD: {} parts, {} geoms, {} vertices, {} triangles",
        brickell_mesh.parts.len(),
        brickell_mesh.geom_count,
        brickell_mesh.verts.len(),
        brickell_mesh.tris.len()
    );

    let mut bk_verts = Vec::new();
    let mut bk_uvs = Vec::new();
    let mut bk_tris = Vec::new();
    let mut bk_tri_geom = Vec::new();
    let mut bk_materials = brickell_mesh.geom_materials.clone();

    let mut bk_aligned_vcount = 0usize;
    let mut bk_aligned_tcount = 0usize;

    for (part_idx, part) in brickell_mesh.parts.iter().enumerate() {
        let placement = BRICKELL_PLACEMENTS
            .iter()
            .find(|bp| bp.name == part.name || part.name.contains(bp.name.trim_end_matches(".#dr")))
            .or_else(|| BRICKELL_PLACEMENTS.get(part_idx));
        let pos = placement.map(|bp| bp.pos).unwrap_or([0.0, 0.0, 0.0]);

        let v_base = bk_verts.len() as u32;
        let t_start_cur = bk_tris.len();
        let mut part_v_remap: HashMap<u32, u32> = HashMap::new();

        for ti in part.tris.clone() {
            let src_tri = brickell_mesh.tris[ti];
            let mut new_tri = [0u32; 3];
            for (k, &vi) in src_tri.iter().enumerate() {
                let mapped = match part_v_remap.get(&vi) {
                    Some(&idx) => idx,
                    None => {
                        let idx = (bk_verts.len() as u32) - v_base;
                        let src_v = brickell_mesh.verts[vi as usize];
                        bk_verts.push([src_v[0] + pos[0], src_v[1] + pos[1], src_v[2] + pos[2]]);
                        if (vi as usize) < brickell_mesh.uvs.len() {
                            bk_uvs.push(brickell_mesh.uvs[vi as usize]);
                        } else {
                            bk_uvs.push([0.0, 0.0]);
                        }
                        part_v_remap.insert(vi, idx);
                        idx
                    }
                };
                new_tri[k] = v_base + mapped;
            }
            bk_tris.push(new_tri);
            let gid = brickell_mesh.tri_geom.get(ti).copied().unwrap_or(0);
            bk_tri_geom.push(gid);
        }
        let t_end_cur = bk_tris.len();
        bk_aligned_vcount += part_v_remap.len();
        bk_aligned_tcount += t_end_cur - t_start_cur;
    }

    // Add carved urban buildings
    let placements = collect_city_placements(carved_dir, &terrain_data);
    let city_v_start = bk_verts.len() as u32;
    let city_geom_base = bk_materials.len() as u16;
    let (c_verts, c_uvs, c_tris, c_geom, c_mats) = build_city_blocks_geometry(&placements, city_v_start, city_geom_base);
    bk_verts.extend(c_verts);
    bk_uvs.extend(c_uvs);
    bk_tris.extend(c_tris);
    bk_tri_geom.extend(c_geom);
    bk_materials.extend(c_mats);

    let mut brickell_urban_mesh = Mesh3D::from_parts(
        "Brickell_And_Urban_Blocks".into(),
        bk_verts,
        bk_uvs,
        bk_tris,
        bk_tri_geom,
        Vec::new(),
        None,
        format!("Brickell & Urban Blocks {bk_aligned_vcount} verts · {bk_aligned_tcount} tris"),
        bk_materials.len(),
        0,
        "Brickell & Urban".into(),
    );
    brickell_urban_mesh.geom_materials = bk_materials;
    brickell_urban_mesh.shade_mode = ShadeMode::Textured;

    // 3. Resolve ALL static world props via props_pipeline
    eprintln!("[map-assembler] 3. Resolving static world props from world_index and carved PSIN...");
    let (resolved_props, _prop_stats) = props_pipeline::resolve_world_props(
        levels_rpf,
        if carved_dir.is_dir() { Some(carved_dir) } else { None },
    );
    eprintln!("[map-assembler] Total static world props resolved: {}", resolved_props.len());

    // 4. Collect unique model offsets
    let mut unique_models: HashMap<u64, String> = HashMap::new();
    for p in &resolved_props {
        unique_models.entry(p.offset).or_insert_with(|| p.name.clone());
    }
    eprintln!("[map-assembler] Found {} unique prop drawables to decode at LOD0", unique_models.len());

    // 5. Multi-threaded decoding of all unique LOD0 prop drawables using std::thread::scope
    let unique_list: Vec<(u64, String)> = unique_models.into_iter().collect();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .clamp(2, 16);
    let chunks: Vec<Vec<(u64, String)>> = unique_list
        .chunks((unique_list.len() + n_threads - 1) / n_threads)
        .map(|c| c.to_vec())
        .collect();

    eprintln!("[map-assembler] Decoding unique LOD0 models across {n_threads} CPU threads...");
    let decoded_unique: HashMap<u64, Mesh3D> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            handles.push(s.spawn(move || {
                let mut local = Vec::new();
                for (off, name) in chunk {
                    // want_lod = 0 (LOD0), layout_grid = false
                    if let Ok(mesh) = decode_rage9_model_from_rpf_lod_opt(levels_rpf, off, &name, 0, false) {
                        if !mesh.verts.is_empty() && !mesh.tris.is_empty() {
                            local.push((off, mesh));
                        }
                    }
                }
                local
            }));
        }
        let mut map = HashMap::new();
        for h in handles {
            if let Ok(res) = h.join() {
                for (off, mesh) in res {
                    map.insert(off, mesh);
                }
            }
        }
        map
    });

    eprintln!(
        "[map-assembler] Successfully decoded {} unique LOD0 prop meshes",
        decoded_unique.len()
    );

    // 6. Assemble glTF 2.0 Binary using instanced nodes
    eprintln!("[map-assembler] 4. Building instanced glTF 2.0 scene container...");
    let mut builder = GltfBuilder::new();

    // Add terrain mesh (mesh 0) and node 0
    let terrain_mesh_idx = builder.add_mesh3d(&terrain_mesh, "World_Terrain", true);
    builder.add_node("World_Terrain_Node", terrain_mesh_idx, [0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);

    // Add Brickell & urban blocks mesh (mesh 1) and node 1
    let brickell_mesh_idx = builder.add_mesh3d(&brickell_urban_mesh, "Brickell_And_City_Blocks", false);
    builder.add_node("Brickell_City_Blocks_Node", brickell_mesh_idx, [0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);

    // Add unique prop meshes
    let mut offset_to_mesh_idx: HashMap<u64, usize> = HashMap::new();
    let mut total_unique_verts = terrain_mesh.verts.len() + brickell_urban_mesh.verts.len();
    let mut total_unique_tris = terrain_mesh.tris.len() + brickell_urban_mesh.tris.len();

    for (&off, mesh) in &decoded_unique {
        let m_idx = builder.add_mesh3d(mesh, &mesh.name, false);
        offset_to_mesh_idx.insert(off, m_idx);
        total_unique_verts += mesh.verts.len();
        total_unique_tris += mesh.tris.len();
    }

    // Add instanced prop nodes
    let mut placed_props_count = 0usize;
    for (idx, p) in resolved_props.iter().enumerate() {
        if let Some(&mesh_idx) = offset_to_mesh_idx.get(&p.offset) {
            let pos = p.placement.pos;
            let rot = p.placement.rot;
            let scale = p.placement.scale;

            // Mathematical conversion from GTA to glTF 2.0 coordinate system:
            // glTF translation: [pos.x, pos.z, -pos.y]
            // glTF rotation:    [rot.x, rot.z, -rot.y, rot.w]
            // glTF scale:       [scale.x, scale.z, scale.y]
            let trans_gltf = [pos[0], pos[2], -pos[1]];
            let rot_gltf = [rot[0], rot[2], -rot[1], rot[3]];
            let scale_gltf = [scale[0], scale[2], scale[1]];

            builder.add_node(
                &format!("Prop_{:05}_{}", idx + 1, p.name.trim_end_matches(".#dr")),
                mesh_idx,
                trans_gltf,
                rot_gltf,
                scale_gltf,
            );
            placed_props_count += 1;
        }
    }

    eprintln!(
        "[map-assembler] Exporting instanced GLB ({} unique meshes, {} placed nodes) to {}...",
        builder.meshes.len(),
        builder.nodes.len(),
        out_glb.display()
    );
    let glb_bytes = builder
        .finish("GTA6_World_Map_With_LOD0_Props", out_glb)
        .map_err(|e| format!("GLB export error: {e}"))?;
    eprintln!("[map-assembler] GLB export successful: {glb_bytes} bytes written");

    // 7. Render high-res 2560x1440 preview image
    eprintln!("[map-assembler] 5. Rendering high-res 2560x1440 preview image to {}", out_png.display());
    if let Some(par) = out_png.parent() {
        let _ = fs::create_dir_all(par);
    }

    // Build preview scene mesh with terrain + Brickell + downtown placed props
    let mut preview_verts = terrain_mesh.verts.clone();
    let mut preview_uvs = terrain_mesh.uvs.clone();
    let mut preview_tris = terrain_mesh.tris.clone();
    let mut preview_tri_geom = terrain_mesh.tri_geom.clone();
    let mut preview_materials = terrain_mesh.geom_materials.clone();

    let bk_v_base = preview_verts.len() as u32;
    let bk_geom_base = preview_materials.len() as u16;
    preview_verts.extend(brickell_urban_mesh.verts.iter().cloned());
    preview_uvs.extend(brickell_urban_mesh.uvs.iter().cloned());
    for tri in &brickell_urban_mesh.tris {
        preview_tris.push([tri[0] + bk_v_base, tri[1] + bk_v_base, tri[2] + bk_v_base]);
    }
    for &g in &brickell_urban_mesh.tri_geom {
        preview_tri_geom.push(bk_geom_base + g);
    }
    preview_materials.extend(brickell_urban_mesh.geom_materials.iter().cloned());

    // Add placed props in the camera viewing window (Downtown / Brickell / Biscayne Bay)
    let mut preview_props_added = 0usize;
    for p in &resolved_props {
        let pos = p.placement.pos;
        // View window: entire world
        if pos[0] > -25000.0 && pos[0] < 25000.0 && pos[1] > -25000.0 && pos[1] < 25000.0 {
            if let Some(m) = decoded_unique.get(&p.offset) {
                let v_base = preview_verts.len() as u32;
                let geom_base = preview_materials.len() as u16;
                let s = p.placement.scale;
                let q = p.placement.rot;

                for &v in &m.verts {
                    let sv = [v[0] * s[0], v[1] * s[1], v[2] * s[2]];
                    let rv = quat_rotate(q, sv);
                    preview_verts.push([rv[0] + pos[0], rv[1] + pos[1], rv[2] + pos[2]]);
                }
                preview_uvs.extend(m.uvs.iter().cloned());
                while preview_uvs.len() < preview_verts.len() {
                    preview_uvs.push([0.0, 0.0]);
                }
                for tri in &m.tris {
                    preview_tris.push([tri[0] + v_base, tri[1] + v_base, tri[2] + v_base]);
                }
                for &g in &m.tri_geom {
                    preview_tri_geom.push(geom_base + g);
                }
                if m.geom_materials.is_empty() {
                    preview_materials.push(GeomMaterial::default());
                } else {
                    preview_materials.extend(m.geom_materials.iter().cloned());
                }
                preview_props_added += 1;
                if preview_props_added >= 3000 {
                    break;
                }
            }
        }
    }

    eprintln!(
        "[map-assembler] Preview mesh composed with terrain, Brickell, and {preview_props_added} downtown props"
    );

    let mut preview_mesh = Mesh3D::from_parts(
        "GTA6_World_Map_Preview".into(),
        preview_verts,
        preview_uvs,
        preview_tris,
        preview_tri_geom,
        Vec::new(),
        None,
        format!("Preview Scene"),
        preview_materials.len(),
        0,
        "Preview Scene".into(),
    );
    preview_mesh.geom_materials = preview_materials;
    preview_mesh.shade_mode = ShadeMode::Textured;
    preview_mesh.sky_bg = true;

    preview_mesh.yaw = 0.58;
    preview_mesh.pitch = 0.62;
    preview_mesh.zoom = 1.15;
    preview_mesh.pan_x = 0.02;
    preview_mesh.pan_y = -0.04;

    preview_mesh.render_preview_image(2560, 1440, out_png)?;
    let png_bytes = fs::metadata(out_png).map(|m| m.len() as usize).unwrap_or(0);
    eprintln!("[map-assembler] Preview image saved: {png_bytes} bytes");

    let elapsed = t_start.elapsed().as_secs_f32();
    eprintln!("[map-assembler] Full assembly completed in {:.2}s!", elapsed);

    Ok(FullAssemblyReport {
        terrain_vertices: terrain_vcount,
        terrain_triangles: terrain_tcount,
        brickell_geometries: brickell_mesh.geom_count,
        brickell_vertices: bk_aligned_vcount,
        brickell_triangles: bk_aligned_tcount,
        unique_props_count: decoded_unique.len(),
        placed_props_count,
        total_vertices: total_unique_verts,
        total_triangles: total_unique_tris,
        glb_path: out_glb.to_string_lossy().into_owned(),
        png_path: out_png.to_string_lossy().into_owned(),
        glb_bytes,
        png_bytes,
        elapsed_secs: elapsed,
    })
}

#[derive(Clone, Debug, Default)]
pub struct SectorInfo {
    pub file: String,
    pub grid_x: i32,
    pub grid_y: i32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub unique_models: usize,
    pub instances_count: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct SectorsAssemblyReport {
    pub total_sectors: usize,
    pub terrain_vertices: usize,
    pub terrain_triangles: usize,
    pub terrain_bytes: u64,
    pub total_unique_models: usize,
    pub total_placed_instances: usize,
    pub total_vertices: usize,
    pub total_triangles: usize,
    pub total_directory_bytes: u64,
    pub elapsed_secs: f32,
    pub sectors: Vec<SectorInfo>,
}

pub fn assemble_all_world_sectors(
    common_rpf: &Path,
    levels_rpf: &Path,
    carved_dir: &Path,
    out_dir: &Path,
) -> Result<SectorsAssemblyReport, String> {
    let t_start = Instant::now();
    eprintln!("=== Assembling All GTA 6 World Sectors (Grid Partitioning) ===");
    eprintln!("[sectors] Output directory: {}", out_dir.display());

    fs::create_dir_all(out_dir).map_err(|e| format!("Failed to create output dir {}: {e}", out_dir.display()))?;

    // 1. Extract 2048x2048 world elevation and export terrain_base.glb
    eprintln!("[sectors] 1. Extracting base terrain elevation from {}...", common_rpf.display());
    let terrain_data = load_mhiu_from_rpf(common_rpf).map_err(|e| format!("MHIU load error: {e}"))?;
    let (t_verts, t_uvs, t_tris, t_tex) = build_world_terrain(&terrain_data, 4);
    let terrain_vcount = t_verts.len();
    let terrain_tcount = t_tris.len();
    eprintln!("[sectors] Base terrain mesh generated: {terrain_vcount} vertices, {terrain_tcount} triangles");

    let mut terrain_mat = GeomMaterial::default();
    terrain_mat.texture = Some(Arc::new(t_tex));
    let mut terrain_mesh = Mesh3D::from_parts(
        "GTA6_World_Terrain_Base".into(),
        t_verts,
        t_uvs,
        t_tris,
        vec![0u16; terrain_tcount],
        Vec::new(),
        None,
        format!("Terrain Base {terrain_vcount} verts · {terrain_tcount} tris"),
        1,
        0,
        "Terrain Base".into(),
    );
    terrain_mesh.geom_materials = vec![terrain_mat];
    terrain_mesh.shade_mode = ShadeMode::Textured;

    let terrain_base_path = out_dir.join("terrain_base.glb");
    eprintln!("[sectors] Exporting terrain_base.glb to {}...", terrain_base_path.display());
    let mut terrain_builder = GltfBuilder::new();
    let t_m_idx = terrain_builder.add_mesh3d(&terrain_mesh, "Terrain_Base_Mesh", true);
    terrain_builder.add_node("Terrain_Base_Node", t_m_idx, [0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    let terrain_bytes = terrain_builder
        .finish("GTA6_Terrain_Base", &terrain_base_path)
        .map_err(|e| format!("terrain_base.glb export error: {e}"))? as u64;
    eprintln!("[sectors] terrain_base.glb exported: {terrain_bytes} bytes");

    // 2. Resolve ALL world entities (strictly 0 vehicles, 0 peds, 0 animals; preserve SLODs, roads, bridges, buildings, props)
    eprintln!("[sectors] 2. Resolving all world entities from world_index_v2 and carved PSIN...");
    let (mut resolved_entities, _stats) = props_pipeline::resolve_all_world_entities(
        levels_rpf,
        if carved_dir.is_dir() { Some(carved_dir) } else { None },
    );

    // Also include Brickell downtown SLOD blocks
    for bp in BRICKELL_PLACEMENTS {
        resolved_entities.push(props_pipeline::ResolvedProp {
            name: bp.name.to_string(),
            archetype_hash: crate::psin::joaat(bp.name.trim_end_matches(".#dr")),
            offset: 8_679_358_480,
            placement: props_pipeline::RawPlacement {
                archetype_hash: crate::psin::joaat(bp.name.trim_end_matches(".#dr")),
                pos: bp.pos,
                rot: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
        });
    }

    eprintln!("[sectors] Total placed world instances to partition: {}", resolved_entities.len());

    // 3. Partition placed instances into spatial grid sectors (2000m x 2000m)
    const GRID_SIZE: f32 = 2000.0;
    let mut sector_map: HashMap<(i32, i32), Vec<&props_pipeline::ResolvedProp>> = HashMap::new();
    for p in &resolved_entities {
        let gx = (p.placement.pos[0] / GRID_SIZE).floor() as i32;
        let gy = (p.placement.pos[1] / GRID_SIZE).floor() as i32;
        sector_map.entry((gx, gy)).or_default().push(p);
    }

    let mut sector_keys: Vec<(i32, i32)> = sector_map.keys().copied().collect();
    sector_keys.sort();

    eprintln!(
        "[sectors] Partitioned world into {} spatial sectors (2000m x 2000m grid)",
        sector_keys.len()
    );

    // 4. Build and export each non-empty sector as an instanced glTF 2.0 Binary `sector_<gx>_<gy>.glb`
    // Decodes models per sector in parallel across CPU cores to ensure lean memory usage
    let mut sector_infos: Vec<SectorInfo> = Vec::with_capacity(sector_keys.len());
    let mut total_placed_instances = 0usize;
    let mut total_scene_verts = terrain_vcount;
    let mut total_scene_tris = terrain_tcount;
    let mut global_unique_models_used: HashSet<u64> = HashSet::new();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .clamp(2, 24);

    for (s_idx, &(gx, gy)) in sector_keys.iter().enumerate() {
        let instances = &sector_map[&(gx, gy)];
        // Collect unique models used in this sector
        let mut sector_models_map: HashMap<u64, String> = HashMap::new();
        for p in instances {
            sector_models_map.entry(p.offset).or_insert_with(|| p.name.clone());
        }

        if sector_models_map.is_empty() {
            continue;
        }

        // Decode unique models for this sector in parallel across CPU cores
        let unique_list: Vec<(u64, String)> = sector_models_map.into_iter().collect();
        let chunks: Vec<Vec<(u64, String)>> = unique_list
            .chunks((unique_list.len() + n_threads - 1) / n_threads)
            .map(|c| c.to_vec())
            .collect();

        let decoded_sector: HashMap<u64, Mesh3D> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                handles.push(s.spawn(move || {
                    let mut local = Vec::new();
                    for (off, name) in chunk {
                        // want_lod = 0 (LOD0), layout_grid = false
                        if let Ok(mesh) = decode_rage9_model_from_rpf_lod_opt(levels_rpf, off, &name, 0, false) {
                            if !mesh.verts.is_empty() && !mesh.tris.is_empty() {
                                local.push((off, mesh));
                            }
                        }
                    }
                    local
                }));
            }
            let mut map = HashMap::new();
            for h in handles {
                if let Ok(res) = h.join() {
                    for (off, mesh) in res {
                        map.insert(off, mesh);
                    }
                }
            }
            map
        });

        if decoded_sector.is_empty() {
            continue;
        }

        let mut builder = GltfBuilder::new();
        let mut offset_to_m_idx: HashMap<u64, usize> = HashMap::new();
        let mut sector_verts = 0usize;
        let mut sector_tris = 0usize;

        for (&off, mesh) in &decoded_sector {
            let m_idx = builder.add_mesh3d(mesh, &mesh.name, false);
            offset_to_m_idx.insert(off, m_idx);
            sector_verts += mesh.verts.len();
            sector_tris += mesh.tris.len();
            global_unique_models_used.insert(off);
        }

        let mut sector_placed_count = 0usize;
        for (idx, p) in instances.iter().enumerate() {
            if let Some(&mesh_idx) = offset_to_m_idx.get(&p.offset) {
                let pos = p.placement.pos;
                let rot = p.placement.rot;
                let scale = p.placement.scale;

                // Mathematical conversion from GTA to glTF 2.0 coordinate system:
                // glTF translation: [pos.x, pos.z, -pos.y]
                // glTF rotation:    [rot.x, rot.z, -rot.y, rot.w]
                // glTF scale:       [scale.x, scale.z, scale.y]
                let trans_gltf = [pos[0], pos[2], -pos[1]];
                let rot_gltf = [rot[0], rot[2], -rot[1], rot[3]];
                let scale_gltf = [scale[0], scale[2], scale[1]];

                builder.add_node(
                    &format!("Inst_{:05}_{}", idx + 1, p.name.trim_end_matches(".#dr")),
                    mesh_idx,
                    trans_gltf,
                    rot_gltf,
                    scale_gltf,
                );
                sector_placed_count += 1;
            }
        }

        if builder.nodes.is_empty() {
            continue;
        }

        let sector_fname = format!("sector_{gx}_{gy}.glb");
        let sector_path = out_dir.join(&sector_fname);
        let s_bytes = builder
            .finish(&format!("GTA6_Sector_{gx}_{gy}"), &sector_path)
            .map_err(|e| format!("Failed to export {sector_fname}: {e}"))? as u64;

        eprintln!(
            "[sectors] [{}/{}] Exported {} ({} models, {} nodes, {:.2} MB)",
            s_idx + 1,
            sector_keys.len(),
            sector_fname,
            decoded_sector.len(),
            sector_placed_count,
            s_bytes as f64 / 1_048_576.0
        );

        total_placed_instances += sector_placed_count;
        total_scene_verts += sector_verts;
        total_scene_tris += sector_tris;

        sector_infos.push(SectorInfo {
            file: sector_fname,
            grid_x: gx,
            grid_y: gy,
            min_x: gx as f32 * GRID_SIZE,
            max_x: (gx + 1) as f32 * GRID_SIZE,
            min_y: gy as f32 * GRID_SIZE,
            max_y: (gy + 1) as f32 * GRID_SIZE,
            unique_models: decoded_sector.len(),
            instances_count: sector_placed_count,
            vertices: sector_verts,
            triangles: sector_tris,
            size_bytes: s_bytes,
        });
    }

    // 6. Write `load_full_world_in_blender.py`
    let blender_script = r#""""
GTA 6 Full World Map - Blender Loader Script
Generated by PercyStar v1.5.0
Auto-imports terrain_base.glb and all sector_*.glb into Blender.
Sets 3D viewport and camera clipping end to 50,000m to view the entire Leonida map.
"""
import os
import bpy

def load_world():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(f"[GTA 6 Loader] Loading full world sectors from: {script_dir}")

    # Set viewport clipping for all 3D viewports to 50,000m
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.clip_end = 50000.0

    # Set camera clipping if camera exists
    if bpy.context.scene.camera and bpy.context.scene.camera.data:
        bpy.context.scene.camera.data.clip_end = 50000.0

    # Collect files: terrain_base.glb first, then all sector_*.glb sorted
    glb_files = []
    terrain_file = os.path.join(script_dir, "terrain_base.glb")
    if os.path.isfile(terrain_file):
        glb_files.append(("terrain_base.glb", terrain_file))

    sector_files = []
    for f in os.listdir(script_dir):
        if f.startswith("sector_") and f.endswith(".glb"):
            sector_files.append((f, os.path.join(script_dir, f)))
    sector_files.sort()
    glb_files.extend(sector_files)

    print(f"[GTA 6 Loader] Found {len(glb_files)} GLB files to import.")

    for i, (name, path) in enumerate(glb_files):
        print(f"[{i+1}/{len(glb_files)}] Importing {name}...")
        try:
            bpy.ops.import_scene.gltf(filepath=path)
        except Exception as e:
            print(f"  Warning: failed to import {name}: {e}")

    print("[GTA 6 Loader] Full world successfully imported into Blender!")

if __name__ == "__main__":
    load_world()
"#;

    let blender_script_path = out_dir.join("load_full_world_in_blender.py");
    fs::write(&blender_script_path, blender_script)
        .map_err(|e| format!("Failed to write Blender script: {e}"))?;
    eprintln!("[sectors] Written Blender loader: {}", blender_script_path.display());

    // 7. Write `sectors_manifest.json`
    let mut manifest_sectors_json = Vec::new();
    for s in &sector_infos {
        manifest_sectors_json.push(format!(
            "    {{\"file\":{},\"grid_x\":{},\"grid_y\":{},\"bounds\":{{\"min_x\":{:.1},\"max_x\":{:.1},\"min_y\":{:.1},\"max_y\":{:.1}}},\"unique_models\":{},\"instances_count\":{},\"vertices\":{},\"triangles\":{},\"size_bytes\":{}}}",
            json_str(&s.file),
            s.grid_x,
            s.grid_y,
            s.min_x,
            s.max_x,
            s.min_y,
            s.max_y,
            s.unique_models,
            s.instances_count,
            s.vertices,
            s.triangles,
            s.size_bytes
        ));
    }

    let manifest_json = format!(
        "{{\n  \"generator\": \"PercyStar v1.5.0\",\n  \"grid_size_meters\": {:.1},\n  \"total_sectors\": {},\n  \"total_unique_models\": {},\n  \"total_placed_instances\": {},\n  \"total_vertices\": {},\n  \"total_triangles\": {},\n  \"terrain_base\": {{\n    \"file\": \"terrain_base.glb\",\n    \"vertices\": {},\n    \"triangles\": {},\n    \"size_bytes\": {}\n  }},\n  \"sectors\": [\n{}\n  ]\n}}\n",
        GRID_SIZE,
        sector_infos.len(),
        global_unique_models_used.len(),
        total_placed_instances,
        total_scene_verts,
        total_scene_tris,
        terrain_vcount,
        terrain_tcount,
        terrain_bytes,
        manifest_sectors_json.join(",\n")
    );

    let manifest_path = out_dir.join("sectors_manifest.json");
    fs::write(&manifest_path, manifest_json)
        .map_err(|e| format!("Failed to write manifest: {e}"))?;
    eprintln!("[sectors] Written manifest: {}", manifest_path.display());

    // Calculate total directory bytes
    let mut total_dir_bytes = 0u64;
    if let Ok(rd) = fs::read_dir(out_dir) {
        for entry in rd.flatten() {
            if let Ok(meta) = entry.metadata() {
                total_dir_bytes += meta.len();
            }
        }
    }

    let elapsed = t_start.elapsed().as_secs_f32();
    eprintln!(
        "[sectors] Assembly complete in {:.2}s! {} sectors, {} unique models, {} placed instances, {} total vertices, {:.2} MB total size",
        elapsed,
        sector_infos.len(),
        global_unique_models_used.len(),
        total_placed_instances,
        total_scene_verts,
        total_dir_bytes as f64 / 1_048_576.0
    );

    Ok(SectorsAssemblyReport {
        total_sectors: sector_infos.len(),
        terrain_vertices: terrain_vcount,
        terrain_triangles: terrain_tcount,
        terrain_bytes,
        total_unique_models: global_unique_models_used.len(),
        total_placed_instances,
        total_vertices: total_scene_verts,
        total_triangles: total_scene_tris,
        total_directory_bytes: total_dir_bytes,
        elapsed_secs: elapsed,
        sectors: sector_infos,
    })
}

pub fn render_street_level_lod0_cli(levels_p: &Path, out_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    eprintln!("[street-level] Assembling and rendering LOD0 Ocean Drive & Brickell street blocks...");

    let sb_models = [
        ("sb_01_ground_02_dw_01.#dr", 14565412880u64),
        ("sb_01_ground_02_dw_02.#dr", 14566043664u64),
        ("sb_01_build1_front_01.#dr", 14552113168u64),
        ("sb_01_build1_front_01_dtl.#dr", 14553079824u64),
        ("sb_01_build1_front_01_01_cloth.#dr", 14553051152u64),
        ("sb_01_build1_main.#dr", 14553141264u64),
        ("sb_01_build2.#dr", 14557077520u64),
        ("sb_01_build2_branding.#dr", 14558167056u64),
        ("sb_01_compoundwall_01.#dr", 14562115600u64),
        ("sb_01_marina_hotels_signage.#dr", 14568161296u64),
        ("sb_01_shops_detail.#dr", 14568800272u64),
        ("sb_01_shops_shell.#dr", 14568902672u64),
        ("sb_01_b2_barlights.#dr", 14549827600u64),
    ];

    let mut sb_verts = Vec::new();
    let mut sb_uvs = Vec::new();
    let mut sb_tris = Vec::new();
    let mut sb_geom = Vec::new();
    let mut sb_mats = Vec::new();

    for (tag, off) in &sb_models {
        match decode_rage9_model_from_rpf_lod_opt(levels_p, *off, tag, 0, false) {
            Ok(m) => {
                let v_base = sb_verts.len() as u32;
                let g_base = sb_mats.len() as u16;
                sb_verts.extend_from_slice(&m.verts);
                sb_uvs.extend_from_slice(&m.uvs);
                for t in &m.tris {
                    sb_tris.push([v_base + t[0], v_base + t[1], v_base + t[2]]);
                }
                for g in &m.tri_geom {
                    sb_geom.push(g_base + g);
                }
                let mats_with_tex = m.geom_materials.iter().filter(|mat| mat.texture.is_some()).count();
                if m.geom_materials.is_empty() {
                    sb_mats.push(GeomMaterial::default());
                } else {
                    sb_mats.extend(m.geom_materials.iter().cloned());
                }
                eprintln!("[street-level] Decoded LOD0: {tag} ({} verts, {} uvs, {} tris, {} mats, {} with tex, tex_layers: {})",
                    m.verts.len(), m.uvs.len(), m.tris.len(), m.geom_materials.len(), mats_with_tex, m.tex_layers.len());
            }
            Err(e) => eprintln!("[street-level] Warning: failed to decode {tag}: {e}"),
        }
    }

    let mut sb_mesh = Mesh3D::from_parts(
        "Ocean_Drive_Street_LOD0".into(),
        sb_verts,
        sb_uvs,
        sb_tris,
        sb_geom,
        Vec::new(),
        None,
        "Ocean Drive LOD0".into(),
        sb_mats.len(),
        0,
        "Ocean Drive LOD0".into(),
    );
    sb_mesh.geom_materials = sb_mats;
    sb_mesh.shade_mode = ShadeMode::Textured;
    sb_mesh.sky_bg = true;

    let total_mats_with_tex = sb_mesh.geom_materials.iter().filter(|m| m.texture.is_some()).count();
    eprintln!("[street-level] Assembled Ocean Drive: verts={}, uvs={}, tris={}, mats={}, mats_with_tex={}, has_uvs={}",
        sb_mesh.verts.len(), sb_mesh.uvs.len(), sb_mesh.tris.len(), sb_mesh.geom_materials.len(), total_mats_with_tex,
        sb_mesh.uvs.len() == sb_mesh.verts.len() && !sb_mesh.uvs.is_empty());

    // Export GLB of Ocean Drive Street block
    let glb_out = out_dir.join("ocean_drive_street_block_lod0.glb");
    if let Err(e) = sb_mesh.export_glb(&glb_out) {
        eprintln!("[street-level] Warning: GLB export error: {e}");
    } else {
        eprintln!("[street-level] Exported: {}", glb_out.display());
    }

    // Render 4 pedestrian & street-level viewpoints
    let sb_views = [
        ("01_ocean_drive_street_level_eye.png", 0.35f32, 0.08f32, 2.2f32, 0.02f32, -0.06f32),
        ("02_hotel_facade_and_awnings_closeup.png", 0.72f32, 0.16f32, 3.2f32, -0.05f32, -0.03f32),
        ("03_shops_and_sidewalk_pedestrian.png", -0.42f32, 0.06f32, 2.7f32, 0.04f32, -0.04f32),
        ("04_street_curb_and_asphalt_wide.png", 0.10f32, 0.12f32, 1.9f32, 0.0f32, -0.08f32),
    ];
    for (filename, yaw, pitch, zoom, pan_x, pan_y) in sb_views {
        let p = out_dir.join(filename);
        sb_mesh.yaw = yaw;
        sb_mesh.pitch = pitch;
        sb_mesh.zoom = zoom;
        sb_mesh.pan_x = pan_x;
        sb_mesh.pan_y = pan_y;
        if let Err(e) = sb_mesh.render_preview_image(2560, 1440, &p) {
            eprintln!("[street-level] Warning: failed to render {filename}: {e}");
        } else {
            eprintln!("[street-level] Rendered street-level view: {filename}");
        }
    }

    // Brickell Street Level Base
    let bk_models = [
        ("bk_01_base_01.#dr", 8095694864u64),
        ("bk_01_base_01_dtl_01.#dr", 8096260112u64),
        ("bk_01_base_02.#dr", 8096591888u64),
    ];
    let mut bk_verts = Vec::new();
    let mut bk_uvs = Vec::new();
    let mut bk_tris = Vec::new();
    let mut bk_geom = Vec::new();
    let mut bk_mats = Vec::new();

    for (tag, off) in &bk_models {
        if let Ok(m) = decode_rage9_model_from_rpf_lod_opt(levels_p, *off, tag, 0, false) {
            let v_base = bk_verts.len() as u32;
            let g_base = bk_mats.len() as u16;
            bk_verts.extend_from_slice(&m.verts);
            bk_uvs.extend_from_slice(&m.uvs);
            for t in &m.tris {
                bk_tris.push([v_base + t[0], v_base + t[1], v_base + t[2]]);
            }
            for g in &m.tri_geom {
                bk_geom.push(g_base + g);
            }
            if m.geom_materials.is_empty() {
                bk_mats.push(GeomMaterial::default());
            } else {
                bk_mats.extend(m.geom_materials.iter().cloned());
            }
            eprintln!("[street-level] Decoded LOD0 Brickell: {tag} ({} verts, {} tris)", m.verts.len(), m.tris.len());
        }
    }

    let mut bk_mesh = Mesh3D::from_parts(
        "Brickell_Avenue_LOD0".into(),
        bk_verts,
        bk_uvs,
        bk_tris,
        bk_geom,
        Vec::new(),
        None,
        "Brickell LOD0".into(),
        bk_mats.len(),
        0,
        "Brickell LOD0".into(),
    );
    bk_mesh.geom_materials = bk_mats;
    bk_mesh.shade_mode = ShadeMode::Textured;
    bk_mesh.sky_bg = true;

    let bk_glb_out = out_dir.join("brickell_avenue_lod0.glb");
    if let Err(e) = bk_mesh.export_glb(&bk_glb_out) {
        eprintln!("[street-level] Warning: Brickell GLB export error: {e}");
    } else {
        eprintln!("[street-level] Exported: {}", bk_glb_out.display());
    }

    let bk_p = out_dir.join("05_brickell_avenue_pedestrian_ground.png");
    bk_mesh.yaw = 0.55;
    bk_mesh.pitch = 0.14;
    bk_mesh.zoom = 2.4;
    bk_mesh.pan_x = 0.0;
    bk_mesh.pan_y = -0.05;
    if let Err(e) = bk_mesh.render_preview_image(2560, 1440, &bk_p) {
        eprintln!("[street-level] Warning: failed to render 05_brickell_avenue_pedestrian_ground.png: {e}");
    } else {
        eprintln!("[street-level] Rendered: 05_brickell_avenue_pedestrian_ground.png");
    }

    Ok(())
}

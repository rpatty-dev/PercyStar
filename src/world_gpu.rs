use crate::model3d::{Mesh3D, Texture2D};
use crate::worldmap::WorldMapData;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Icon, Window, WindowId};

mod smoke;
pub use smoke::run_gpu_world_smoke;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuInstance {
    pub model_0: [f32; 4],
    pub model_1: [f32; 4],
    pub model_2: [f32; 4],
    pub model_3: [f32; 4],
    pub color: [f32; 4],
}

impl GpuInstance {
    pub fn from_mat4(m: Mat4, color: [f32; 4]) -> Self {
        let cols = m.to_cols_array_2d();
        Self {
            model_0: cols[0],
            model_1: cols[1],
            model_2: cols[2],
            model_3: cols[3],
            color,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    flags: [f32; 4], // x=rt_on, y=time_hours, z=map_min_x, w=map_span_x
    map_y: [f32; 4], // x=map_min_y, y=map_span_y, z=viewport_w, w=viewport_h
    hud: [f32; 4],   // x=load_progress, y=loading_active, z=elapsed_sec, w=fog_density
}

pub struct WorldBatchInput {
    pub mesh: Mesh3D,
    pub instances: Vec<(Vec3, Quat, Vec3, [f32; 4])>,
}

pub struct WorldLaunchConfig {
    pub terrain: Option<Arc<WorldMapData>>,
    pub batches: Vec<WorldBatchInput>,
    pub levels_rpf: Option<std::path::PathBuf>,
    pub map_count: usize,
    pub start_pos: [f32; 3],
}

pub enum WorldStreamMsg {
    Batch(WorldBatchInput),
    MapCount(usize),
    TotalArchetypes(usize),
    ArchetypeDone,
}

pub const DEFAULT_VRAM_BUDGET_BYTES: usize = 5800 * 1024 * 1024; // 5.8 GB target ceiling (~5.5 - 6.0 GB)

#[derive(Debug)]
pub struct StreamThrottleState {
    pub stream_radius: f32,
    pub vram_pressure: bool,
    pub evicted_hashes: Vec<u32>,
}

impl StreamThrottleState {
    pub fn new() -> Self {
        Self {
            stream_radius: 35000.0,
            vram_pressure: false,
            evicted_hashes: Vec::new(),
        }
    }
}

pub fn compute_throttle_tier(total_vram_bytes: usize, budget_bytes: usize) -> (f32, bool, &'static str) {
    let ratio = total_vram_bytes as f32 / budget_bytes.max(1) as f32;
    if ratio >= 0.88 || total_vram_bytes >= 5_500 * 1024 * 1024 {
        (350.0, true, "CRITICAL: EVICTING (350m)")
    } else if ratio >= 0.80 || total_vram_bytes >= 4_800 * 1024 * 1024 {
        (450.0, true, "NEAR CAPACITY: THROTTLED (450m)")
    } else if ratio >= 0.70 {
        (1500.0, false, "HIGH LOAD: 1.5km")
    } else if ratio >= 0.55 {
        (3000.0, false, "MODERATE: 3km")
    } else if ratio >= 0.40 {
        (6000.0, false, "HEADROOM: 6km")
    } else {
        (35000.0, false, "OPTIMAL: UNCAPPED (35km)")
    }
}

pub fn archetype_hash_from_mesh_name(name: &str) -> u32 {
    if let Some(rest) = name.strip_prefix("arch_") {
        u32::from_str_radix(rest, 16).unwrap_or(0)
    } else if name == "mhiu_world_terrain" {
        0
    } else if name.starts_with("nav_") || name.contains("road") || name.contains("street") {
        1
    } else if name == "__placeholder_boxes__" {
        2
    } else {
        let mut h: u32 = 0x811c9dc5;
        for b in name.as_bytes() {
            h = (h ^ (*b as u32)).wrapping_mul(0x01000193);
        }
        h
    }
}

pub fn launch_world_gpu_process(
    common_rpf: Option<&Path>,
    levels_rpf: Option<&Path>,
    ctx: eframe::egui::Context,
) -> Result<Receiver<Result<(), String>>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--gpu-world");
    if let Some(c) = common_rpf {
        cmd.arg("--common").arg(c);
    }
    if let Some(l) = levels_rpf {
        cmd.arg("--levels").arg(l);
    }
    use std::process::Stdio;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().map_err(|e| format!("Failed to launch 3D GPU World: {e}"))?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader, Read};
        let stderr = child.stderr.take().unwrap();
        let errors = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut text);
            text
        });
        let mut ready = false;
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            match line {
                Ok(line) if line == "WORLD_READY" => {
                    ready = true;
                    let _ = tx.send(Ok(()));
                    ctx.request_repaint();
                }
                Ok(line) if line.starts_with("WORLD_ERROR ") => {
                    let _ = tx.send(Err(line[12..].to_string()));
                    ctx.request_repaint();
                }
                Err(error) => {
                    let _ = tx.send(Err(format!("GPU viewer output failed: {error}")));
                    ctx.request_repaint();
                }
                _ => {}
            }
        }
        let status = child.wait();
        let errors = errors.join().unwrap_or_default();
        if !ready || status.as_ref().map_or(true, |s| !s.success()) {
            let _ = tx.send(Err(format!("GPU viewer exited ({status:?}): {}", errors.trim())));
            ctx.request_repaint();
        }
    });
    Ok(rx)
}

static WORLD_LOGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub fn log_world(msg: impl Into<String>) {
    let s = msg.into();
    eprintln!("{s}");
    if let Ok(mut logs) = WORLD_LOGS.lock() {
        if logs.len() >= 400 {
            logs.remove(0);
        }
        logs.push(s);
    }
}

pub fn run_gpu_world_cli(common_rpf: Option<&Path>, levels_rpf: Option<&Path>) -> Result<(), String> {
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn SetPriorityClass(hProcess: isize, dwPriorityClass: u32) -> i32;
        }
        // below_normal keeps windows audio and desktop apps stutter-free during streaming
        SetPriorityClass(GetCurrentProcess(), 0x0000_4000);
    }
    let fallback_dir = crate::load_last_folder();
    let resolved_common = common_rpf.map(Path::to_path_buf).or_else(|| {
        fallback_dir.as_ref().map(|d| d.join("common.rpf")).filter(|p| p.exists())
    });
    let resolved_levels = levels_rpf.map(Path::to_path_buf).or_else(|| {
        fallback_dir.as_ref().map(|d| d.join("levels.rpf")).filter(|p| p.exists())
    });
    if resolved_common.is_none() && resolved_levels.is_none() {
        return Err("No world archives found; open a game folder first".into());
    }
    let terrain = resolved_common.as_deref()
        .map(|p| crate::worldmap::load_mhiu_from_rpf(p).map(Arc::new)).transpose()?;
    let cfg = WorldLaunchConfig {
        terrain,
        batches: Vec::new(),
        levels_rpf: resolved_levels,
        map_count: 0,
        start_pos: [-127.276, -1855.924, 24.0],
    };
    let event_loop = EventLoop::builder().build().map_err(|e| format!("Cannot start GPU event loop: {e}"))?;
    let (tx, rx) = mpsc::sync_channel(16);
    let cam_track = Arc::new(std::sync::Mutex::new((Vec3::from_array(cfg.start_pos), 1u64)));
    let cam_track_bg = Arc::clone(&cam_track);
    let stream_ctl = Arc::new(std::sync::Mutex::new(StreamThrottleState::new()));
    let stream_ctl_bg = Arc::clone(&stream_ctl);
    if let Some(lp) = cfg.levels_rpf.clone() {
        std::thread::spawn(move || {
            // 1. Stream real CNavMesh road tiles concurrently across full world map
            let nav_rpf = lp.parent().map(|p| p.join("navmeshes.rpf")).filter(|p| p.exists());
            if let Some(ref np) = nav_rpf {
                let np = np.clone();
                let nav_tx = tx.clone();
                let cam_track_bg = Arc::clone(&cam_track_bg);
                std::thread::spawn(move || {
                    if let Some(base) = std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from) {
                        let tsv = base.join("PercyStar").join("scan_cache_v9.tsv");
                        if let Ok(f) = std::fs::File::open(tsv) {
                            use std::io::BufRead;
                            let r = std::io::BufReader::new(f);
                            let start = cam_track_bg.lock().unwrap().0;
                            let mut nav_tiles: Vec<(f32, String, u64)> = Vec::new();
                            for line in r.lines().map_while(Result::ok) {
                                if !line.contains("\tNAVM\t") { continue; }
                                let parts: Vec<&str> = line.split('\t').collect();
                                if parts.len() < 6 { continue; }
                                let tile_nm = parts[3];
                                let Ok(off) = parts[5].parse::<u64>() else { continue };
                                if let (Some(s_p), Some(e_p)) = (tile_nm.find('('), tile_nm.find(')')) {
                                    let cparts: Vec<&str> = tile_nm[s_p + 1..e_p].split(',').collect();
                                    if cparts.len() == 2 {
                                        if let (Ok(tx_c), Ok(ty_c)) = (cparts[0].parse::<f32>(), cparts[1].parse::<f32>()) {
                                            let dx = tx_c - start.x;
                                            let dy = ty_c - start.y;
                                            let d = (dx * dx + dy * dy).sqrt();
                                            if d <= 35000.0 {
                                                nav_tiles.push((d, tile_nm.to_string(), off));
                                            }
                                        }
                                    }
                                }
                            }
                            nav_tiles.sort_by(|a, b| a.0.total_cmp(&b.0));
                            log_world(format!("[world-gpu] Streaming {} CNavMesh road tiles across full world...", nav_tiles.len().min(2000)));
                            for (_dist, tile_nm, off) in nav_tiles.into_iter().take(2000) {
                                if let Ok(mesh) = crate::tex::decode_navmesh_roads_from_rpf(&np, off, &tile_nm) {
                                    let _ = nav_tx.send(WorldStreamMsg::Batch(WorldBatchInput {
                                        mesh,
                                        instances: vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, [1.0; 4])],
                                    }));
                                }
                            }
                        }
                    }
                });
            }
            // 2. build/load cached 0DRPz map placements + model index
            log_world(format!("[world-gpu] loading/building world index from {}", lp.display()));
            let ix = crate::mapdata::build_or_load_world_index(&lp);
            log_world(format!("[world-gpu] index ready: {} maps, {} models, {} entities", ix.map_count, ix.models.len(), ix.entities.len()));
            let _ = tx.send(WorldStreamMsg::MapCount(ix.map_count));

            let mut grouped: HashMap<u32, (Vec3, Vec<(Vec3, Quat, Vec3, [f32; 4])>)> = HashMap::new();
            let mut box_insts: Vec<(Vec3, Quat, Vec3, [f32; 4])> = Vec::new();

            // 1. Exact instance deduplication (eliminates duplicates from overlapping 0DRPz boundary chunks)
            let mut seen_exact = HashSet::new();
            let mut hd_ents = Vec::new();
            let mut lod_ents = Vec::new();
            let mut slod_ents = Vec::new();

            for e in &ix.entities {
                let p = Vec3::from_array(e.pos);
                if !p.is_finite() || !e.scale_xy.is_finite() || !e.scale_z.is_finite() {
                    continue;
                }
                if !ix.models.contains_key(&e.archetype_hash) {
                    continue;
                }
                let q_raw = Quat::from_xyzw(e.rot[0], e.rot[1], e.rot[2], e.rot[3]);
                let q = if !q_raw.is_finite() || q_raw.length_squared() < 1e-6 {
                    Quat::IDENTITY
                } else {
                    q_raw.normalize()
                };
                let sc = Vec3::new(e.scale_xy, e.scale_xy, e.scale_z);

                let key = (e.archetype_hash, (p.x * 2.0).round() as i32, (p.y * 2.0).round() as i32, (p.z * 2.0).round() as i32);
                if !seen_exact.insert(key) {
                    continue;
                }

                if ix.slod_models.contains(&e.archetype_hash) {
                    slod_ents.push((e.archetype_hash, p, q, sc));
                } else if ix.lod_models.contains(&e.archetype_hash) || e.child_lod_dist > 0.0 {
                    lod_ents.push((e.archetype_hash, p, q, sc));
                } else {
                    hd_ents.push((e.archetype_hash, p, q, sc));
                }
            }

            // 2. Spatial grids for hierarchy suppression
            // HD grid (10m cells) to suppress overlapping LOD shells
            let mut hd_grid: HashMap<(i32, i32), Vec<Vec3>> = HashMap::new();
            for &(_arch, p, _, _) in &hd_ents {
                let k = ((p.x / 10.0).floor() as i32, (p.y / 10.0).floor() as i32);
                hd_grid.entry(k).or_default().push(p);
            }

            // 3. Clean LODs: suppress LOD if HD model exists within 8m; dedup LODs within 2m
            let mut clean_lods = Vec::new();
            let mut lod_seen_pos: HashSet<(i32, i32, i32)> = HashSet::new();
            for (arch, p, q, sc) in lod_ents {
                let lk = ((p.x / 2.0).round() as i32, (p.y / 2.0).round() as i32, (p.z / 3.0).round() as i32);
                if !lod_seen_pos.insert(lk) {
                    continue;
                }
                let gx = (p.x / 10.0).floor() as i32;
                let gy = (p.y / 10.0).floor() as i32;
                let mut has_hd = false;
                'check_hd: for dx in -1..=1 {
                    for dy in -1..=1 {
                        if let Some(pts) = hd_grid.get(&(gx + dx, gy + dy)) {
                            for &hp in pts {
                                if (hp - p).length_squared() < 64.0 { // 8m in 3D
                                    has_hd = true;
                                    break 'check_hd;
                                }
                            }
                        }
                    }
                }
                if !has_hd {
                    clean_lods.push((arch, p, q, sc));
                }
            }

            // 4. Building grid (30m cells) containing HD + clean LODs to suppress SLOD shells
            let mut building_grid: HashMap<(i32, i32), Vec<Vec3>> = HashMap::new();
            for &(_arch, p, _, _) in hd_ents.iter().chain(clean_lods.iter()) {
                let k = ((p.x / 30.0).floor() as i32, (p.y / 30.0).floor() as i32);
                building_grid.entry(k).or_default().push(p);
            }

            // 5. Clean SLODs: suppress SLOD composite shells if HD or LOD buildings exist within 35m
            let mut clean_slods = Vec::new();
            let mut slod_seen_pos: HashSet<(i32, i32)> = HashSet::new();
            for (arch, p, q, sc) in slod_ents {
                let sk = ((p.x / 6.0).round() as i32, (p.y / 6.0).round() as i32);
                if !slod_seen_pos.insert(sk) {
                    continue;
                }
                let gx = (p.x / 30.0).floor() as i32;
                let gy = (p.y / 30.0).floor() as i32;
                let mut has_building = false;
                'check_b: for dx in -2..=2 {
                    for dy in -2..=2 {
                        if let Some(pts) = building_grid.get(&(gx + dx, gy + dy)) {
                            for &bp in pts {
                                let dx_f = bp.x - p.x;
                                let dy_f = bp.y - p.y;
                                if dx_f * dx_f + dy_f * dy_f < 1225.0 { // 35m in 2D
                                    has_building = true;
                                    break 'check_b;
                                }
                            }
                        }
                    }
                }
                if !has_building {
                    clean_slods.push((arch, p, q, sc));
                }
            }

            // Populate grouped instances from final clean list
            for (arch, p, q, sc) in hd_ents.into_iter().chain(clean_lods.into_iter()).chain(clean_slods.into_iter()) {
                let entry = grouped.entry(arch).or_insert_with(|| (p, Vec::new()));
                entry.1.push((p, q, sc, [1.0; 4]));
            }

            // 6. Generate debug placeholder boxes ONLY for unmodeled entities far away from buildings (>20m)
            for e in &ix.entities {
                if ix.models.contains_key(&e.archetype_hash) || e.child_lod_dist > 0.0 {
                    continue;
                }
                let p = Vec3::from_array(e.pos);
                if !p.is_finite() || !e.scale_xy.is_finite() || !e.scale_z.is_finite() {
                    continue;
                }
                let gx = (p.x / 30.0).floor() as i32;
                let gy = (p.y / 30.0).floor() as i32;
                let mut near_building = false;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if let Some(pts) = building_grid.get(&(gx + dx, gy + dy)) {
                            for &bp in pts {
                                let dx_f = bp.x - p.x;
                                let dy_f = bp.y - p.y;
                                if dx_f * dx_f + dy_f * dy_f < 400.0 { // 20m
                                    near_building = true;
                                    break;
                                }
                            }
                        }
                        if near_building { break; }
                    }
                    if near_building { break; }
                }
                if near_building || box_insts.len() >= 2_000 {
                    continue;
                }
                let q_raw = Quat::from_xyzw(e.rot[0], e.rot[1], e.rot[2], e.rot[3]);
                let q = if !q_raw.is_finite() || q_raw.length_squared() < 1e-6 { Quat::IDENTITY } else { q_raw.normalize() };
                let h = e.archetype_hash;
                let r = 0.45 + ((h & 0xFF) as f32 / 255.0) * 0.45;
                let g = 0.45 + (((h >> 8) & 0xFF) as f32 / 255.0) * 0.45;
                let b = 0.50 + (((h >> 16) & 0xFF) as f32 / 255.0) * 0.45;
                box_insts.push((p, q, Vec3::new(3.0 * e.scale_xy, 3.0 * e.scale_xy, 3.0 * e.scale_z), [r, g, b, 1.0]));
            }

            log_world(format!("[world-gpu] Clean world ready: {} unique archetypes after HD/LOD/SLOD hierarchy deduplication", grouped.len()));
            if !box_insts.is_empty() {
                let bmesh = Mesh3D::from_parts(
                    "__placeholder_boxes__".into(),
                    vec![
                        [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
                        [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0],
                    ],
                    vec![[0.0, 0.0]; 8],
                    vec![
                        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
                    ],
                    vec![0; 12],
                    Vec::new(),
                    None,
                    String::new(),
                    1,
                    0,
                    "BOX".into(),
                );
                let _ = tx.send(WorldStreamMsg::Batch(WorldBatchInput { mesh: bmesh, instances: box_insts }));
            }

            let all_groups: Vec<(u32, u64, Vec<(Vec3, Quat, Vec3, [f32; 4])>)> = grouped
                .into_iter()
                .filter_map(|(h, (_rep, insts))| ix.models.get(&h).map(|&moff| (h, moff, insts)))
                .collect();
            // 200m spatial grid for O(cells) distance-tiered streaming up to 6 km
            const STREAM_CELL: f32 = 200.0;
            let mut stream_grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
            for (idx, (_, _, insts)) in all_groups.iter().enumerate() {
                for &(p, _, _, _) in insts {
                    let key = ((p.x / STREAM_CELL).floor() as i32, (p.y / STREAM_CELL).floor() as i32);
                    let cell_vec = stream_grid.entry(key).or_default();
                    if cell_vec.last().copied() != Some(idx) {
                        cell_vec.push(idx);
                    }
                }
            }
            for v in stream_grid.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
            log_world(format!("[world-gpu] {} unique drawable archetypes across {} 200m cells indexed for live streaming", all_groups.len(), stream_grid.len()));

            let mut loaded = std::collections::HashSet::<u32>::new();
            let mut last_gen = 0u64;
            loop {
                let (cam_p, cur_gen) = *cam_track_bg.lock().unwrap();
                let (stream_radius, vram_pressure) = {
                    let mut sc = stream_ctl_bg.lock().unwrap();
                    let evicted = std::mem::take(&mut sc.evicted_hashes);
                    for h in evicted {
                        loaded.remove(&h);
                    }
                    (sc.stream_radius, sc.vram_pressure)
                };

                if loaded.len() >= all_groups.len() && cur_gen == last_gen && !vram_pressure {
                    std::thread::sleep(std::time::Duration::from_millis(80));
                    continue;
                }
                last_gen = cur_gen;

                let max_d2 = stream_radius * stream_radius;
                let mut cells: Vec<((i32, i32), f32)> = stream_grid.keys().map(|&key| {
                    let x0 = key.0 as f32 * STREAM_CELL;
                    let y0 = key.1 as f32 * STREAM_CELL;
                    let rx = (x0 - cam_p.x).max(0.0).max(cam_p.x - (x0 + STREAM_CELL));
                    let ry = (y0 - cam_p.y).max(0.0).max(cam_p.y - (y0 + STREAM_CELL));
                    (key, rx * rx + ry * ry)
                })
                .filter(|&(_, d2)| d2 <= max_d2)
                .collect();
                cells.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

                let take_n = if vram_pressure {
                    80
                } else if loaded.is_empty() {
                    2500
                } else if stream_radius <= 3000.0 {
                    300
                } else {
                    1200
                };
                let mut seen_idx = std::collections::HashSet::new();
                let mut batch_list: Vec<(u32, u64, Vec<(Vec3, Quat, Vec3, [f32; 4])>)> = Vec::with_capacity(take_n);
                'outer: for (key, _) in cells {
                    if let Some(indices) = stream_grid.get(&key) {
                        for &idx in indices {
                            let (h, moff, ref insts) = all_groups[idx];
                            if loaded.contains(&h) || !seen_idx.insert(idx) {
                                continue;
                            }
                            batch_list.push((h, moff, insts.clone()));
                            if batch_list.len() >= take_n {
                                break 'outer;
                            }
                        }
                    }
                }
                if batch_list.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(80));
                    continue;
                }
                let _ = tx.send(WorldStreamMsg::TotalArchetypes(loaded.len() + batch_list.len()));
                let groups = Arc::new(batch_list);
                let cursor = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let workers = std::thread::available_parallelism().map(|n| n.get().clamp(4, 16)).unwrap_or(4);
                // a camera move stops the batch early and whatever was not reached is queued again
                let mut finished_count = 0usize;
                let worker_done: Vec<Vec<u32>> = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..workers)
                        .map(|_| {
                            let groups = Arc::clone(&groups);
                            let cursor = Arc::clone(&cursor);
                            let tx = tx.clone();
                            let lp = lp.clone();
                            let cam_track_bg = Arc::clone(&cam_track_bg);
                            s.spawn(move || {
                                crate::tex::set_thread_texture_cap(4096);
                                let mut local_done = Vec::new();
                                loop {
                                    if cam_track_bg.lock().unwrap().1 != last_gen {
                                        break;
                                    }
                                    let i = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if i >= groups.len() {
                                        break;
                                    }
                                    let (h, moff, ref insts) = groups[i];
                                    if let Ok(mesh) = crate::model3d::decode_rage9_model_from_rpf_lod_opt(&lp, moff, &format!("arch_{h:08x}"), 0, false) {
                                        let lo = mesh.name.to_ascii_lowercase();
                                        let is_helper = lo.contains("_col.") || lo.ends_with("_col")
                                            || lo.contains("_collision")
                                            || lo.contains("_camera_collision")
                                            || lo.contains("_player_only_col")
                                            || lo.contains("_shadow")
                                            || lo.contains("_pxy")
                                            || lo.contains("_lit_proxy")
                                            || lo.contains("_occl");
                                        if !is_helper
                                            && tx.send(WorldStreamMsg::Batch(WorldBatchInput { mesh, instances: insts.clone() })).is_err() {
                                                break;
                                            }
                                    }
                                    local_done.push(h);
                                    if tx.send(WorldStreamMsg::ArchetypeDone).is_err() {
                                        break;
                                    }
                                }
                                local_done
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
                });
                for chunk in worker_done {
                    finished_count += chunk.len();
                    loaded.extend(chunk);
                }
                if finished_count < groups.len() {
                    // let the next pass run even if the camera has stopped moving again
                    last_gen = 0;
                }
                let _ = tx.send(WorldStreamMsg::TotalArchetypes(loaded.len()));
            }
        });
    }
    let mut app = GpuWorldApp::new(cfg, rx, cam_track);
    app.stream_ctl = Some(stream_ctl);
    event_loop.run_app(&mut app).map_err(|e| format!("GPU event loop failed: {e}"))?;
    match app.failure { Some(error) => Err(error), None => Ok(()) }
}

const WORLD_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    flags: vec4<f32>,
    map_y: vec4<f32>,
    hud: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;

@group(2) @binding(0) var shadow_tex: texture_depth_2d;
@group(2) @binding(1) var shadow_samp: sampler_comparison;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) inst_col: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) inst_col: vec4<f32>,
    @location(4) shadow_coord: vec4<f32>,
};

@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = model * vec4<f32>(in.pos, 1.0);
    return u.light_vp * wp;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = (model * vec4<f32>(in.pos, 1.0)).xyz;
    let wn = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.world_n = wn;
    out.uv = in.uv;
    out.inst_col = in.inst_col;
    out.shadow_coord = u.light_vp * vec4<f32>(wp, 1.0);
    return out;
}

fn sample_shadow(sc: vec4<f32>, ndl: f32) -> f32 {
    let proj = sc.xyz / max(sc.w, 0.0001);
    let uv = proj.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    let bias = max(0.0015 * (1.0 - ndl), 0.0003);
    let texel = 1.0 / 2048.0;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            sum += textureSampleCompare(
                shadow_tex,
                shadow_samp,
                uv + vec2<f32>(f32(x), f32(y)) * texel,
                proj.z - bias
            );
        }
    }
    return sum / 9.0;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn neutral_col(c: vec3<f32>, k: f32) -> vec3<f32> {
    return mix(c, vec3<f32>(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))), k);
}

// cook-torrance pbr lighting adapted from sixng shader.wgsl
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex_c = textureSample(albedo_tex, albedo_samp, in.uv);
    if (tex_c.a < 0.4) {
        discard;
    }
    let albedo = tex_c.rgb * in.inst_col.rgb;
    let n = normalize(in.world_n);
    let l = normalize(u.sun_dir.xyz);
    let v = normalize(u.cam_pos.xyz - in.world_pos);
    let h = normalize(l + v);

    let ndl = max(dot(n, l), 0.0);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.001);

    var sh = sample_shadow(in.shadow_coord, ndl);
    let rt_on = u.flags.x > 0.5;
    if (rt_on) {
        sh = smoothstep(0.08, 0.92, sh);
    }

    let rough = 0.62;
    let metal = 0.06;
    let a2 = rough * rough * rough * rough;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let d_ggx = a2 / (3.14159 * dd * dd);
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g_smith = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    let fr = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    let spec = d_ggx * g_smith * fr / max(4.0 * ndl * ndv, 0.001);
    let kd = (1.0 - fr) * (1.0 - metal);

    let sun_elev = clamp(l.z * 2.2, 0.0, 1.0);
    let sun_col = mix(vec3<f32>(1.0, 0.48, 0.22), vec3<f32>(1.05, 0.98, 0.90), sun_elev) * (2.2 * sun_elev + 0.18);
    let direct = (kd * albedo / 3.14159 + spec) * sun_col * ndl * sh * 2.4;

    let hemi = n.z * 0.5 + 0.5;
    let amb_d = mix(neutral_col(u.sky_horizon.rgb, 0.78) * 0.55, neutral_col(u.sky_zenith.rgb, 0.78), hemi);
    let fr_v = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
    let env = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.5) * 0.45;
    let rt_bounce = select(0.0, 0.14 * (1.0 - ndv) * (0.4 + 0.6 * hemi), rt_on);

    let lit = direct + kd * albedo * amb_d + env * fr_v + amb_d * rt_bounce;

    let dist = length(u.cam_pos.xyz - in.world_pos);
    let fog = 1.0 - exp(-dist * 0.000028 * max(u.hud.w, 0.0));
    let fog_col = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.25);
    let col = mix(lit, fog_col, clamp(fog, 0.0, 0.65)) * max(u.sky_zenith.w, 0.2);
    return vec4<f32>(aces_tonemap(col), 1.0);
}

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(pts[vid], 0.9999, 1.0);
}

@fragment
fn fs_sky(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let t = clamp(1.0 - frag_pos.y / max(u.map_y.w, 1.0), 0.0, 1.0);
    let col = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, pow(t, 0.7));
    return vec4<f32>(aces_tonemap(col), 1.0);
}

struct MinimapOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_minimap(@builtin(vertex_index) vid: u32) -> MinimapOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    );
    let q = quad[vid];
    let w = max(u.map_y.z, 640.0);
    let h = max(u.map_y.w, 480.0);
    let px = 18.0 + q.x * 190.0;
    let py = h - 18.0 - 190.0 + q.y * 190.0;
    let ndc_x = (px / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (py / h) * 2.0;
    var out: MinimapOut;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = vec2<f32>(q.x, 1.0 - q.y);
    return out;
}

@fragment
fn fs_minimap(in: MinimapOut) -> @location(0) vec4<f32> {
    let c = textureSample(albedo_tex, albedo_samp, in.uv);
    let border = in.uv.x < 0.02 || in.uv.x > 0.98 || in.uv.y < 0.02 || in.uv.y > 0.98;
    if (border) {
        return vec4<f32>(0.92, 0.94, 0.98, 0.95);
    }
    let player_u = (u.cam_pos.x - u.flags.z) / max(u.flags.w, 1.0);
    let player_v = 1.0 - (u.cam_pos.y - u.map_y.x) / max(u.map_y.y, 1.0);
    let d = length(in.uv - vec2<f32>(player_u, player_v));
    if (d < 0.028) {
        return vec4<f32>(1.0, 0.66, 0.11, 1.0);
    }
    return vec4<f32>(c.rgb, 0.92);
}

struct WaterOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_water(@builtin(vertex_index) vid: u32) -> WaterOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let p = quad[vid] * 26000.0;
    let wp = vec3<f32>(p.x, p.y, -0.5);
    var out: WaterOut;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    return out;
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    let dist = length(u.cam_pos.xyz - in.world_pos);
    let wave_fade = exp(-dist * 0.0015);
    let t = u.hud.z;
    let w1 = sin(in.world_pos.x * 0.11 + t * 1.4) * cos(in.world_pos.y * 0.13 - t * 1.1);
    let w2 = sin((in.world_pos.x * 0.07 - in.world_pos.y * 0.09) - t * 0.8);
    let n = normalize(vec3<f32>((w1 + w2) * 0.035 * wave_fade, (w1 - w2) * 0.035 * wave_fade, 1.0));
    let v = normalize(u.cam_pos.xyz - in.world_pos);
    let l = normalize(u.sun_dir.xyz);
    let h = normalize(l + v);
    let fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
    let sun_elev = clamp(l.z * 2.2, 0.0, 1.0);
    let deep = vec3<f32>(0.02, 0.11, 0.19) * (0.35 + 0.65 * sun_elev);
    let sky_r = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.45);
    let glint = pow(max(dot(n, h), 0.0), 140.0) * 1.6 * sun_elev;
    let water_col = mix(deep, sky_r, clamp(fres * 0.85 + 0.08, 0.0, 0.92)) + vec3<f32>(glint);
    let fog = clamp(1.0 - exp(-dist * 0.00010), 0.0, 0.90);
    let fog_col = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.25);
    return vec4<f32>(aces_tonemap(mix(water_col, fog_col, fog)), 1.0);
}

struct ProgOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_progress(@builtin(vertex_index) vid: u32) -> ProgOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    );
    let q = quad[vid];
    let w = max(u.map_y.z, 640.0);
    let h = max(u.map_y.w, 480.0);
    let bw = min(460.0, w * 0.5);
    let bh = 14.0;
    let px = (w - bw) * 0.5 + q.x * bw;
    let py = 18.0 + q.y * bh;
    var out: ProgOut;
    out.pos = vec4<f32>((px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0, 1.0);
    out.uv = q;
    return out;
}

@fragment
fn fs_progress(in: ProgOut) -> @location(0) vec4<f32> {
    if (u.hud.y < 0.5) {
        discard;
    }
    let border = in.uv.x < 0.006 || in.uv.x > 0.994 || in.uv.y < 0.12 || in.uv.y > 0.88;
    if (border) {
        return vec4<f32>(0.85, 0.90, 0.96, 0.85);
    }
    if (in.uv.x <= clamp(u.hud.x, 0.0, 1.0)) {
        return vec4<f32>(0.22, 0.80, 0.98, 0.95);
    }
    return vec4<f32>(0.08, 0.10, 0.14, 0.80);
}
"#;

#[derive(Clone, Debug)]
struct LoadedArchetype {
    _hash: u32,
    geom_bytes: usize,
    unshared_tex_bytes: usize,
    tri_count: usize,
    instance_count: usize,
    tex_keys: Vec<usize>,
    center: Vec3,
    radius: f32,
    is_terrain: bool,
    is_road: bool,
    _is_box: bool,
}

struct GpuMeshBatch {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
    index_count: u32,
    index_format: wgpu::IndexFormat,
    inst_first: u32,
    inst_count: u32,
    tex_bind_group: wgpu::BindGroup,
    bound_center: Vec3,
    bound_radius: f32,
    // radius of one placed copy, decides draw distance and shadow casting
    obj_radius: f32,
    is_box: bool,
    is_terrain: bool,
    is_road: bool,
    archetype_hash: u32,
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    shadow_bg: wgpu::BindGroup,
    tex_bgl: wgpu::BindGroupLayout,
    tex_samp: wgpu::Sampler,
    white_bg: wgpu::BindGroup,
    land_bg: wgpu::BindGroup,
    road_bg: wgpu::BindGroup,
    sidewalk_bg: wgpu::BindGroup,
    shadow_pipeline: wgpu::RenderPipeline,
    main_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    minimap_pipeline: wgpu::RenderPipeline,
    progress_pipeline: wgpu::RenderPipeline,
    minimap_bg: wgpu::BindGroup,
    egui_renderer: egui_wgpu::Renderer,
    batches: Vec<GpuMeshBatch>,
    tri_count: usize,
    instance_count: usize,
    model_count: usize,
    map_count: usize,
    stream_done: usize,
    stream_total: usize,
    tex_bytes: usize,
    geom_bytes: usize,
    _vram_oom: bool,
    // texture bind groups by pixel buffer address with the textures kept alive behind them
    tex_uploads: HashMap<usize, wgpu::BindGroup>,
    tex_ref_counts: HashMap<usize, usize>,
    tex_sizes: HashMap<usize, usize>,
    tex_keep: HashMap<usize, Arc<crate::model3d::Texture2D>>,
    vram_budget_bytes: usize,
    is_evicting: bool,
    eviction_count: usize,
    last_cam_pos: Vec3,
    loaded_archetypes: HashMap<u32, LoadedArchetype>,
    pending_evicted: Vec<u32>,
    frame_idx: u64,
    start_instant: Instant,
    has_hw_rt: bool,
    map_min: [f32; 2],
    map_span: [f32; 2],
}

fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_depth"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgl: &wgpu::BindGroupLayout,
    samp: &wgpu::Sampler,
    tex: &Texture2D,
    max_dim: usize,
) -> Option<(wgpu::BindGroup, usize)> {
    let w0 = tex.w.max(1);
    let h0 = tex.h.max(1);
    let scale = (w0.max(h0) + max_dim - 1) / max_dim.max(1);
    let (w, h, raw) = if scale > 1 && tex.pixels.len() >= w0 * h0 {
        let nw = (w0 / scale).max(1);
        let nh = (h0 / scale).max(1);
        let mut buf = Vec::with_capacity(nw * nh * 4);
        for y in 0..nh {
            let sy = (y * scale).min(h0 - 1);
            for x in 0..nw {
                let sx = (x * scale).min(w0 - 1);
                buf.extend_from_slice(&tex.pixels[sy * w0 + sx].to_array());
            }
        }
        (nw, nh, buf)
    } else {
        let mut buf = Vec::with_capacity(tex.pixels.len() * 4);
        for c in &tex.pixels {
            buf.extend_from_slice(&c.to_array());
        }
        (w0, h0, buf)
    };
    let bytes = raw.len();
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let gpu_tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("albedo"),
            size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &raw,
    );
    if let Some(err) = pollster_block_on(device.pop_error_scope()) {
        log_world(format!("[world-gpu] Texture upload OOM ({w}x{h}, {} KiB): {err}", raw.len() / 1024));
        return None;
    }
    let view = gpu_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tex_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
        ],
    });
    Some((bg, bytes))
}

fn compute_smooth_normals(mesh: &Mesh3D) -> Vec<[f32; 3]> {
    let mut accum = vec![Vec3::ZERO; mesh.verts.len()];
    for (ti, &[a, b, c]) in mesh.tris.iter().enumerate() {
        let n = mesh.normals.get(ti).copied().unwrap_or([0.0, 0.0, 1.0]);
        let nv = Vec3::from_array(n);
        if (a as usize) < accum.len() { accum[a as usize] += nv; }
        if (b as usize) < accum.len() { accum[b as usize] += nv; }
        if (c as usize) < accum.len() { accum[c as usize] += nv; }
    }
    accum
        .into_iter()
        .map(|n| {
            let nn = n.normalize_or_zero();
            if nn.length_squared() > 0.01 { nn.to_array() } else { [0.0, 0.0, 1.0] }
        })
        .collect()
}

impl GpuState {
    fn new(window: Arc<Window>, cfg: &WorldLaunchConfig) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter found");

        let has_hw_rt = adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let mut req_features = wgpu::Features::empty();
        if has_hw_rt {
            req_features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
        let (device, queue) = pollster_block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("percystar_world_device"),
                required_features: req_features,
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_texture = create_depth_texture(&device, config.width, config.height);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sun_shadow_map"),
            size: wgpu::Extent3d { width: 2048, height: 2048, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("u_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("u_bg"),
            layout: &u_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bg"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
            ],
        });

        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tex_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world_shader"),
            source: wgpu::ShaderSource::Wgsl(WORLD_WGSL.into()),
        });

        let v_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4],
            },
        ];

        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pl"),
            bind_group_layouts: &[&u_bgl, &tex_bgl],
            push_constant_ranges: &[],
        });

        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main_pl"),
            bind_group_layouts: &[&u_bgl, &tex_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipe"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_shadow"), buffers: &v_layouts, compilation_options: Default::default() },
            fragment: None,
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let main_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &v_layouts, compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_sky"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_water"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_water"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let minimap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("minimap_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_minimap"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_minimap"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let progress_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("progress_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_progress"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_progress"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let white_tex = Texture2D::new(1, 1, vec![eframe::egui::Color32::from_rgb(190, 195, 205)]);
        let (white_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &white_tex, 256).unwrap();
        let (minimap_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &white_tex, 256).unwrap();

        let land_tex = Texture2D::new(4, 4, vec![
            eframe::egui::Color32::from_rgb(58, 88, 42), eframe::egui::Color32::from_rgb(52, 82, 38), eframe::egui::Color32::from_rgb(63, 93, 46), eframe::egui::Color32::from_rgb(55, 85, 40),
            eframe::egui::Color32::from_rgb(60, 90, 44), eframe::egui::Color32::from_rgb(50, 80, 36), eframe::egui::Color32::from_rgb(57, 87, 41), eframe::egui::Color32::from_rgb(65, 95, 48),
            eframe::egui::Color32::from_rgb(53, 83, 39), eframe::egui::Color32::from_rgb(62, 92, 45), eframe::egui::Color32::from_rgb(56, 86, 42), eframe::egui::Color32::from_rgb(51, 81, 37),
            eframe::egui::Color32::from_rgb(59, 89, 43), eframe::egui::Color32::from_rgb(54, 84, 40), eframe::egui::Color32::from_rgb(61, 91, 46), eframe::egui::Color32::from_rgb(57, 87, 41),
        ]);
        let (land_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &land_tex, 256).unwrap();

        let road_tex = Texture2D::new(4, 4, vec![
            eframe::egui::Color32::from_rgb(34, 35, 38), eframe::egui::Color32::from_rgb(38, 39, 43), eframe::egui::Color32::from_rgb(32, 33, 36), eframe::egui::Color32::from_rgb(36, 37, 41),
            eframe::egui::Color32::from_rgb(39, 40, 44), eframe::egui::Color32::from_rgb(31, 32, 35), eframe::egui::Color32::from_rgb(37, 38, 42), eframe::egui::Color32::from_rgb(35, 36, 40),
            eframe::egui::Color32::from_rgb(33, 34, 37), eframe::egui::Color32::from_rgb(37, 38, 42), eframe::egui::Color32::from_rgb(35, 36, 39), eframe::egui::Color32::from_rgb(38, 39, 43),
            eframe::egui::Color32::from_rgb(36, 37, 40), eframe::egui::Color32::from_rgb(32, 33, 36), eframe::egui::Color32::from_rgb(39, 40, 44), eframe::egui::Color32::from_rgb(34, 35, 38),
        ]);
        let (road_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &road_tex, 256).unwrap();

        let sidewalk_tex = Texture2D::new(4, 4, vec![
            eframe::egui::Color32::from_rgb(175, 170, 160), eframe::egui::Color32::from_rgb(168, 163, 153), eframe::egui::Color32::from_rgb(172, 167, 157), eframe::egui::Color32::from_rgb(180, 175, 165),
            eframe::egui::Color32::from_rgb(165, 160, 150), eframe::egui::Color32::from_rgb(178, 173, 163), eframe::egui::Color32::from_rgb(170, 165, 155), eframe::egui::Color32::from_rgb(174, 169, 159),
            eframe::egui::Color32::from_rgb(172, 167, 157), eframe::egui::Color32::from_rgb(169, 164, 154), eframe::egui::Color32::from_rgb(176, 171, 161), eframe::egui::Color32::from_rgb(167, 162, 152),
            eframe::egui::Color32::from_rgb(177, 172, 162), eframe::egui::Color32::from_rgb(171, 166, 156), eframe::egui::Color32::from_rgb(166, 161, 151), eframe::egui::Color32::from_rgb(173, 168, 158),
        ]);
        let (sidewalk_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &sidewalk_tex, 256).unwrap();

        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        let mut st = Self {
            window,
            surface,
            device,
            queue,
            config,
            depth_view,
            depth_texture,
            shadow_view,
            uniform_buf,
            uniform_bg,
            shadow_bg,
            tex_bgl,
            tex_samp,
            white_bg,
            land_bg,
            road_bg,
            sidewalk_bg,
            shadow_pipeline,
            main_pipeline,
            sky_pipeline,
            water_pipeline,
            minimap_pipeline,
            progress_pipeline,
            minimap_bg,
            egui_renderer,
            batches: Vec::new(),
            tri_count: 0,
            instance_count: 0,
            model_count: 0,
            map_count: cfg.map_count,
            stream_done: 0,
            stream_total: 0,
            tex_bytes: 0,
            geom_bytes: 0,
            _vram_oom: false,
            tex_uploads: HashMap::new(),
            tex_ref_counts: HashMap::new(),
            tex_sizes: HashMap::new(),
            tex_keep: HashMap::new(),
            vram_budget_bytes: DEFAULT_VRAM_BUDGET_BYTES,
            is_evicting: false,
            eviction_count: 0,
            last_cam_pos: Vec3::from_array(cfg.start_pos),
            loaded_archetypes: HashMap::new(),
            pending_evicted: Vec::new(),
            frame_idx: 0,
            start_instant: Instant::now(),
            has_hw_rt,
            map_min: [-13000.0, -9000.0],
            map_span: [20000.0, 20000.0],
        };

        if let Some(ref terrain) = cfg.terrain {
            st.map_min = [terrain.bmin[0], terrain.bmin[1]];
            st.map_span = [terrain.bmax[0] - terrain.bmin[0], terrain.bmax[1] - terrain.bmin[1]];
            let tmesh = terrain.build_world_terrain_mesh(4);
            if let Some(ref t) = tmesh.texture {
                if let Some((mbg, bytes)) = upload_texture(&st.device, &st.queue, &st.tex_bgl, &st.tex_samp, t, 1024) {
                    st.minimap_bg = mbg;
                    st.tex_bytes += bytes;
                }
            }
            st.add_batch(&WorldBatchInput {
                mesh: tmesh,
                instances: vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, [1.0; 4])],
            });
            st.model_count = st.model_count.saturating_sub(1);
        }

        for b in &cfg.batches {
            st.add_batch(b);
        }
        st
    }

    // splits mesh3d by tri_geom material so every ttpf texture and paint group renders cleanly
    fn add_batch(&mut self, b: &WorldBatchInput) {
        if b.mesh.verts.is_empty() || b.mesh.tris.is_empty() || b.instances.is_empty() {
            return;
        }
        let is_terrain = b.mesh.name == "mhiu_world_terrain";
        let is_road = b.mesh.name.starts_with("nav_") || b.mesh.name.contains("road") || b.mesh.name.contains("street");
        let arch_hash = archetype_hash_from_mesh_name(&b.mesh.name);
        let use_u16 = b.mesh.verts.len() <= u16::MAX as usize;
        let idx_fmt = if use_u16 { wgpu::IndexFormat::Uint16 } else { wgpu::IndexFormat::Uint32 };
        let est_geom_bytes = b.mesh.verts.len() * std::mem::size_of::<GpuVertex>()
            + b.mesh.tris.len() * if use_u16 { 6 } else { 12 }
            + b.instances.len() * std::mem::size_of::<GpuInstance>();

        // Check active VRAM budget and proactively prune if approaching capacity
        if !is_terrain && (self.geom_bytes + self.tex_bytes + est_geom_bytes > self.vram_budget_bytes) {
            let bytes_to_free = (self.geom_bytes + self.tex_bytes + est_geom_bytes)
                .saturating_sub((self.vram_budget_bytes as f32 * 0.75) as usize);
            let evicted = self.prune_distant_batches(self.last_cam_pos, bytes_to_free);
            self.pending_evicted.extend(evicted);
            if self.geom_bytes + self.tex_bytes + est_geom_bytes > self.vram_budget_bytes {
                return;
            }
        }

        let normals = compute_smooth_normals(&b.mesh);
        // placements of one model can span the whole map, sorting them by cell lets each cell be culled on its own
        const CELL: f32 = 1024.0;
        let cell_of = |p: Vec3| ((p.x / CELL).floor() as i32, (p.y / CELL).floor() as i32, (p.z / CELL).floor() as i32);
        let mut placed = b.instances.clone();
        placed.sort_by_key(|i| cell_of(i.0));
        let insts: Vec<GpuInstance> = placed
            .iter()
            .map(|&(pos, rot, scale, col)| GpuInstance::from_mat4(Mat4::from_scale_rotation_translation(scale, rot, pos), col))
            .collect();

        let mut by_geom: HashMap<u16, Vec<u32>> = HashMap::new();
        for (ti, &[i0, i1, i2]) in b.mesh.tris.iter().enumerate() {
            let g = if b.mesh.texture.is_some() { 0 } else { b.mesh.tri_geom.get(ti).copied().unwrap_or(0) };
            let v = by_geom.entry(g).or_default();
            v.push(i0);
            v.push(i1);
            v.push(i2);
        }

        let gv: Vec<GpuVertex> = b
            .mesh
            .verts
            .iter()
            .enumerate()
            .map(|(i, &pos)| GpuVertex {
                pos,
                normal: normals[i],
                uv: b.mesh.uvs.get(i).copied().unwrap_or([pos[0] * 0.15, pos[1] * 0.15]),
            })
            .collect();

        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let inst_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("m_inst"),
            contents: bytemuck::cast_slice(&insts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("m_vb"),
            contents: bytemuck::cast_slice(&gv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut ibs = Vec::with_capacity(by_geom.len());
        for (g, indices) in by_geom {
            let ib = if use_u16 {
                let idx16: Vec<u16> = indices.iter().map(|&i| i as u16).collect();
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("m_ib16"),
                    contents: bytemuck::cast_slice(&idx16),
                    usage: wgpu::BufferUsages::INDEX,
                })
            } else {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("m_ib32"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
            };
            ibs.push((g, indices.len() as u32, ib));
        }
        if let Some(err) = pollster_block_on(self.device.pop_error_scope()) {
            log_world(format!(
                "[world-gpu] GPU VRAM allocation limit reached ({} MiB geom, {} MiB tex): {err}. Pruning distant meshes to recover...",
                self.geom_bytes / (1024 * 1024),
                self.tex_bytes / (1024 * 1024)
            ));
            let evicted = self.prune_distant_batches(self.last_cam_pos, 600 * 1024 * 1024);
            self.pending_evicted.extend(evicted);
            return;
        }

        let mut lmin = Vec3::splat(f32::MAX);
        let mut lmax = Vec3::splat(f32::MIN);
        for v in &b.mesh.verts {
            let p = Vec3::from_array(*v);
            lmin = lmin.min(p);
            lmax = lmax.max(p);
        }
        let local_center = (lmin + lmax) * 0.5;
        let local_rad = ((lmax - lmin) * 0.5).length();
        // one entry per run of placements in the same cell: first instance, count, bounds, largest copy
        let mut cells: Vec<(u32, u32, Vec3, f32, f32)> = Vec::new();
        let mut start = 0usize;
        while start < placed.len() {
            let key = cell_of(placed[start].0);
            let mut end = start;
            let mut wmin = Vec3::splat(f32::MAX);
            let mut wmax = Vec3::splat(f32::MIN);
            let mut obj_radius = 0.0f32;
            while end < placed.len() && cell_of(placed[end].0) == key {
                let (pos, rot, scale, _) = placed[end];
                let c = pos + rot * (local_center * scale);
                let r = local_rad * scale.abs().max_element();
                wmin = wmin.min(c - Vec3::splat(r));
                wmax = wmax.max(c + Vec3::splat(r));
                obj_radius = obj_radius.max(r);
                end += 1;
            }
            cells.push((start as u32, (end - start) as u32, (wmin + wmax) * 0.5, ((wmax - wmin) * 0.5).length().max(1.0), obj_radius.max(1.0)));
            start = end;
        }

        let mut arch_tex_keys = Vec::new();
        let mut arch_unshared_tex_bytes = 0usize;

        for (g, index_count, ib) in ibs {
            let shared = if b.mesh.texture.is_some() { None } else { b.mesh.geom_materials.get(g as usize).and_then(|m| m.texture.clone()) };
            let tex = b.mesh.texture.as_ref().or(shared.as_deref());
            let tex_bg = match tex {
                Some(t) if self.geom_bytes + self.tex_bytes < self.vram_budget_bytes => {
                    let key = t.pixels.as_ptr() as usize;
                    if shared.is_some() {
                        arch_tex_keys.push(key);
                        match self.tex_uploads.get(&key) {
                            Some(bg) => {
                                *self.tex_ref_counts.entry(key).or_default() += 1;
                                bg.clone()
                            }
                            None => {
                                match upload_texture(&self.device, &self.queue, &self.tex_bgl, &self.tex_samp, t, 4096) {
                                    Some((bg, bytes)) => {
                                        self.tex_bytes += bytes;
                                        self.tex_sizes.insert(key, bytes);
                                        *self.tex_ref_counts.entry(key).or_default() += 1;
                                        if let Some(keep) = shared.clone() {
                                            self.tex_keep.insert(key, keep);
                                            self.tex_uploads.insert(key, bg.clone());
                                        }
                                        bg
                                    }
                                    None => {
                                        self.white_bg.clone()
                                    }
                                }
                            }
                        }
                    } else {
                        match upload_texture(&self.device, &self.queue, &self.tex_bgl, &self.tex_samp, t, 4096) {
                            Some((bg, bytes)) => {
                                self.tex_bytes += bytes;
                                arch_unshared_tex_bytes += bytes;
                                bg
                            }
                            None => {
                                self.white_bg.clone()
                            }
                        }
                    }
                }
                _ => {
                    let m_name = b.mesh.name.to_ascii_lowercase();
                    if g == 2 || m_name.contains("road") || m_name.contains("street") || m_name.contains("highway") {
                        self.road_bg.clone()
                    } else if g == 3 || m_name.contains("sidewalk") || m_name.contains("pavement") || m_name.contains("curb") {
                        self.sidewalk_bg.clone()
                    } else if g == 0 && (m_name.starts_with("nav_") || is_terrain) {
                        self.land_bg.clone()
                    } else {
                        self.white_bg.clone()
                    }
                }
            };

            for &(inst_first, inst_count, bound_center, bound_radius, obj_radius) in &cells {
                self.batches.push(GpuMeshBatch {
                    vb: vb.clone(),
                    ib: ib.clone(),
                    inst_buf: inst_buf.clone(),
                    index_count,
                    index_format: idx_fmt,
                    inst_first,
                    inst_count,
                    tex_bind_group: tex_bg.clone(),
                    bound_center,
                    bound_radius,
                    obj_radius,
                    is_box: b.mesh.name == "__placeholder_boxes__",
                    is_terrain,
                    is_road,
                    archetype_hash: arch_hash,
                });
            }
        }

        let (arch_center, arch_radius) = if cells.is_empty() {
            (Vec3::ZERO, 100.0)
        } else {
            let mut min_p = Vec3::splat(f32::MAX);
            let mut max_p = Vec3::splat(f32::MIN);
            for &(_, _, c, r, _) in &cells {
                min_p = min_p.min(c - Vec3::splat(r));
                max_p = max_p.max(c + Vec3::splat(r));
            }
            let c = (min_p + max_p) * 0.5;
            let r = ((max_p - min_p) * 0.5).length().max(1.0);
            (c, r)
        };
        arch_tex_keys.sort_unstable();
        arch_tex_keys.dedup();
        self.loaded_archetypes.insert(arch_hash, LoadedArchetype {
            _hash: arch_hash,
            geom_bytes: est_geom_bytes,
            unshared_tex_bytes: arch_unshared_tex_bytes,
            tri_count: b.mesh.tris.len() * insts.len(),
            instance_count: insts.len(),
            tex_keys: arch_tex_keys,
            center: arch_center,
            radius: arch_radius,
            is_terrain,
            is_road,
            _is_box: b.mesh.name == "__placeholder_boxes__",
        });

        self.geom_bytes += est_geom_bytes;
        self.tri_count += b.mesh.tris.len() * insts.len();
        self.instance_count += insts.len();
        self.model_count += 1;
    }

    pub fn prune_distant_batches(&mut self, cam_pos: Vec3, bytes_to_free: usize) -> Vec<u32> {
        if bytes_to_free == 0 || self.loaded_archetypes.is_empty() {
            return Vec::new();
        }
        self.is_evicting = true;
        const MIN_EVICT_DIST: f32 = 450.0; // preserve meshes within close range (300m - 500m)

        let mut candidates: Vec<(u32, f32, usize)> = self.loaded_archetypes
            .iter()
            .filter(|(_, arch)| !arch.is_terrain && !arch.is_road)
            .map(|(&h, arch)| {
                let dist = ((arch.center - cam_pos).length() - arch.radius).max(0.0);
                let est_bytes = arch.geom_bytes + arch.unshared_tex_bytes;
                (h, dist, est_bytes)
            })
            .filter(|&(_, dist, _)| dist >= MIN_EVICT_DIST)
            .collect();

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut evicted = Vec::new();
        let mut freed_bytes = 0usize;
        let mut hashes_to_remove = HashSet::new();

        for (h, _dist, est_bytes) in candidates {
            hashes_to_remove.insert(h);
            evicted.push(h);
            freed_bytes += est_bytes;
            if freed_bytes >= bytes_to_free {
                break;
            }
        }

        if evicted.is_empty() {
            self.is_evicting = false;
            return Vec::new();
        }

        self.batches.retain(|b| !hashes_to_remove.contains(&b.archetype_hash));

        for &h in &evicted {
            if let Some(arch) = self.loaded_archetypes.remove(&h) {
                self.geom_bytes = self.geom_bytes.saturating_sub(arch.geom_bytes);
                self.tex_bytes = self.tex_bytes.saturating_sub(arch.unshared_tex_bytes);
                self.tri_count = self.tri_count.saturating_sub(arch.tri_count);
                self.instance_count = self.instance_count.saturating_sub(arch.instance_count);
                self.model_count = self.model_count.saturating_sub(1);

                for key in arch.tex_keys {
                    let should_remove = if let Some(rc) = self.tex_ref_counts.get_mut(&key) {
                        *rc = rc.saturating_sub(1);
                        *rc == 0
                    } else {
                        false
                    };
                    if should_remove {
                        self.tex_ref_counts.remove(&key);
                        self.tex_uploads.remove(&key);
                        self.tex_keep.remove(&key);
                        if let Some(tbytes) = self.tex_sizes.remove(&key) {
                            self.tex_bytes = self.tex_bytes.saturating_sub(tbytes);
                        }
                    }
                }
            }
        }

        self.eviction_count += evicted.len();
        log_world(format!(
            "[world-gpu] VRAM budget: pruned {} distant archetypes, freed ~{} MiB (active VRAM: {} MiB geom, {} MiB tex)",
            evicted.len(),
            freed_bytes / (1024 * 1024),
            self.geom_bytes / (1024 * 1024),
            self.tex_bytes / (1024 * 1024)
        ));
        self.is_evicting = false;
        evicted
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture = create_depth_texture(&self.device, w, h);
        self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }
}

struct GpuWorldApp {
    cfg: Option<WorldLaunchConfig>,
    terrain_ref: Option<Arc<WorldMapData>>,
    stream_rx: Receiver<WorldStreamMsg>,
    cam_track: Arc<std::sync::Mutex<(Vec3, u64)>>,
    stream_ctl: Option<Arc<std::sync::Mutex<StreamThrottleState>>>,
    last_stream_cam: Vec3,
    gpu: Option<GpuState>,
    egui_ctx: eframe::egui::Context,
    egui_events: Vec<eframe::egui::Event>,
    pointer_pos: eframe::egui::Pos2,
    cam_pos: Vec3,
    vel_z: f32,
    yaw: f32,
    pitch: f32,
    time_h: f32,
    auto_sun: bool,
    rt_on: bool,
    walk_mode: bool,
    menu_open: bool,
    show_models: bool,
    show_boxes: bool,
    show_water: bool,
    draw_radius: f32,
    fog_density: f32,
    exposure: f32,
    keys: [bool; 9], // W A S D Q E Shift Space Ctrl
    dragging: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    last_frame: Instant,
    last_title: Instant,
    smoke_frames: Option<u32>,
    ready: bool,
    failure: Option<String>,
    show_console: bool,
    logs: Vec<String>,
}

impl GpuWorldApp {
    fn new(cfg: WorldLaunchConfig, stream_rx: Receiver<WorldStreamMsg>, cam_track: Arc<std::sync::Mutex<(Vec3, u64)>>) -> Self {
        Self {
            cam_pos: Vec3::from_array(cfg.start_pos),
            last_stream_cam: Vec3::from_array(cfg.start_pos),
            terrain_ref: cfg.terrain.clone(),
            cfg: Some(cfg),
            stream_rx,
            cam_track,
            stream_ctl: None,
            gpu: None,
            egui_ctx: eframe::egui::Context::default(),
            egui_events: Vec::new(),
            pointer_pos: eframe::egui::Pos2::ZERO,
            vel_z: 0.0,
            yaw: 0.65,
            pitch: -0.08,
            time_h: 13.50,
            auto_sun: false,
            rt_on: true,
            walk_mode: false,
            menu_open: true,
            show_console: false,
            logs: Vec::new(),
            show_models: true,
            show_boxes: false,
            show_water: true,
            draw_radius: 35000.0,
            fog_density: 0.70,
            exposure: 1.0,
            keys: [false; 9],
            dragging: false,
            last_cursor: None,
            last_frame: Instant::now(),
            last_title: Instant::now(),
            smoke_frames: None,
            ready: false,
            failure: None,
        }
    }

    fn sample_terrain_z(&self, x: f32, y: f32) -> f32 {
        let Some(ref t) = self.terrain_ref else { return 0.0 };
        let u = ((x - t.bmin[0]) / (t.bmax[0] - t.bmin[0]).max(1.0)).clamp(0.0, 1.0);
        let v = ((y - t.bmin[1]) / (t.bmax[1] - t.bmin[1]).max(1.0)).clamp(0.0, 1.0);
        let ix = ((u * (t.w.saturating_sub(1)) as f32) as usize).min(t.w - 1);
        let iy = ((v * (t.h.saturating_sub(1)) as f32) as usize).min(t.h - 1);
        let h = t.heights[iy * t.w + ix] as f32 / 255.0;
        t.bmin[2] + h * (t.bmax[2] - t.bmin[2])
    }
}

impl ApplicationHandler for GpuWorldApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let icon = image::load_from_memory(include_bytes!("../assets/logo.png")).ok().and_then(|image| {
                let image = image.into_rgba8();
                let (w, h) = image.dimensions();
                Icon::from_rgba(image.into_raw(), w, h).ok()
            });
        let attrs = Window::default_attributes()
            .with_title("PercyStar 3D World (wgpu)")
            .with_window_icon(icon.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 860.0));
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_taskbar_icon(icon)
        };
        let win = match event_loop.create_window(attrs) {
            Ok(win) => win,
            Err(error) => {
                self.failure = Some(format!("Could not create GPU World window: {error}"));
                event_loop.exit();
                return;
            }
        };
        let win = Arc::new(win);
        let cfg = self.cfg.take().unwrap_or(WorldLaunchConfig {
            terrain: None,
            batches: Vec::new(),
            levels_rpf: None,
            map_count: 0,
            start_pos: [0.0, 0.0, 120.0],
        });
        self.gpu = Some(GpuState::new(win, &cfg));
        self.gpu.as_ref().unwrap().window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                if let Some(g) = &mut self.gpu {
                    g.resize(sz.width, sz.height);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let egui_btn = match button {
                    MouseButton::Left => Some(eframe::egui::PointerButton::Primary),
                    MouseButton::Right => Some(eframe::egui::PointerButton::Secondary),
                    MouseButton::Middle => Some(eframe::egui::PointerButton::Middle),
                    _ => None,
                };
                if let Some(eb) = egui_btn {
                    self.egui_events.push(eframe::egui::Event::PointerButton {
                        pos: self.pointer_pos,
                        button: eb,
                        pressed,
                        modifiers: Default::default(),
                    });
                }
                let over_ui = self.menu_open && self.egui_ctx.wants_pointer_input();
                if matches!(button, MouseButton::Left | MouseButton::Right) && (!pressed || !over_ui) {
                    self.dragging = pressed;
                    if !self.dragging {
                        self.last_cursor = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = self.gpu.as_ref().map(|g| g.window.scale_factor() as f32).unwrap_or(1.0);
                self.pointer_pos = eframe::egui::pos2(position.x as f32 / scale, position.y as f32 / scale);
                self.egui_events.push(eframe::egui::Event::PointerMoved(self.pointer_pos));
                if self.dragging {
                    if let Some(prev) = self.last_cursor {
                        let dx = (position.x - prev.x) as f32;
                        let dy = (position.y - prev.y) as f32;
                        self.yaw -= dx * 0.004;
                        self.pitch = (self.pitch - dy * 0.004).clamp(-1.45, 1.45);
                    }
                    self.last_cursor = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x * 24.0, y * 24.0),
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
                };
                self.egui_events.push(eframe::egui::Event::MouseWheel {
                    unit: eframe::egui::MouseWheelUnit::Point,
                    delta: eframe::egui::vec2(dx, dy),
                    modifiers: Default::default(),
                });
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys[0] = down,
                        KeyCode::KeyA => self.keys[1] = down,
                        KeyCode::KeyS => self.keys[2] = down,
                        KeyCode::KeyD => self.keys[3] = down,
                        KeyCode::KeyQ | KeyCode::KeyC => self.keys[4] = down,
                        KeyCode::KeyE => self.keys[5] = down,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys[6] = down,
                        KeyCode::Space => self.keys[7] = down,
                        KeyCode::ControlLeft | KeyCode::ControlRight => self.keys[8] = down,
                        KeyCode::Home | KeyCode::F1 if down => self.menu_open = !self.menu_open,
                        KeyCode::Backquote | KeyCode::KeyL if down => self.show_console = !self.show_console,
                        KeyCode::KeyB if down => self.show_boxes = !self.show_boxes,
                        KeyCode::KeyV if down => self.show_water = !self.show_water,
                        KeyCode::F6 if down => self.auto_sun = !self.auto_sun,
                        KeyCode::KeyR if down => self.rt_on = !self.rt_on,
                        KeyCode::KeyG if down => {
                            self.walk_mode = !self.walk_mode;
                            self.vel_z = 0.0;
                        }
                        KeyCode::BracketRight if down => self.time_h = (self.time_h + 0.5) % 24.0,
                        KeyCode::BracketLeft if down => self.time_h = (self.time_h + 23.5) % 24.0,
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                if let Ok(mut pending) = WORLD_LOGS.lock() {
                    if !pending.is_empty() {
                        self.logs.extend(pending.drain(..));
                        if self.logs.len() > 400 {
                            let drop_cnt = self.logs.len() - 400;
                            self.logs.drain(0..drop_cnt);
                        }
                    }
                }

                if let Some(g) = &mut self.gpu {
                    g.last_cam_pos = self.cam_pos;
                    let upload_start = Instant::now();
                    for _ in 0..64 {
                        if upload_start.elapsed().as_micros() >= 12000 {
                            break;
                        }
                        let Ok(msg) = self.stream_rx.try_recv() else { break };
                        match msg {
                            WorldStreamMsg::Batch(batch) => g.add_batch(&batch),
                            WorldStreamMsg::MapCount(mc) => g.map_count = mc,
                            WorldStreamMsg::TotalArchetypes(t) => g.stream_total = t,
                            WorldStreamMsg::ArchetypeDone => g.stream_done += 1,
                        }
                    }

                    // Dynamic VRAM budget check & distant batch pruning
                    let total_vram = g.geom_bytes + g.tex_bytes;
                    let (target_radius, vram_pressure, _) = compute_throttle_tier(total_vram, g.vram_budget_bytes);

                    let mut evicted = Vec::new();
                    let ratio = total_vram as f32 / g.vram_budget_bytes as f32;
                    if ratio >= 0.88 || total_vram >= 5_500 * 1024 * 1024 {
                        let safe_target = (g.vram_budget_bytes as f32 * 0.75) as usize;
                        if total_vram > safe_target {
                            let bytes_to_free = total_vram - safe_target;
                            evicted = g.prune_distant_batches(self.cam_pos, bytes_to_free);
                        }
                    }
                    evicted.extend(g.pending_evicted.drain(..));
                    if !evicted.is_empty() {
                        g.stream_done = g.stream_done.saturating_sub(evicted.len());
                        if let Some(ref ctl) = self.stream_ctl {
                            let mut sc = ctl.lock().unwrap();
                            sc.evicted_hashes.extend(evicted);
                        }
                        if let Ok(mut ct) = self.cam_track.lock() {
                            ct.1 += 1;
                        }
                    }

                    if let Some(ref ctl) = self.stream_ctl {
                        let mut sc = ctl.lock().unwrap();
                        let rad_diff = (sc.stream_radius - target_radius).abs();
                        if rad_diff > 25.0 || sc.vram_pressure != vram_pressure {
                            sc.stream_radius = target_radius;
                            sc.vram_pressure = vram_pressure;
                            if let Ok(mut ct) = self.cam_track.lock() {
                                ct.1 += 1;
                            }
                        }
                    }
                }

                if self.auto_sun {
                    self.time_h = (self.time_h + dt * 0.35) % 24.0;
                }
                let fwd = Vec3::new(self.yaw.cos() * self.pitch.cos(), self.yaw.sin() * self.pitch.cos(), self.pitch.sin()).normalize();
                let fwd_flat = Vec3::new(self.yaw.cos(), self.yaw.sin(), 0.0).normalize();
                let right = fwd.cross(Vec3::Z).normalize_or(Vec3::Y);
                let move_fwd = if self.walk_mode { fwd_flat } else { fwd };
                let speed = if self.walk_mode {
                    if self.keys[6] { 45.0 } else if self.keys[8] { 4.0 } else { 14.0 }
                } else if self.keys[6] {
                    650.0
                } else if self.keys[8] {
                    28.0
                } else {
                    140.0
                } * dt;
                if self.keys[0] { self.cam_pos += move_fwd * speed; }
                if self.keys[2] { self.cam_pos -= move_fwd * speed; }
                if self.keys[3] { self.cam_pos += right * speed; }
                if self.keys[1] { self.cam_pos -= right * speed; }
                if self.walk_mode {
                    let ground_z = self.sample_terrain_z(self.cam_pos.x, self.cam_pos.y) + 1.8;
                    self.vel_z -= 18.0 * dt;
                    self.cam_pos.z += self.vel_z * dt;
                    if self.cam_pos.z <= ground_z {
                        self.cam_pos.z = ground_z;
                        self.vel_z = if self.keys[7] { 8.5 } else { 0.0 };
                    }
                } else {
                    if self.keys[5] { self.cam_pos.z += speed; }
                    if self.keys[4] { self.cam_pos.z -= speed; }
                }

                if self.cam_pos.distance_squared(self.last_stream_cam) > 60.0 * 60.0 {
                    self.last_stream_cam = self.cam_pos;
                    if let Ok(mut ct) = self.cam_track.lock() {
                        ct.0 = self.cam_pos;
                        ct.1 += 1;
                    }
                }

                let Some(g) = &mut self.gpu else { return };
                let az = (self.time_h / 24.0) * std::f32::consts::TAU - 1.2;
                let el = ((self.time_h - 6.0) / 12.0 * std::f32::consts::PI).sin() * 1.15;
                let sun_dir = Vec3::new(az.cos() * el.cos(), az.sin() * el.cos(), el.sin()).normalize();

                if now - self.last_title >= std::time::Duration::from_millis(120) {
                    self.last_title = now;
                    let rt_lbl = if !self.rt_on { "OFF" } else if g.has_hw_rt { "HW-RT" } else { "SS-RT/GI" };
                    let mode_lbl = if self.walk_mode { "WALK(G)" } else { "FLY(G)" };
                    let total_vram_gb = (g.geom_bytes + g.tex_bytes) as f32 / (1024.0 * 1024.0 * 1024.0);
                    let ceiling_gb = g.vram_budget_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
                    let vram_pct = ((g.geom_bytes + g.tex_bytes) as f32 / g.vram_budget_bytes as f32 * 100.0).clamp(0.0, 100.0);
                    let (_, _, throttle_lbl) = compute_throttle_tier(g.geom_bytes + g.tex_bytes, g.vram_budget_bytes);
                    let load_lbl = if g.stream_total > 0 && g.stream_done < g.stream_total {
                        format!(" | Load {}/{} ({:.0}%)", g.stream_done, g.stream_total, (g.stream_done as f32 / g.stream_total as f32) * 100.0)
                    } else if g.stream_total > 0 {
                        " | World Ready".to_string()
                    } else {
                        " | Indexing...".to_string()
                    };
                    g.window.set_title(&format!(
                        "PercyStar 3D World | pos {:.0} {:.0} {:.0} | {} | Sun az {:.1} el {:+.1} (F6) | Time {:.2}h | {} maps | {} models | {} instances | {:.2} M tris | VRAM {:.2}/{:.1}GB ({:.0}%) [{}] | RT: {}{}",
                        self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, mode_lbl,
                        az.to_degrees().rem_euclid(360.0), el.to_degrees(),
                        self.time_h, g.map_count, g.model_count, g.instance_count,
                        g.tri_count as f64 / 1e6,
                        total_vram_gb, ceiling_gb, vram_pct, throttle_lbl,
                        rt_lbl, load_lbl
                    ));
                }

                let aspect = g.config.width as f32 / g.config.height.max(1) as f32;
                let proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, 45000.0);
                let view = Mat4::look_to_rh(self.cam_pos, fwd, Vec3::Z);
                let view_proj = proj * view;

                let light_target = self.cam_pos + fwd * 300.0;
                let light_eye = light_target + sun_dir * 3500.0;
                let light_view = Mat4::look_at_rh(light_eye, light_target, Vec3::Z);
                let light_proj = Mat4::orthographic_rh(-1800.0, 1800.0, -1800.0, 1800.0, 50.0, 9000.0);
                let light_vp = light_proj * light_view;

                let day_fac = (sun_dir.z * 2.5).clamp(0.0, 1.0);
                let sky_zenith = [0.06 + day_fac * 0.20, 0.09 + day_fac * 0.42, 0.16 + day_fac * 0.72, self.exposure];
                let sky_horizon = [0.14 + day_fac * 0.54, 0.16 + day_fac * 0.62, 0.24 + day_fac * 0.68, 1.0];
                let load_prog = if g.stream_total > 0 { (g.stream_done as f32 / g.stream_total as f32).clamp(0.0, 1.0) } else { 0.02 };
                let load_active = if g.stream_total == 0 || g.stream_done < g.stream_total { 1.0 } else { 0.0 };

                let uni = FrameUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    light_vp: light_vp.to_cols_array_2d(),
                    cam_pos: [self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, 1.0],
                    sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                    sky_zenith,
                    sky_horizon,
                    flags: [if self.rt_on { 1.0 } else { 0.0 }, self.time_h, g.map_min[0], g.map_span[0]],
                    map_y: [g.map_min[1], g.map_span[1], g.config.width as f32, g.config.height as f32],
                    hud: [load_prog, load_active, g.start_instant.elapsed().as_secs_f32(), self.fog_density],
                };
                g.queue.write_buffer(&g.uniform_buf, 0, bytemuck::bytes_of(&uni));

                let Ok(frame) = g.surface.get_current_texture() else { return };
                let view_tex = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world_enc") });
                g.frame_idx = g.frame_idx.wrapping_add(1);

                if g.frame_idx % 2 == 1 {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &g.shadow_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&g.shadow_pipeline);
                    rp.set_bind_group(0, &g.uniform_bg, &[]);
                    for b in &g.batches {
                        if b.is_road || b.is_terrain || b.obj_radius < 3.0 {
                            continue;
                        }
                        if b.bound_radius < 5000.0 && (b.bound_center - light_target).length_squared() > (1200.0 + b.bound_radius).powi(2) {
                            continue;
                        }
                        rp.set_bind_group(1, &b.tex_bind_group, &[]);
                        rp.set_vertex_buffer(0, b.vb.slice(..));
                        rp.set_vertex_buffer(1, b.inst_buf.slice(..));
                        rp.set_index_buffer(b.ib.slice(..), b.index_format);
                        rp.draw_indexed(0..b.index_count, 0, b.inst_first..b.inst_first + b.inst_count);
                    }
                }

                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_tex,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &g.depth_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&g.sky_pipeline);
                    rp.set_bind_group(0, &g.uniform_bg, &[]);
                    rp.set_bind_group(1, &g.minimap_bg, &[]);
                    rp.set_bind_group(2, &g.shadow_bg, &[]);
                    rp.draw(0..3, 0..1);

                    // Smoke checks must not mistake the ocean plane for loaded geometry.
                    if self.smoke_frames.is_none() && self.show_water {
                        rp.set_pipeline(&g.water_pipeline);
                        rp.draw(0..6, 0..1);
                    }

                    rp.set_pipeline(&g.main_pipeline);
                    let mut _drawn = 0usize;
                    for b in &g.batches {
                        if b.is_box && !self.show_boxes {
                            continue;
                        }
                        if !b.is_box && !b.is_terrain && !b.is_road && !self.show_models {
                            continue;
                        }
                        if !b.is_terrain && b.bound_radius < 5000.0 {
                            let to_b = b.bound_center - self.cam_pos;
                            if to_b.dot(fwd) < -(b.bound_radius + 120.0) {
                                continue;
                            }
                            let d2 = to_b.length_squared();
                            let max_d = if b.obj_radius < 1.5 {
                                1200.0f32.min(self.draw_radius)
                            } else {
                                self.draw_radius
                            };
                            if d2 > (max_d + b.bound_radius).powi(2) {
                                continue;
                            }
                            _drawn += 1;
                        }
                        rp.set_bind_group(1, &b.tex_bind_group, &[]);
                        rp.set_vertex_buffer(0, b.vb.slice(..));
                        rp.set_vertex_buffer(1, b.inst_buf.slice(..));
                        rp.set_index_buffer(b.ib.slice(..), b.index_format);
                        rp.draw_indexed(0..b.index_count, 0, b.inst_first..b.inst_first + b.inst_count);
                    }

                    rp.set_pipeline(&g.minimap_pipeline);
                    rp.set_bind_group(1, &g.minimap_bg, &[]);
                    rp.draw(0..6, 0..1);

                    rp.set_pipeline(&g.progress_pipeline);
                    rp.draw(0..6, 0..1);
                }

                if self.menu_open {
                    let scale = g.window.scale_factor() as f32;
                    let screen_desc = egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [g.config.width, g.config.height],
                        pixels_per_point: scale,
                    };
                    let raw_in = eframe::egui::RawInput {
                        screen_rect: Some(eframe::egui::Rect::from_min_size(
                            eframe::egui::Pos2::ZERO,
                            eframe::egui::vec2(g.config.width as f32 / scale, g.config.height as f32 / scale),
                        )),
                        events: std::mem::take(&mut self.egui_events),
                        ..Default::default()
                    };
                    let full_out = self.egui_ctx.run(raw_in, |ctx| {
                        let mut vis = eframe::egui::Visuals::dark();
                        vis.window_fill = eframe::egui::Color32::from_rgba_premultiplied(20, 13, 31, 238);
                        vis.panel_fill = eframe::egui::Color32::from_rgb(20, 13, 31);
                        vis.window_stroke = eframe::egui::Stroke::new(1.2_f32, eframe::egui::Color32::from_rgb(255, 72, 165));
                        vis.widgets.inactive.bg_fill = eframe::egui::Color32::from_rgb(36, 22, 54);
                        vis.widgets.hovered.bg_fill = eframe::egui::Color32::from_rgb(56, 32, 82);
                        vis.widgets.active.bg_fill = eframe::egui::Color32::from_rgb(255, 168, 28);
                        vis.selection.bg_fill = eframe::egui::Color32::from_rgb(255, 168, 28);
                        ctx.set_visuals(vis);

                        eframe::egui::Window::new(
                            eframe::egui::RichText::new("PERCYSTAR · WORLD & RESHADE (Home / F1)")
                                .strong()
                                .color(eframe::egui::Color32::from_rgb(255, 168, 28)),
                        )
                        .default_pos([g.config.width as f32 / scale - 370.0, 18.0])
                        .default_width(350.0)
                        .resizable(false)
                        .show(ctx, |ui| {
                            let total_vram_bytes = g.geom_bytes + g.tex_bytes;
                            let total_vram_gb = total_vram_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
                            let ceiling_gb = g.vram_budget_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
                            let vram_pct = (total_vram_bytes as f32 / g.vram_budget_bytes as f32 * 100.0).clamp(0.0, 100.0);
                            let stream_rad = self.stream_ctl.as_ref().map(|c| c.lock().unwrap().stream_radius).unwrap_or(35000.0);
                            let (_, _, throttle_lbl) = compute_throttle_tier(total_vram_bytes, g.vram_budget_bytes);
                            let status_display = if g.is_evicting {
                                "EVICTING (Pruning Distant Meshes)"
                            } else {
                                throttle_lbl
                            };
                            let status_color = if g.is_evicting || vram_pct >= 85.0 {
                                eframe::egui::Color32::from_rgb(255, 75, 75)
                            } else if vram_pct >= 75.0 {
                                eframe::egui::Color32::from_rgb(255, 185, 45)
                            } else if vram_pct >= 55.0 {
                                eframe::egui::Color32::from_rgb(230, 220, 70)
                            } else {
                                eframe::egui::Color32::from_rgb(85, 235, 145)
                            };

                            ui.horizontal(|ui| {
                                ui.label(
                                    eframe::egui::RichText::new(format!(
                                        "VRAM: {:.2} / {:.1} GB ({:.0}%)",
                                        total_vram_gb, ceiling_gb, vram_pct
                                    ))
                                    .strong()
                                    .size(11.5)
                                    .color(if vram_pct >= 80.0 {
                                        eframe::egui::Color32::from_rgb(255, 110, 110)
                                    } else {
                                        eframe::egui::Color32::from_rgb(95, 215, 255)
                                    }),
                                );
                                ui.label(
                                    eframe::egui::RichText::new(format!("• [{status_display}]"))
                                        .strong()
                                        .size(11.0)
                                        .color(status_color),
                                );
                            });
                            ui.label(
                                eframe::egui::RichText::new(format!(
                                    "  ↳ Geom {}MB · Tex {}MB · Stream {:.0}m",
                                    g.geom_bytes / (1024 * 1024),
                                    g.tex_bytes / (1024 * 1024),
                                    stream_rad,
                                ))
                                .size(10.5)
                                .color(eframe::egui::Color32::from_rgb(180, 175, 210)),
                            );
                            ui.label(
                                eframe::egui::RichText::new(format!(
                                    "{} maps · {}/{} models · {} inst · {:.2}M tris",
                                    g.map_count, g.model_count, g.stream_total, g.instance_count,
                                    g.tri_count as f64 / 1e6
                                ))
                                .size(10.5)
                                .color(eframe::egui::Color32::from_rgb(150, 150, 180)),
                            );
                            ui.horizontal(|ui| {
                                ui.label("VRAM Budget:");
                                for (lbl, bytes) in [
                                    ("4.0 GB", 4000 * 1024 * 1024),
                                    ("5.5 GB", 5500 * 1024 * 1024),
                                    ("5.8 GB (Def)", 5800 * 1024 * 1024),
                                    ("7.5 GB", 7500 * 1024 * 1024),
                                ] {
                                    if ui.selectable_label(g.vram_budget_bytes == bytes, lbl).clicked() {
                                        g.vram_budget_bytes = bytes;
                                    }
                                }
                            });
                            ui.separator();
                            ui.add(eframe::egui::Slider::new(&mut self.time_h, 0.0..=24.0).text("Time of Day (h)"));
                            ui.horizontal(|ui| {
                                for (lbl, h) in [("Dawn", 6.2), ("Noon", 12.5), ("Golden", 18.0), ("Night", 23.0)] {
                                    if ui.small_button(lbl).clicked() {
                                        self.time_h = h;
                                        self.auto_sun = false;
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Radius:");
                                for (lbl, r) in [("1.5km", 1500.0), ("3km", 3000.0), ("6km", 6000.0), ("16km", 16000.0), ("35km (Ultra)", 35000.0)] {
                                    if ui.selectable_label((self.draw_radius - r).abs() < 1.0, lbl).clicked() {
                                        self.draw_radius = r;
                                    }
                                }
                            });
                            ui.add(eframe::egui::Slider::new(&mut self.fog_density, 0.0..=2.0).text("Fog Density"));
                            ui.add(eframe::egui::Slider::new(&mut self.exposure, 0.3..=2.2).text("Exposure"));
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.auto_sun, "Auto Sun (F6)");
                                ui.checkbox(&mut self.rt_on, "RT Shadows (R)");
                                ui.checkbox(&mut self.show_water, "Water (V)");
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.show_models, "Real Models");
                                ui.checkbox(&mut self.show_boxes, "Boxes (B)");
                                ui.checkbox(&mut self.walk_mode, "Walk (G)");
                                ui.checkbox(&mut self.show_console, "Console (~ / L)");
                            });
                            ui.horizontal(|ui| {
                                let btn = eframe::egui::Button::new(
                                    eframe::egui::RichText::new("Go to Downtown Brickell")
                                        .strong()
                                        .color(eframe::egui::Color32::BLACK),
                                )
                                .fill(eframe::egui::Color32::from_rgb(255, 168, 28));
                                if ui.add(btn).clicked() {
                                    self.cam_pos = Vec3::new(-127.276, -1855.924, 24.0);
                                    self.yaw = 0.65;
                                    self.pitch = -0.08;
                                }
                                ui.label(
                                    eframe::egui::RichText::new(format!("Pos: {:.0}, {:.0}, {:.0}", self.cam_pos.x, self.cam_pos.y, self.cam_pos.z))
                                        .size(11.0)
                                        .color(eframe::egui::Color32::from_rgb(200, 190, 220)),
                                );
                            });
                        });
                        if self.show_console {
                            eframe::egui::Window::new("PercyStar Debug Console (~ / L)")
                                .default_pos([20.0, 480.0])
                                .default_size([720.0, 260.0])
                                .resizable(true)
                                .show(ctx, |ui| {
                                    ui.horizontal(|ui| {
                                        if ui.button("Clear").clicked() {
                                            self.logs.clear();
                                        }
                                        if ui.button("Copy All").clicked() {
                                            ui.ctx().copy_text(self.logs.join("\n"));
                                        }
                                        ui.label(format!("{} log lines", self.logs.len()));
                                    });
                                    ui.separator();
                                    eframe::egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                                        for line in &self.logs {
                                            let color = if line.contains("ERROR") || line.contains("failed") || line.contains("error") {
                                                eframe::egui::Color32::from_rgb(255, 90, 90)
                                            } else if line.contains("WARN") || line.contains("limit") {
                                                eframe::egui::Color32::from_rgb(255, 200, 60)
                                            } else if line.contains("Streaming") || line.contains("ready") || line.contains("Ready") {
                                                eframe::egui::Color32::from_rgb(90, 230, 140)
                                            } else {
                                                eframe::egui::Color32::from_rgb(220, 220, 230)
                                            };
                                            ui.label(eframe::egui::RichText::new(line).monospace().size(11.0).color(color));
                                        }
                                    });
                                });
                        }
                    });
                    for (id, delta) in &full_out.textures_delta.set {
                        g.egui_renderer.update_texture(&g.device, &g.queue, *id, delta);
                    }
                    let tris = self.egui_ctx.tessellate(full_out.shapes, scale);
                    g.egui_renderer.update_buffers(&g.device, &g.queue, &mut enc, &tris, &screen_desc);
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("egui_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view_tex,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        }).forget_lifetime();
                        g.egui_renderer.render(&mut rp, &tris, &screen_desc);
                    }
                    for id in &full_out.textures_delta.free {
                        g.egui_renderer.free_texture(id);
                    }
                } else {
                    self.egui_events.clear();
                }

                g.queue.submit(Some(enc.finish()));
                frame.present();
                if !self.ready {
                    use std::io::Write;
                    println!("WORLD_READY");
                    let _ = std::io::stdout().flush();
                    self.ready = true;
                }
                if let Some(frames) = &mut self.smoke_frames {
                    if !g.batches.is_empty() {
                        *frames -= 1;
                        if *frames == 0 {
                            g.assert_visible_geometry();
                            event_loop.exit();
                        }
                    }
                }
                g.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn pollster_block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn dummy_raw() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker { RawWaker::new(p, &VTABLE) }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(dummy_raw()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        if let Poll::Ready(res) = fut.as_mut().poll(&mut cx) {
            return res;
        }
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod terrain_sampling_tests {
    use super::*;
    #[test]
    fn camera_keeps_configured_start_and_samples_south_to_north() {
        let cfg = WorldLaunchConfig {
            terrain: Some(Arc::new(WorldMapData {
                w: 2,
                h: 2,
                bmin: [-10.0, -10.0, 0.0],
                bmax: [10.0, 10.0, 255.0],
                heights: vec![10, 10, 200, 200],
                tex: None,
            })),
            batches: vec![],
            levels_rpf: None,
            map_count: 1,
            start_pos: [-127.276, -1855.924, 24.0],
        };
        let (_, rx) = mpsc::sync_channel(16);
        let start = Vec3::from_array(cfg.start_pos);
        let app = GpuWorldApp::new(cfg, rx, Arc::new(std::sync::Mutex::new((start, 1))));
        assert_eq!(app.cam_pos, start);
        assert!((app.sample_terrain_z(0.0, -10.0) - 10.0).abs() < 0.001);
        assert!((app.sample_terrain_z(0.0, 10.0) - 200.0).abs() < 0.001);
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;
    #[test]
    fn missing_world_archive_reports_an_error() {
        let missing = Path::new("target/missing-world-regression.rpf");
        assert!(!missing.exists());
        assert!(run_gpu_world_cli(Some(missing), None).is_err());
    }
}

#[cfg(test)]
mod vram_budget_tests {
    use super::*;

    #[test]
    fn vram_throttling_tiers_match_budget_headroom() {
        let budget = 5800 * 1024 * 1024;
        // Test <40% usage -> 35km uncapped
        let (r0, p0, s0) = compute_throttle_tier(1000 * 1024 * 1024, budget);
        assert_eq!(r0, 35000.0);
        assert!(!p0);
        assert!(s0.contains("UNCAPPED"));

        // Test 45% usage -> 6km
        let (r1, p1, s1) = compute_throttle_tier(2600 * 1024 * 1024, budget);
        assert_eq!(r1, 6000.0);
        assert!(!p1);
        assert!(s1.contains("6km"));

        // Test 60% usage -> 3km
        let (r2, p2, s2) = compute_throttle_tier(3500 * 1024 * 1024, budget);
        assert_eq!(r2, 3000.0);
        assert!(!p2);
        assert!(s2.contains("3km"));

        // Test 75% usage -> 1.5km
        let (r3, p3, s3) = compute_throttle_tier(4300 * 1024 * 1024, budget);
        assert_eq!(r3, 1500.0);
        assert!(!p3);
        assert!(s3.contains("1.5km"));

        // Test >80% usage -> 450m close range + pressure
        let (r4, p4, s4) = compute_throttle_tier(4800 * 1024 * 1024, budget);
        assert_eq!(r4, 450.0);
        assert!(p4);
        assert!(s4.contains("THROTTLED"));

        // Test >88% usage -> 350m close range + critical eviction
        let (r5, p5, s5) = compute_throttle_tier(5200 * 1024 * 1024, budget);
        assert_eq!(r5, 350.0);
        assert!(p5);
        assert!(s5.contains("CRITICAL"));
    }

    #[test]
    fn archetype_hash_extraction_from_mesh_name() {
        assert_eq!(archetype_hash_from_mesh_name("arch_0001abcd"), 0x0001abcd);
        assert_eq!(archetype_hash_from_mesh_name("arch_deadbeef"), 0xdeadbeef);
        assert_eq!(archetype_hash_from_mesh_name("mhiu_world_terrain"), 0);
        assert_eq!(archetype_hash_from_mesh_name("nav_tile_12_34"), 1);
        assert_eq!(archetype_hash_from_mesh_name("__placeholder_boxes__"), 2);
    }
}

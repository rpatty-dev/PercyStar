use crate::mapdata;
use crate::model3d::{decode_rage9_model_from_rpf_lod_opt, GeomMaterial, Mesh3D, MeshPart, ShadeMode};
use crate::psin::{joaat, parse_psin_doc, PsinDocument, PsinValue};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct RawPlacement {
    pub archetype_hash: u32,
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct ResolvedProp {
    pub name: String,
    pub archetype_hash: u32,
    pub offset: u64,
    pub placement: RawPlacement,
}

#[derive(Clone, Debug, Default)]
pub struct PropPipelineStats {
    pub total_entities_scanned: usize,
    pub psin_files_scanned: usize,
    pub psin_placements_found: usize,
    pub world_index_entities_found: usize,
    pub vehicles_filtered: usize,
    pub peds_filtered: usize,
    pub slods_filtered: usize,
    pub static_props_resolved: usize,
    pub lod0_props_decoded: usize,
    pub total_vertices: usize,
    pub total_triangles: usize,
    pub total_materials: usize,
    pub sample_glb_path: String,
    pub sample_glb_bytes: usize,
}

// Strictly identify and filter out vehicle models
#[inline]
pub fn is_vehicle_model(name: &str, offset: u64) -> bool {
    // 70 GB offset threshold is the vehicle partition in levels.rpf
    if offset >= 70_000_000_000 {
        return true;
    }
    let n = name.to_ascii_lowercase();
    n.contains("veh_")
        || n.contains("aphid_")
        || n.contains("nimbe_")
        || n.contains("stanier")
        || n.contains("cheetah")
        || n.contains("comet")
        || n.contains("banshee")
        || n.contains("infernus")
        || n.contains("sultan")
        || n.contains("buffalo")
        || n.contains("_wheel")
        || (n.starts_with("car_") && !n.contains("park") && !n.contains("wash") && !n.contains("dealer") && !n.contains("pet"))
        || n.starts_with("truck_")
        || (n.starts_with("bike_") && !n.contains("rack") && !n.contains("stand"))
        || (n.starts_with("boat_") && !n.contains("house") && !n.contains("dock"))
        || n.starts_with("heli_")
        || n.starts_with("plane_")
}

// Strictly identify and filter out ped/character models
#[inline]
pub fn is_ped_model(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("char_")
        || n.contains("ped_")
        || n.contains("cs_")
        || n.contains("mp_")
        || n.contains("a_m_")
        || n.contains("a_f_")
        || n.contains("ig_")
        || n.contains("skin_")
        || n.contains("skel_")
        || n.contains("skeleton")
        || n.contains("body_")
}

// Strictly identify and filter out animal/creature models
#[inline]
pub fn is_animal_model(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("a_c_")
        || n.contains("animal")
        || n.contains("creature")
        || n.starts_with("dog_")
        || n.starts_with("cat_")
        || n.starts_with("bird_")
        || n.starts_with("deer_")
        || n.starts_with("cow_")
        || n.starts_with("horse_")
        || n.starts_with("pig_")
        || n.starts_with("shark_")
        || n.starts_with("alligator_")
        || n.starts_with("gator_")
        || n.starts_with("croc_")
        || n.starts_with("dolphin_")
        || n.starts_with("fish_")
        || n.starts_with("rat_")
        || n.starts_with("snake_")
}

// Identify and filter out distant terrain or SLOD city blocks
#[inline]
pub fn is_slod_or_terrain(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("_slod")
        || n.starts_with("slod_")
        || n.starts_with("terrain_")
        || n.starts_with("mhiu_")
        || n.contains("_lod1")
        || n.contains("_lod2")
        || n.contains("_lod3")
}

// Determine if a model is a static world prop candidate
pub fn is_static_world_prop(name: &str, offset: u64) -> bool {
    if is_vehicle_model(name, offset) || is_ped_model(name) || is_slod_or_terrain(name) {
        return false;
    }
    let n = name.to_ascii_lowercase();
    n.contains("prop_")
        || n.contains("am_")
        || n.contains("street")
        || n.contains("lamp")
        || n.contains("fence")
        || n.contains("bench")
        || n.contains("door")
        || n.contains("barrier")
        || n.contains("sign")
        || n.contains("foliage")
        || n.contains("bin")
        || n.contains("trash")
        || n.contains("light")
        || n.contains("hydrant")
        || n.contains("crate")
        || n.contains("box")
        || n.contains("barrel")
        || n.contains("table")
        || n.contains("chair")
        || n.contains("atm")
        || n.contains("shed")
        || n.contains("canopy")
        || n.contains("kiosk")
        || n.contains("post")
        || n.contains("planter")
        || n.contains("pipe")
        || n.contains("cable")
        || n.contains("vent")
        || n.contains("antenna")
        || n.contains("ac_unit")
        || n.contains("duct")
        || n.contains("billboard")
        || n.contains("bollard")
        || n.contains("traffic")
        || n.contains("parking")
        || n.contains("fire")
        || n.contains("tree")
        || n.contains("bush")
        || n.contains("plant")
        || n.contains("detail")
        || n.contains("furniture")
        || n.contains("pallet")
        || n.contains("tool")
        || n.contains("locker")
        || n.contains("switch")
        || n.contains("int_")
}

// Apply world quaternion rotation to a 3D vertex
#[inline]
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

// Extract placements from a decoded PSIN document (handles CEntityDef and rage__fwPropInstanceListDef)
pub fn extract_props_from_psin(doc: &PsinDocument) -> Vec<RawPlacement> {
    let mut out = Vec::new();
    let h_prop_list = joaat("rage__fwPropInstanceListDef");
    let h_centity = joaat("CEntityDef");
    let h_arch = joaat("archetypeName");
    let h_arch2 = joaat("ArchetypeName");
    let h_model = joaat("modelName");
    let h_model2 = joaat("ModelName");
    let h_pos = joaat("position");
    let h_pos2 = joaat("Position");
    let h_rot = joaat("rotation");
    let h_rot2 = joaat("Rotation");
    let h_sxy = joaat("scaleXY");
    let h_sz = joaat("scaleZ");
    let h_scale = joaat("scale");
    let h_scale2 = joaat("Scale");

    for p in doc.extract_placements() {
        out.push(RawPlacement {
            archetype_hash: p.archetype_hash,
            pos: p.position,
            rot: p.rotation,
            scale: p.scale,
        });
    }

    for inst in &doc.instances {
        if inst.tag == h_prop_list || inst.tag == h_centity {
            let mut arch = None;
            let mut pos = None;
            let mut rot = [0.0f32, 0.0, 0.0, 1.0];
            let mut scale = [1.0f32; 3];

            for f in &inst.fields {
                if f.name_hash == h_arch || f.name_hash == h_arch2 || f.name_hash == h_model || f.name_hash == h_model2 {
                    if let PsinValue::Hash(h) | PsinValue::U32(h) = f.value {
                        arch = Some(h);
                    }
                } else if f.name_hash == h_pos || f.name_hash == h_pos2 {
                    if let PsinValue::Vec3(p) = f.value {
                        pos = Some(p);
                    }
                } else if f.name_hash == h_rot || f.name_hash == h_rot2 {
                    if let PsinValue::Vec4(q) = f.value {
                        rot = q;
                    }
                } else if f.name_hash == h_sxy {
                    if let PsinValue::F32(s) = f.value {
                        scale[0] = s;
                        scale[1] = s;
                    }
                } else if f.name_hash == h_sz {
                    if let PsinValue::F32(s) = f.value {
                        scale[2] = s;
                    }
                } else if f.name_hash == h_scale || f.name_hash == h_scale2 {
                    match f.value {
                        PsinValue::Vec3(s) => scale = s,
                        PsinValue::F32(s) => scale = [s, s, s],
                        _ => {}
                    }
                }
            }
            if let (Some(a), Some(p)) = (arch, pos) {
                out.push(RawPlacement {
                    archetype_hash: a,
                    pos: p,
                    rot,
                    scale,
                });
            }
        }
    }
    out
}

// Load model name-to-offset dictionary from scan_cache_v9.tsv
pub fn load_model_name_map() -> HashMap<u32, (String, u64)> {
    let mut map = HashMap::with_capacity(250_000);
    let Some(base) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) else {
        return map;
    };
    let tsv = base.join("PercyStar").join("scan_cache_v9.tsv");
    let Ok(f) = File::open(tsv) else {
        return map;
    };
    let r = BufReader::new(f);
    for line in r.lines().map_while(Result::ok) {
        if !line.contains("\tlevels.rpf\t") {
            continue;
        }
        let mut cols = line.split('\t');
        let (_k, _p, _rpf, Some(name), Some(kind), Some(off_s)) = (
            cols.next(),
            cols.next(),
            cols.next(),
            cols.next(),
            cols.next(),
            cols.next(),
        ) else {
            continue;
        };
        if kind != "MODEL" && kind != "VEH" {
            continue;
        }
        let Ok(off) = off_s.parse::<u64>() else {
            continue;
        };
        let stem = name.split(".#").next().unwrap_or(name);
        let h = joaat(stem);
        map.entry(h).or_insert_with(|| (name.to_string(), off));
    }
    map
}

// High-speed parallel resolution of world prop placements and archetype hashes
pub fn resolve_world_props(
    levels_rpf: &Path,
    carved_dir: Option<&Path>,
) -> (Vec<ResolvedProp>, PropPipelineStats) {
    let mut stats = PropPipelineStats::default();
    let t0 = Instant::now();

    // 1. Build or load WorldIndex (cached PSW2 / PSW1 fallback)
    eprintln!("[props-pipeline] Loading world index for {}...", levels_rpf.display());
    let world_ix = mapdata::build_or_load_world_index(levels_rpf);
    stats.world_index_entities_found = world_ix.entities.len();
    eprintln!(
        "[props-pipeline] World index loaded: {} maps, {} models, {} entity placements",
        world_ix.map_count,
        world_ix.models.len(),
        world_ix.entities.len()
    );

    // 2. Load model names and offsets from scan_cache_v9.tsv
    eprintln!("[props-pipeline] Loading model name cache...");
    let mut model_name_map = load_model_name_map();
    for (&h, &off) in &world_ix.models {
        model_name_map.entry(h).or_insert_with(|| (format!("model_0x{:08x}", h), off));
    }
    eprintln!("[props-pipeline] Total known models in catalog: {}", model_name_map.len());

    // 3. Multi-threaded scanning of carved PSIN files
    let mut all_raw_placements: Vec<RawPlacement> = Vec::with_capacity(world_ix.entities.len() + 10_000);

    // Add entities from WorldIndex
    for e in &world_ix.entities {
        all_raw_placements.push(RawPlacement {
            archetype_hash: e.archetype_hash,
            pos: e.pos,
            rot: e.rot,
            scale: [e.scale_xy, e.scale_xy, e.scale_z],
        });
    }

    if let Some(cdir) = carved_dir {
        if cdir.is_dir() {
            let mut psin_files = Vec::new();
            if let Ok(rd) = fs::read_dir(cdir) {
                for entry in rd.flatten() {
                    let p = entry.path();
                    if p.is_file()
                        && p.extension()
                            .and_then(|s| s.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("bin"))
                            .unwrap_or(false)
                    {
                        psin_files.push(p);
                    }
                }
            }
            stats.psin_files_scanned = psin_files.len();
            eprintln!(
                "[props-pipeline] Scanning {} carved PSIN files in parallel across CPU cores...",
                psin_files.len()
            );

            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                .clamp(2, 16);

            let chunks: Vec<Vec<PathBuf>> = psin_files
                .chunks((psin_files.len() + n_threads - 1) / n_threads)
                .map(|c| c.to_vec())
                .collect();

            let psin_placements: Vec<RawPlacement> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    handles.push(s.spawn(move || {
                        let mut local_placements = Vec::new();
                        for path in chunk {
                            if let Ok(data) = fs::read(&path) {
                                if let Ok(doc) = parse_psin_doc(&data) {
                                    let pl = extract_props_from_psin(&doc);
                                    local_placements.extend(pl);
                                }
                            }
                        }
                        local_placements
                    }));
                }
                let mut merged = Vec::new();
                for h in handles {
                    if let Ok(res) = h.join() {
                        merged.extend(res);
                    }
                }
                merged
            });

            stats.psin_placements_found = psin_placements.len();
            eprintln!(
                "[props-pipeline] Extracted {} prop placements from carved PSIN files",
                psin_placements.len()
            );
            all_raw_placements.extend(psin_placements);
        }
    }

    stats.total_entities_scanned = all_raw_placements.len();

    // 4. Resolve archetype hashes against models and apply strict filtering
    eprintln!("[props-pipeline] Filtering and resolving {} entity placements...", all_raw_placements.len());
    let mut resolved = Vec::new();

    for p in all_raw_placements {
        let Some((name, offset)) = model_name_map.get(&p.archetype_hash) else {
            continue;
        };

        // Strict vehicle filter
        if is_vehicle_model(name, *offset) {
            stats.vehicles_filtered += 1;
            continue;
        }

        // Strict ped filter
        if is_ped_model(name) {
            stats.peds_filtered += 1;
            continue;
        }

        // Filter out slods / terrain blocks
        if is_slod_or_terrain(name) {
            stats.slods_filtered += 1;
            continue;
        }

        // Coordinate validity check
        if !p.pos[0].is_finite()
            || !p.pos[1].is_finite()
            || !p.pos[2].is_finite()
            || p.pos[0].abs() > 20_000.0
            || p.pos[1].abs() > 20_000.0
            || p.pos[2].abs() > 2_000.0
        {
            continue;
        }

        resolved.push(ResolvedProp {
            name: name.clone(),
            archetype_hash: p.archetype_hash,
            offset: *offset,
            placement: p,
        });
    }

    stats.static_props_resolved = resolved.len();
    eprintln!(
        "[props-pipeline] Resolution complete in {:.2}s: {} static props resolved (filtered: {} vehicles, {} peds, {} slods)",
        t0.elapsed().as_secs_f32(),
        stats.static_props_resolved,
        stats.vehicles_filtered,
        stats.peds_filtered,
        stats.slods_filtered
    );

    (resolved, stats)
}

#[derive(Clone, Debug, Default)]
pub struct AllWorldPipelineStats {
    pub total_entities_scanned: usize,
    pub world_index_entities_found: usize,
    pub psin_files_scanned: usize,
    pub psin_placements_found: usize,
    pub vehicles_filtered: usize,
    pub peds_filtered: usize,
    pub animals_filtered: usize,
    pub world_entities_resolved: usize,
    pub unique_models_count: usize,
}

// High-speed parallel resolution of ALL world entities EXCEPT vehicles, peds, and animals
// Strictly preserves all terrain meshes, roads, highways, bridges, sidewalks, city districts, buildings, and LOD0 props
pub fn resolve_all_world_entities(
    levels_rpf: &Path,
    carved_dir: Option<&Path>,
) -> (Vec<ResolvedProp>, AllWorldPipelineStats) {
    let mut stats = AllWorldPipelineStats::default();
    let t0 = Instant::now();

    // 1. Build or load WorldIndex (cached PSW2 / PSW1 fallback)
    eprintln!("[props-pipeline] Loading world index for {}...", levels_rpf.display());
    let world_ix = mapdata::build_or_load_world_index(levels_rpf);
    stats.world_index_entities_found = world_ix.entities.len();
    eprintln!(
        "[props-pipeline] World index loaded: {} maps, {} models, {} entity placements",
        world_ix.map_count,
        world_ix.models.len(),
        world_ix.entities.len()
    );

    // 2. Load model names and offsets from scan_cache_v9.tsv
    eprintln!("[props-pipeline] Loading model name cache...");
    let mut model_name_map = load_model_name_map();
    for (&h, &off) in &world_ix.models {
        model_name_map.entry(h).or_insert_with(|| (format!("model_0x{:08x}", h), off));
    }
    eprintln!("[props-pipeline] Total known models in catalog: {}", model_name_map.len());

    // 3. Multi-threaded scanning of carved PSIN files
    let mut all_raw_placements: Vec<RawPlacement> = Vec::with_capacity(world_ix.entities.len() + 10_000);

    for e in &world_ix.entities {
        all_raw_placements.push(RawPlacement {
            archetype_hash: e.archetype_hash,
            pos: e.pos,
            rot: e.rot,
            scale: [e.scale_xy, e.scale_xy, e.scale_z],
        });
    }

    if let Some(cdir) = carved_dir {
        if cdir.is_dir() {
            let mut psin_files = Vec::new();
            if let Ok(rd) = fs::read_dir(cdir) {
                for entry in rd.flatten() {
                    let p = entry.path();
                    if p.is_file()
                        && p.extension()
                            .and_then(|s| s.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("bin"))
                            .unwrap_or(false)
                    {
                        psin_files.push(p);
                    }
                }
            }
            stats.psin_files_scanned = psin_files.len();
            eprintln!(
                "[props-pipeline] Scanning {} carved PSIN files in parallel across CPU cores...",
                psin_files.len()
            );

            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                .clamp(2, 24);

            let chunks: Vec<Vec<PathBuf>> = psin_files
                .chunks((psin_files.len() + n_threads - 1) / n_threads)
                .map(|c| c.to_vec())
                .collect();

            let psin_placements: Vec<RawPlacement> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    handles.push(s.spawn(move || {
                        let mut local_placements = Vec::new();
                        for path in chunk {
                            if let Ok(data) = fs::read(&path) {
                                if let Ok(doc) = parse_psin_doc(&data) {
                                    let pl = extract_props_from_psin(&doc);
                                    local_placements.extend(pl);
                                }
                            }
                        }
                        local_placements
                    }));
                }
                let mut merged = Vec::new();
                for h in handles {
                    if let Ok(res) = h.join() {
                        merged.extend(res);
                    }
                }
                merged
            });

            stats.psin_placements_found = psin_placements.len();
            eprintln!(
                "[props-pipeline] Extracted {} placements from carved PSIN files",
                psin_placements.len()
            );
            all_raw_placements.extend(psin_placements);
        }
    }

    stats.total_entities_scanned = all_raw_placements.len();

    // 4. Resolve archetype hashes against models and apply strict filtering (strictly 0 vehicles, 0 peds, 0 animals)
    // DO NOT filter out SLODs, roads, bridges, terrain meshes, or buildings!
    eprintln!("[props-pipeline] Resolving and strictly filtering {} placements (0 vehicles, 0 peds, 0 animals)...", all_raw_placements.len());
    let mut resolved = Vec::with_capacity(all_raw_placements.len());
    let mut unique_models = HashSet::new();

    for p in all_raw_placements {
        let Some((name, offset)) = model_name_map.get(&p.archetype_hash) else {
            continue;
        };

        // Strict vehicle filter
        if is_vehicle_model(name, *offset) {
            stats.vehicles_filtered += 1;
            continue;
        }

        // Strict ped filter
        if is_ped_model(name) {
            stats.peds_filtered += 1;
            continue;
        }

        // Strict animal filter
        if is_animal_model(name) {
            stats.animals_filtered += 1;
            continue;
        }

        // Coordinate sanity check
        if !p.pos[0].is_finite()
            || !p.pos[1].is_finite()
            || !p.pos[2].is_finite()
            || p.pos[0].abs() > 30_000.0
            || p.pos[1].abs() > 30_000.0
            || p.pos[2].abs() > 5_000.0
        {
            continue;
        }

        unique_models.insert(*offset);
        resolved.push(ResolvedProp {
            name: name.clone(),
            archetype_hash: p.archetype_hash,
            offset: *offset,
            placement: p,
        });
    }

    stats.world_entities_resolved = resolved.len();
    stats.unique_models_count = unique_models.len();
    eprintln!(
        "[props-pipeline] Resolution complete in {:.2}s: {} world entities resolved across {} unique models (filtered: {} vehicles, {} peds, {} animals)",
        t0.elapsed().as_secs_f32(),
        stats.world_entities_resolved,
        stats.unique_models_count,
        stats.vehicles_filtered,
        stats.peds_filtered,
        stats.animals_filtered
    );

    (resolved, stats)
}

// Multi-threaded decoding of LOD0 prop geometries + textures and assembling verification GLB
pub fn decode_and_assemble_props_sample(
    levels_rpf: &Path,
    props: &[ResolvedProp],
    sample_count: usize,
    out_glb: &Path,
) -> Result<PropPipelineStats, String> {
    if props.is_empty() {
        return Err("No resolved props available for sample assembly".into());
    }

    let t0 = Instant::now();
    let sample_count = sample_count.clamp(1, 100);

    // Pick diverse unique static props
    let mut chosen_props: Vec<ResolvedProp> = Vec::with_capacity(sample_count);
    let mut seen_offsets = std::collections::HashSet::new();

    // Prefer recognized static prop names first
    for p in props {
        if is_static_world_prop(&p.name, p.offset) && seen_offsets.insert(p.offset) {
            chosen_props.push(p.clone());
            if chosen_props.len() >= sample_count {
                break;
            }
        }
    }
    // If not filled, add remaining valid non-vehicle/ped props
    if chosen_props.len() < sample_count {
        for p in props {
            if seen_offsets.insert(p.offset) {
                chosen_props.push(p.clone());
                if chosen_props.len() >= sample_count {
                    break;
                }
            }
        }
    }

    eprintln!(
        "[props-pipeline] Selected {} unique LOD0 props for verified assembly",
        chosen_props.len()
    );

    // Multi-threaded decoding using std::thread::scope across all CPU cores
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .clamp(2, 16);

    let chunks: Vec<Vec<ResolvedProp>> = chosen_props
        .chunks((chosen_props.len() + n_threads - 1) / n_threads)
        .map(|c| c.to_vec())
        .collect();

    let decoded_meshes: Vec<(Mesh3D, ResolvedProp)> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            handles.push(s.spawn(move || {
                let mut local = Vec::new();
                for prop in chunk {
                    // want_lod = 0 (LOD0), layout_grid = false (preserve native local coords)
                    match decode_rage9_model_from_rpf_lod_opt(
                        levels_rpf,
                        prop.offset,
                        &prop.name,
                        0,
                        false,
                    ) {
                        Ok(mesh) => {
                            if !mesh.verts.is_empty() && !mesh.tris.is_empty() {
                                local.push((mesh, prop));
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[props-pipeline] Warning: failed to decode prop {} at 0x{:x}: {}",
                                prop.name, prop.offset, e
                            );
                        }
                    }
                }
                local
            }));
        }
        let mut merged = Vec::new();
        for h in handles {
            if let Ok(res) = h.join() {
                merged.extend(res);
            }
        }
        merged
    });

    if decoded_meshes.is_empty() {
        return Err("Failed to decode any LOD0 props from levels.rpf".into());
    }

    eprintln!(
        "[props-pipeline] Successfully decoded {} LOD0 prop meshes in parallel ({:.2}s)",
        decoded_meshes.len(),
        t0.elapsed().as_secs_f32()
    );

    // Assemble decoded LOD0 meshes with world placement transforms into single composite Mesh3D
    let mut combined_verts: Vec<[f32; 3]> = Vec::new();
    let mut combined_uvs: Vec<[f32; 2]> = Vec::new();
    let mut combined_tris: Vec<[u32; 3]> = Vec::new();
    let mut combined_tri_geom: Vec<u16> = Vec::new();
    let mut combined_materials: Vec<GeomMaterial> = Vec::new();
    let mut combined_parts: Vec<MeshPart> = Vec::new();

    for (idx, (mesh, prop)) in decoded_meshes.iter().enumerate() {
        let v_base = combined_verts.len() as u32;
        let t_start = combined_tris.len();
        let geom_base = combined_materials.len() as u16;

        // Apply scale, quaternion rotation, and world position
        let p = &prop.placement;
        let s = p.scale;
        let q = p.rot;
        let pos = p.pos;

        for &v in &mesh.verts {
            let sv = [v[0] * s[0], v[1] * s[1], v[2] * s[2]];
            let rv = quat_rotate(q, sv);
            let wv = [rv[0] + pos[0], rv[1] + pos[1], rv[2] + pos[2]];
            combined_verts.push(wv);
        }

        // UVs
        combined_uvs.extend_from_slice(&mesh.uvs);
        while combined_uvs.len() < combined_verts.len() {
            combined_uvs.push([0.0, 0.0]);
        }

        // Triangles with remapped vertex indices
        for tri in &mesh.tris {
            combined_tris.push([tri[0] + v_base, tri[1] + v_base, tri[2] + v_base]);
        }

        // Triangle geometry material mapping
        for &g in &mesh.tri_geom {
            combined_tri_geom.push(geom_base + g);
        }

        // Materials & textures
        if mesh.geom_materials.is_empty() {
            let def_mat = GeomMaterial::default();
            combined_materials.push(def_mat);
        } else {
            combined_materials.extend_from_slice(&mesh.geom_materials);
        }

        combined_parts.push(MeshPart {
            name: format!("Prop_{}_{}", idx + 1, prop.name.trim_end_matches(".#dr")),
            geoms: mesh.geom_count.max(1),
            tris: t_start..combined_tris.len(),
        });
    }

    let total_geoms = combined_materials.len();
    let total_v = combined_verts.len();
    let total_t = combined_tris.len();

    eprintln!(
        "[props-pipeline] Assembled sample scene: {} vertices, {} triangles, {} materials across {} parts",
        total_v,
        total_t,
        total_geoms,
        combined_parts.len()
    );

    let mut scene_mesh = Mesh3D::from_parts(
        "GTA6_LOD0_World_Props_Verified_Sample".into(),
        combined_verts,
        combined_uvs,
        combined_tris,
        combined_tri_geom,
        Vec::new(),
        None,
        format!("{total_geoms} materials · {total_v} verts · {total_t} tris"),
        total_geoms,
        0,
        "LOD0 World Props Sample".into(),
    );
    scene_mesh.geom_materials = combined_materials;
    scene_mesh.parts = combined_parts;
    scene_mesh.shade_mode = ShadeMode::Textured;

    // Export GLB
    if let Some(par) = out_glb.parent() {
        let _ = fs::create_dir_all(par);
    }
    eprintln!("[props-pipeline] Exporting verification sample GLB to {}...", out_glb.display());
    scene_mesh
        .export_glb(out_glb)
        .map_err(|e| format!("GLB export error: {e}"))?;

    let glb_bytes = fs::metadata(out_glb)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    if glb_bytes < 1024 {
        return Err(format!("Exported GLB is abnormally small: {glb_bytes} bytes"));
    }

    eprintln!(
        "[props-pipeline] Successfully verified and exported GLB: {} bytes",
        glb_bytes
    );

    Ok(PropPipelineStats {
        total_entities_scanned: 0,
        psin_files_scanned: 0,
        psin_placements_found: 0,
        world_index_entities_found: 0,
        vehicles_filtered: 0,
        peds_filtered: 0,
        slods_filtered: 0,
        static_props_resolved: props.len(),
        lod0_props_decoded: decoded_meshes.len(),
        total_vertices: total_v,
        total_triangles: total_t,
        total_materials: total_geoms,
        sample_glb_path: out_glb.to_string_lossy().into_owned(),
        sample_glb_bytes: glb_bytes,
    })
}

// CLI entrypoint for `--verify-props`
pub fn verify_props_cli() -> Result<(), String> {
    let levels_rpf = PathBuf::from(r"S:\gta\Gta6fordevelopersacces\levels.rpf");
    let carved_dir = PathBuf::from(r"C:\pulpit\xx\carved_data");
    let out_glb = PathBuf::from(r"C:\pulpit\xx\decoders\samples\props_sample_verified.glb");

    if !levels_rpf.exists() {
        return Err(format!("Archive not found: {}", levels_rpf.display()));
    }

    println!("=== PercyStar LOD0 World Props Pipeline Verification ===");
    println!("Target Archive: {}", levels_rpf.display());
    println!("Carved PSIN Directory: {}", carved_dir.display());
    println!("Output Sample GLB: {}", out_glb.display());

    let (resolved_props, stats) = resolve_world_props(
        &levels_rpf,
        if carved_dir.is_dir() { Some(&carved_dir) } else { None },
    );

    println!("\n--- Step 1: Resolution & Filtering Metrics ---");
    println!("Total Placements Scanned:   {}", stats.total_entities_scanned);
    println!("WorldIndex Entities:         {}", stats.world_index_entities_found);
    println!("Carved PSIN Placements:      {}", stats.psin_placements_found);
    println!("Vehicles Filtered Out:       {}", stats.vehicles_filtered);
    println!("Peds Filtered Out:           {}", stats.peds_filtered);
    println!("SLODs / Terrains Filtered:   {}", stats.slods_filtered);
    println!("Static World Props Resolved: {}", stats.static_props_resolved);

    println!("\n--- Step 2: Multi-Threaded LOD0 Decoding & Sample Assembly ---");
    let assembly_stats = decode_and_assemble_props_sample(&levels_rpf, &resolved_props, 25, &out_glb)?;

    println!("Props Decoded at LOD0:       {}", assembly_stats.lod0_props_decoded);
    println!("Total Sample Vertices:       {}", assembly_stats.total_vertices);
    println!("Total Sample Triangles:      {}", assembly_stats.total_triangles);
    println!("Total Sample Materials:      {}", assembly_stats.total_materials);
    println!("Sample GLB Exported:         {} ({} bytes)", assembly_stats.sample_glb_path, assembly_stats.sample_glb_bytes);

    // Validate GLB binary header
    let mut f = File::open(&out_glb).map_err(|e| format!("Failed to read exported GLB: {e}"))?;
    let mut hdr = [0u8; 12];
    f.read_exact(&mut hdr).map_err(|e| format!("Failed to read GLB header: {e}"))?;
    if &hdr[0..4] != b"glTF" {
        return Err("Exported file is not a valid glTF binary!".into());
    }
    let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
    let length = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);
    println!("glTF Version:                {}", version);
    println!("glTF Header Stated Length:   {} bytes", length);

    println!("\n[OK] LOD0 World Props Pipeline verified successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_rotation_identity() {
        let q = [0.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 2.0, 3.0];
        let res = quat_rotate(q, v);
        assert!((res[0] - v[0]).abs() < 1e-5);
        assert!((res[1] - v[1]).abs() < 1e-5);
        assert!((res[2] - v[2]).abs() < 1e-5);
    }

    #[test]
    fn test_quat_rotation_90_deg_z() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let q = [0.0f32, 0.0, s, s]; // 90 deg around Z
        let v = [1.0f32, 0.0, 0.0];
        let res = quat_rotate(q, v);
        assert!((res[0] - 0.0).abs() < 1e-5);
        assert!((res[1] - 1.0).abs() < 1e-5);
        assert!((res[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_vehicle_filtering() {
        assert!(is_vehicle_model("veh_cheetah.#dr", 1000));
        assert!(is_vehicle_model("stanier_body.#dr", 2000));
        assert!(is_vehicle_model("aphid_chassis.#dr", 3000));
        assert!(is_vehicle_model("car_sedan.#dr", 4000));
        assert!(is_vehicle_model("any_model.#dr", 75_000_000_000));
        assert!(!is_vehicle_model("prop_am_pallet_box.#dr", 1000));
        assert!(!is_vehicle_model("street_lamp_01.#dr", 2000));
    }

    #[test]
    fn test_ped_filtering() {
        assert!(is_ped_model("char_protagonist.#dr"));
        assert!(is_ped_model("ped_a_m_tourist.#dr"));
        assert!(is_ped_model("cs_cop_01.#dr"));
        assert!(is_ped_model("mp_f_freemode.#dr"));
        assert!(!is_ped_model("prop_barrier_01.#dr"));
    }

    #[test]
    fn test_animal_filtering() {
        assert!(is_animal_model("a_c_dog.#dr"));
        assert!(is_animal_model("a_c_alligator.#dr"));
        assert!(is_animal_model("shark_tiger.#dr"));
        assert!(is_animal_model("deer_buck.#dr"));
        assert!(!is_animal_model("prop_tree_palm_01.#dr"));
        assert!(!is_animal_model("bk_04_slod.#dr"));
    }

    #[test]
    fn test_static_world_prop_classification() {
        assert!(is_static_world_prop("prop_am_safetyswitch_02a_.#dr", 5499166736));
        assert!(is_static_world_prop("prop_street_bench_01.#dr", 100000));
        assert!(is_static_world_prop("ky24_mod1b_prop_am_locker_02a_rsref001.#dr", 5499125776));
        assert!(!is_static_world_prop("veh_cheetah.#dr", 1000));
        assert!(!is_static_world_prop("bk_04_slod.#dr", 5000));
    }
}

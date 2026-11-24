mod anim;
mod audio_player;
mod awc;
mod bink2;
mod bink2_container;
mod boot;
mod dat;
mod dxbc;
mod mapdata;
mod meta7a;
mod names;
mod psin;
mod psin_xml;
mod rsc9;
mod sgd5;
mod world_gpu;
mod worldmap;
mod model3d;
mod rdna2;
mod tex;
mod vehicle_data;
mod vehicle_variations;
mod bench;
use std::collections::HashSet;
use model3d::{
    Mesh3D, decode_rage9_model_from_rpf, decode_rage9_model_from_rpf_lod, extract_0drpz_mesh, extract_rage9_model_name,
    has_rage9_meshlets, is_vehicle_virtual, is_zlib_head, parse_obj,
    probe_large_rage9_model_in_buf, read_virtual_from_file, render_mesh3d_viewport,
};
use rdna2::{
    Ps09Cache, Ps09ResourceName, Ps09ShaderEntry, ShaderStage, format_gfx103_disasm,
    format_spirv_abi_summary, lift_to_hlsl, lift_to_spirv,
};
use tex::{
    DecodedTexture, build_dds_dx10, decode_0drpz_text, decode_mhiu_heightmap, decode_navmesh_from_rpf,
    decode_rbf0,
    decode_ttpf_texture, decode_txd_from_rpf, export_texture_png, inspect_gfx, probe_txd_header,
    scan_ttpf_descriptors,
};
use eframe::egui::{self, Color32, ColorImage, CornerRadius, FontData, FontDefinitions, FontFamily, RichText, Stroke, TextureHandle, TextureOptions};
use flate2::{Decompress, FlushDecompress, Status};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const REPO_RELEASES_URL: &str = "https://github.com/rpatty-dev/PercyStar/releases";
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

// threshold in levels rpf where vehicle fragtypes begin
const VEH_ZONE_OFF: u64 = 0x10b2_0000_00;

// win32 docking + PlaySound; non-windows builds use the stubs in the other `plat` below
#[cfg(windows)]
mod plat {
use std::path::PathBuf;

#[link(name = "winmm")]
extern "system" {
    fn PlaySoundW(pszSound: *const u16, hmod: isize, fdwSound: u32) -> i32;
}

#[link(name = "user32")]
extern "system" {
    fn EnumWindows(lpEnumFunc: unsafe extern "system" fn(isize, isize) -> i32, lParam: isize) -> i32;
    fn GetWindowThreadProcessId(hWnd: isize, lpdwProcessId: *mut u32) -> u32;
    fn IsWindowVisible(hWnd: isize) -> i32;
    fn SetParent(hWndChild: isize, hWndNewParent: isize) -> isize;
    fn MoveWindow(hWnd: isize, X: i32, Y: i32, nWidth: i32, nHeight: i32, bRepaint: i32) -> i32;
    fn GetActiveWindow() -> isize;
    fn SetFocus(hWnd: isize) -> isize;
    fn SetForegroundWindow(hWnd: isize) -> i32;
    fn GetWindowLongW(hWnd: isize, nIndex: i32) -> i32;
    fn SetWindowLongW(hWnd: isize, nIndex: i32, dwNewLong: i32) -> i32;
}

struct EnumCtx {
    pid: u32,
    hwnd: isize,
}

unsafe extern "system" fn enum_cb(hwnd: isize, lparam: isize) -> i32 {
    let ctx = &mut *(lparam as *mut EnumCtx);
    let mut wpid = 0u32;
    GetWindowThreadProcessId(hwnd, &mut wpid);
    if wpid == ctx.pid && IsWindowVisible(hwnd) != 0 {
        ctx.hwnd = hwnd;
        return 0;
    }
    1
}

pub fn find_hwnd_for_pid(pid: u32) -> Option<isize> {
    let mut ctx = EnumCtx { pid, hwnd: 0 };
    unsafe { EnumWindows(enum_cb, &mut ctx as *mut _ as isize); }
    (ctx.hwnd != 0).then_some(ctx.hwnd)
}

// strips popup frame so binkplay docks cleanly insid egui panel
pub fn dock_child_hwnd(child_hwnd: isize, parent_hwnd: isize, x: i32, y: i32, w: i32, h: i32) {
    const GWL_STYLE: i32 = -16;
    const WS_CHILD: i32 = 0x4000_0000;
    const WS_POPUP: i32 = -0x8000_0000_i32;
    const WS_CAPTION: i32 = 0x00C0_0000;
    const WS_THICKFRAME: i32 = 0x0004_0000;
    unsafe {
        let style = GetWindowLongW(child_hwnd, GWL_STYLE);
        SetWindowLongW(child_hwnd, GWL_STYLE, (style & !WS_POPUP & !WS_CAPTION & !WS_THICKFRAME) | WS_CHILD);
        SetParent(child_hwnd, parent_hwnd);
        MoveWindow(child_hwnd, x, y, w.max(64), h.max(64), 1);
        SetForegroundWindow(parent_hwnd);
        SetFocus(parent_hwnd);
    }
}

pub fn find_binkplay() -> PathBuf {
    if let Ok(p) = std::env::var("BINKPLAY_PATH") {
        let pb = PathBuf::from(p.trim().trim_matches('"'));
        if pb.is_file() {
            return pb;
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("binkplay.exe"));
            candidates.push(dir.join("tools").join("binkplay.exe"));
        }
    }
    // never the working directory it may be an untrusted dump folder
    if let Some(path_os) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_os) {
            candidates.push(dir.join("binkplay.exe"));
        }
    }
    for env_key in ["ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA", "USERPROFILE"] {
        if let Ok(base) = std::env::var(env_key) {
            let b = PathBuf::from(base);
            candidates.push(b.join("RADVideo").join("binkplay.exe"));
            candidates.push(b.join("RAD Video Tools").join("binkplay.exe"));
        }
    }
    candidates.push(PathBuf::from(r"C:\Program Files (x86)\RADVideo\binkplay.exe"));
    candidates.push(PathBuf::from(r"C:\Program Files\RADVideo\binkplay.exe"));
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files (x86)\RADVideo\binkplay.exe"))
}

pub fn stop_wav() {
    unsafe { PlaySoundW(std::ptr::null(), 0, 0); }
}

pub fn active_window() -> isize {
    unsafe { GetActiveWindow() }
}

pub fn move_window(hwnd: isize, x: i32, y: i32, w: i32, h: i32) {
    unsafe { MoveWindow(hwnd, x, y, w, h, 1); }
}
}

// macos/linux: no HWND docking, binkplay launched as a separate window, wav via afplay/aplay
#[cfg(not(windows))]
mod plat {
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

static WAV_PROC: Mutex<Option<Child>> = Mutex::new(None);

pub fn find_hwnd_for_pid(_pid: u32) -> Option<isize> { None }
pub fn dock_child_hwnd(_c: isize, _p: isize, _x: i32, _y: i32, _w: i32, _h: i32) {}
pub fn active_window() -> isize { 0 }
pub fn move_window(_hwnd: isize, _x: i32, _y: i32, _w: i32, _h: i32) {}

pub fn find_binkplay() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("binkplay")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("binkplay"))
}

pub fn stop_wav() {
    if let Some(mut c) = WAV_PROC.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}
}
use plat::{active_window, dock_child_hwnd, find_binkplay, find_hwnd_for_pid, move_window, stop_wav};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum AppTab {
    #[default]
    Explorer,
    ModelViewer,
    WorldMap,
    Shaders,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum ShaderViewMode {
    #[default]
    Asm,
    Hlsl,
    SpirvAbi,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum ShaderSource {
    #[default]
    Ps09,
    Sgd5,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum ModelCat {
    #[default]
    All,
    Vehicles,
    AircraftBoats,
    Weapons,
    Characters,
    Clothing,
    Animals,
    Props,
    World,
    NavMesh,
}

const MODEL_CATS: [(ModelCat, &str); 10] = [
    (ModelCat::All, "All Models"),
    (ModelCat::Vehicles, "Vehicles"),
    (ModelCat::AircraftBoats, "Air & Sea"),
    (ModelCat::Weapons, "Weapons"),
    (ModelCat::Characters, "Characters"),
    (ModelCat::Clothing, "Clothing"),
    (ModelCat::Animals, "Animals"),
    (ModelCat::Props, "Props & Items"),
    (ModelCat::World, "World / Map"),
    (ModelCat::NavMesh, "NavMesh 3D"),
];

fn model_cat_idx(c: ModelCat) -> usize {
    match c {
        ModelCat::All => 0,
        ModelCat::Vehicles => 1,
        ModelCat::AircraftBoats => 2,
        ModelCat::Weapons => 3,
        ModelCat::Characters => 4,
        ModelCat::Clothing => 5,
        ModelCat::Animals => 6,
        ModelCat::Props => 7,
        ModelCat::World => 8,
        ModelCat::NavMesh => 9,
    }
}

fn model_cat_tag(c: ModelCat) -> (&'static str, Color32) {
    match c {
        ModelCat::Vehicles => ("VEH", Color32::from_rgb(255, 85, 175)),
        ModelCat::AircraftBoats => ("AIR", Color32::from_rgb(72, 210, 255)),
        ModelCat::Weapons => ("WPN", Color32::from_rgb(255, 120, 82)),
        ModelCat::Characters => ("CHR", Color32::from_rgb(252, 175, 23)),
        ModelCat::Clothing => ("CLO", Color32::from_rgb(215, 130, 255)),
        ModelCat::Animals => ("ANM", Color32::from_rgb(95, 235, 145)),
        ModelCat::Props => ("PRP", Color32::from_rgb(130, 185, 255)),
        ModelCat::World => ("WLD", Color32::from_rgb(160, 170, 190)),
        ModelCat::NavMesh => ("NAV", Color32::from_rgb(125, 235, 110)),
        ModelCat::All => ("MDL", Color32::from_rgb(180, 185, 200)),
    }
}

fn is_map_zone_prefix(p0: &str) -> bool {
    let b = p0.as_bytes();
    if b.len() >= 3 && b.len() <= 6 && b[0].is_ascii_alphabetic() {
        let alpha_end = b.iter().position(|c| !c.is_ascii_alphabetic()).unwrap_or(b.len());
        if (1..=2).contains(&alpha_end) && alpha_end < b.len() && b[alpha_end].is_ascii_digit() {
            let dig_end = b[alpha_end..]
                .iter()
                .position(|c| !c.is_ascii_digit())
                .map(|i| alpha_end + i)
                .unwrap_or(b.len());
            let digits = dig_end - alpha_end;
            let trail = b.len() - dig_end;
            if digits >= 1 && trail <= 1 && (trail == 0 || b[dig_end].is_ascii_alphabetic()) {
                return matches!(
                    &p0[..alpha_end],
                    "n" | "s" | "tw" | "bk" | "dn" | "vk" | "ky" | "fi" | "vi" | "ap" | "ds"
                        | "fh" | "ss" | "tr" | "sb" | "le" | "id" | "sn" | "nm" | "po" | "nb"
                        | "ma" | "lo" | "td" | "ec" | "sj"
                );
            }
        }
    }
    false
}

fn match_known_stem(p0: &str, stems: &[&str]) -> bool {
    stems.contains(&p0) || (p0.len() > 4 && p0.get(1..).is_some_and(|rest| stems.contains(&rest)))
}

fn classify_model_cat(kind: &str, name: &str) -> ModelCat {
    if kind == "NAVM" {
        return ModelCat::NavMesh;
    }
    let lo = name.to_ascii_lowercase();
    let p0 = lo.split('_').next().unwrap_or("").split('.').next().unwrap_or("");
    let has = |pats: &[&str]| pats.iter().any(|p| lo.contains(p));
    let starts = |pats: &[&str]| pats.iter().any(|p| lo.starts_with(p));

    // 1. Animals: exact RAGE 9 creature taxonomy ONLY (crm_=mammal, crb_=bird, crf_=fish, crr_=reptile, cra_=amphibian, crc_=crustacean, cri_=insect)
    if starts(&["crm_", "crb_", "crf_", "crr_", "cra_", "crc_", "cri_"]) {
        return ModelCat::Animals;
    }

    // 2. Weapons: explicit weapon prefixes, vehicle weapon mounts, & gunshop weapon displays
    if starts(&[
        "w_preset_", "wpn_", "w_ar_", "w_pi_", "w_sg_", "w_sr_", "w_sb_", "w_me_",
        "w_lr_", "w_ex_", "w_mg_", "w_am_", "weap_", "weapon_",
    ]) || has(&[
        "_weap", "_weapon", "mint_widget_gun_", "prop_am_weapon_",
        "mint_display_weapon_", "mint_display_bag_rifle_", "mint_gun_counter_",
    ]) {
        return ModelCat::Weapons;
    }

    // 3. Props & World map sectors FIRST so props/buildings never leak into vehicles/clothing/characters/air&sea
    if starts(&["prop_", "cvar_"]) {
        return ModelCat::Props;
    }
    let has_ped_rig = has(&["_mr1_", "_fr1_", "_mb1_", "_fb1_", "_mo1_", "_fo1_"]);
    if !has_ped_rig
        && (starts(&[
            "prefab_", "mint_", "int_", "veg_", "maptile_", "glass_", "sea_tr_", "sea_city_",
            "sea_cruise_", "a_bfyb", "a_tl_", "ie_", "lo_", "ma_", "nb_", "ds_", "fh_", "bk_",
            "dn_", "vk_", "ky_", "fi_", "vi_", "ap_", "ss_", "tr_", "sb_", "le_", "id_", "sn_",
            "nm_", "po_", "tw_",
        ]) || is_map_zone_prefix(p0))
    {
        return ModelCat::World;
    }

    // 4. Ped collision/bounds helpers (filtered out when "Hide LODs & Bounds" is ON)
    if lo.contains("bipedbounds") || starts(&["collision_", "char_boundprims", "char_z_z_physics"]) {
        if has(&[
            "skirt", "shirt", "pant", "shoe", "boot", "hat", "cap", "glasses", "shades",
            "watch", "ring", "mask", "helmet", "vest", "dress", "jacket", "coat", "short",
            "jean", "bra", "bikini", "undies", "glove", "apron", "armor", "bag", "sock",
            "belt", "neckwear", "scarf", "hoodie", "suit", "tights", "legging", "swim",
            "sandal", "flipflop", "loafer", "sneaker", "bandana", "bracelet", "earring",
            "piercing", "headwear", "accs_", "bodice", "corset",
        ]) {
            return ModelCat::Clothing;
        }
        return ModelCat::Characters;
    }

    // 5. Clothing vs Characters: strictly ped-rigged (_mr1_/_fr1_/etc.) & gmt_ garments — never car skirts or boat parts
    if lo.starts_with("gmt_") {
        return ModelCat::Clothing;
    }
    if starts(&[
        "head_", "hc_", "cs_", "ig_", "player_", "u_m_", "u_f_", "a_m_", "a_f_", "s_m_",
        "s_f_", "g_m_", "g_f_", "legacyped", "gore", "bodygenital", "bodynipple", "sc_alien",
        "hand_", "feet_", "eyebrow", "eyelash", "teeth", "biceps", "fchar_", "char_",
    ]) {
        return ModelCat::Characters;
    }
    if has_ped_rig {
        if has(&[
            "hair", "beard", "brows", "eyelash", "eyebrow", "head_", "bodylower",
            "bodyupper", "bodygenital", "bodynipple", "legacyped", "teeth", "face_", "torso_",
        ]) {
            return ModelCat::Characters;
        }
        return ModelCat::Clothing;
    }

    // 6. Air & Sea: exact aircraft & watercraft stems + cmn_boat_
    const AIR_SEA_STEMS: &[&str] = &[
        // Aircraft
        "polmav", "dodo", "swift", "duster", "duster2", "mammatus", "nimbus", "maverick",
        "shamal", "supervolito", "buzzard", "seasparrow", "mallard", "lazer", "valkyrie",
        "volatus", "cargobob", "crake", "jet2", "216cstreamer", "streamer216", "frogger",
        "hydra", "luxor", "velum", "vestra", "cuban800", "titan", "blimp", "besra", "miljet",
        "havok", "akula", "annihilator", "skylift", "seabreeze", "tula", "rogue", "pyro",
        "nokota", "molotok", "starling", "mogul", "howard", "microlight", "conada", "raiju",
        "alkonost", "cargoplane", "ufo",
        // Boats & watercraft (note: "toros" is an SUV -> Vehicles, "toro" is a boat)
        "airgator", "airboat", "cruise", "cruiser", "pontoon", "dinghy", "marquis", "seashark",
        "squalo", "speeder700gt", "speeder", "tropic", "suntrap", "longfin", "jetmax", "tug",
        "predator", "submersible", "kraken", "avisa", "kurtz", "patrolboat", "bombora",
        "cretacean", "seaenchantress", "seacab", "flatdeck", "sedna", "kayak", "canoe",
        "skiff", "yacht", "toro", "coastguard", "boat",
    ];
    if starts(&["cmn_boat", "veh_boat", "veh_heli", "veh_plane", "sea_carrier_", "sea_yacht_"])
        || match_known_stem(p0, AIR_SEA_STEMS)
        || (has(&["_plane.", "_heli.", "_boat."]) && !has(&["groundplane", "emissiveplane"]))
    {
        return ModelCat::AircraftBoats;
    }

    // 7. Vehicles: cars, trucks, bikes, trailers, rims, & vehicle body/tuning mods (incl. car skirts)
    const VEH_MOD_TOKENS: &[&str] = &[
        "_histr_", "_skirt", "_bumper", "_fender", "_bonnet", "_spoiler", "_exhaust", "_grille",
        "_rollcage", "_tb_", "_chassis", "_diffuser", "_splitter", "_canard", "_bullbar",
        "_lightbar", "_snorkel", "_mudflap", "_sidestep", "_sunstrip", "_louvre", "_wheel_",
        "_stwheel", "_shifter", "_curbfeeler", "_headlight", "_taillight", "_towhook",
        "_intercooler", "_airfilter", "_turbo", "_caliper", "_windscreen", "_windshield",
        "_hood_", "_trunk_", "_boot_", "_door_fl", "_door_fr", "_door_rl", "_door_rr",
        "_mirror_l", "_mirror_r", "_roof_rack", "_livery", "_engine_", "_seats_", "_dash_",
        "_bed_", "_canopy", "_wing_", "_arch_", "_winch", "_rack_",
    ];
    const VEH_STEMS: &[&str] = &[
        "truck", "ambulance", "bus", "taxi", "police", "firetruck", "forklift", "tractor",
        "excavator", "telehandler", "tanker", "freight", "van", "buggy", "burrito", "rancher",
        "yosemite", "bison", "walton", "bobcat", "dominator", "tulip", "buffalo", "stanier",
        "sandking", "stallion", "reclaimer", "caracara", "impaler", "gauntlet", "coquette",
        "schafter", "sentinel", "elegy", "futo", "banshee", "comet", "cheetah", "toros",
        "phoenix", "ruiner", "slamvan", "sanchez", "manchez", "esskey", "sovereign", "carbonrs",
        "nightblade", "blade500", "cliffhanger", "lectro", "enduro", "zombie", "fcr", "chino",
        "picador", "vamos", "buccaneer", "emperor", "rumpo", "granger", "woodland", "kamacho",
        "riata", "rebel", "contender", "marshall", "drafter", "jugular", "tailgater", "blista",
        "penumbra", "growler", "cypher", "sultan", "paragon", "sugoi", "rebla", "zion", "akane",
        "astron", "cavalcade", "xls", "vstr", "hustler", "deviant", "packer", "phantom", "youga",
        "boxville", "journey", "jbus", "speedo", "manana", "hermes", "minivan", "landstalker",
        "tempesta", "patriot", "rhino", "airtug", "dump", "crawler", "parkland", "lowline",
        "montagne", "alvinov", "sfb1000", "declasse", "double", "ganado", "sungrazer", "bogan",
        "carbonizzare", "greta", "animus", "fr36", "mb210", "guardian", "yakuruna",
        "transgressor", "allander", "fortune", "apc", "bike", "moto", "coupe", "sedan", "suv",
    ];
    if kind == "VEH"
        || starts(&["veh_", "cmn_", "rim_", "truckbeds_", "trailer", "car_", "van_"])
        || has(VEH_MOD_TOKENS)
        || VEH_STEMS.iter().any(|s| p0.contains(s))
    {
        return ModelCat::Vehicles;
    }

    // 8. Remaining Props
    if has(&[
        "chair", "table", "lamp", "sign_", "phone", "laptop", "bottle", "crate", "barrel",
        "bench", "trash", "vending", "atm_", "sofa", "couch", "stool", "cabinet", "lure",
        "inflatable",
    ]) {
        return ModelCat::Props;
    }

    ModelCat::World
}

#[derive(Clone, PartialEq, Eq)]
enum Category {
    All,
    Vehicles,
    Models,
    Textures,
    Ui,
    NavMesh,
    Videos,
    Audio,
    Xml,
    Resources,
    Binary,
}

const MODEL_TEX_TIP: &str = "Open every texture this model's materials use as flat images for inspection and export";

// lod switch requested from the 3d viewer header
#[derive(Clone, Copy)]
enum LodPick {
    // entry index of a sibling detail file
    File(usize),
    // lod slot inside the current file
    Level(usize),
}

// file names of the three detail levels of a model stem_hi stem_mid stem ordered high to low
fn lod_sibling_names(name: &str) -> Option<[String; 3]> {
    let (stem, ext) = name.rsplit_once('.')?;
    let lower = stem.to_ascii_lowercase();
    let cut = if lower.ends_with("_hi") { 3 } else if lower.ends_with("_mid") { 4 } else { 0 };
    let base = &stem[..stem.len() - cut];
    if base.is_empty() {
        return None;
    }
    Some([format!("{base}_hi.{ext}"), format!("{base}_mid.{ext}"), format!("{base}.{ext}")])
}

fn model_file_stem(name: &str) -> String {
    let stem = name.rsplit_once('.').map(|x| x.0).unwrap_or(name).to_ascii_lowercase();
    stem.strip_suffix("_hi")
        .or_else(|| stem.strip_suffix("_mid"))
        .unwrap_or(&stem)
        .to_string()
}

// detail file and in file lod buttons for the viewer header returns the clicked switch
fn lod_switch_ui(ui: &mut egui::Ui, mesh: &Mesh3D, siblings: &[(&'static str, usize)], selected: Option<usize>) -> Option<LodPick> {
    let mut pick = None;
    let levels = mesh.lod_geoms.iter().filter(|&&n| n > 0).count();
    if siblings.is_empty() && levels < 2 {
        return None;
    }
    ui.separator();
    ui.label(RichText::new("Detail:").size(11.0).color(Color32::from_rgb(130, 138, 156)));
    for &(lbl, idx) in siblings {
        if ui.selectable_label(selected == Some(idx), lbl).on_hover_text("Separate detail level file of this model").clicked() && selected != Some(idx) {
            pick = Some(LodPick::File(idx));
        }
    }
    if levels >= 2 {
        for (l, &n) in mesh.lod_geoms.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let tip = format!("{n} geometries stored in this file");
            if ui.selectable_label(mesh.lod == l, model3d::LOD_NAMES[l]).on_hover_text(tip).clicked() && mesh.lod != l {
                pick = Some(LodPick::Level(l));
            }
        }
    }
    pick
}

#[derive(Clone, Debug)]
struct RpfEntry {
    source_rpf: PathBuf,
    rpf_name: String,
    name: String,
    kind: &'static str,
    offset: u64,
    comp_size: u64,
    raw_size: u64,
    tracks: u32,
    chunks: Vec<(u64, u64)>,
}

#[derive(Default)]
struct ScanState {
    scanning: bool,
    header_info: String,
    status: String,
    entries: Vec<RpfEntry>,
    rpf_names: Vec<String>,
    file_sig: HashMap<PathBuf, (u64, Option<SystemTime>)>,
    audio_bank_meta: HashMap<u64, dat::ResolvedAwcMeta>,
    ver: u64,
}

#[derive(Default)]
struct MediaState {
    loading: bool,
    // bumped whenever the selection changes so a decode that finishes late is dropped
    job: u64,
    status: String,
    loaded_tracks: Option<(Vec<audio_player::LoadedWavTrack>, Arc<Vec<u8>>)>,
    loaded_bank_ctx: Option<audio_player::AudioBankContext>,
    bink_child_pid: Option<u32>,
    bink_hwnd: Option<isize>,
    bink_clip: Option<PathBuf>,
    loaded_mesh: Option<Mesh3D>,
    loaded_vehicle_parts: Vec<(usize, usize, Mesh3D)>,
    loaded_vehicle_configurations: Option<(u64, vehicle_variations::VehicleConfigurationDb)>,
    loaded_textures: Option<Vec<DecodedTexture>>,
    // textures bound by the viewed model with a summary note the model stays loaded behind them
    loaded_model_textures: Option<(Vec<DecodedTexture>, String)>,
}

#[derive(Clone)]
struct VehiclePartCandidate {
    entry: usize,
    name: String,
    enabled: bool,
    loading: bool,
}

#[derive(Clone)]
struct ResolvedVehicleVariation {
    label: String,
    colour: Option<[f32; 3]>,
    entries: Vec<usize>,
    part_names: Vec<String>,
    unresolved: Vec<u32>,
}

// physics components of the viewed vehicle filled in by a worker thread
#[derive(Default)]
struct VehicleDataState {
    // archive and file name of the model the data belongs to
    model: Option<(PathBuf, String)>,
    loading: bool,
    data: Option<vehicle_data::VehicleData>,
    // scanned vehicle model names by hash used to label parents
    names: vehicle_data::Names,
}

#[derive(Default)]
struct UpdateState {
    available: Option<(String, String)>,
    checking: bool,
    installing: bool,
    show_modal: bool,
    status: String,
}


#[derive(Default)]
struct Sgd5JobState {
    running: bool,
    status: String,
    files: Option<Arc<Vec<sgd5::Sgd5File>>>,
}
fn parse_semver(s: &str) -> Option<(u32, u32, u32)> {
    let trimmed = s.trim().trim_start_matches(['v', 'V']);
    let core = trimmed.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let maj = parts.next()?.parse::<u32>().ok()?;
    let min = parts.next()?.parse::<u32>().ok()?;
    let pat = parts.next().unwrap_or("0").parse::<u32>().ok()?;
    Some((maj, min, pat))
}

// archive entry names become file names so path separators and reserved characters are neutralized
// a name can then never leave the folder it is joined onto
fn safe_file_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
        .collect();
    let cleaned = cleaned.trim().trim_end_matches('.').to_string();
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        return "unnamed".into();
    }
    cleaned
}

// release page of a version built from the pinned repository url
fn release_page_url(ver: &str) -> String {
    format!("{REPO_RELEASES_URL}/tag/v{ver}")
}

// opens an https url in the default browser without going through a shell
fn open_url(url: &str) {
    if !url.starts_with("https://") {
        return;
    }
    let _ = if cfg!(windows) {
        silent_cmd("explorer.exe").arg(url).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(url).spawn()
    } else {
        Command::new("xdg-open").arg(url).spawn()
    };
}

fn silent_cmd(prog: &str) -> Command {
    #[cfg(windows)]
    {
        let mut c = Command::new(prog);
        c.creation_flags(CREATE_NO_WINDOW);
        c
    }
    #[cfg(not(windows))]
    Command::new(prog)
}

// checks github latest release redirect without api rate limits
fn spawn_update_check(upd: Arc<Mutex<UpdateState>>, ctx: egui::Context) {
    {
        let mut g = upd.lock().unwrap();
        if g.checking || g.installing { return; }
        g.checking = true;
    }
    std::thread::spawn(move || {
        let (curl, null) = if cfg!(windows) { ("curl.exe", "NUL") } else { ("curl", "/dev/null") };
        let out = silent_cmd(curl)
            .args(["-fsSL", "--proto", "=https", "--proto-redir", "=https", "--max-time", "20"])
            .args(["-o", null, "-w", "%{url_effective}", &format!("{REPO_RELEASES_URL}/latest")])
            .output();
        let mut found = None;
        if let Ok(res) = out {
            if res.status.success() {
                let eff = String::from_utf8_lossy(&res.stdout).trim().to_string();
                if let Some((_, tag)) = eff.rsplit_once("/tag/") {
                    if let (Some(rem), Some(cur)) = (parse_semver(tag), parse_semver(APP_VERSION)) {
                        if rem > cur {
                            // only the parsed numbers reach urls and file names never the raw tag text
                            let ver = format!("{}.{}.{}", rem.0, rem.1, rem.2);
                            found = Some((ver.clone(), release_page_url(&ver)));
                        }
                    }
                }
            }
        }
        {
            let mut g = upd.lock().unwrap();
            g.checking = false;
            g.available = found;
        }
        ctx.request_repaint();
    });
}

// downloads new release zip and swaps running windows binary cleanly
fn spawn_install_update(ver: String, upd: Arc<Mutex<UpdateState>>, ctx: egui::Context) {
    {
        let mut g = upd.lock().unwrap();
        if g.installing { return; }
        g.installing = true;
        g.status = format!("Downloading v{ver}...");
    }
    ctx.request_repaint();
    std::thread::spawn(move || {
        let res: Result<(), String> = (|| {
            let ver = parse_semver(&ver).map(|(a, b, c)| format!("{a}.{b}.{c}")).ok_or("bad version")?;
            let tmp = std::env::temp_dir().join("percystar_self_update");
            // a leftover binary from an earlier attempt must never be the one that gets installed
            if tmp.exists() {
                fs::remove_dir_all(&tmp).map_err(|e| format!("Cannot clear update folder: {e}"))?;
            }
            fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
            let zip_p = tmp.join("update.zip");
            let dl_url = format!("{REPO_RELEASES_URL}/download/v{ver}/PercyStar_v{ver}_Win64.zip");
            let dl = silent_cmd("curl.exe")
                .args(["-fsSL", "--proto", "=https", "--proto-redir", "=https", "--max-time", "900", "-o"])
                .arg(&zip_p)
                .arg(&dl_url)
                .output()
                .map_err(|e| format!("curl error: {e}"))?;
            if !dl.status.success() {
                return Err(format!("Download failed ({})", dl.status));
            }

            {
                let mut g = upd.lock().unwrap();
                g.status = format!("Extracting v{ver}...");
            }
            ctx.request_repaint();
            // paths travel through the environment so a quote in the profile name cannot end the command
            let ex = silent_cmd("powershell")
                .args(["-NoProfile", "-NonInteractive", "-Command"])
                .arg("Expand-Archive -LiteralPath $env:PERCY_ZIP -DestinationPath $env:PERCY_DST -Force")
                .env("PERCY_ZIP", &zip_p)
                .env("PERCY_DST", &tmp)
                .output()
                .map_err(|e| format!("Expand error: {e}"))?;
            if !ex.status.success() {
                return Err("Archive extraction failed".into());
            }

            let new_exe = tmp.join("PercyStar.exe");
            let head = fs::read(&new_exe).map_err(|_| "PercyStar.exe missing in release zip".to_string())?;
            if head.len() < 100_000 || !head.starts_with(b"MZ") {
                return Err("Downloaded binary looks invalid".into());
            }

            let cur_exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let old_exe = cur_exe.with_extension("exe.old");
            let _ = fs::remove_file(&old_exe);
            fs::rename(&cur_exe, &old_exe).map_err(|e| format!("Rename failed: {e}"))?;
            // install the bytes that were checked not whatever is on disk by now
            if let Err(e) = fs::write(&cur_exe, &head) {
                let _ = fs::remove_file(&cur_exe);
                let _ = fs::rename(&old_exe, &cur_exe);
                return Err(format!("Replace failed: {e}"));
            }
            let _ = fs::remove_dir_all(&tmp);
            Command::new(&cur_exe).spawn().map_err(|e| format!("Relaunch failed: {e}"))?;
            std::process::exit(0);
        })();
        if let Err(err) = res {
            let mut g = upd.lock().unwrap();
            g.installing = false;
            g.status = err;
            ctx.request_repaint();
        }
    });
}

struct Rpf9App {
    target_path: Option<PathBuf>,
    is_folder: bool,
    selected_rpf: String,
    state: Arc<Mutex<ScanState>>,
    media: Arc<Mutex<MediaState>>,
    category: Category,
    selected: Option<usize>,
    preview: String,
    filter: String,
    logo_tex: Option<TextureHandle>,
    watch_gen: Arc<Mutex<u64>>,
    bink_proc: Option<Child>,
    parent_hwnd: isize,
    last_dock_rect: [i32; 4],
    clip_seq: u64,
    mesh3d: Option<Mesh3D>,
    show_3d: bool,
    anim_playing: bool,
    anim_time: f32,
    anim_dur: f32,
    anim_tracks: usize,
    anim_seed: u32,
    anim_name: String,
    textures: Vec<DecodedTexture>,
    tex_sel: usize,
    tex_handle: Option<(usize, TextureHandle)>,
    tex_zoom: f32,
    tex_channel: tex::TexChannel,
    tex_rebuild_z: bool,
    tex_view_key: (tex::TexChannel, bool),
    vehicles_only: bool,
    audio_named: bool,
    multi_selected: HashSet<usize>,
    multi_anchor: Option<usize>,
    marquee_start: Option<egui::Pos2>,
    // cachd ui list so huge scans never re filter or clone evry fram
    view: Vec<usize>,
    view_key: Option<(u64, String, Category, String, bool, bool)>,
    hay: Vec<String>,
    hay_ver: u64,
    hay_audio_named: bool,
    counts: [usize; 11],
    cached_rpf_names: Vec<String>,
    last_rebuild: Instant,
    update: Arc<Mutex<UpdateState>>,
    // dedicated Model Viewer tab state (cached for instant browsing across 145k+ models)
    tab: AppTab,
    mv_cat: ModelCat,
    mv_filter: String,
    mv_hide_lods: bool,
    // detail level files of the selected model label and entry index ordered high to low
    lod_siblings: Vec<(&'static str, usize)>,
    // lod switch clicked in the viewer header applied at the start of the next frame
    lod_pick: Option<LodPick>,
    // archive entry of the viewed model none for obj navmesh and animation previews
    model_entry: Option<usize>,
    // Base vehicle plus independently stored tuning fragments selected by the user.
    vehicle_base_mesh: Option<Mesh3D>,
    vehicle_parts: Vec<VehiclePartCandidate>,
    vehicle_part_meshes: HashMap<usize, Mesh3D>,
    vehicle_part_filter: String,
    vehicle_configuration_db: vehicle_variations::VehicleConfigurationDb,
    vehicle_configuration_loaded: bool,
    vehicle_configuration_loading: bool,
    vehicle_configuration_generation: u64,
    vehicle_variations: Vec<ResolvedVehicleVariation>,
    vehicle_variation_selected: usize,
    vehicle_variation_filter: String,
    vehicle_data: Arc<Mutex<VehicleDataState>>,
    // textures button clicked in the viewer header applied at the start of the next frame
    tex_pick: bool,
    // shown textures belong to the viewed model so the viewer offers a way back to it
    tex_from_model: bool,
    // camera yaw pitch zoom pan carried across a lod switch
    lod_keep_cam: Option<[f32; 5]>,
    export_busy: Arc<AtomicBool>,
    // the entry on screen is a psin file shown as xml
    psin_selected: bool,
    export_cancel: Arc<AtomicBool>,
    mv_drawer_open: bool,
    mv_entries: Vec<(usize, ModelCat, String)>,
    mv_counts: [usize; 10],
    mv_video_count: usize,
    mv_ver: u64,
    mv_view: Vec<usize>,
    mv_view_key: Option<(u64, ModelCat, String, bool)>,
    mv_rng: u64,
    audio_player: audio_player::AudioPlayerState,
    explorer_scroll_to_sel: bool,
    // list ordering shared by explorer and model viewer
    sort_mode: SortMode,
    view_sort: SortMode,
    mv_sort: SortMode,
    world_map: worldmap::WorldMapState,
    // dedicated rdna2 gfx103 ps09 shader disassembler and hlsl spirv lifter tab
    sh_cache: Option<Ps09Cache>,
    sh_status: String,
    sh_stage_filter: Option<ShaderStage>,
    sh_search: String,
    sh_selected: Option<usize>,
    sh_mode: ShaderViewMode,
    sh_cached_text: (Option<(usize, ShaderViewMode)>, String),
    sh_source: ShaderSource,
    sgd5_job: Arc<Mutex<Sgd5JobState>>,
    sgd5_cache: Option<Arc<Vec<sgd5::Sgd5File>>>,
    sgd5_group: usize,
    sgd5_stage_filter: Option<&'static str>,
    sgd5_search: String,
    sgd5_selected: Option<usize>,
    sgd5_cached_text: (Option<(usize, usize, ShaderViewMode)>, String),
}

fn sgd5_stage(stage: &str) -> ShaderStage {
    match stage {
        "VS" => ShaderStage::Vertex,
        "PS" => ShaderStage::Pixel,
        "GS" => ShaderStage::Geometry,
        "DS" => ShaderStage::Domain,
        "HS" => ShaderStage::Hull,
        "CS" => ShaderStage::Compute,
        _ => ShaderStage::Unknown,
    }
}

fn sgd5_resource_declaration(resource_type: u16) -> Option<&'static str> {
    match resource_type {
        0x0102 => Some("Texture2D<float4>"),
        0x0142 => Some("Texture2DArray<float4>"),
        0x0202 => Some("TextureCube<float4>"),
        0x0302 => Some("Texture3D<float4>"),
        0x0401 => Some("Buffer<float4>"),
        0x0405 => Some("StructuredBuffer<uint>"),
        0x0407 => Some("ByteAddressBuffer"),
        0x0423 => Some("SamplerState"),
        0x0430 => Some("cbuffer"),
        _ => None,
    }
}

fn sgd5_resource_names(file: &sgd5::Sgd5File, shader: &sgd5::Sgd5Shader) -> Vec<Ps09ResourceName> {
    shader.reflection_records.iter().filter_map(|record| {
        let declaration = sgd5_resource_declaration(record.resource_type)?;
        let register_prefix = match record.resource_type {
            0x0423 => "s",
            0x0430 => "b",
            _ => "t",
        };
        let name = record
            .name
            .clone()
            .or_else(|| file.resolve_name(record.name_hash).map(|name| name.to_string()))
            ;
        Some(Ps09ResourceName {
            declaration: declaration.to_string(),
            register: format!("{register_prefix}{}", record.slot),
            name,
            hash: record.name_hash,
        })
    }).collect()
}

fn sgd5_ps09_entry(file: &sgd5::Sgd5File, shader: &sgd5::Sgd5Shader, fssm: &dxbc::FssmShader) -> Ps09ShaderEntry {
    Ps09ShaderEntry {
        index: shader.index,
        code_offset: u32::try_from(fssm.code_offset).unwrap_or(0),
        code_size: u32::try_from(fssm.code_size).unwrap_or(0),
        stage: sgd5_stage(shader.stage),
        num_threads: None,
        rsrc1: None,
        rsrc2: None,
        bindings: Vec::new(),
        resource_names: sgd5_resource_names(file, shader),
        pso_hash: 0,
        sga_hash: Some(shader.sga_hash),
        has_rt: false,
        has_ngg: false,
        insn_count: u32::try_from(rdna2::decode_gfx103(fssm.code(), 4000).len()).unwrap_or(0),
    }
}

fn sgd5_stage_color(stage: &str) -> Color32 {
    match stage {
        "VS" => Color32::from_rgb(205, 145, 255),
        "PS" => Color32::from_rgb(95, 235, 145),
        "GS" => Color32::from_rgb(120, 220, 255),
        "DS" => Color32::from_rgb(255, 190, 120),
        "HS" => Color32::from_rgb(255, 140, 180),
        "CS" => Color32::from_rgb(72, 210, 255),
        _ => Color32::from_rgb(170, 170, 190),
    }
}

impl Rpf9App {
    fn new(cc: &eframe::CreationContext<'_>, logo_rgba: Option<(Vec<u8>, u32, u32)>, startup: Option<PathBuf>) -> Self {
        setup_style(&cc.egui_ctx);
        let logo_tex = logo_rgba.map(|(raw, w, h)| {
            let ci = ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &raw);
            cc.egui_ctx.load_texture("percystar_logo", ci, TextureOptions::LINEAR)
        });
        let mut app = Self {
            target_path: None,
            is_folder: false,
            selected_rpf: "All RPFs".into(),
            state: Arc::new(Mutex::new(ScanState {
                header_info: "No archive or folder loaded".into(),
                status: "Open an .rpf/.ttpf or select a game folder".into(),
                ..Default::default()
            })),
            media: Arc::new(Mutex::new(MediaState::default())),
            category: Category::All,
            selected: None,
            preview: String::new(),
            filter: String::new(),
            logo_tex,
            watch_gen: Arc::new(Mutex::new(0)),
            bink_proc: None,
            parent_hwnd: 0,
            last_dock_rect: [0; 4],
            clip_seq: 0,
            mesh3d: None,
            show_3d: false,
            anim_playing: false,
            anim_time: 0.0,
            anim_dur: 0.0,
            anim_tracks: 0,
            anim_seed: 0,
            anim_name: String::new(),
            textures: Vec::new(),
            tex_sel: 0,
            tex_handle: None,
            tex_zoom: 1.0,
            tex_channel: tex::TexChannel::Rgba,
            tex_rebuild_z: false,
            tex_view_key: (tex::TexChannel::Rgba, false),
            vehicles_only: false,
            audio_named: true,
            multi_selected: HashSet::new(),
            multi_anchor: None,
            marquee_start: None,
            view: Vec::new(),
            view_key: None,
            hay: Vec::new(),
            hay_ver: u64::MAX,
            hay_audio_named: true,
            counts: [0; 11],
            cached_rpf_names: Vec::new(),
            last_rebuild: Instant::now(),
            update: Arc::new(Mutex::new(UpdateState::default())),
            tab: AppTab::Explorer,
            mv_cat: ModelCat::All,
            mv_filter: String::new(),
            mv_hide_lods: true,
            lod_siblings: Vec::new(),
            lod_pick: None,
            model_entry: None,
            vehicle_base_mesh: None,
            vehicle_parts: Vec::new(),
            vehicle_part_meshes: HashMap::new(),
            vehicle_part_filter: String::new(),
            vehicle_configuration_db: HashMap::new(),
            vehicle_configuration_loaded: false,
            vehicle_configuration_loading: false,
            vehicle_configuration_generation: 0,
            vehicle_variations: Vec::new(),
            vehicle_variation_selected: 0,
            vehicle_variation_filter: String::new(),
            vehicle_data: Arc::default(),
            tex_pick: false,
            tex_from_model: false,
            lod_keep_cam: None,
            export_busy: Arc::new(AtomicBool::new(false)),
            psin_selected: false,
            export_cancel: Arc::new(AtomicBool::new(false)),
            mv_drawer_open: true,
            mv_entries: Vec::new(),
            mv_counts: [0; 10],
            mv_video_count: 0,
            mv_ver: u64::MAX,
            mv_view: Vec::new(),
            mv_view_key: None,
            sort_mode: SortMode::Relevance,
            view_sort: SortMode::Relevance,
            mv_sort: SortMode::Relevance,
            world_map: worldmap::WorldMapState::default(),
            mv_rng: 0x9E37_79B9_7F4A_7C15,
            audio_player: audio_player::AudioPlayerState::default(),
            explorer_scroll_to_sel: false,
            sh_cache: None,
            sh_status: String::new(),
            sh_stage_filter: None,
            sh_search: String::new(),
            sh_selected: None,
            sh_mode: ShaderViewMode::Asm,
            sh_cached_text: (None, String::new()),
            sh_source: ShaderSource::Ps09,
            sgd5_job: Arc::new(Mutex::new(Sgd5JobState::default())),
            sgd5_cache: None,
            sgd5_group: 0,
            sgd5_stage_filter: None,
            sgd5_search: String::new(),
            sgd5_selected: None,
            sgd5_cached_text: (None, String::new()),
        };
        psin::seed_names();
        if let Some(dir) = names_dir() {
            names::load_dir(&dir);
        }
        load_ui_state(&mut app);
        spawn_update_check(Arc::clone(&app.update), cc.egui_ctx.clone());
        // a path on the command line (open with, drag onto the exe) wins over the last folder
        if let Some(path) = startup {
            app.open_any_path(path);
        } else if let Some(last) = load_last_folder() {
            if last.is_dir() {
                app.start_folder_scan(last);
            }
        }
        app
    }

    // opens a game folder an archive an obj model or a ps09 shader cache by what the path is
    fn open_any_path(&mut self, p: PathBuf) {
        if p.is_dir() {
            self.start_folder_scan(p);
            return;
        }
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("obj") {
            self.tab = AppTab::Explorer;
            self.open_obj_path(&p);
        } else if ext.eq_ignore_ascii_case("ps09") {
            self.reset_selection();
            self.tab = AppTab::Shaders;
            self.load_ps09_path(&p);
        } else {
            self.start_single_scan(p);
        }
    }

    fn ordered_selected_indices(&self) -> Vec<usize> {
        if self.multi_selected.is_empty() {
            return self.selected.into_iter().collect();
        }
        let mut out: Vec<usize> = self
            .view
            .iter()
            .copied()
            .filter(|i| self.multi_selected.contains(i))
            .collect();
        if out.is_empty() {
            if let Some(s) = self.selected {
                out.push(s);
            }
        }
        out
    }

    fn stop_bink_player(&mut self) {
        if let Some(mut c) = self.bink_proc.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        let mut m = self.media.lock().unwrap();
        m.bink_child_pid = None;
        m.bink_hwnd = None;
        m.bink_clip = None;
    }

    fn reset_selection(&mut self) {
        self.selected = None;
        self.lod_siblings.clear();
        self.lod_pick = None;
        self.model_entry = None;
        self.vehicle_base_mesh = None;
        self.vehicle_parts.clear();
        self.vehicle_part_meshes.clear();
        self.vehicle_part_filter.clear();
        self.vehicle_variations.clear();
        self.vehicle_variation_selected = 0;
        self.vehicle_variation_filter.clear();
        self.tex_pick = false;
        self.tex_from_model = false;
        self.lod_keep_cam = None;
        self.psin_selected = false;
        self.preview.clear();
        self.last_dock_rect = [0; 4];
        self.mesh3d = None;
        self.show_3d = false;
        self.anim_dur = 0.0;
        self.anim_playing = false;
        self.textures.clear();
        self.tex_sel = 0;
        self.tex_handle = None;
        self.tex_zoom = 1.0;
        stop_wav();
        self.audio_player.clear_all();
        self.stop_bink_player();
        let mut m = self.media.lock().unwrap();
        m.job += 1;
        m.loaded_tracks = None;
        m.status.clear();
        m.loading = false;
        m.loaded_mesh = None;
        m.loaded_vehicle_parts.clear();
        m.loaded_textures = None;
        m.loaded_model_textures = None;
    }

    fn browse_videos(&mut self) {
        self.reset_selection();
        self.tab = AppTab::Explorer;
        self.category = Category::Videos;
        self.selected_rpf = "All RPFs".into();
        self.filter.clear();
    }

    fn open_obj_path(&mut self, path: &Path) {
        let Some(mesh) = parse_obj(path) else {
            self.state.lock().unwrap().status = format!("Could not read an OBJ model from {}", path.display());
            return;
        };
        self.reset_selection();
        self.mesh3d = Some(mesh);
        self.show_3d = true;
    }

    fn import_catalog_csv(&mut self, csv_path: &Path) {
        let base_dir = self.target_path.as_ref().map(|p| {
            if self.is_folder { p.clone() } else { p.parent().unwrap_or(p).to_path_buf() }
        }).or_else(load_last_folder).unwrap_or_else(|| PathBuf::from(r"D:\Grand Theft Auto VI Build"));

        let mut rpf_map: HashMap<String, PathBuf> = HashMap::new();
        let mut all_paths = Vec::new();
        collect_rpfs(&base_dir, &mut all_paths);
        for p in all_paths {
            if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
                rpf_map.insert(fname.to_ascii_lowercase(), p);
            }
        }

        let Ok(f) = File::open(csv_path) else {
            self.state.lock().unwrap().status = format!("Cannot open catalog: {}", csv_path.display());
            return;
        };
        let mut added = 0usize;
        let mut st = self.state.lock().unwrap();
        let mut existing: std::collections::HashSet<(String, u64)> = st.entries.iter()
            .map(|e| (e.rpf_name.to_ascii_lowercase(), e.offset))
            .collect();

        for (li, line) in BufReader::new(f).lines().map_while(Result::ok).enumerate() {
            if li == 0 && line.starts_with("archive,") { continue; }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 4 { continue; }
            let arch = cols[0].trim();
            let off_s = cols[1].trim().trim_start_matches("0x").trim_start_matches("0X");
            let Ok(off) = u64::from_str_radix(off_s, 16) else { continue };
            let raw_name = cols[2].trim();
            let kind_ext = cols[3].trim();
            if raw_name.is_empty() { continue; }

            let arch_file = Path::new(arch).file_name().and_then(|s| s.to_str()).unwrap_or(arch);
            let arch_lo = arch_file.to_ascii_lowercase();
            let src_path = rpf_map.get(&arch_lo).cloned().unwrap_or_else(|| base_dir.join(arch));
            if !existing.insert((arch_lo, off)) { continue; }

            let name = if raw_name.contains(".#") {
                raw_name.to_string()
            } else {
                format!("{raw_name}.#{kind_ext}")
            };
            st.entries.push(RpfEntry {
                source_rpf: src_path,
                rpf_name: arch_file.to_string(),
                name,
                kind: "MODEL",
                offset: off,
                comp_size: 0,
                raw_size: 0,
                tracks: 0,
                chunks: Vec::new(),
            });
            if !st.rpf_names.iter().any(|r| r.eq_ignore_ascii_case(arch_file)) {
                st.rpf_names.push(arch_file.to_string());
            }
            added += 1;
        }
        st.rpf_names.sort();
        st.ver += 1;
        st.status = format!("Imported {added} catalog models ({} total entries)", st.entries.len());
    }

    fn bump_gen(&self) -> u64 {
        let mut g = self.watch_gen.lock().unwrap();
        *g += 1;
        *g
    }

    fn start_single_scan(&mut self, path: PathBuf) {
        self.target_path = Some(path.clone());
        self.is_folder = false;
        self.selected_rpf = "All RPFs".into();
        self.reset_selection();
        self.vehicle_configuration_db.clear();
        self.vehicle_configuration_loaded = false;
        self.vehicle_configuration_loading = false;
        self.vehicle_configuration_generation = 0;
        let _gen = self.bump_gen();
        {
            let mut st = self.state.lock().unwrap();
            st.scanning = true;
            st.entries.clear();
            st.ver += 1;
            st.rpf_names.clear();
            st.file_sig.clear();
            st.status = "Scanning RPF9 archive...".into();
        }
        let state = Arc::clone(&self.state);
        std::thread::spawn(move || {
            if path.file_name().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("levels.rpf")).unwrap_or(false) {
                let _ = scan_levels_vehicle_zone(&path, Some(&state));
            }
            match scan_single_rpf(&path, Some(&state)) {
                Ok((hdr, mut entries)) => {
                    let mut st = state.lock().unwrap();
                    st.header_info = hdr;
                    st.rpf_names = vec![path.file_name().unwrap_or_default().to_string_lossy().into_owned()];
                    let n = entries.len();
                    st.entries.append(&mut entries);
                    st.ver += 1;
                    st.scanning = false;
                    st.status = format!("{n} entries ready");
                }
                Err(e) => {
                    let mut st = state.lock().unwrap();
                    st.scanning = false;
                    st.status = format!("Error: {e}");
                }
            }
        });
    }

    fn start_folder_scan(&mut self, folder: PathBuf) {
        if load_last_folder().as_ref() != Some(&folder) {
            self.selected_rpf = "All RPFs".into();
        }
        save_last_folder(&folder);
        self.target_path = Some(folder.clone());
        self.is_folder = true;
        self.reset_selection();
        self.vehicle_configuration_db.clear();
        self.vehicle_configuration_loaded = false;
        self.vehicle_configuration_loading = false;
        self.vehicle_configuration_generation = 0;
        let gen = self.bump_gen();
        {
            let mut st = self.state.lock().unwrap();
            st.entries.clear();
            st.ver += 1;
            st.rpf_names.clear();
            st.file_sig.clear();
            st.scanning = true;
            st.header_info = format!("Folder: {}", folder.display());
            st.status = "Scanning folder for .rpf archives...".into();
        }

        let state = Arc::clone(&self.state);
        let watch_gen = Arc::clone(&self.watch_gen);
        std::thread::spawn(move || {
            let mut disk_cache = load_disk_cache();
            loop {
                if *watch_gen.lock().unwrap() != gen { break; }
                let mut paths = Vec::new();
                collect_rpfs(&folder, &mut paths);
                let mut rpfs: Vec<(PathBuf, fs::Metadata)> = paths.into_iter().filter_map(|p| fs::metadata(&p).ok().map(|m| (p, m))).collect();

                // scan levels veh zone first so cars show up fast
                if let Some((levels_p, meta)) = rpfs.iter().find(|(p, _)| {
                    p.file_name().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("levels.rpf")).unwrap_or(false)
                }).cloned() {
                    {
                        if meta.len() > VEH_ZONE_OFF {
                            let mtime_s = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                            let vkey = format!("VEHZONE|{}|{}|{}", levels_p.display(), meta.len(), mtime_s);
                            let already_loaded = state.lock().unwrap().entries.iter().any(|e| e.kind == "VEH" && e.source_rpf == levels_p);
                            if !already_loaded {
                                if let Some(cached_vehs) = disk_cache.get(&vkey) {
                                    let mut st = state.lock().unwrap();
                                    st.entries.extend(cached_vehs.iter().cloned());
                                    st.ver += 1;
                                    if !st.rpf_names.contains(&"levels.rpf".to_string()) {
                                        st.rpf_names.push("levels.rpf".into());
                                        st.rpf_names.sort();
                                    }
                                    st.status = format!("Loaded {} cached vehicles!", cached_vehs.len());
                                } else {
                                    state.lock().unwrap().status = "Fast-indexing Vehicles (levels.rpf)...".into();
                                    let veh_entries = scan_levels_vehicle_zone(&levels_p, Some(&state));
                                    if !veh_entries.is_empty() {
                                        {
                                            let mut st = state.lock().unwrap();
                                            if !st.rpf_names.contains(&"levels.rpf".to_string()) {
                                                st.rpf_names.push("levels.rpf".into());
                                                st.rpf_names.sort();
                                            }
                                            st.status = format!("Indexed {} vehicle/mod models!", veh_entries.len());
                                        }
                                        disk_cache.insert(vkey, veh_entries);
                                        save_disk_cache(&disk_cache);
                                    }
                                }
                            }
                        }
                    }
                }

                // smallest archives first so xmls and audio pop in under 2 secs
                rpfs.sort_by_key(|(_, m)| m.len());

                let mut changed = Vec::new();
                let mut cache_hits = 0usize;
                let mut disk_cache_dirty = false;
                {
                    let mut st = state.lock().unwrap();
                    for (p, meta) in &rpfs {
                        {
                            let mtime_s = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                            let sig = (meta.len(), meta.modified().ok());
                            if st.file_sig.get(p) != Some(&sig) {
                                let key = format!("{}|{}|{}", p.display(), meta.len(), mtime_s);
                                if let Some(cached_entries) = disk_cache.get_mut(&key) {
                                    let is_audio = p.file_name().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("audio.rpf")).unwrap_or(false);
                                    if is_audio {
                                        let needs_enrich = !cached_entries.iter().any(|e| e.kind == "DAT")
                                            || cached_entries.iter().filter(|e| e.kind == "AWC").any(|e| e.name.contains("_Radio_Music_") || e.name.contains("_Score_Streams_"))
                                            || cached_entries.iter().filter(|e| e.kind == "AWC").take(10).all(|e| e.name.starts_with("sound_00") && e.name.matches('_').count() == 2);
                                        if needs_enrich {
                                            for e in cached_entries.iter_mut() {
                                                if e.kind == "AWC" {
                                                    e.name = display_entry_name(e, false).into_owned();
                                                }
                                            }
                                            st.audio_bank_meta = enrich_audio_rpf_entries(p, meta.len(), cached_entries);
                                            disk_cache_dirty = true;
                                        } else if st.audio_bank_meta.is_empty() {
                                            st.audio_bank_meta = enrich_audio_rpf_entries(p, meta.len(), cached_entries);
                                        }
                                    }
                                    let is_levels = p.file_name().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("levels.rpf")).unwrap_or(false);
                                    st.entries.retain(|e| &e.source_rpf != p || (is_levels && e.offset >= VEH_ZONE_OFF));
                                    st.entries.extend(cached_entries.iter().cloned());
                                    st.ver += 1;
                                    st.file_sig.insert(p.clone(), sig);
                                    let rname = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                                    if !st.rpf_names.contains(&rname) {
                                        st.rpf_names.push(rname);
                                    }
                                    cache_hits += 1;
                                } else {
                                    changed.push((p.clone(), sig, key));
                                }
                            }
                        }
                    }
                    if cache_hits > 0 {
                        st.rpf_names.sort();
                        st.status = format!("Loaded {cache_hits} cached RPFs ({} entries)...", st.entries.len());
                    }
                }
                if disk_cache_dirty {
                    save_disk_cache(&disk_cache);
                }

                if !changed.is_empty() {
                    state.lock().unwrap().scanning = true;
                    let total = changed.len();
                    let next_idx = Arc::new(Mutex::new(0usize));
                    let cache_arc = Arc::new(Mutex::new(disk_cache));
                    let last_save = Arc::new(Mutex::new(Instant::now()));
                    std::thread::scope(|s| {
                        for _ in 0..4 {
                            let state = Arc::clone(&state);
                            let watch_gen = Arc::clone(&watch_gen);
                            let next_idx = Arc::clone(&next_idx);
                            let cache_arc = Arc::clone(&cache_arc);
                            let last_save = Arc::clone(&last_save);
                            let changed = &changed;
                            s.spawn(move || loop {
                                if *watch_gen.lock().unwrap() != gen { return; }
                                let idx = {
                                    let mut g = next_idx.lock().unwrap();
                                    let i = *g;
                                    *g += 1;
                                    i
                                };
                                if idx >= total { return; }
                                let (p, sig, key) = &changed[idx];
                                let rname = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                                state.lock().unwrap().status = format!("Scanning [{}/{}] {}...", idx + 1, total, rname);
                                if let Ok((_, new_entries)) = scan_single_rpf(p, None) {
                                    {
                                        let mut st = state.lock().unwrap();
                                        let is_levels = rname.eq_ignore_ascii_case("levels.rpf");
                                        st.entries.retain(|e| &e.source_rpf != p || (is_levels && e.offset >= VEH_ZONE_OFF));
                                        st.entries.extend(new_entries.iter().cloned());
                                        st.ver += 1;
                                        st.file_sig.insert(p.clone(), *sig);
                                        if !st.rpf_names.contains(&rname) {
                                            st.rpf_names.push(rname);
                                            st.rpf_names.sort();
                                        }
                                    }
                                    let mut c = cache_arc.lock().unwrap();
                                    c.insert(key.clone(), new_entries);
                                    let mut ls = last_save.lock().unwrap();
                                    if ls.elapsed() > Duration::from_secs(45) {
                                        save_disk_cache(&c);
                                        *ls = Instant::now();
                                    }
                                }
                            });
                        }
                    });
                    disk_cache = Arc::try_unwrap(cache_arc).unwrap().into_inner().unwrap();
                    save_disk_cache(&disk_cache);
                    let mut st = state.lock().unwrap();
                    st.scanning = false;
                    st.status = format!("{} RPFs · {} entries (cached + auto-updating)", st.rpf_names.len(), st.entries.len());
                } else {
                    let mut st = state.lock().unwrap();
                    st.scanning = false;
                    if cache_hits > 0 {
                        st.status = format!("{} RPFs · {} entries (instant cache)", st.rpf_names.len(), st.entries.len());
                    }
                }

                for _ in 0..20 {
                    if *watch_gen.lock().unwrap() != gen { return; }
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        });
    }

    // other detail level files of e in the same archive stem_hi stem_mid stem
    fn find_lod_siblings(&self, e: &RpfEntry) -> Vec<(&'static str, usize)> {
        let Some(names) = lod_sibling_names(&e.name) else { return Vec::new() };
        let mut found: [Option<usize>; 3] = [None; 3];
        let st = self.state.lock().unwrap();
        for (i, o) in st.entries.iter().enumerate() {
            if o.kind != "MODEL" && o.kind != "VEH" {
                continue;
            }
            for (k, n) in names.iter().enumerate() {
                if found[k].is_none() && o.name.eq_ignore_ascii_case(n) && o.source_rpf == e.source_rpf {
                    found[k] = Some(i);
                }
            }
        }
        let out: Vec<_> = ["Hi", "Mid", "Low"].into_iter().zip(found).filter_map(|(l, f)| f.map(|i| (l, i))).collect();
        if out.len() < 2 { Vec::new() } else { out }
    }

    fn find_vehicle_parts(&self, e: &RpfEntry) -> Vec<VehiclePartCandidate> {
        if e.kind != "VEH" {
            return Vec::new();
        }
        let base = model_file_stem(&e.name);
        let prefix = format!("{base}_");
        let want_hi = e.name.rsplit_once('.').map(|x| x.0).unwrap_or(&e.name).to_ascii_lowercase().ends_with("_hi");
        let st = self.state.lock().unwrap();
        let mut by_stem: HashMap<String, (usize, bool)> = HashMap::new();
        for (idx, part) in st.entries.iter().enumerate() {
            if part.source_rpf != e.source_rpf || (part.kind != "MODEL" && part.kind != "VEH") {
                continue;
            }
            let raw = part.name.rsplit_once('.').map(|x| x.0).unwrap_or(&part.name).to_ascii_lowercase();
            let related = raw.starts_with(&prefix);
            let shared_emergency = raw.starts_with("cmn_siren_");
            if (!related && !shared_emergency)
                || matches!(raw.as_str(), x if x == format!("{base}_hi") || x == format!("{base}_mid"))
            {
                continue;
            }
            let logical = raw.strip_suffix("_hi").or_else(|| raw.strip_suffix("_mid")).unwrap_or(&raw).to_string();
            let detail_match = raw.ends_with("_hi") == want_hi;
            by_stem
                .entry(logical)
                .and_modify(|slot| {
                    if detail_match && !slot.1 {
                        *slot = (idx, true);
                    }
                })
                .or_insert((idx, detail_match));
        }
        let mut out: Vec<_> = by_stem
            .into_values()
            .map(|(entry, _)| VehiclePartCandidate {
                entry,
                name: st.entries[entry].name.clone(),
                enabled: false,
                loading: false,
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn ensure_vehicle_configuration_db(&mut self) {
        if self.vehicle_configuration_loading || self.vehicle_configuration_loaded {
            return;
        }
        let entries: Vec<RpfEntry> = self
            .state
            .lock()
            .unwrap()
            .entries
            .iter()
            .filter(|entry| {
                entry.rpf_name.eq_ignore_ascii_case("packs.rpf")
                    && matches!(entry.kind, "META7A" | "ANIM")
            })
            .cloned()
            .collect();
        if entries.is_empty() {
            return;
        }
        let generation = *self.watch_gen.lock().unwrap();
        self.vehicle_configuration_generation = generation;
        self.vehicle_configuration_loading = true;
        let media = Arc::clone(&self.media);
        std::thread::spawn(move || {
            let mut database = vehicle_variations::VehicleConfigurationDb::new();
            for entry in entries {
                let parsed = File::open(&entry.source_rpf)
                    .map_err(|error| error.to_string())
                    .and_then(|mut file| read_virtual_from_file(&mut file, entry.offset).map(|result| result.0))
                    .ok()
                    .filter(|bytes| meta7a::has_schema_enums(bytes))
                    .map(|bytes| vehicle_variations::parse_configurations(&bytes));
                if let Some(configurations) = parsed {
                    vehicle_variations::merge_configurations(&mut database, configurations);
                }
            }
            media.lock().unwrap().loaded_vehicle_configurations = Some((generation, database));
        });
    }

    fn refresh_vehicle_variations(&mut self) {
        let Some(model_index) = self.model_entry else {
            self.vehicle_variations.clear();
            return;
        };
        let st = self.state.lock().unwrap();
        let Some(model) = st.entries.get(model_index) else { return };
        if model.kind != "VEH" {
            self.vehicle_variations.clear();
            return;
        }
        let base = model_file_stem(&model.name);
        let model_hash = psin::joaat(&base);
        let Some(configurations) = self.vehicle_configuration_db.get(&model_hash) else {
            self.vehicle_variations.clear();
            self.vehicle_variation_selected = 0;
            return;
        };
        let wants_hi = model.name.rsplit_once('.').map(|part| part.0).unwrap_or(&model.name).to_ascii_lowercase().ends_with("_hi");
        let mut entries_by_hash: HashMap<u32, (usize, bool)> = HashMap::new();
        for (index, entry) in st.entries.iter().enumerate() {
            if (entry.kind != "MODEL" && entry.kind != "VEH") || entry.source_rpf != model.source_rpf {
                continue;
            }
            let raw = entry.name.rsplit_once('.').map(|part| part.0).unwrap_or(&entry.name).to_ascii_lowercase();
            let logical = raw.strip_suffix("_hi").or_else(|| raw.strip_suffix("_mid")).unwrap_or(&raw);
            let detail_match = raw.ends_with("_hi") == wants_hi;
            for hash in [psin::joaat(logical), psin::joaat(&raw)] {
                let preferred = vehicle_variations::prefer_lod_entry(entries_by_hash.get(&hash).copied(), index, detail_match);
                entries_by_hash.insert(hash, preferred);
            }
        }

        let mount_prefix = format!("{base}_siren_roofbar1_mount_");
        let implicit_mount = st
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| (entry.kind == "MODEL" || entry.kind == "VEH") && entry.source_rpf == model.source_rpf)
            .filter_map(|(index, entry)| {
                let raw = entry.name.rsplit_once('.').map(|part| part.0).unwrap_or(&entry.name).to_ascii_lowercase();
                raw.starts_with(&mount_prefix).then_some((index, raw.ends_with("_hi") == wants_hi, raw))
            })
            .min_by(|a, b| (!a.1, &a.2).cmp(&(!b.1, &b.2)))
            .map(|item| item.0);

        let entry_names: HashMap<usize, String> = entries_by_hash
            .values()
            .map(|&(entry, _)| (entry, st.entries[entry].name.clone()))
            .collect();
        self.vehicle_variations = configurations
            .iter()
            .enumerate()
            .map(|(index, configuration)| {
                let resolved = vehicle_variations::resolve_configuration_parts(
                    model_hash,
                    &configuration.parts,
                    &entries_by_hash,
                    &entry_names,
                );
                let entries = vehicle_variations::with_implicit_roofbar_mount(
                    resolved.entries,
                    implicit_mount,
                    resolved.needs_roofbar_mount,
                );
                let part_names = entries
                    .iter()
                    .filter_map(|&entry| st.entries.get(entry))
                    .map(|entry| model_file_stem(&entry.name))
                    .collect();
                ResolvedVehicleVariation {
                    label: format!("Configuration {:02}", index + 1),
                    colour: configuration.colour_argb.map(vehicle_variations::argb_to_linear_rgb),
                    entries,
                    part_names,
                    unresolved: resolved.unresolved,
                }
            })
            .collect();
        self.vehicle_variation_selected = self.vehicle_variation_selected.min(self.vehicle_variations.len());
    }

    fn apply_vehicle_variation(&mut self, selected: usize) {
        self.vehicle_variation_selected = selected.min(self.vehicle_variations.len());
        if self.vehicle_variation_selected > 0 {
            for part in &mut self.vehicle_parts {
                part.enabled = false;
            }
        }
        let entries = self
            .vehicle_variation_selected
            .checked_sub(1)
            .and_then(|index| self.vehicle_variations.get(index))
            .map(|variation| variation.entries.clone())
            .unwrap_or_default();
        for entry in entries {
            if !self.vehicle_part_meshes.contains_key(&entry) {
                self.spawn_vehicle_part_decode(entry);
            }
        }
        self.rebuild_vehicle_parts();
    }

    fn rebuild_vehicle_parts(&mut self) {
        let Some(mut mesh) = self.vehicle_base_mesh.clone() else { return };
        let mut appended = std::collections::HashSet::new();
        if let Some(variation) = self
            .vehicle_variation_selected
            .checked_sub(1)
            .and_then(|index| self.vehicle_variations.get(index))
        {
            for &entry in &variation.entries {
                if let Some(part) = self.vehicle_part_meshes.get(&entry) {
                    mesh.append_vehicle_part(part.clone());
                    appended.insert(entry);
                }
            }
        }
        for choice in &self.vehicle_parts {
            if choice.enabled && appended.insert(choice.entry) {
                if let Some(part) = self.vehicle_part_meshes.get(&choice.entry) {
                    mesh.append_vehicle_part(part.clone());
                }
            }
        }
        if let Some(current) = self.mesh3d.as_ref() {
            (mesh.yaw, mesh.pitch, mesh.zoom, mesh.pan_x, mesh.pan_y) =
                (current.yaw, current.pitch, current.zoom, current.pan_x, current.pan_y);
            mesh.paint_color = self
                .vehicle_variation_selected
                .checked_sub(1)
                .and_then(|index| self.vehicle_variations.get(index))
                .and_then(|variation| variation.colour)
                .unwrap_or(current.paint_color);
            mesh.els_enabled = current.els_enabled && mesh.has_els;
            mesh.els_flash = current.els_flash;
        }
        self.mesh3d = Some(mesh);
    }

    fn spawn_vehicle_part_decode(&self, entry: usize) {
        let Some(e) = self.state.lock().unwrap().entries.get(entry).cloned() else { return };
        let Some(owner) = self.model_entry else { return };
        let media = Arc::clone(&self.media);
        std::thread::spawn(move || {
            let result = decode_rage9_model_from_rpf_lod(&e.source_rpf, e.offset, &e.name, 0);
            let mut m = media.lock().unwrap();
            match result {
                Ok(mesh) => m.loaded_vehicle_parts.push((owner, entry, mesh)),
                Err(err) => m.status = format!("Vehicle part decode error ({}): {err}", e.name),
            }
        });
    }

    fn spawn_model_decode(&self, e: &RpfEntry, lod: usize) {
        let src = e.source_rpf.clone();
        let off = e.offset;
        let nm = e.name.clone();
        spawn_media_job(
            &self.media,
            format!("Decoding RAGE9 meshlets: {}...", e.name),
            true,
            move |_| {
                let mut mesh = decode_rage9_model_from_rpf_lod(&src, off, &nm, lod).map_err(|err| format!("Model decode error ({nm}): {err}"))?;
                // the viewer also applies normal maps unless the model binds an unusually large set
                let normals = mesh.tex_layers.iter().filter(|l| l.kind == model3d::TexLayerKind::Normal).count();
                if (1..=64).contains(&normals) {
                    mesh.set_tex_layers(|l| l.default_on || l.kind == model3d::TexLayerKind::Normal);
                }
                Ok(mesh)
            },
            |m, res| match res {
                Ok(mesh) => {
                    m.status.clear();
                    m.loaded_mesh = Some(mesh);
                }
                Err(err) => m.status = err,
            },
        );
    }

    fn vehicle_parts_ui(&mut self, ui: &mut egui::Ui) {
        if self.vehicle_parts.is_empty() {
            return;
        }
        let enabled = self.vehicle_parts.iter().filter(|p| p.enabled).count();
        let mut loads = Vec::new();
        let mut rebuild = false;
        egui::CollapsingHeader::new(format!("Advanced / Manual Parts ({enabled}/{})", self.vehicle_parts.len()))
            .id_salt("vehicle_parts")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.vehicle_part_filter)
                            .hint_text("Filter bumpers, sirens, liveries, wheels…")
                            .desired_width(320.0),
                    );
                    if ui.button("Clear selected").clicked() {
                        for part in &mut self.vehicle_parts {
                            part.enabled = false;
                        }
                        rebuild = true;
                    }
                });
                let filter = self.vehicle_part_filter.trim().to_ascii_lowercase();
                egui::ScrollArea::vertical()
                    .id_salt("vehicle_parts_list")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for part in &mut self.vehicle_parts {
                            if !filter.is_empty() && !part.name.to_ascii_lowercase().contains(&filter) {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                let changed = ui.checkbox(&mut part.enabled, &part.name).changed();
                                if part.loading {
                                    ui.spinner();
                                }
                                if changed {
                                    if part.enabled && !self.vehicle_part_meshes.contains_key(&part.entry) {
                                        part.loading = true;
                                        loads.push(part.entry);
                                    }
                                    rebuild = true;
                                }
                            });
                        }
                    });
                ui.label(
                    RichText::new("Parts are separate RAGE 9 fragments and are overlaid in vehicle model space.")
                        .size(10.5)
                        .color(Color32::from_rgb(135, 145, 165)),
                );
            });
        for entry in loads {
            self.spawn_vehicle_part_decode(entry);
        }
        if rebuild {
            self.rebuild_vehicle_parts();
        }
    }

    fn vehicle_variations_ui(&mut self, ui: &mut egui::Ui) {
        let is_vehicle = self
            .model_entry
            .and_then(|index| self.state.lock().unwrap().entries.get(index).map(|entry| entry.kind == "VEH"))
            .unwrap_or(false);
        if !is_vehicle {
            return;
        }
        self.ensure_vehicle_configuration_db();
        if self.vehicle_configuration_loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Reading authored vehicle configurations from packs.rpf…");
            });
            return;
        }
        if self.vehicle_configuration_db.is_empty() {
            ui.label(
                RichText::new("Authored variations unavailable: packs.rpf metadata was not found in the current scan.")
                    .size(11.0)
                    .color(Color32::from_rgb(180, 145, 110)),
            );
            return;
        }
        if self.vehicle_variations.is_empty() {
            ui.label(
                RichText::new("No authored fixed configurations were found for this vehicle.")
                    .size(11.0)
                    .color(Color32::from_rgb(150, 160, 180)),
            );
            return;
        }

        let selected_label = self
            .vehicle_variation_selected
            .checked_sub(1)
            .and_then(|index| self.vehicle_variations.get(index))
            .map(|variation| variation.label.as_str())
            .unwrap_or("Stock / base model");
        let mut selection = None;
        egui::CollapsingHeader::new(format!("Vehicle Variation · {selected_label}"))
            .id_salt("vehicle_variations")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.vehicle_variation_filter)
                            .hint_text("Filter configurations…")
                            .desired_width(260.0),
                    );
                    if ui.selectable_label(self.vehicle_variation_selected == 0, "Stock / base").clicked() {
                        selection = Some(0);
                    }
                });
                let filter = self.vehicle_variation_filter.trim().to_ascii_lowercase();
                egui::ScrollArea::vertical()
                    .id_salt("vehicle_variations_list")
                    .max_height(150.0)
                    .show(ui, |ui| {
                        for (index, variation) in self.vehicle_variations.iter().enumerate() {
                            if !filter.is_empty() && !variation.label.to_ascii_lowercase().contains(&filter) {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                if let Some(rgb) = variation.colour {
                                    let colour = Color32::from_rgb(
                                        (rgb[0] * 255.0).round() as u8,
                                        (rgb[1] * 255.0).round() as u8,
                                        (rgb[2] * 255.0).round() as u8,
                                    );
                                    ui.colored_label(colour, "■");
                                }
                                if ui
                                    .selectable_label(self.vehicle_variation_selected == index + 1, &variation.label)
                                    .clicked()
                                {
                                    selection = Some(index + 1);
                                }
                                ui.label(
                                    RichText::new(format!(
                                        "{} models{}",
                                        variation.part_names.len(),
                                        if variation.unresolved.is_empty() {
                                            String::new()
                                        } else {
                                            format!(" · {} ids with no model file", variation.unresolved.len())
                                        }
                                    ))
                                    .size(10.0)
                                    .color(Color32::from_rgb(135, 145, 165)),
                                );
                            });
                        }
                    });
                if let Some(variation) = self
                    .vehicle_variation_selected
                    .checked_sub(1)
                    .and_then(|index| self.vehicle_variations.get(index))
                {
                    if !variation.part_names.is_empty() {
                        ui.label(
                            RichText::new(format!("Models in this build: {}", variation.part_names.join(", ")))
                                .size(10.5)
                                .color(Color32::from_rgb(180, 210, 190)),
                        );
                    }
                    if !variation.unresolved.is_empty() {
                        ui.label(
                            RichText::new(format!(
                                "{} configuration ids do not match any model filename in the scanned archives.",
                                variation.unresolved.len()
                            ))
                            .size(10.5)
                            .color(Color32::from_rgb(170, 150, 130)),
                        );
                    }
                }
            });
        if let Some(selected) = selection {
            self.apply_vehicle_variation(selected);
        }
    }

    // read only physics and handling records of the viewed vehicle from data.rpf next to its archive
    fn vehicle_data_ui(&mut self, ui: &mut egui::Ui) {
        let Some(index) = self.model_entry else { return };
        let shared = Arc::clone(&self.vehicle_data);
        let mut slot = shared.lock().unwrap_or_else(|p| p.into_inner());
        {
            let st = self.state.lock().unwrap();
            let Some(model) = st.entries.get(index).filter(|entry| entry.kind == "VEH") else { return };
            let current = slot.model.as_ref().is_some_and(|(rpf, name)| *rpf == model.source_rpf && *name == model.name);
            if !current {
                let names: vehicle_data::Names = st
                    .entries
                    .iter()
                    .filter(|entry| entry.kind == "VEH")
                    .map(|entry| {
                        let stem = model_file_stem(&entry.name);
                        (psin::joaat(&stem), stem)
                    })
                    .collect();
                let key = (model.source_rpf.clone(), model.name.clone());
                let hash = psin::joaat(&model_file_stem(&model.name));
                *slot = VehicleDataState { model: Some(key.clone()), loading: true, ..Default::default() };
                let (shared, ctx) = (Arc::clone(&shared), ui.ctx().clone());
                std::thread::spawn(move || {
                    // a missing data.rpf or a model without an entry simply leaves the panel hidden
                    let data = std::panic::catch_unwind(|| {
                        let db = vehicle_data::load_from_folder(key.0.parent()?)?;
                        db.lookup(hash, &names).map(|data| (data, names))
                    })
                    .ok()
                    .flatten();
                    let mut slot = shared.lock().unwrap_or_else(|p| p.into_inner());
                    if slot.model.as_ref() == Some(&key) {
                        slot.loading = false;
                        if let Some((data, names)) = data {
                            slot.data = Some(data);
                            slot.names = names;
                        }
                    }
                    ctx.request_repaint();
                });
            }
        }
        if slot.loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Reading vehicle data from data.rpf…");
            });
            return;
        }
        if let Some(data) = slot.data.as_ref() {
            vehicle_data::panel_ui(ui, data, &slot.names);
        }
    }

    // decoded texture viewer with zoom export and a picker strip
    fn render_texture_viewer(&mut self, ui: &mut egui::Ui) {
        if self.tex_sel >= self.textures.len() {
            self.tex_sel = 0;
        }
        let t = &self.textures[self.tex_sel];
        let mut back = false;
        let mut export_all = false;
        ui.horizontal(|ui| {
            if self.tex_from_model && ui.button("◀ Back to model").clicked() {
                back = true;
            }
            ui.label(RichText::new(&t.name).strong().color(Color32::from_rgb(255, 168, 28)));
            ui.label(RichText::new(format!("{}x{} · {} · mips={}", t.w, t.h, t.fmt_name, t.mips)).color(Color32::from_rgb(170, 210, 235)));
            ui.add(egui::Slider::new(&mut self.tex_zoom, 0.25..=4.0).text("Zoom").fixed_decimals(2));
            if ui.button("Export PNG").clicked() {
                let safe: String = t.name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
                if let Some(p) = rfd::FileDialog::new().set_file_name(format!("{safe}.png")).add_filter("PNG Image", &["png"]).save_file() {
                    if let Err(err) = export_texture_png(t, &p) {
                        self.state.lock().unwrap().status = err;
                    } else {
                        self.state.lock().unwrap().status = format!("Saved {}", p.display());
                    }
                }
            }
            if ui.button("Export DDS").clicked() {
                let safe: String = t.name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
                if let Some(p) = rfd::FileDialog::new().set_file_name(format!("{safe}.dds")).add_filter("DirectDraw Surface", &["dds"]).save_file() {
                    let dds = build_dds_dx10(t.w, t.h, t.dxgi, &t.raw_dds_payload);
                    if let Err(err) = fs::write(&p, &dds) {
                        self.state.lock().unwrap().status = format!("DDS write error: {err}");
                    } else {
                        self.state.lock().unwrap().status = format!("Saved {}", p.display());
                    }
                }
            }
            if self.textures.len() > 1 && ui.button("Export All PNG").clicked() {
                export_all = true;
            }
        });
        if export_all {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                let mut ok = 0;
                for (i, t) in self.textures.iter().enumerate() {
                    let safe: String = t.name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
                    if export_texture_png(t, &dir.join(format!("{i:02}_{safe}.png"))).is_ok() {
                        ok += 1;
                    }
                }
                self.state.lock().unwrap().status = format!("Saved {ok}/{} textures to {}", self.textures.len(), dir.display());
            }
        }
        if back {
            self.textures.clear();
            self.tex_handle = None;
            self.tex_from_model = false;
            self.show_3d = true;
            return;
        }
        if self.textures.len() > 1 {
            ui.horizontal(|ui| {
                if ui.button("◀").clicked() {
                    self.tex_sel = if self.tex_sel == 0 { self.textures.len() - 1 } else { self.tex_sel - 1 };
                }
                ui.label(RichText::new(format!("{}/{}", self.tex_sel + 1, self.textures.len())).color(Color32::from_rgb(190, 180, 215)));
                if ui.button("▶").clicked() {
                    self.tex_sel = (self.tex_sel + 1) % self.textures.len();
                }
                egui::ScrollArea::horizontal().id_salt("txd_strip").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (i, item) in self.textures.iter().enumerate() {
                            if ui.selectable_label(self.tex_sel == i, format!("{} ({}x{})", item.name, item.w, item.h)).clicked() {
                                self.tex_sel = i;
                            }
                        }
                    });
                });
            });
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Channels:").size(11.0).color(Color32::from_rgb(150, 145, 175)));
            for (ch, lbl) in tex::TexChannel::ALL {
                if ui.selectable_label(self.tex_channel == ch, lbl).clicked() {
                    self.tex_channel = ch;
                }
            }
            ui.separator();
            ui.checkbox(&mut self.tex_rebuild_z, "Normal: rebuild Z").on_hover_text(
                "Recomputes blue Z = sqrt(1 - x^2 - y^2) from R/G for two-channel normal maps",
            );
        });
        ui.separator();
        let view_key = (self.tex_channel, self.tex_rebuild_z);
        let need_upload = self.tex_view_key != view_key
            || self.tex_handle.as_ref().map(|(idx, _)| *idx != self.tex_sel).unwrap_or(true);
        if need_upload {
            self.tex_view_key = view_key;
            let cur = &self.textures[self.tex_sel];
            // decoded texels are straight egui wants them premultiplied
            let pixels = if view_key == (tex::TexChannel::Rgba, false) {
                tex::premultiplied_for_egui(&cur.pixels)
            } else {
                tex::channel_view(&cur.pixels, view_key.0, view_key.1)
            };
            let ci = ColorImage { size: [cur.w, cur.h], pixels };
            let h = ui.ctx().load_texture(format!("percystar_tex_{}", self.tex_sel), ci, TextureOptions::LINEAR);
            self.tex_handle = Some((self.tex_sel, h));
        }
        if let Some((_, handle)) = &self.tex_handle {
            let cur = &self.textures[self.tex_sel];
            let avail = ui.available_size();
            let fit = (avail.x / cur.w as f32).min(avail.y / cur.h as f32).min(1.0).max(0.05);
            let disp = egui::vec2(cur.w as f32 * fit * self.tex_zoom, cur.h as f32 * fit * self.tex_zoom);
            egui::ScrollArea::both().show(ui, |ui| {
                ui.image((handle.id(), disp));
            });
        }
    }

    // lists and decodes every pooled texture the selected model binds in its shaders
    fn load_model_textures(&mut self) {
        let Some(e) = self.model_entry.and_then(|i| self.state.lock().unwrap().entries.get(i).cloned()) else { return };
        let name = e.name.clone();
        spawn_media_job(
            &self.media,
            format!("Resolving textures of {}...", e.name),
            false,
            move |progress| {
                let refs = model3d::list_model_texture_refs(&e.source_rpf, e.offset)?;
                let total = refs.len();
                let pooled: Vec<_> = refs.iter().filter(|r| r.hit.is_some()).collect();
                let mut texs = Vec::with_capacity(pooled.len());
                for (i, r) in pooled.iter().enumerate() {
                    progress(format!("Decoding textures of {} ({}/{})...", e.name, i + 1, pooled.len()));
                    let Some(hit) = &r.hit else { continue };
                    if let Ok(mut t) = decode_ttpf_texture(&hit.path, hit.tex_idx) {
                        let pool = hit.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                        t.name = format!("{} · {pool} #{}", r.role(), hit.tex_idx);
                        texs.push(t);
                    }
                }
                if texs.is_empty() {
                    return Err(format!("none of its {total} texture references are in the TTPF pools"));
                }
                let note = format!(
                    "{}: loaded {} of {total} bound textures ({} are shared engine defaults outside the TTPF pools)",
                    e.name,
                    texs.len(),
                    total - pooled.len()
                );
                Ok((texs, note))
            },
            move |m, res| match res {
                Ok(v) => {
                    m.status.clear();
                    m.loaded_model_textures = Some(v);
                }
                Err(err) => m.status = format!("Model textures ({name}): {err}"),
            },
        );
    }

    // switches the viewed model to another detail file or in file lod level keeping the camera
    fn apply_lod_pick(&mut self, pick: LodPick) {
        let cam = self.mesh3d.as_ref().map(|m| [m.yaw, m.pitch, m.zoom, m.pan_x, m.pan_y]);
        match pick {
            LodPick::File(idx) => {
                let show = self.show_3d;
                self.select_entry(idx);
                self.show_3d = show;
            }
            LodPick::Level(lod) => {
                let Some(e) = self.selected.and_then(|i| self.state.lock().unwrap().entries.get(i).cloned()) else { return };
                self.spawn_model_decode(&e, lod);
            }
        }
        self.lod_keep_cam = cam;
    }

    fn select_entry(&mut self, idx: usize) {
        let Some(e) = self.state.lock().unwrap().entries.get(idx).cloned() else { return };
        self.reset_selection();
        self.selected = Some(idx);

        if e.kind == "MODEL" || e.kind == "VEH" {
            self.lod_siblings = self.find_lod_siblings(&e);
            self.vehicle_parts = self.find_vehicle_parts(&e);
            self.model_entry = Some(idx);
            if e.kind == "VEH" {
                self.ensure_vehicle_configuration_db();
                self.refresh_vehicle_variations();
            }
            self.spawn_model_decode(&e, 0);
            return;
        }

        if e.kind == "NAVM" {
            let (src, off, nm) = (e.source_rpf.clone(), e.offset, e.name.clone());
            spawn_media_job(
                &self.media,
                format!("Decoding CNavMesh tile: {}...", e.name),
                true,
                move |_| decode_navmesh_from_rpf(&src, off, &nm).map_err(|err| format!("NavMesh decode error ({nm}): {err}")),
                |m, res| match res {
                    Ok(mesh) => {
                        m.status.clear();
                        m.loaded_mesh = Some(mesh);
                    }
                    Err(err) => m.status = err,
                },
            );
            return;
        }

        if e.kind == "TXD" || e.kind == "TTPF" {
            let (src, off, nm) = (e.source_rpf.clone(), e.offset, e.name.clone());
            let tidx = e.tracks as usize;
            let is_txd = e.kind == "TXD";
            let status = if is_txd {
                format!("Deswizzling Xbox Scarlett TXD: {}...", e.name)
            } else {
                format!("Decoding RDS5/ETPF texture: {}...", e.name)
            };
            spawn_media_job(
                &self.media,
                status,
                true,
                move |_| {
                    if is_txd {
                        decode_txd_from_rpf(&src, off).map_err(|err| format!("TXD decode error ({nm}): {err}"))
                    } else {
                        decode_ttpf_texture(&src, tidx).map(|t| vec![t]).map_err(|err| format!("TTPF decode error ({nm}): {err}"))
                    }
                },
                |m, res| match res {
                    Ok(texs) => {
                        m.status.clear();
                        m.loaded_textures = Some(texs);
                    }
                    Err(err) => m.status = err,
                },
            );
            return;
        }

        if e.kind == "BK2" {
            // per process folder and only our own clips are removed
            let tmp = std::env::temp_dir().join(format!("percystar_preview_bk2_{}", std::process::id()));
            let _ = fs::create_dir_all(&tmp);
            if let Ok(rd) = fs::read_dir(&tmp) {
                for old in rd.flatten() {
                    let is_clip = old.file_name().to_string_lossy().starts_with("clip_") && old.path().extension().is_some_and(|x| x == "bk2");
                    if is_clip && old.file_type().is_ok_and(|t| t.is_file()) {
                        let _ = fs::remove_file(old.path());
                    }
                }
            }
            // bump seq name so windows file locks never block switching vids
            self.clip_seq += 1;
            let bk2_p = tmp.join(format!("clip_{}.bk2", self.clip_seq));
            match read_entry_bytes(&e) {
                Ok(bytes) => {
                    if let Err(err) = fs::write(&bk2_p, &bytes) {
                        self.media.lock().unwrap().status = format!("Write error: {err}");
                        return;
                    }
                }
                Err(err) => {
                    self.media.lock().unwrap().status = format!("Read error: {err}");
                    return;
                }
            }
            let bink_path = find_binkplay();
            match Command::new(&bink_path)
                .arg(&bk2_p)
                .spawn()
            {
                Ok(child) => {
                    let pid = child.id();
                    self.bink_proc = Some(child);
                    let mut m = self.media.lock().unwrap();
                    m.bink_child_pid = Some(pid);
                    m.bink_clip = Some(bk2_p);
                    m.status = format!("Playing {} (D3D11 BinkPlay)", e.name);
                }
                Err(err) => {
                    self.media.lock().unwrap().status = format!(
                        "binkplay.exe error ({}): {err} — place binkplay.exe next to PercyStar.exe, add RADVideo to PATH, or set BINKPLAY_PATH",
                        bink_path.display()
                    );
                }
            }
        } else if e.kind == "AWC" {
            let bank_ctx = if e.rpf_name.eq_ignore_ascii_case("audio.rpf") {
                self.state.lock().unwrap().audio_bank_meta.get(&e.offset).map(|meta| {
                    audio_player::AudioBankContext {
                        pack_name: meta.pack_name.clone(),
                        used_by: meta.used_by.clone(),
                    }
                })
            } else {
                None
            };
            self.media.lock().unwrap().loaded_bank_ctx = bank_ctx;
            spawn_media_job(
                &self.media,
                format!("Decoding {} Opus streams...", e.name),
                true,
                move |progress| Ok::<_, String>(decode_awc_for_player(&e, progress)),
                |m, res| match res {
                    Ok((status, tracks)) => {
                        m.status = status;
                        m.loaded_tracks = tracks;
                    }
                    Err(err) => {
                        m.status = format!("Decoded 0 playable .wav stream(s) natively — {err}");
                        m.loaded_tracks = None;
                    }
                },
            );
        } else {
            match read_entry_bytes(&e) {
                Ok(data) => {
                    let sample = &data[..data.len().min(200)];
                    let is_text = e.kind == "XML"
                        || (!sample.is_empty()
                            && sample.iter().all(|&b| (32..127).contains(&b) || matches!(b, 9 | 10 | 13 | 0x80..)));
                    if e.kind == "HMAP" {
                        if let Some(t) = decode_mhiu_heightmap(&data) {
                            self.textures = vec![t];
                            self.tex_sel = 0;
                            self.tex_handle = None;
                        } else {
                            self.preview = format_hex(&data[..data.len().min(2048)]);
                        }
                    } else if e.kind == "PSIN" {
                        // typed xml when the schema is readable, the plain dump otherwise
                        match psin_xml::export_xml(&data, PSIN_PREVIEW_CHARS) {
                            Ok(xml) => {
                                self.preview = xml;
                                self.psin_selected = true;
                            }
                            Err(_) => self.preview = psin::parse_psin(&data),
                        }
                    } else if e.kind == "ANIM" {
                        self.preview = anim::parse_anim_dict(&data);
                        let (dur, trk, seed) = anim::extract_anim_summary(&data);
                        self.anim_dur = dur;
                        self.anim_tracks = trk;
                        self.anim_seed = seed;
                        self.anim_name = e.name.clone();
                        self.anim_time = 0.0;
                        self.anim_playing = true;
                        self.mesh3d = Some(anim::build_stickman_mesh(&e.name, 0.0, dur, trk, seed));
                        self.show_3d = true;
                    } else if e.kind == "DAT" {
                        self.preview = dat::parse_dat(&data);
                    } else if e.kind == "RSC9" {
                        self.preview = rsc9::parse_rsc9(&data);
                    } else if e.kind == "RBF0" {
                        self.preview = decode_rbf0(&data).unwrap_or_else(|| format_hex(&data[..data.len().min(2048)]));
                    } else if e.kind == "GFX" {
                        self.preview = inspect_gfx(&data);
                    } else if e.kind == "GXT" {
                        self.preview = decode_0drpz_text(&data).unwrap_or_else(|| format_hex(&data[..data.len().min(2048)]));
                    } else if is_text {
                        self.preview = String::from_utf8_lossy(&data[..data.len().min(250_000)]).into_owned();
                    } else {
                        if let Some(txt) = decode_0drpz_text(&data) {
                            self.preview = txt;
                        } else {
                            if data.len() >= 24 && &data[16..21] == b"0DRPz" {
                                self.mesh3d = extract_0drpz_mesh(&e.name, &data);
                            }
                            self.preview = format_hex(&data[..data.len().min(2048)]);
                        }
                    }
                }
                Err(err) => self.preview = format!("Read error: {err}"),
            }
        }
    }

    // throtled view rebuilder runs 4x a sec during live scans
    fn refresh_view(&mut self) {
        let ver = self.state.lock().unwrap().ver;
        let key = (ver, self.filter.clone(), self.category.clone(), self.selected_rpf.clone(), self.vehicles_only, self.audio_named);
        if self.view_sort != self.sort_mode {
            self.view_key = None;
        }
        if let Some(old) = &self.view_key {
            if *old == key { return; }
            let only_ver = old.1 == key.1 && old.2 == key.2 && old.3 == key.3 && old.4 == key.4 && old.5 == key.5;
            if only_ver && self.last_rebuild.elapsed() < Duration::from_millis(1000) {
                return;
            }
            if !only_ver {
                save_ui_state(self);
            }
        }
        self.rebuild_view(key);
    }

    fn rebuild_view(&mut self, key: (u64, String, Category, String, bool, bool)) {
        let st = self.state.lock().unwrap();
        if self.hay_ver != st.ver || self.hay_audio_named != self.audio_named {
            self.hay = build_hay(&st.entries);
            self.cached_rpf_names = st.rpf_names.clone();
            self.hay_ver = st.ver;
            self.hay_audio_named = self.audio_named;
            let mut c = [0usize; 11];
            for e in &st.entries {
                c[0] += 1;
                c[match e.kind {
                    "VEH" => 1,
                    "MODEL" => 2,
                    "TXD" | "TTPF" | "HMAP" => 3,
                    "RBF0" | "GFX" | "GXT" => 4,
                    "NAVM" => 5,
                    "BK2" => 6,
                    "AWC" => 7,
                    "XML" => 8,
                    "PSIN" | "ANIM" | "RSC9" | "META7A" => 9,
                    _ => 10,
                }] += 1;
            }
            self.counts = c;
        }
        let q = key.1.to_lowercase();
        let (inc, exc): (Vec<&str>, Vec<&str>) = q.split_whitespace().partition(|t| !t.starts_with('-'));
        let exc: Vec<&str> = exc.iter().map(|t| &t[1..]).filter(|t| !t.is_empty()).collect();
        let all_rpf = self.selected_rpf == "All RPFs";
        let (entries, hay, category, vehicles_only, selected_rpf) = (&st.entries, &self.hay, &self.category, self.vehicles_only, &self.selected_rpf);
        let mut view: Vec<usize> = par_filter(entries.len().min(hay.len()), |i| {
                let e = &entries[i];
                let cat_ok = match category {
                    Category::All => true,
                    Category::Vehicles => e.kind == "VEH",
                    Category::Models => e.kind == "VEH" || (!vehicles_only && e.kind == "MODEL"),
                    Category::Textures => e.kind == "TXD" || e.kind == "TTPF" || e.kind == "HMAP",
                    Category::Ui => e.kind == "RBF0" || e.kind == "GFX" || e.kind == "GXT",
                    Category::NavMesh => e.kind == "NAVM",
                    Category::Videos => e.kind == "BK2",
                    Category::Audio => e.kind == "AWC",
                    Category::Xml => e.kind == "XML",
                    Category::Resources => matches!(e.kind, "PSIN" | "ANIM" | "RSC9" | "META7A"),
                    Category::Binary => e.kind == "ZLIB",
                };
                let h = &hay[i];
                cat_ok
                    && (all_rpf || e.rpf_name == *selected_rpf)
                    && inc.iter().all(|t| h.contains(t))
                    && !exc.iter().any(|t| h.contains(t))
            });
        match self.sort_mode {
            SortMode::Relevance => {
                if let Some(first) = inc.first() {
                    view.sort_unstable_by_key(|&i| (!self.hay[i].starts_with(first), self.hay[i].len(), i));
                }
            }
            SortMode::SizeDesc => view.sort_unstable_by_key(|&i| (std::cmp::Reverse(st.entries[i].raw_size), i)),
            SortMode::SizeAsc => view.sort_unstable_by_key(|&i| (st.entries[i].raw_size, i)),
            SortMode::Name => view.sort_unstable_by(|&a, &b| self.hay[a].cmp(&self.hay[b]).then(a.cmp(&b))),
        }
        drop(st);
        self.view = view;
        self.view_sort = self.sort_mode;
        self.view_key = Some(key);
        self.last_rebuild = Instant::now();
    }

    fn find_local_catalog(&self, fname: &str) -> Option<PathBuf> {
        let mut dirs = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(p) = exe.parent() {
                dirs.push(p.join("models_catalog"));
                dirs.push(p.to_path_buf());
            }
        }
        if let Some(tp) = &self.target_path {
            dirs.push(tp.join("models_catalog"));
            if let Some(par) = tp.parent() {
                dirs.push(par.join("models_catalog"));
            }
        }
        dirs.push(PathBuf::from(r"F:\Ext6\models_catalog"));
        for d in dirs {
            let p = d.join(fname);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    fn refresh_mv_view(&mut self) {
        let ver = self.state.lock().unwrap().ver;
        if self.mv_ver != ver {
            let st = self.state.lock().unwrap();
            let mut counts = [0usize; 10];
            let video_count = st.entries.iter().filter(|e| e.kind == "BK2").count();
            let models = par_filter(st.entries.len(), |i| matches!(st.entries[i].kind, "MODEL" | "VEH" | "NAVM"));
            let entries: Vec<(usize, ModelCat, String)> = par_map(&models, |&idx| {
                let e = &st.entries[idx];
                (idx, classify_model_cat(e.kind, &e.name), format!("{}|{}", e.name, e.rpf_name).to_ascii_lowercase())
            });
            for (_, cat, _) in &entries {
                if *cat != ModelCat::NavMesh {
                    counts[0] += 1;
                }
                counts[model_cat_idx(*cat)] += 1;
            }
            drop(st);
            self.mv_entries = entries;
            self.mv_counts = counts;
            self.mv_video_count = video_count;
            self.mv_ver = ver;
            self.mv_view_key = None;
        }
        let key = (ver, self.mv_cat, self.mv_filter.clone(), self.mv_hide_lods);
        if self.mv_sort != self.sort_mode {
            self.mv_view_key = None;
        }
        if self.mv_view_key.as_ref() == Some(&key) {
            return;
        }
        let q = key.2.to_ascii_lowercase();
        let (inc, exc): (Vec<&str>, Vec<&str>) = q.split_whitespace().partition(|t| !t.starts_with('-'));
        let exc: Vec<&str> = exc.iter().map(|t| &t[1..]).filter(|t| !t.is_empty()).collect();
        let (mv_entries, mv_cat, hide_lods) = (&self.mv_entries, self.mv_cat, self.mv_hide_lods);
        let mut view: Vec<usize> = par_filter(mv_entries.len(), |i| {
                let (_, cat, hay) = &mv_entries[i];
                let cat_ok = match mv_cat {
                    ModelCat::All => *cat != ModelCat::NavMesh,
                    other => *cat == other,
                };
                let lod_ok = !hide_lods
                    || mv_cat == ModelCat::NavMesh
                    || (!hay.contains("_slod")
                        && !hay.contains("_lod.")
                        && !hay.contains("_lod_")
                        && !hay.contains("batchlod")
                        && !hay.contains("_mid.")
                        && !hay.contains("bipedbounds")
                        && !hay.contains("boundprims")
                        && !hay.contains("collision_")
                        && !hay.contains("_bounds."));
                cat_ok
                    && lod_ok
                    && inc.iter().all(|t| hay.contains(t))
                    && !exc.iter().any(|t| hay.contains(t))
            });
        {
            let st = self.state.lock().unwrap();
            let size = |mi: usize| st.entries.get(self.mv_entries[mi].0).map_or(0, |e| e.raw_size);
            match self.sort_mode {
                SortMode::Relevance => {
                    if let Some(first) = inc.first() {
                        view.sort_unstable_by_key(|&i| (!self.mv_entries[i].2.starts_with(first), self.mv_entries[i].2.len(), i));
                    }
                }
                SortMode::SizeDesc => view.sort_unstable_by_key(|&i| (std::cmp::Reverse(size(i)), i)),
                SortMode::SizeAsc => view.sort_unstable_by_key(|&i| (size(i), i)),
                SortMode::Name => view.sort_unstable_by(|&a, &b| self.mv_entries[a].2.cmp(&self.mv_entries[b].2).then(a.cmp(&b))),
            }
        }
        self.mv_view = view;
        self.mv_sort = self.sort_mode;
        self.mv_view_key = Some(key);
    }

    fn next_rand(&mut self) -> usize {
        let mut x = self.mv_rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.mv_rng = x;
        x as usize
    }

    fn render_model_viewer_tab(&mut self, ctx: &egui::Context, media_loading: bool, media_status: &str) {
        self.refresh_mv_view();
        let mut to_select: Option<usize> = None;
        let mut browse_videos = false;

        if self.mv_drawer_open {
            model_viewer_sidebar(ctx)
                .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("MODELS").size(12.0).strong().color(Color32::from_rgb(255, 168, 28)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} shown · {} total", self.mv_view.len(), self.mv_counts[0]))
                                .size(11.0)
                                .color(Color32::from_rgb(155, 145, 180)),
                        );
                    });
                });
                ui.add_space(5.0);

                let sr = model_search_row(ui, &mut self.mv_filter);
                if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F)) {
                    sr.request_focus();
                }
                if sr.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.mv_filter.clear();
                }

                ui.add_space(5.0);
                ui.horizontal_wrapped(|ui| {
                    for (cat, label) in MODEL_CATS {
                        let cnt = self.mv_counts[model_cat_idx(cat)];
                        let active = self.mv_cat == cat;
                        let (_, col) = model_cat_tag(cat);
                        let txt = RichText::new(format!("{label} {cnt}"))
                            .size(11.0)
                            .strong()
                            .color(if active { Color32::WHITE } else { col });
                        let btn = egui::Button::new(txt)
                            .fill(if active { Color32::from_rgb(28, 122, 198) } else { Color32::from_rgb(32, 20, 48) });
                        if ui.add(btn).clicked() {
                            self.mv_cat = cat;
                        }
                    }
                    let videos = RichText::new(format!("Videos {}", self.mv_video_count))
                        .size(11.0)
                        .strong()
                        .color(kind_color("BK2"));
                    browse_videos = ui.add(egui::Button::new(videos).fill(Color32::from_rgb(32, 20, 48)))
                        .on_hover_text("Browse all .bk2 clips in Explorer").clicked();
                });

                ui.add_space(4.0);
                let curated_p = self.find_local_catalog("models_vehicles_weapons_peds.csv");
                let master_p = self.find_local_catalog("models_catalog_master.csv");
                ui.horizontal_wrapped(|ui| {
                    if let Some(cp) = curated_p {
                        if ui.button("★ Curated (32k)").on_hover_text("Load vehicles, weapons, characters, animals & fragtypes").clicked() {
                            self.import_catalog_csv(&cp);
                            self.refresh_mv_view();
                        }
                    }
                    if let Some(mp) = master_p {
                        if ui.button("Master (144k)").on_hover_text("Load all 144,655 discovered 3D models").clicked() {
                            self.import_catalog_csv(&mp);
                            self.refresh_mv_view();
                        }
                    }
                    if ui.button("Import CSV...").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("Models Catalog CSV", &["csv"]).pick_file() {
                            self.import_catalog_csv(&p);
                            self.refresh_mv_view();
                        }
                    }
                    if ui.button("Open .OBJ").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("OBJ 3D Model", &["obj"]).pick_file() {
                            self.open_obj_path(&p);
                        }
                    }
                });

                ui.add_space(5.0);
                self.refresh_mv_view();
                let cur_pos = self.selected.and_then(|sel| {
                    self.mv_view.iter().position(|&mi| self.mv_entries[mi].0 == sel)
                });
                if model_browser_options(ui, &mut self.mv_hide_lods, &mut self.sort_mode, !self.mv_view.is_empty()) {
                    let r = self.next_rand() % self.mv_view.len();
                    to_select = Some(self.mv_entries[self.mv_view[r]].0);
                }
                ui.label(RichText::new("Up/Down arrow keys step through the list").size(10.5).color(Color32::from_rgb(130, 134, 150)));

                // Arrow key / W-S navigation when search box is not focused
                if !ctx.wants_keyboard_input() && !self.mv_view.is_empty() {
                    if ui.input(|i| i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::S)) {
                        let next_i = match cur_pos {
                            Some(p) if p + 1 < self.mv_view.len() => p + 1,
                            _ => 0,
                        };
                        to_select = Some(self.mv_entries[self.mv_view[next_i]].0);
                    } else if ui.input(|i| i.key_pressed(egui::Key::ArrowUp) || i.key_pressed(egui::Key::W)) {
                        let next_i = match cur_pos {
                            Some(0) | None => self.mv_view.len() - 1,
                            Some(p) => p - 1,
                        };
                        to_select = Some(self.mv_entries[self.mv_view[next_i]].0);
                    }
                }

                ui.add_space(4.0);
                ui.separator();

                const ROW_H: f32 = 22.0;
                let mut copied_msg: Option<String> = None;
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                egui::ScrollArea::vertical()
                    .id_salt("mv_list_scroll")
                    .auto_shrink([false, false])
                    .show_rows(ui, ROW_H, self.mv_view.len(), |ui, range| {
                        let rows: Vec<(usize, ModelCat, RpfEntry)> = {
                            let st = self.state.lock().unwrap();
                            self.mv_view[range]
                                .iter()
                                .filter_map(|&mi| {
                                    let (eidx, cat, _) = &self.mv_entries[mi];
                                    st.entries.get(*eidx).map(|e| (*eidx, *cat, e.clone()))
                                })
                                .collect()
                        };
                        for (eidx, cat, e) in &rows {
                            let w = ui.available_width();
                            let (tag, col) = model_cat_tag(*cat);
                            ui.allocate_ui_with_layout(egui::vec2(w, ROW_H), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                ui.set_height(ROW_H);
                                ui.label(RichText::new(tag).strong().size(10.5).color(col));
                                let label = format!("{}  ·  {}  ·  {}", e.name, e.rpf_name, format_size(e.raw_size));
                                let resp = ui.selectable_label(self.selected == Some(*eidx), RichText::new(label).size(12.0));
                                if resp.clicked() {
                                    to_select = Some(*eidx);
                                }
                                if let Some(msg) = entry_context_menu(&resp, e) {
                                    copied_msg = Some(msg);
                                }
                            });
                        }
                    });
                if let Some(msg) = copied_msg {
                    self.state.lock().unwrap().status = msg;
                }
            });
        }

        if browse_videos {
            self.browse_videos();
            ctx.request_repaint();
            return;
        }

        if let Some(i) = to_select {
            self.select_entry(i);
            self.show_3d = true;
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::from_rgb(13, 9, 21)).inner_margin(12.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let dr_btn = egui::Button::new(RichText::new("☰ Models").strong().color(Color32::WHITE))
                        .fill(if self.mv_drawer_open { Color32::from_rgb(28, 122, 198) } else { Color32::from_rgb(36, 24, 54) });
                    if ui.add(dr_btn).clicked() {
                        self.mv_drawer_open = !self.mv_drawer_open;
                    }
                    let has_any = !self.mv_view.is_empty();
                    let cur_pos = self.selected.and_then(|sel| self.mv_view.iter().position(|&mi| self.mv_entries[mi].0 == sel));
                    if ui.add_enabled(has_any, egui::Button::new("◀")).clicked() {
                        let next_i = match cur_pos { Some(0) | None => self.mv_view.len() - 1, Some(p) => p - 1 };
                        self.select_entry(self.mv_entries[self.mv_view[next_i]].0);
                        self.show_3d = true;
                    }
                    if ui.add_enabled(has_any, egui::Button::new("▶")).clicked() {
                        let next_i = match cur_pos { Some(p) if p + 1 < self.mv_view.len() => p + 1, _ => 0 };
                        self.select_entry(self.mv_entries[self.mv_view[next_i]].0);
                        self.show_3d = true;
                    }
                    let sel_label = self.selected.and_then(|s| self.state.lock().unwrap().entries.get(s).map(|e| format!("{}@{}", e.rpf_name, e.offset)))
                        .unwrap_or_else(|| "none".into());
                    ui.label(RichText::new(format!("Model: {sel_label}")).size(12.0).color(Color32::from_rgb(185, 190, 210)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Open .OBJ").clicked() {
                            if let Some(p) = rfd::FileDialog::new().add_filter("OBJ 3D Model", &["obj"]).pick_file() {
                                self.open_obj_path(&p);
                            }
                        }
                    });
                });
                ui.separator();
                if media_loading {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            draw_percystar_spinner(ui, 42.0);
                            ui.add_space(10.0);
                            ui.label(RichText::new(media_status).size(14.0).color(Color32::from_rgb(252, 175, 23)));
                        });
                    });
                } else if !media_status.is_empty() && self.mesh3d.is_none() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.label(RichText::new(media_status).size(14.0).color(Color32::from_rgb(255, 130, 115)));
                        ui.add_space(10.0);
                        if !self.mv_view.is_empty() && ui.button("Try Random Model").clicked() {
                            let r = self.next_rand() % self.mv_view.len();
                            let eidx = self.mv_entries[self.mv_view[r]].0;
                            self.select_entry(eidx);
                        }
                    });
                } else if self.tex_from_model && !self.textures.is_empty() {
                    self.render_texture_viewer(ui);
                } else if self.mesh3d.is_some() {
                    self.vehicle_variations_ui(ui);
                    self.vehicle_parts_ui(ui);
                    self.vehicle_data_ui(ui);
                    let mesh = self.mesh3d.as_mut().unwrap();
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(&mesh.name).size(15.0).strong().color(Color32::from_rgb(255, 168, 28)));
                        ui.label(
                            RichText::new(format!(
                                "{} verts · {} tris · {} geoms · {} meshlets · {}",
                                mesh.verts.len(),
                                mesh.tris.len(),
                                mesh.geom_count,
                                mesh.meshlet_count,
                                mesh.lod_label
                            ))
                            .size(11.5)
                            .color(Color32::from_rgb(170, 200, 230)),
                        );
                        if let Some(pick) = lod_switch_ui(ui, mesh, &self.lod_siblings, self.selected) {
                            self.lod_pick = Some(pick);
                        }
                        if self.model_entry.is_some() && ui.button("Texture Browser").on_hover_text(MODEL_TEX_TIP).clicked() {
                            self.tex_pick = true;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // Quick camera presets
                            for (lbl, yaw, pitch) in [
                                ("Top", 0.0_f32, 1.48_f32),
                                ("Back", std::f32::consts::PI, 0.12),
                                ("Side", std::f32::consts::FRAC_PI_2, 0.12),
                                ("Front", 0.0, 0.12),
                                ("3/4", 2.45, 0.38),
                            ] {
                                if ui.button(lbl).clicked() {
                                    mesh.yaw = yaw;
                                    mesh.pitch = pitch;
                                    mesh.pan_x = 0.0;
                                    mesh.pan_y = 0.0;
                                    mesh.dirty = true;
                                }
                            }
                            ui.label(RichText::new("Cam:").size(11.0).color(Color32::from_rgb(130, 138, 156)));
                        });
                    });
                    ui.separator();
                    render_mesh3d_viewport(ui, mesh);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(RichText::new("PERCYSTAR 3D MODEL VIEWER").size(20.0).strong().color(Color32::from_rgb(255, 168, 28)));
                            ui.add_space(6.0);
                            if self.mv_view.is_empty() {
                                ui.label(
                                    RichText::new("Open a GTA VI folder, import a catalog CSV (32k / 144k models), or open a .OBJ file.")
                                        .size(13.0)
                                        .color(Color32::from_rgb(185, 180, 210)),
                                );
                            } else {
                                ui.label(
                                    RichText::new(format!(
                                        "{} 3D models ready across {} categories — pick any model on the left or click below.",
                                        self.mv_view.len(),
                                        MODEL_CATS.len() - 1
                                    ))
                                    .size(13.0)
                                    .color(Color32::from_rgb(185, 180, 210)),
                                );
                                ui.add_space(12.0);
                                let btn = egui::Button::new(RichText::new("★ View Random 3D Model").size(14.0).strong().color(Color32::BLACK))
                                    .fill(Color32::from_rgb(255, 168, 28));
                                if ui.add(btn).clicked() {
                                    let r = self.next_rand() % self.mv_view.len();
                                    let eidx = self.mv_entries[self.mv_view[r]].0;
                                    self.select_entry(eidx);
                                }
                            }
                        });
                    });
                }
            });
    }

    fn load_ps09_path(&mut self, path: &Path) {
        match load_ps09_from_path(path) {
            Ok(cache) => {
                let st_msg = format!(
                    "Loaded PS\\x00\\x09 ({}, drv {}): {} shaders · {} Gfx PSOs · {} Compute PSOs",
                    cache.header.id,
                    cache.header.driver,
                    cache.shaders.len(),
                    cache.header.graphics_pso_count,
                    cache.header.compute_pso_count,
                );
                self.sh_cache = Some(cache);
                self.sh_selected = Some(0);
                self.sh_cached_text = (None, String::new());
                self.sh_status = st_msg;
            }
            Err(e) => {
                self.sh_status = format!("PS\\x00\\x09 load error: {e}");
            }
        }
    }

    fn find_local_shaders_dir(&self) -> Option<PathBuf> {
        let mut dirs = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(d) = exe.parent() {
                dirs.push(d.join("shaders_rpf").join("zlib"));
                dirs.push(d.join("shaders.rpf"));
            }
        }
        dirs.push(PathBuf::from(r"F:\Ext6\shaders_rpf\zlib"));
        dirs.into_iter().find(|p| p.exists())
    }

    fn spawn_sgd5_extraction(&self, ctx: egui::Context, input: PathBuf, output: PathBuf) {
        {
            let mut job = self.sgd5_job.lock().unwrap();
            if job.running {
                return;
            }
            job.running = true;
            job.status = format!("Extracting SGD5 from {}...", input.display());
        }

        let job_state = self.sgd5_job.clone();
        std::thread::spawn(move || {
            let result = sgd5::extract_sgd5(&input, &output, 0);
            let (status, files) = match result {
                Ok(extraction) => {
                    let effect_count = extraction
                        .files
                        .iter()
                        .map(|file| file.effect_count as usize)
                        .sum::<usize>();
                    let status = format!(
                        "Extracted {} DXBC shaders from {} SGD5 groups ({} effects) to {}",
                        extraction.extracted_count,
                        extraction.files.len(),
                        effect_count,
                        output.display()
                    );
                    (status, Some(Arc::new(extraction.files)))
                }
                Err(error) => (format!("SGD5 extraction failed: {error}"), None),
            };
            {
                let mut job = job_state.lock().unwrap();
                job.running = false;
                job.status = status;
                job.files = files;
            }
            ctx.request_repaint();
        });
    }

    fn spawn_sgd5_load(&mut self, ctx: egui::Context, input: PathBuf) {
        {
            let mut job = self.sgd5_job.lock().unwrap();
            if job.running {
                return;
            }
            job.running = true;
            job.status = format!("Loading SGD5 from {}...", input.display());
        }
        self.sh_source = ShaderSource::Sgd5;

        let job_state = self.sgd5_job.clone();
        std::thread::spawn(move || {
            let result = sgd5::load_sgd5(&input);
            let (status, files) = match result {
                Ok(files) => {
                    let shader_count = files.iter().map(|file| file.shaders.len()).sum::<usize>();
                    (
                        format!(
                            "Loaded {shader_count} SGD5 shaders from {} groups",
                            files.len()
                        ),
                        Some(Arc::new(files)),
                    )
                }
                Err(error) => (format!("SGD5 load failed: {error}"), None),
            };
            {
                let mut job = job_state.lock().unwrap();
                job.running = false;
                job.status = status;
                job.files = files;
            }
            ctx.request_repaint();
        });
    }

    fn poll_sgd5_cache(&mut self) {
        let files = self.sgd5_job.lock().unwrap().files.take();
        if let Some(files) = files {
            self.sgd5_group = 0;
            self.sgd5_selected = Some(0);
            self.sgd5_cached_text = (None, String::new());
            self.sgd5_cache = Some(files);
            if self.sh_cache.is_none() {
                self.sh_source = ShaderSource::Sgd5;
            }
        }
    }

    fn render_shader_tab(&mut self, ctx: &egui::Context) {
        let local_sh = self.find_local_shaders_dir();
        egui::SidePanel::left("sh_sidebar")
            .default_width(355.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(20, 13, 31))
                    .inner_margin(10.0),
            )
            .show(ctx, |ui| {
                ui.label(
                    RichText::new("RDNA2 (GFX10.3) SHADER DATA")
                        .size(12.0)
                        .strong()
                        .color(Color32::from_rgb(72, 210, 255)),
                );
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    for (source, label) in [
                        (ShaderSource::Ps09, "PS09 Runtime Cache"),
                        (ShaderSource::Sgd5, "SGD5 DXBC Groups"),
                    ] {
                        let active = self.sh_source == source;
                        let button = egui::Button::new(
                            RichText::new(label)
                                .size(12.0)
                                .strong()
                                .color(if active { Color32::BLACK } else { Color32::from_rgb(225, 215, 240) }),
                        )
                        .fill(if active { Color32::from_rgb(72, 210, 255) } else { Color32::from_rgb(36, 22, 54) });
                        if ui.add(button).clicked() {
                            self.sh_source = source;
                        }
                    }
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if let Some(lp) = local_sh {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("★ Load Local PS09")
                                        .color(Color32::BLACK)
                                        .strong(),
                                )
                                .fill(Color32::from_rgb(255, 168, 28)),
                            )
                            .clicked()
                        {
                            self.load_ps09_path(&lp);
                        }
                    }
                    if ui.button("Open shaders.rpf (PS09 + SGD5)").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("RAGE 9 Shaders", &["rpf"])
                            .pick_file()
                        {
                            self.load_ps09_path(&p);
                            self.spawn_sgd5_load(ctx.clone(), p);
                            self.sh_source = ShaderSource::Ps09;
                        }
                    }
                    if ui.button("Open PS09 .ps09").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("PS09 Cache", &["ps09"])
                            .pick_file()
                        {
                            self.load_ps09_path(&p);
                            self.sh_source = ShaderSource::Ps09;
                        }
                    }
                    if ui.button("Open SGD5 .bin").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("SGD5 Shader Group", &["bin"])
                            .pick_file()
                        {
                            self.spawn_sgd5_load(ctx.clone(), p);
                        }
                    }
                    if ui.button("Open zlib Dir...").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            self.load_ps09_path(&p);
                        }
                    }
                });
                ui.add_space(4.0);
                let running = self.sgd5_job.lock().unwrap().running;
                if ui
                    .add_enabled(
                        !running,
                        egui::Button::new("Extract SGD5 from .rpf / .bin..."),
                    )
                    .clicked()
                {
                    if let Some(input) = rfd::FileDialog::new()
                        .add_filter("SGD5 Containers", &["rpf", "bin"])
                        .pick_file()
                    {
                        if let Some(output) = rfd::FileDialog::new().pick_folder() {
                            self.spawn_sgd5_extraction(ctx.clone(), input, output);
                            self.sh_source = ShaderSource::Sgd5;
                        }
                    }
                }
                let sgd5_status = self.sgd5_job.lock().unwrap().status.clone();
                if !sgd5_status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(sgd5_status)
                            .size(10.5)
                            .color(Color32::from_rgb(165, 225, 185)),
                    );
                }
                if self.sh_source == ShaderSource::Sgd5 {
                if let Some(files) = self.sgd5_cache.clone() {
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        for group_index in 0..files.len() {
                            let label = format!("Group {group_index}");
                            let active = self.sgd5_group == group_index;
                            let button = egui::Button::new(
                                RichText::new(label)
                                    .size(11.5)
                                    .strong()
                                    .color(if active { Color32::BLACK } else { Color32::from_rgb(225, 215, 240) }),
                            )
                            .fill(if active { Color32::from_rgb(255, 168, 28) } else { Color32::from_rgb(38, 24, 58) });
                            if ui.add(button).clicked() {
                                self.sgd5_group = group_index;
                                self.sgd5_selected = Some(0);
                                self.sgd5_cached_text = (None, String::new());
                                self.sh_source = ShaderSource::Sgd5;
                            }
                        }
                    });
                    let group = &files[self.sgd5_group.min(files.len() - 1)];
                    let mut stage_counts = std::collections::BTreeMap::new();
                    for shader in &group.shaders {
                        *stage_counts.entry(shader.stage).or_insert(0usize) += 1;
                    }
                    ui.horizontal_wrapped(|ui| {
                        let options = [
                            (None, "All"),
                            (Some("VS"), "VS"),
                            (Some("PS"), "PS"),
                            (Some("GS"), "GS"),
                            (Some("DS"), "DS"),
                            (Some("HS"), "HS"),
                            (Some("CS"), "CS"),
                        ];
                        for (option, label) in options {
                            let count = option.map_or(group.shaders.len(), |stage| {
                                stage_counts.get(stage).copied().unwrap_or(0)
                            });
                            let color = option.map_or(
                                Color32::from_rgb(225, 215, 240),
                                sgd5_stage_color,
                            );
                            let active = self.sgd5_stage_filter == option;
                            let button = egui::Button::new(
                                RichText::new(format!("{label} ({count})"))
                                    .size(11.0)
                                    .strong()
                                    .color(if active { Color32::BLACK } else { color }),
                            )
                            .fill(if active { color } else { Color32::from_rgb(36, 22, 54) });
                            if ui.add(button).clicked() {
                                self.sgd5_stage_filter = option;
                                self.sgd5_selected = None;
                                self.sh_source = ShaderSource::Sgd5;
                            }
                        }
                    });
                    ui.add(
                        egui::TextEdit::singleline(&mut self.sgd5_search)
                            .desired_width(f32::INFINITY)
                            .hint_text("Filter SGD5 shaders by name, SGA hash, or stage..."),
                    );
                    let query = self.sgd5_search.trim().to_ascii_lowercase();
                    let filtered: Vec<usize> = group
                        .shaders
                        .iter()
                        .enumerate()
                        .filter(|(_, shader)| {
                            self.sgd5_stage_filter
                                .is_none_or(|stage| shader.stage == stage)
                                && (query.is_empty()
                                    || shader.name.to_ascii_lowercase().contains(&query)
                                    || format!("0x{:016x}", shader.sga_hash).contains(&query)
                                    || group.effect_names_for_shader(shader.stage, shader.index)
                                        .iter()
                                        .any(|name| name.to_ascii_lowercase().contains(&query)))
                        })
                        .map(|(index, _)| index)
                        .collect();
                    ui.label(format!("{} shaders", filtered.len()));
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show_rows(ui, 20.0, filtered.len(), |ui, range| {
                            for index in filtered[range].iter().copied() {
                                let shader = &group.shaders[index];
                                let selected = self.sgd5_selected == Some(index);
                                let color = sgd5_stage_color(shader.stage);
                                let effects = group.effect_names_for_shader(shader.stage, shader.index);
                                let label = format!(
                                    "[{}] #{:05} {} {:5}B{}",
                                    shader.stage,
                                    shader.index,
                                    shader.name,
                                    shader.uncompressed_size,
                                    if effects.is_empty() { String::new() }
                                    else { format!(" · {}", effects.join(", ")) }
                                );
                                let button = egui::Button::new(
                                    RichText::new(label)
                                        .monospace()
                                        .size(11.0)
                                        .color(if selected { Color32::BLACK } else { color }),
                                )
                                .fill(if selected { color } else { Color32::from_rgb(24, 15, 36) });
                                if ui.add(button).clicked() {
                                    self.sgd5_selected = Some(index);
                                    self.sh_source = ShaderSource::Sgd5;
                                }
                            }
                        });
                }
                }
                if self.sh_source == ShaderSource::Ps09 {
                if !self.sh_status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(&self.sh_status)
                            .size(10.5)
                            .color(Color32::from_rgb(165, 225, 185)),
                    );
                }
                ui.add_space(6.0);

                // Stage filter pills
                let stage_counts = if let Some(ref c) = self.sh_cache {
                    let (mut cs, mut ps, mut vs, mut ngg, mut rt) = (0usize, 0, 0, 0, 0);
                    for s in &c.shaders {
                        match s.stage {
                            ShaderStage::Compute => cs += 1,
                            ShaderStage::Pixel => ps += 1,
                            ShaderStage::Vertex => vs += 1,
                            ShaderStage::MeshNgg => ngg += 1,
                            ShaderStage::RayTracing => rt += 1,
                            _ => {}
                        }
                    }
                    (c.shaders.len(), cs, ps, vs, ngg, rt)
                } else {
                    (0, 0, 0, 0, 0, 0)
                };
                ui.horizontal_wrapped(|ui| {
                    for (opt, lbl, cnt) in [
                        (None, "All", stage_counts.0),
                        (Some(ShaderStage::Pixel), "PS", stage_counts.2),
                        (Some(ShaderStage::Vertex), "VS", stage_counts.3),
                        (Some(ShaderStage::Compute), "CS", stage_counts.1),
                        (Some(ShaderStage::MeshNgg), "NGG", stage_counts.4),
                        (Some(ShaderStage::RayTracing), "RT", stage_counts.5),
                    ] {
                        let active = self.sh_stage_filter == opt;
                        let btn = egui::Button::new(
                            RichText::new(format!("{lbl} ({cnt})"))
                                .size(11.0)
                                .strong()
                                .color(if active {
                                    Color32::BLACK
                                } else {
                                    Color32::from_rgb(220, 210, 240)
                                }),
                        )
                        .fill(if active {
                            Color32::from_rgb(72, 210, 255)
                        } else {
                            Color32::from_rgb(36, 22, 54)
                        });
                        if ui.add(btn).clicked() {
                            self.sh_stage_filter = opt;
                            self.sh_source = ShaderSource::Ps09;
                        }
                    }
                });

                ui.add_space(6.0);
                ui.add(
                    egui::TextEdit::singleline(&mut self.sh_search)
                        .desired_width(f32::INFINITY)
                        .hint_text("Filter by #index, PSO hash, stage, or threads..."),
                );
                ui.add_space(6.0);

                if let Some(ref cache) = self.sh_cache {
                    let q = self.sh_search.trim().to_ascii_lowercase();
                    let filtered: Vec<usize> = cache
                        .shaders
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| {
                            let st_ok = self.sh_stage_filter.is_none_or(|f| s.stage == f);
                            if !st_ok {
                                return false;
                            }
                            if q.is_empty() {
                                return true;
                            }
                            let hay = format!(
                                "#{} {} {:016x} {}b {}bind",
                                s.index,
                                s.stage.tag(),
                                s.pso_hash,
                                s.code_size,
                                s.bindings.len()
                            )
                            .to_ascii_lowercase();
                            hay.contains(&q)
                        })
                        .map(|(i, _)| i)
                        .collect();

                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show_rows(ui, 20.0, filtered.len(), |ui, range| {
                            for ri in range {
                                let idx = filtered[ri];
                                let s = &cache.shaders[idx];
                                let sel = self.sh_selected == Some(idx);
                                let col = match s.stage {
                                    ShaderStage::RayTracing => Color32::from_rgb(255, 72, 165),
                                    ShaderStage::MeshNgg => Color32::from_rgb(255, 168, 28),
                                    ShaderStage::Compute => Color32::from_rgb(72, 210, 255),
                                    ShaderStage::Pixel => Color32::from_rgb(95, 235, 145),
                                    ShaderStage::Vertex => Color32::from_rgb(205, 145, 255),
                                    _ => Color32::from_rgb(170, 170, 190),
                                };
                                let lbl = format!(
                                    "[{}] #{:05} · {:4}B · {}i · {}b",
                                    s.stage.tag(),
                                    s.index,
                                    s.code_size,
                                    s.insn_count,
                                    s.bindings.len()
                                );
                                if ui
                                    .selectable_label(
                                        sel,
                                        RichText::new(lbl).monospace().size(11.5).color(col),
                                    )
                                    .clicked()
                                {
                                    self.sh_selected = Some(idx);
                                    self.sh_source = ShaderSource::Ps09;
                                }
                            }
                        });
                }
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::from_rgb(14, 9, 24)).inner_margin(12.0))
            .show(ctx, |ui| {
                if self.sh_cache.is_some() && self.sgd5_cache.is_some() {
                    ui.horizontal_wrapped(|ui| {
                        for (source, label) in [
                            (ShaderSource::Ps09, "PS09 Runtime Cache"),
                            (ShaderSource::Sgd5, "SGD5 DXBC Groups"),
                        ] {
                            let active = self.sh_source == source;
                            let button = egui::Button::new(
                                RichText::new(label)
                                    .size(12.0)
                                    .strong()
                                    .color(if active {
                                        Color32::BLACK
                                    } else {
                                        Color32::from_rgb(225, 215, 240)
                                    }),
                            )
                            .fill(if active {
                                Color32::from_rgb(255, 168, 28)
                            } else {
                                Color32::from_rgb(38, 24, 58)
                            });
                            if ui.add(button).clicked() {
                                self.sh_source = source;
                            }
                        }
                    });
                    ui.separator();
                }

                if self.sh_source == ShaderSource::Sgd5 {
                if let Some(files) = self.sgd5_cache.clone() {
                    let group_index = self.sgd5_group.min(files.len() - 1);
                    if let Some(shader_index) = self.sgd5_selected {
                        let file = &files[group_index];
                        if let Some(shader) = file.shaders.get(shader_index) {
                            let effect_names = file.effect_names_for_shader(shader.stage, shader.index);
                            match sgd5::decompress_shader(file, shader) {
                                Ok(blob) => match dxbc::extract_fssm(&blob) {
                                    Ok(fssm) => {
                                        let code = fssm.code().to_vec();
                                        let entry = sgd5_ps09_entry(file, shader, &fssm);
                                        let stem = format!(
                                            "sgd5_{:05}_{}_{}",
                                            shader.index,
                                            shader.stage.to_lowercase(),
                                            shader.name.replace(['/', '\\', ':'], "_")
                                        );
                                        ui.horizontal_wrapped(|ui| {
                                            ui.label(
                                                RichText::new(format!(
                                                    "SGD5 group {} · {} · {} · {} code bytes",
                                                    group_index,
                                                    shader.stage,
                                                    shader.name,
                                                    code.len()
                                                ))
                                                .strong()
                                                .size(13.0)
                                                .color(Color32::from_rgb(255, 168, 28)),
                                            );
                                            if !effect_names.is_empty() {
                                                ui.label(
                                                    RichText::new(format!("Effects: {}", effect_names.join(", ")))
                                                        .size(11.0)
                                                        .color(Color32::from_rgb(120, 220, 255)),
                                                );
                                            }
                                            ui.separator();
                                            for (mode, label) in [
                                                (ShaderViewMode::Asm, "GFX10.3 ASM"),
                                                (ShaderViewMode::Hlsl, "Lifted HLSL (SM 6.6)"),
                                                (ShaderViewMode::SpirvAbi, "SPIR-V 1.5 / SGPR ABI"),
                                            ] {
                                                let active = self.sh_mode == mode;
                                                let button = egui::Button::new(
                                                    RichText::new(label)
                                                        .size(12.0)
                                                        .strong()
                                                        .color(if active { Color32::BLACK } else { Color32::from_rgb(225, 215, 240) }),
                                                )
                                                .fill(if active { Color32::from_rgb(255, 168, 28) } else { Color32::from_rgb(38, 24, 58) });
                                                if ui.add(button).clicked() {
                                                    self.sh_mode = mode;
                                                }
                                            }
                                            ui.separator();
                                            if ui.button("Export .dxbc").clicked() {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .set_file_name(format!("{stem}.dxbc"))
                                                    .save_file()
                                                {
                                                    let _ = fs::write(&path, &blob);
                                                }
                                            }
                                            if ui.button("Export .asm").clicked() {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .set_file_name(format!("{stem}.asm"))
                                                    .save_file()
                                                {
                                                    let _ = fs::write(&path, format_gfx103_disasm(&entry, &code));
                                                }
                                            }
                                            if ui.button("Export .hlsl").clicked() {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .set_file_name(format!("{stem}.hlsl"))
                                                    .save_file()
                                                {
                                                    let _ = fs::write(&path, lift_to_hlsl(&entry, &code));
                                                }
                                            }
                                            if ui.button("Export .spv").clicked() {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .set_file_name(format!("{stem}.spv"))
                                                    .save_file()
                                                {
                                                    let _ = fs::write(&path, lift_to_spirv(&entry, &code));
                                                }
                                            }
                                        });
                                        ui.add_space(8.0);
                                        let key = Some((group_index, shader_index, self.sh_mode));
                                        if self.sgd5_cached_text.0 != key {
                                            let text = match self.sh_mode {
                                                ShaderViewMode::Asm => format_gfx103_disasm(&entry, &code),
                                                ShaderViewMode::Hlsl => lift_to_hlsl(&entry, &code),
                                                ShaderViewMode::SpirvAbi => format_spirv_abi_summary(&entry, &code),
                                            };
                                            self.sgd5_cached_text = (key, text);
                                        }
                                        egui::ScrollArea::both()
                                            .auto_shrink([false, false])
                                            .show(ui, |ui| {
                                                ui.add(
                                                    egui::TextEdit::multiline(
                                                        &mut self.sgd5_cached_text.1.as_str(),
                                                    )
                                                    .font(egui::TextStyle::Monospace)
                                                    .desired_width(f32::INFINITY),
                                                );
                                            });
                                        return;
                                    }
                                    Err(error) => {
                                        ui.label(
                                            RichText::new(format!("FSSM extraction failed: {error}"))
                                                .color(Color32::from_rgb(255, 100, 120)),
                                        );
                                        return;
                                    }
                                },
                                Err(error) => {
                                    ui.label(
                                        RichText::new(format!("DXBC decompression failed: {error}"))
                                            .color(Color32::from_rgb(255, 100, 120)),
                                    );
                                    return;
                                }
                            }
                        }
                    }
                }
                }
                if self.sh_source == ShaderSource::Ps09 {
                if let (Some(ref cache), Some(sel_idx)) = (&self.sh_cache, self.sh_selected) {
                    if let Some(entry) = cache.shaders.get(sel_idx) {
                        let code = cache.shader_bytes(entry);
                        ui.horizontal_wrapped(|ui| {
                            for (mode, lbl) in [
                                (ShaderViewMode::Asm, "GFX10.3 ASM"),
                                (ShaderViewMode::Hlsl, "Lifted HLSL (SM 6.6)"),
                                (ShaderViewMode::SpirvAbi, "SPIR-V 1.5 / SGPR ABI"),
                            ] {
                                let active = self.sh_mode == mode;
                                let btn = egui::Button::new(
                                    RichText::new(lbl)
                                        .size(12.0)
                                        .strong()
                                        .color(if active { Color32::BLACK } else { Color32::from_rgb(225, 215, 240) }),
                                )
                                .fill(if active { Color32::from_rgb(255, 168, 28) } else { Color32::from_rgb(38, 24, 58) });
                                if ui.add(btn).clicked() {
                                    self.sh_mode = mode;
                                }
                            }
                            ui.separator();
                            if ui.button("Export .hlsl").clicked() {
                                if let Some(p) = rfd::FileDialog::new()
                                    .set_file_name(format!("shader_{:05}_{}.hlsl", entry.index, entry.stage.tag()))
                                    .save_file()
                                {
                                    let _ = fs::write(&p, lift_to_hlsl(entry, code));
                                }
                            }
                            if ui.button("Export .spv").clicked() {
                                if let Some(p) = rfd::FileDialog::new()
                                    .set_file_name(format!("shader_{:05}_{}.spv", entry.index, entry.stage.tag()))
                                    .save_file()
                                {
                                    let _ = fs::write(&p, lift_to_spirv(entry, code));
                                }
                            }
                            if ui.button("Export .asm").clicked() {
                                if let Some(p) = rfd::FileDialog::new()
                                    .set_file_name(format!("shader_{:05}_{}.asm", entry.index, entry.stage.tag()))
                                    .save_file()
                                {
                                    let _ = fs::write(&p, format_gfx103_disasm(entry, code));
                                }
                            }
                        });
                        ui.add_space(8.0);
                        let key = Some((sel_idx, self.sh_mode));
                        if self.sh_cached_text.0 != key {
                            let txt = match self.sh_mode {
                                ShaderViewMode::Asm => format_gfx103_disasm(entry, code),
                                ShaderViewMode::Hlsl => lift_to_hlsl(entry, code),
                                ShaderViewMode::SpirvAbi => format_spirv_abi_summary(entry, code),
                            };
                            self.sh_cached_text = (key, txt);
                        }
                        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.sh_cached_text.1.as_str())
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                        return;
                    }
                }
                }
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new(
                            "Open shaders.rpf to load the PS09 runtime cache and SGD5 DXBC groups, then select a shader to inspect its AMD RDNA2 (GFX10.3) code.",
                        )
                        .size(13.5)
                        .color(Color32::from_rgb(185, 180, 210)),
                    );
                });
            });
    }


    // export, edit and name tools for the psin file on screen
    fn psin_tools_ui(&mut self, ui: &mut egui::Ui) {
        let complete = !self.preview.contains(PSIN_CUT_MARK);
        ui.horizontal(|ui| {
            if ui.button("Export XML…").on_hover_text("Save the whole file as editable XML").clicked() {
                self.psin_export_xml();
            }
            if ui
                .add_enabled(complete, egui::Button::new("Save edits shown here…"))
                .on_hover_text("Write the values edited in the text below into a patched copy of the file")
                .on_disabled_hover_text("The view is cut short for this file: export the XML, edit it, then use Apply XML file")
                .clicked()
            {
                let xml = self.preview.clone();
                self.psin_apply_xml(xml);
            }
            if ui.button("Apply XML file…").on_hover_text("Write the values of an edited XML export into a patched copy of the file").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("XML", &["xml"]).pick_file() {
                    match fs::read_to_string(&p) {
                        Ok(xml) => self.psin_apply_xml(xml),
                        Err(err) => self.state.lock().unwrap().status = format!("Read error: {err}"),
                    }
                }
            }
            ui.separator();
            if ui
                .button("Recover names")
                .on_hover_text("Match the hashed names of every metadata file against words found in the open archives and in the names folder")
                .clicked()
            {
                self.recover_names();
            }
            ui.label(RichText::new(format!("{} names known", names::len())).size(11.0).color(Color32::from_rgb(150, 160, 180)));
        });
    }

    fn selected_entry(&self) -> Option<RpfEntry> {
        self.selected.and_then(|i| self.state.lock().unwrap().entries.get(i).cloned())
    }

    fn psin_export_xml(&self) {
        let Some(e) = self.selected_entry() else { return };
        let stem = safe_file_name(&e.name);
        let Some(out) = rfd::FileDialog::new().set_file_name(format!("{stem}.xml")).add_filter("XML", &["xml"]).save_file() else { return };
        let state = Arc::clone(&self.state);
        state.lock().unwrap().status = format!("Exporting {} as XML...", e.name);
        std::thread::spawn(move || {
            let res = read_entry_bytes(&e)
                .map_err(|err| err.to_string())
                .and_then(|data| psin_xml::export_xml(&data, 0))
                .and_then(|xml| fs::write(&out, xml).map_err(|err| err.to_string()));
            state.lock().unwrap().status = match res {
                Ok(()) => format!("Saved {}", out.display()),
                Err(err) => format!("XML export failed: {err}"),
            };
        });
    }

    // patches a copy of the selected file with the values in the xml and saves it where the user picks
    fn psin_apply_xml(&self, xml: String) {
        let Some(e) = self.selected_entry() else { return };
        let res = read_entry_bytes(&e).map_err(|err| err.to_string()).and_then(|data| psin_xml::apply_xml(&data, &xml));
        let status = match res {
            Err(err) => format!("XML not applied: {err}"),
            Ok(done) if done.changes.is_empty() => "The XML has the same values as the file, nothing to save".to_string(),
            Ok(done) => {
                let name = format!("{}.patched.psin", safe_file_name(e.name.trim_end_matches(".psin")));
                match rfd::FileDialog::new().set_file_name(name).save_file() {
                    None => "Patched copy not saved".to_string(),
                    Some(out) => match fs::write(&out, &done.data) {
                        Ok(()) => format!(
                            "{} value(s) changed, first: {}. Saved {} (same size as the original, its checksum section is not updated)",
                            done.changes.len(),
                            done.changes[0],
                            out.display()
                        ),
                        Err(err) => format!("Write error: {err}"),
                    },
                }
            }
        };
        self.state.lock().unwrap().status = status;
    }

    // looks for the words behind hashed struct, field and enum names in the open archives
    fn recover_names(&self) {
        if self.export_busy.swap(true, Ordering::SeqCst) {
            return;
        }
        let busy = Arc::clone(&self.export_busy);
        let state = Arc::clone(&self.state);
        let entries: Vec<RpfEntry> =
            state.lock().unwrap().entries.iter().filter(|e| matches!(e.kind, "PSIN" | "XML" | "RBF0" | "DAT" | "GFX" | "ZLIB")).cloned().collect();
        std::thread::spawn(move || {
            let mut last_note = Instant::now();
            let (targets, tokens) = gather_names(&entries, |n| {
                if last_note.elapsed() > Duration::from_millis(200) {
                    last_note = Instant::now();
                    state.lock().unwrap().status = format!("Reading metadata {}/{}...", n + 1, entries.len());
                }
            });
            state.lock().unwrap().status = format!("Matching {} words against the hashed names...", tokens.len());
            let found = names::recover(&tokens, &targets);
            names::add(found.exact.iter().map(|f| f.1.as_str()));
            names::add_built(found.built.iter().map(|f| f.1.as_str()));
            let all = targets.all();
            let resolved = all.iter().filter(|h| names::lookup(**h).is_some()).count();
            let note = match names_dir() {
                Some(dir) if names::save_list(&dir, names::RECOVERED_FILE, &found.exact).is_ok() && names::save_list(&dir, names::BUILT_FILE, &found.built).is_ok() => {
                    format!(" Saved in {}; extra word lists (.txt, one name per line) placed there are used too.", dir.display())
                }
                _ => String::new(),
            };
            state.lock().unwrap().status = format!(
                "Names: {resolved} of {} hashed names resolved. {} found written out, {} built from word combinations and shown with a ? (about {:.0} of those are chance matches).{note}",
                all.len(),
                found.exact.len(),
                found.built.len(),
                found.built_noise
            );
            busy.store(false, Ordering::SeqCst);
        });
    }

    fn export_entries(&self, indices: &[usize]) {
        let Some(out_dir) = rfd::FileDialog::new().pick_folder() else { return };
        let entries: Vec<RpfEntry> = {
            let st = self.state.lock().unwrap();
            indices.iter().filter_map(|&i| st.entries.get(i).cloned()).collect()
        };
        if self.export_busy.swap(true, Ordering::SeqCst) {
            return;
        }
        self.export_cancel.store(false, Ordering::SeqCst);
        let (busy, cancel) = (Arc::clone(&self.export_busy), Arc::clone(&self.export_cancel));
        let state = Arc::clone(&self.state);
        let is_folder = self.is_folder;
        std::thread::spawn(move || {
        let total = entries.len();
        let mut ok = 0usize;
        let mut last_note = Instant::now();
        for (n, e) in entries.into_iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if last_note.elapsed() > Duration::from_millis(200) {
                last_note = Instant::now();
                state.lock().unwrap().status = format!("Exporting {}/{total}: {}", n + 1, e.name);
            }
            if e.kind == "MODEL" || e.kind == "VEH" {
                if let Ok(mesh) = decode_rage9_model_from_rpf(&e.source_rpf, e.offset, &e.name) {
                    let fname = format!("{}.obj", safe_file_name(&e.name.replace('#', "")));
                    if mesh.export_obj(&out_dir.join(fname)).is_ok() {
                        ok += 1;
                    }
                }
            } else if e.kind == "NAVM" {
                if let Ok(mesh) = decode_navmesh_from_rpf(&e.source_rpf, e.offset, &e.name) {
                    let fname = format!("{}.obj", safe_file_name(&e.name.replace('#', "")));
                    if mesh.export_obj(&out_dir.join(fname)).is_ok() {
                        ok += 1;
                    }
                }
            } else if e.kind == "TXD" {
                if let Ok(texs) = decode_txd_from_rpf(&e.source_rpf, e.offset) {
                    for t in &texs {
                        let safe: String = t.name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
                        if export_texture_png(t, &out_dir.join(format!("{safe}.png"))).is_ok() {
                            ok += 1;
                        }
                    }
                }
            } else if e.kind == "TTPF" {
                if let Ok(t) = decode_ttpf_texture(&e.source_rpf, e.tracks as usize) {
                    if export_texture_png(&t, &out_dir.join(format!("{}.png", safe_file_name(&t.name)))).is_ok() {
                        ok += 1;
                    }
                }
            } else if e.kind == "RBF0" {
                if let Ok(data) = read_entry_bytes(&e) {
                    let xml = decode_rbf0(&data).unwrap_or_else(|| String::from_utf8_lossy(&data).into_owned());
                    let fname = format!("{}.xml", safe_file_name(e.name.trim_end_matches(".rbf")));
                    if fs::write(out_dir.join(fname), xml).is_ok() {
                        ok += 1;
                    }
                }
            } else if let Ok(data) = read_entry_bytes(&e) {
                let fname = if is_folder { format!("{}_{}", e.rpf_name.trim_end_matches(".rpf"), e.name) } else { e.name.clone() };
                if fs::write(out_dir.join(safe_file_name(&fname)), &data).is_ok() {
                    ok += 1;
                }
            }
        }
        let stopped = if cancel.load(Ordering::Relaxed) { " (cancelled)" } else { "" };
        state.lock().unwrap().status = format!("Exported {ok} file(s) to {}{stopped}", out_dir.display());
        busy.store(false, Ordering::SeqCst);
        });
    }
}

// runs a selection decode off the ui thread
// the result is dropped when a newer job started meanwhile and a panicking decoder still clears the spinner
fn spawn_media_job<T: 'static>(
    media: &Arc<Mutex<MediaState>>,
    status: String,
    fresh: bool,
    work: impl FnOnce(&(dyn Fn(String) + Sync)) -> Result<T, String> + Send + 'static,
    done: impl FnOnce(&mut MediaState, Result<T, String>) + Send + 'static,
) {
    let job = {
        let mut m = media.lock().unwrap();
        if fresh {
            m.job += 1;
        }
        m.loading = true;
        m.status = status;
        m.job
    };
    let media = Arc::clone(media);
    std::thread::spawn(move || {
        let progress = |text: String| {
            if let Ok(mut m) = media.try_lock() {
                if m.job == job {
                    m.status = text;
                }
            }
        };
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&progress)))
            .unwrap_or_else(|_| Err("decoder panicked".into()));
        let mut m = media.lock().unwrap_or_else(|p| p.into_inner());
        if m.job != job {
            return;
        }
        m.loading = false;
        done(&mut m, res);
    });
}

// preview length of a psin shown as xml and the comment the exporter leaves when it stops there
const PSIN_PREVIEW_CHARS: usize = 200_000;
const PSIN_CUT_MARK: &str = "<!-- preview cut here";

// hashed names of the metadata entries and every word found in the readable ones
fn gather_names(entries: &[RpfEntry], mut progress: impl FnMut(usize)) -> (names::NameTargets, HashSet<String>) {
    let mut targets = names::NameTargets::default();
    let mut tokens = HashSet::new();
    for (n, e) in entries.iter().enumerate() {
        progress(n);
        names::harvest_tokens(e.name.as_bytes(), &mut tokens);
        if e.raw_size > 64 << 20 {
            continue;
        }
        let Ok(data) = read_entry_bytes(e) else { continue };
        if e.kind == "PSIN" {
            psin_xml::collect_name_targets(&data, &mut targets);
            psin_xml::collect_strings(&data, &mut tokens);
        } else {
            names::harvest_tokens(&data, &mut tokens);
            if e.kind == "RBF0" {
                if let Some(xml) = decode_rbf0(&data) {
                    names::harvest_tokens(xml.as_bytes(), &mut tokens);
                }
            }
        }
    }
    (targets, tokens)
}

// word lists for hashed names live here, one name per line in any txt file
fn names_dir() -> Option<PathBuf> {
    app_data_dir().map(|d| d.join("names"))
}

fn app_data_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var("LOCALAPPDATA").ok().map(|d| PathBuf::from(d).join("PercyStar"))
    } else if cfg!(target_os = "macos") {
        std::env::var("HOME").ok().map(|d| PathBuf::from(d).join("Library/Application Support/PercyStar"))
    } else {
        std::env::var("HOME").ok().map(|d| PathBuf::from(d).join(".local/share/PercyStar"))
    }
}

fn cache_file_path() -> Option<PathBuf> {
    // v9 scanner rejects metadata and mid resource chunks that v8 listed as model entries
    app_data_dir().map(|d| d.join("scan_cache_v9.tsv"))
}

pub(crate) fn load_last_folder() -> Option<PathBuf> {
    let s = fs::read_to_string(app_data_dir()?.join("last_folder.txt")).ok()?;
    let t = s.trim();
    (!t.is_empty()).then(|| PathBuf::from(t))
}

fn save_last_folder(folder: &Path) {
    if let Some(dir) = app_data_dir() {
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(dir.join("last_folder.txt"), folder.display().to_string());
    }
}

const CATS: [Category; 11] = [
    Category::All, Category::Vehicles, Category::Models, Category::Textures,
    Category::Ui, Category::NavMesh, Category::Videos, Category::Audio,
    Category::Xml, Category::Resources, Category::Binary,
];

fn save_ui_state(app: &Rpf9App) {
    let Some(dir) = app_data_dir() else { return };
    let _ = fs::create_dir_all(&dir);
    let cat = CATS.iter().position(|c| *c == app.category).unwrap_or(0);
    let _ = fs::write(dir.join("ui_state.txt"), format!("{cat}\n{}\n{}\n{}", app.vehicles_only as u8, app.selected_rpf, app.filter));
}

fn load_ui_state(app: &mut Rpf9App) {
    let Some(s) = app_data_dir().and_then(|d| fs::read_to_string(d.join("ui_state.txt")).ok()) else { return };
    let mut l = s.splitn(4, '\n');
    if let Some(c) = l.next().and_then(|x| x.parse::<usize>().ok()).and_then(|i| CATS.get(i)) {
        app.category = c.clone();
    }
    app.vehicles_only = l.next() == Some("1");
    if let Some(r) = l.next().filter(|r| !r.is_empty()) {
        app.selected_rpf = r.into();
    }
    if let Some(f) = l.next() {
        app.filter = f.into();
    }
}

fn load_disk_cache() -> HashMap<String, Vec<RpfEntry>> {
    cache_file_path().map(|p| load_disk_cache_from(&p)).unwrap_or_default()
}

fn load_disk_cache_from(p: &Path) -> HashMap<String, Vec<RpfEntry>> {
    let mut map: HashMap<String, Vec<RpfEntry>> = HashMap::new();
    let Ok(bytes) = fs::read(p) else { return map };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    for row in par_map(&lines, |l| parse_cache_row(l)).into_iter().flatten() {
        let (key, e) = row;
        match map.get_mut(key) {
            Some(v) => v.push(e),
            None => {
                map.insert(key.to_string(), vec![e]);
            }
        }
    }
    map
}

fn parse_cache_row(line: &str) -> Option<(&str, RpfEntry)> {
    let mut cols = [""; 12];
    let mut n = 0;
    for c in line.split('\t') {
        if n == cols.len() {
            break;
        }
        cols[n] = c;
        n += 1;
    }
    if n < 9 {
        return None;
    }
    let kind: &'static str = match cols[4] {
        "VEH" => "VEH",
        "MODEL" => "MODEL",
        "TXD" => "TXD",
        "TTPF" => "TTPF",
        "RBF0" => "RBF0",
        "GFX" => "GFX",
        "GXT" => "GXT",
        "NAVM" => "NAVM",
        "BK2" => "BK2",
        "AWC" => "AWC",
        "XML" => "XML",
        "DAT" => "DAT",
        "HMAP" => "HMAP",
        "PSIN" => "PSIN",
        "ANIM" => "ANIM",
        "RSC9" => "RSC9",
        "META7A" => "META7A",
        _ => "ZLIB",
    };
    // tolerate old 11 col cache rows and newer 10 col ones
    let tracks_col = if n >= 11 { 9 } else { 8 };
    let chunks_col = tracks_col + 1;
    let chunks = if n > chunks_col && !cols[chunks_col].is_empty() {
        cols[chunks_col].split(';').filter_map(|pair| {
            let mut it = pair.split(',');
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        }).collect()
    } else {
        Vec::new()
    };
    Some((cols[0], RpfEntry {
        source_rpf: PathBuf::from(cols[1]),
        rpf_name: cols[2].to_string(),
        name: cols[3].to_string(),
        kind,
        offset: cols[5].parse().unwrap_or(0),
        comp_size: cols[6].parse().unwrap_or(0),
        raw_size: cols[7].parse().unwrap_or(0),
        tracks: cols[tracks_col].parse().unwrap_or(0),
        chunks,
    }))
}

// atomic write via tmp file so ctrl c never corrupts the cahce
fn save_disk_cache(map: &HashMap<String, Vec<RpfEntry>>) {
    if let Some(p) = cache_file_path() {
        save_disk_cache_to(map, &p);
    }
}

fn save_disk_cache_to(map: &HashMap<String, Vec<RpfEntry>>, p: &Path) {
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = p.with_extension("tmp");
    let Ok(f) = File::create(&tmp) else { return };
    let mut w = BufWriter::new(f);
    // rows are formatted on all cores in blocks then written in order
    let rows: Vec<(&str, &RpfEntry)> = map.iter().flat_map(|(k, v)| v.iter().map(move |e| (k.as_str(), e))).collect();
    let per = rows.len().div_ceil(std::thread::available_parallelism().map_or(1, |n| n.get())).max(8192);
    let blocks: Vec<Vec<u8>> = std::thread::scope(|s| {
        let parts: Vec<_> = rows
            .chunks(per)
            .map(|c| {
                s.spawn(move || {
                    let mut b = Vec::with_capacity(c.len() * 200);
                    for (key, e) in c {
                        let _ = write!(
                            b,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t",
                            key, e.source_rpf.display(), e.rpf_name, e.name, e.kind,
                            e.offset, e.comp_size, e.raw_size, e.tracks
                        );
                        for (j, (a, z)) in e.chunks.iter().enumerate() {
                            let _ = write!(b, "{}{a},{z}", if j > 0 { ";" } else { "" });
                        }
                        b.push(b'\n');
                    }
                    b
                })
            })
            .collect();
        parts.into_iter().map(|p| p.join().unwrap_or_default()).collect()
    });
    // a failed write must never replace the good cache with a cut one
    if blocks.iter().all(|b| w.write_all(b).is_ok()) && w.flush().is_ok() {
        drop(w);
        let _ = fs::rename(&tmp, p);
    } else {
        drop(w);
        let _ = fs::remove_file(&tmp);
    }
}

fn collect_rpfs(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            // skip .Trashes/.Spotlight and SMB "._x.rpf" AppleDouble forks
            if entry.file_name().as_encoded_bytes().first() == Some(&b'.') { continue; }
            // d_type from readdir: no extra stat per entry over SMB, and symlinks are not followed
            let Ok(ft) = entry.file_type() else { continue };
            let p = entry.path();
            if ft.is_dir() {
                collect_rpfs(&p, out);
            } else if ft.is_file() && p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("rpf") || s.eq_ignore_ascii_case("ttpf")).unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

type AwcPlayerTracks = Option<(Vec<audio_player::LoadedWavTrack>, Arc<Vec<u8>>)>;

// status line and decoded tracks of one awc entry for the audio player
fn decode_awc_for_player(e: &RpfEntry, progress: &(dyn Fn(String) + Sync)) -> (String, AwcPlayerTracks) {
    match read_entry_bytes(e) {
        Ok(bytes) => {
            let res = awc::decode_awc_bytes_in_memory(&bytes, "stream", |done, total| {
                progress(format!("Scanning AWC waveforms ({done}/{total})..."));
            });
            if res.tracks.is_empty() {
                let mut lines = vec![
                    format!("Decoded 0 playable .wav stream(s) natively — {}", res.summary),
                    format!(
                        "Entry: {} ({} bytes read, comp={}, raw={}, offset=0x{:X}, RPF: {})",
                        e.name, bytes.len(), e.comp_size, e.raw_size, e.offset, e.rpf_name
                    ),
                ];
                lines.extend(res.details);
                (lines.join("\n"), None)
            } else {
                (format!("Decoded {} playable .wav stream(s) natively", res.tracks.len()), Some((res.tracks, Arc::new(bytes))))
            }
        }
        Err(err) => (
            format!(
                "Decoded 0 playable .wav stream(s) natively — Read error: {err}\nEntry: {} (comp={}, raw={}, offset=0x{:X}, RPF: {})",
                e.name, e.comp_size, e.raw_size, e.offset, e.rpf_name
            ),
            None,
        ),
    }
}

fn setup_style(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let mut font_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(windir) = std::env::var("WINDIR") {
        font_dirs.push(PathBuf::from(windir).join("Fonts"));
    }
    if cfg!(windows) {
        font_dirs.push(PathBuf::from(r"C:\Windows\Fonts"));
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        font_dirs.push(PathBuf::from(local).join(r"Microsoft\Windows\Fonts"));
    }
    if let Ok(home) = std::env::var("HOME") {
        font_dirs.push(PathBuf::from(&home).join("Library/Fonts"));
        font_dirs.push(PathBuf::from(home).join(".local/share/fonts"));
    }
    font_dirs.push(PathBuf::from("/Library/Fonts"));
    font_dirs.push(PathBuf::from("/System/Library/Fonts"));
    font_dirs.push(PathBuf::from("/usr/share/fonts"));

    'dirs: for base in &font_dirs {
        for name in ["SF-Pro-Display-Regular.otf", "SF-Pro-Display-Medium.otf", "SF-Pro-Display-Semibold.otf"] {
            if let Ok(bytes) = fs::read(base.join(name)) {
                fonts.font_data.insert("sf_pro".into(), Arc::new(FontData::from_owned(bytes)));
                fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "sf_pro".into());
                break 'dirs;
            }
        }
    }

    // load system unicode cjk korean japanese cyrillic fallback fonts so all 13 localizations render cleanly
    let fallbacks: &[(&str, &[&str])] = &[
        ("fb_ui", &["segoeui.ttf", "arial.ttf", "SFNS.ttf", "DejaVuSans.ttf"]),
        ("fb_cjk", &["msyh.ttc", "YuGothR.ttc", "msgothic.ttc", "simsun.ttc", "PingFang.ttc", "Hiragino Sans GB.ttc", "NotoSansCJK-Regular.ttc"]),
        ("fb_kr", &["malgun.ttf", "AppleSDGothicNeo.ttc"]),
    ];
    for (key, candidates) in fallbacks {
        'cand: for cand in *candidates {
            for base in &font_dirs {
                if let Ok(bytes) = fs::read(base.join(cand)) {
                    fonts.font_data.insert((*key).into(), Arc::new(FontData::from_owned(bytes)));
                    fonts.families.entry(FontFamily::Proportional).or_default().push((*key).into());
                    fonts.families.entry(FontFamily::Monospace).or_default().push((*key).into());
                    break 'cand;
                }
            }
        }
    }
    ctx.set_fonts(fonts);

    // neon sunset theme matching dusk purple magenta and golden amber
    // the palette is dark only so the windows light mode must not switch egui to its light style
    ctx.set_theme(egui::Theme::Dark);
    let mut vis = egui::Visuals::dark();
    vis.panel_fill = Color32::from_rgb(18, 12, 28);
    vis.window_fill = Color32::from_rgb(24, 16, 36);
    vis.extreme_bg_color = Color32::from_rgb(12, 8, 20);
    vis.widgets.inactive.bg_fill = Color32::from_rgb(34, 22, 50);
    vis.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(68, 42, 98));
    vis.widgets.inactive.corner_radius = CornerRadius::same(6);
    vis.widgets.hovered.bg_fill = Color32::from_rgb(56, 34, 82);
    vis.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(255, 72, 165));
    vis.widgets.hovered.corner_radius = CornerRadius::same(6);
    vis.widgets.active.bg_fill = Color32::from_rgb(255, 168, 28);
    vis.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::BLACK);
    vis.widgets.active.corner_radius = CornerRadius::same(6);
    vis.selection.bg_fill = Color32::from_rgba_unmultiplied(255, 72, 165, 58);
    vis.selection.stroke = Stroke::new(1.0_f32, Color32::from_rgb(255, 168, 28));
    ctx.set_visuals(vis);
}

fn draw_percystar_spinner(ui: &mut egui::Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let t = ui.input(|i| i.time) as f32;
    let p = ui.painter();
    let c = rect.center();
    p.circle_stroke(c, size * 0.44, Stroke::new(2.0_f32, Color32::from_rgb(255, 85, 175)));
    let mut pts = Vec::with_capacity(10);
    for i in 0..10 {
        let ang = t * 3.5 + (i as f32) * std::f32::consts::PI / 5.0 - std::f32::consts::FRAC_PI_2;
        let r = if i % 2 == 0 { size * 0.34 } else { size * 0.14 };
        pts.push(egui::pos2(c.x + r * ang.cos(), c.y + r * ang.sin()));
    }
    p.add(egui::Shape::convex_polygon(pts, Color32::from_rgb(252, 175, 23), Stroke::NONE));
    ui.ctx().request_repaint();
}

fn kind_color(kind: &str) -> Color32 {
    match kind {
        "VEH" => Color32::from_rgb(255, 85, 175),
        "MODEL" => Color32::from_rgb(130, 175, 255),
        "TXD" => Color32::from_rgb(64, 235, 225),
        "TTPF" => Color32::from_rgb(48, 205, 255),
        "RBF0" => Color32::from_rgb(255, 145, 72),
        "GFX" => Color32::from_rgb(255, 115, 130),
        "GXT" => Color32::from_rgb(255, 225, 95),
        "NAVM" => Color32::from_rgb(125, 235, 110),
        "XML" => Color32::from_rgb(72, 210, 255),
        "AWC" => Color32::from_rgb(90, 235, 165),
        "BK2" => Color32::from_rgb(252, 175, 23),
        "DAT" => Color32::from_rgb(135, 240, 165),
        "HMAP" => Color32::from_rgb(72, 220, 155),
        "PSIN" => Color32::from_rgb(210, 120, 255),
        "ANIM" => Color32::from_rgb(100, 225, 195),
        "RSC9" => Color32::from_rgb(120, 185, 255),
        _ => Color32::from_rgb(145, 152, 168),
    }
}

// result list ordering relevance is scan order or best name match first while searching
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SortMode {
    Relevance,
    SizeDesc,
    SizeAsc,
    Name,
}

impl SortMode {
    const ALL: [SortMode; 4] = [SortMode::Relevance, SortMode::SizeDesc, SortMode::SizeAsc, SortMode::Name];

    fn label(self) -> &'static str {
        match self {
            SortMode::Relevance => "Relevance",
            SortMode::SizeDesc => "Size: largest first",
            SortMode::SizeAsc => "Size: smallest first",
            SortMode::Name => "Name (A-Z)",
        }
    }
}

fn model_viewer_sidebar(ctx: &egui::Context) -> egui::SidePanel {
    let max_width = (ctx.available_rect().width() * 0.45).min(480.0);
    egui::SidePanel::left("mv_sidebar")
        .default_width(360.0)
        .width_range(240.0_f32.min(max_width)..=max_width)
        .resizable(true)
        .frame(egui::Frame::new().fill(Color32::from_rgb(20, 13, 31)).inner_margin(10.0))
}

fn model_search_row(ui: &mut egui::Ui, filter: &mut String) -> egui::Response {
    ui.horizontal(|ui| {
        // Lay out the button first, then give the field the remaining outer width.
        // TextEdit::desired_width alone excludes its own horizontal margins.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("✕").on_hover_text("Clear filter").clicked() {
                filter.clear();
            }
            ui.add_sized(
                egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
                egui::TextEdit::singleline(filter)
                    .desired_width(f32::INFINITY)
                    .hint_text("Search models (Ctrl+F) · words AND · -exclude"),
            )
        }).inner
    }).inner
}

fn model_browser_options(ui: &mut egui::Ui, hide_lods: &mut bool, sort: &mut SortMode, has_models: bool) -> bool {
    let random = ui.horizontal(|ui| {
        ui.checkbox(hide_lods, "hide LOD copies");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_enabled(has_models, egui::Button::new("Random")).clicked()
        }).inner
    }).inner;
    // ComboBox contains its own horizontal layout and cannot reliably wrap as
    // one widget alongside the checkbox; give it a row of its own.
    sort_combo(ui, "mv_sort_combo", sort);
    random
}

// compact sort dropdown lists pick up the change on next refresh
fn sort_combo(ui: &mut egui::Ui, id: &str, mode: &mut SortMode) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(RichText::new(format!("Sort: {}", mode.label())).size(11.0))
        .width(170.0)
        .show_ui(ui, |ui| {
            for m in SortMode::ALL {
                ui.selectable_value(mode, m, m.label());
            }
        });
}

// right click context menu on file rows to copy name archive path directory or hex offset
fn format_multi_copy(entries: &[&RpfEntry], mode: u8) -> String {
    entries
        .iter()
        .map(|e| match mode {
            0 => e.name.clone(),
            1 => format!("{} / {}", e.rpf_name, e.name),
            _ => format!("{} @ 0x{:08x} ({})", e.rpf_name, e.offset, e.name),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn apply_multi_select(
    view: &[usize],
    multi: &mut HashSet<usize>,
    anchor: &mut Option<usize>,
    clicked_idx: usize,
    ctrl: bool,
    shift: bool,
) {
    if shift {
        if let Some(anc) = *anchor {
            if let (Some(a), Some(b)) = (view.iter().position(|&x| x == anc), view.iter().position(|&x| x == clicked_idx)) {
                if !ctrl {
                    multi.clear();
                }
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                for &idx in &view[lo..=hi] {
                    multi.insert(idx);
                }
                return;
            }
        }
    }
    if ctrl {
        if !multi.remove(&clicked_idx) {
            multi.insert(clicked_idx);
        }
        *anchor = Some(clicked_idx);
    } else {
        multi.clear();
        multi.insert(clicked_idx);
        *anchor = Some(clicked_idx);
    }
}

fn entry_context_menu(resp: &egui::Response, e: &RpfEntry) -> Option<String> {
    let mut copied = None;
    resp.context_menu(|ui| {
        if ui.button("Copy File Name").clicked() {
            ui.ctx().copy_text(e.name.clone());
            copied = Some(format!("Copied name: {}", e.name));
            ui.close_menu();
        }
        let rel = format!("{} / {}", e.rpf_name, e.name);
        if ui.button("Copy Archive / File Path").clicked() {
            ui.ctx().copy_text(rel.clone());
            copied = Some(format!("Copied: {rel}"));
            ui.close_menu();
        }
        let full = e.source_rpf.display().to_string();
        if ui.button("Copy Source Archive Path").clicked() {
            ui.ctx().copy_text(full.clone());
            copied = Some(format!("Copied archive path: {full}"));
            ui.close_menu();
        }
        if let Some(dir) = e.source_rpf.parent() {
            let dir_s = dir.display().to_string();
            if ui.button("Copy Directory Path").clicked() {
                ui.ctx().copy_text(dir_s.clone());
                copied = Some(format!("Copied directory: {dir_s}"));
                ui.close_menu();
            }
        }
        let loc = format!("{} @ 0x{:08x} ({})", e.rpf_name, e.offset, e.name);
        if ui.button("Copy Offset Location").clicked() {
            ui.ctx().copy_text(loc.clone());
            copied = Some(format!("Copied: {loc}"));
            ui.close_menu();
        }
    });
    copied
}

fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1_073_741_824.0 {
        format!("{:.2} GB", b / 1_073_741_824.0)
    } else if b >= 1_048_576.0 {
        format!("{:.1} MB", b / 1_048_576.0)
    } else if b >= 1024.0 {
        format!("{:.1} KB", b / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_hex(data: &[u8]) -> String {
    let mut out = String::new();
    for (row, chunk) in data.chunks(16).enumerate() {
        out.push_str(&format!("{:04x}: ", row * 16));
        for b in chunk { out.push_str(&format!("{b:02x} ")); }
        for _ in chunk.len()..16 { out.push_str("   "); }
        out.push(' ');
        for &b in chunk {
            out.push(if (32..127).contains(&b) { b as char } else { '.' });
        }
        out.push('\n');
    }
    out
}

fn read_entry_bytes(e: &RpfEntry) -> std::io::Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    let mut f = File::open(&e.source_rpf)?;
    if !e.chunks.is_empty() {
        let mut combined = Vec::new();
        for &(coff, clen) in &e.chunks {
            f.seek(SeekFrom::Start(coff))?;
            let mut raw = vec![0u8; clen as usize];
            f.read_exact(&mut raw)?;
            ZlibDecoder::new(&raw[..]).read_to_end(&mut combined)?;
        }
        Ok(combined)
    } else {
        f.seek(SeekFrom::Start(e.offset))?;
        let mut raw = vec![0u8; e.comp_size as usize];
        f.read_exact(&mut raw)?;
        if matches!(e.kind, "XML" | "DAT" | "PSIN" | "ANIM" | "RSC9" | "RBF0" | "GFX" | "GXT" | "ZLIB") {
            let mut out = Vec::new();
            ZlibDecoder::new(&raw[..]).read_to_end(&mut out)?;
            Ok(out)
        } else {
            Ok(raw)
        }
    }
}

fn extract_xml_tag(bytes: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(&bytes[..bytes.len().min(300)]).ok()?;
    for part in s.split('<').skip(1) {
        let tag = part.split(|c: char| c.is_whitespace() || c == '>').next()?;
        if !tag.starts_with('?') && !tag.starts_with('!') && !tag.starts_with('/') && tag.len() >= 3 && tag != "Item" {
            return Some(tag.chars().filter(|c| c.is_alphanumeric() || *c == '_').take(36).collect());
        }
    }
    None
}

fn probe_meta7a_zlib(
    buf: &[u8],
    zlib_at: usize,
    read_n: usize,
    off: u64,
    path: &Path,
    rname: &str,
    probe_dec: &mut Decompress,
    probe_out: &mut [u8],
) -> Option<RpfEntry> {
    if zlib_at + 8 > read_n || !is_zlib_head(&buf[zlib_at..read_n]) {
        return None;
    }
    let end = (zlib_at + 32768).min(read_n);
    probe_dec.reset(true);
    let _ = probe_dec.decompress(&buf[zlib_at..end], probe_out, FlushDecompress::None);
    let olen = probe_dec.total_out() as usize;
    if olen < 0x80 || !meta7a::has_schema_enums(&probe_out[..olen]) {
        return None;
    }
    let vft = u64::from_le_bytes(probe_out[0..8].try_into().ok()?);
    let bmap = u64::from_le_bytes(probe_out[8..16].try_into().ok()?);
    if vft != 0 || !(0x5000_0000..0x5800_0000).contains(&bmap) {
        return None;
    }
    Some(RpfEntry {
        source_rpf: path.to_path_buf(),
        rpf_name: rname.to_string(),
        name: format!("meta7a_{:012x}.#7a", off + zlib_at as u64),
        kind: "META7A",
        offset: off + zlib_at as u64,
        comp_size: probe_dec.total_in(),
        raw_size: olen as u64,
        tracks: 0,
        chunks: Vec::new(),
    })
}

// shared rage9 page detector used by both veh scanner and full archive pass
fn probe_rage9_page(
    buf: &[u8],
    pos: usize,
    read_n: usize,
    off: u64,
    path: &Path,
    rname: &str,
    probe_dec: &mut Decompress,
    probe_out: &mut [u8],
) -> Option<RpfEntry> {
    // packs.rpf vehicle tables are often indexed at the zlib stream itself.
    if let Some(meta) = probe_meta7a_zlib(buf, pos, read_n, off, path, rname, probe_dec, probe_out) {
        return Some(meta);
    }
    let p = pos + 0x10;
    if p + 8 > read_n || !is_zlib_head(&buf[p..read_n]) {
        return None;
    }
    let end = (p + 32768).min(read_n);
    probe_dec.reset(true);
    let _ = probe_dec.decompress(&buf[p..end], probe_out, FlushDecompress::None);
    let olen = probe_dec.total_out() as usize;
    if olen < 0x80 { return None; }
    let vft = u64::from_le_bytes(probe_out[0..8].try_into().ok()?);
    let bmap = u64::from_le_bytes(probe_out[8..16].try_into().ok()?);
    if vft != 0 || !(0x5000_0000..0x5800_0000).contains(&bmap) {
        return None;
    }
    // Schema resources such as vehicle configurations share the 0DRPz tag with
    // clip dictionaries. Enum definitions distinguish them before the anim probe.
    if let Some(meta) = probe_meta7a_zlib(buf, p, read_n, off, path, rname, probe_dec, probe_out) {
        return Some(meta);
    }
    // 0. crClipDictionary (.ycd animation dictionary with 0DRPz tag at +0x10)
    if let Some((aname, clip_cnt)) = anim::probe_anim_dict(&probe_out[..olen]) {
        return Some(RpfEntry {
            source_rpf: path.to_path_buf(),
            rpf_name: rname.to_string(),
            name: aname,
            kind: "ANIM",
            offset: off + p as u64,
            comp_size: probe_dec.total_in(),
            raw_size: olen as u64,
            tracks: clip_cnt,
            chunks: Vec::new(),
        });
    }
    // reject mid resource virtual chunks that start directly with a grmgeometry struct
    if u64::from_le_bytes(probe_out[0x60..0x68].try_into().ok()?) == 0x1601 {
        return None;
    }
    // 1. GTA VI CNavMesh tile (version 0x00010015)
    let ver = u32::from_le_bytes(probe_out[0x14..0x18].try_into().ok()?);
    if ver == 0x0001_0015 && olen >= 0xD0 {
        let n_polys = u32::from_le_bytes(probe_out[0xC4..0xC8].try_into().ok()?);
        let midx = u32::from_le_bytes(probe_out[0xC8..0xCC].try_into().ok()?);
        let nm = if midx < 512 * 512 {
            let gx = (midx % 512) as i32;
            let gy = (midx / 512) as i32;
            let wx = (gx - 256) * 128;
            let wy = (gy - 256) * 128;
            format!("nav_{gx:03}_{gy:03}_({wx},{wy}).ynv")
        } else {
            format!("nav_dyn_{midx:06x}.ynv")
        };
        return Some(RpfEntry {
            source_rpf: path.to_path_buf(),
            rpf_name: rname.to_string(),
            name: nm,
            kind: "NAVM",
            offset: off + p as u64,
            comp_size: probe_dec.total_in(),
            raw_size: olen as u64,
            tracks: n_polys,
            chunks: Vec::new(),
        });
    }
    // 2. pgDictionary<grcTexture> (.#td texture dictionary)
    if let Some((txd_name, tex_cnt)) = probe_txd_header(&probe_out[..olen]) {
        return Some(RpfEntry {
            source_rpf: path.to_path_buf(),
            rpf_name: rname.to_string(),
            name: txd_name,
            kind: "TXD",
            offset: off + p as u64,
            comp_size: probe_dec.total_in(),
            raw_size: olen as u64,
            tracks: tex_cnt as u32,
            chunks: Vec::new(),
        });
    }
    // 3. Drawable / FragType 3D Model (.#dr / .#ft / .#dd)
    let bmap_off = (bmap - 0x5000_0000) as usize;
    let fits_single_chunk = olen < probe_out.len()
        && bmap_off + 10 <= olen
        && probe_out[bmap_off + 8] == 1
        && (1..=64).contains(&probe_out[bmap_off + 9]);
    let (mname, is_veh, vlen) = if fits_single_chunk {
        if !has_rage9_meshlets(&probe_out[..olen]) {
            let rsc_name = rsc9::probe_cutscene_rsc9(&probe_out[..olen])
                .unwrap_or_else(|| format!("rsc9_{:08x}_{}B.rsc9", off + p as u64, olen));
            return Some(RpfEntry {
                source_rpf: path.to_path_buf(),
                rpf_name: rname.to_string(),
                name: rsc_name,
                kind: "RSC9",
                offset: off + p as u64,
                comp_size: probe_dec.total_in(),
                raw_size: olen as u64,
                tracks: 0,
                chunks: Vec::new(),
            });
        }
        let (mname, _) = extract_rage9_model_name(&probe_out[..olen])?;
        (mname.clone(), is_vehicle_virtual(&probe_out[..olen], &mname), olen)
    } else {
        probe_large_rage9_model_in_buf(&buf[p..read_n], path, off + p as u64)?
    };
    Some(RpfEntry {
        source_rpf: path.to_path_buf(),
        rpf_name: rname.to_string(),
        name: mname,
        kind: if is_veh { "VEH" } else { "MODEL" },
        offset: off + p as u64,
        comp_size: probe_dec.total_in(),
        raw_size: vlen as u64,
        tracks: 0,
        chunks: Vec::new(),
    })
}

// runs probe_rage9_page on every 4kb page of buf on all cores, slot i holds the page at i * 0x1000
fn probe_pages(buf: &[u8], read_n: usize, off: u64, path: &Path, rname: &str) -> Vec<Option<RpfEntry>> {
    use std::sync::atomic::AtomicUsize;
    const PAGES_PER_JOB: usize = 64;
    let pages = if read_n >= 44 { (read_n - 44) / 0x1000 + 1 } else { 0 };
    let mut out: Vec<Option<RpfEntry>> = (0..pages).map(|_| None).collect();
    let next = AtomicUsize::new(0);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()).min(pages.div_ceil(PAGES_PER_JOB)).max(1);
    let found: Vec<Vec<(usize, RpfEntry)>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..cores)
            .map(|_| {
                s.spawn(|| {
                    let mut dec = Decompress::new(true);
                    let mut scratch = vec![0u8; 32768];
                    let mut hits = Vec::new();
                    loop {
                        let job = next.fetch_add(1, Ordering::Relaxed) * PAGES_PER_JOB;
                        if job >= pages {
                            return hits;
                        }
                        for i in job..(job + PAGES_PER_JOB).min(pages) {
                            if let Some(e) = probe_rage9_page(buf, i * 0x1000, read_n, off, path, rname, &mut dec, &mut scratch) {
                                hits.push((i, e));
                            }
                        }
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap_or_default()).collect()
    });
    for (i, e) in found.into_iter().flatten() {
        out[i] = Some(e);
    }
    out
}

type ChunkReader = (std::sync::mpsc::Sender<u64>, std::sync::mpsc::Receiver<(u64, std::io::Result<Vec<u8>>)>);

// reads 16mb chunks on its own thread one request ahead so the disk never waits for the probes
// send an offset, receive (offset, bytes); an empty chunk means end of file
fn spawn_chunk_reader(path: &Path) -> std::io::Result<ChunkReader> {
    let mut f = File::open(path)?;
    let (req_tx, req_rx) = std::sync::mpsc::channel::<u64>();
    let (res_tx, res_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for at in req_rx {
            let res = f.seek(SeekFrom::Start(at)).and_then(|_| {
                let mut b = vec![0u8; 16 * 1024 * 1024];
                let n = f.read(&mut b)?;
                b.truncate(n);
                Ok(b)
            });
            if res_tx.send((at, res)).is_err() {
                return;
            }
        }
    });
    Ok((req_tx, res_rx))
}

// single 16mb sequntial stream avoids hdd seek thrash while indexing vehicles
fn scan_levels_vehicle_zone(levels_rpf: &PathBuf, live_state: Option<&Arc<Mutex<ScanState>>>) -> Vec<RpfEntry> {
    let sz = fs::metadata(levels_rpf).map(|m| m.len()).unwrap_or(0);
    let start_off = VEH_ZONE_OFF.min(sz);
    let total_len = sz.saturating_sub(start_off);
    if total_len == 0 { return Vec::new(); }
    let rname = levels_rpf.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let Ok((req, chunks)) = spawn_chunk_reader(levels_rpf) else { return Vec::new() };

    let mut all_entries: Vec<RpfEntry> = Vec::new();
    let mut local_batch: Vec<RpfEntry> = Vec::with_capacity(32);
    let mut off = start_off;
    let mut last_pct = 999u64;
    let _ = req.send(off);

    while off < sz {
        let Ok((_, Ok(buf))) = chunks.recv() else { break };
        let n = buf.len();
        if n == 0 { break; }
        let _ = req.send(off + n as u64);
        for entry in probe_pages(&buf, n, off, levels_rpf, &rname).into_iter().flatten() {
            {
                local_batch.push(entry);
                if local_batch.len() >= 15 {
                    if let Some(st) = live_state {
                        let mut g = st.lock().unwrap();
                        g.entries.extend(local_batch.iter().cloned());
                        g.ver += 1;
                    }
                    all_entries.append(&mut local_batch);
                }
            }
        }
        off += n as u64;
        if let Some(st) = live_state {
            let pct = ((off - start_off) * 100 / total_len).min(100);
            if pct != last_pct {
                last_pct = pct;
                let mut g = st.lock().unwrap();
                let vcnt = g.entries.iter().filter(|e| e.kind == "VEH").count();
                g.status = format!("Fast-indexing Vehicles: {pct}% ({vcnt} vehicles found)...");
            }
        }
    }
    if !local_batch.is_empty() {
        if let Some(st) = live_state {
            let mut g = st.lock().unwrap();
            g.entries.extend(local_batch.iter().cloned());
            g.ver += 1;
        }
        all_entries.extend(local_batch);
    }
    all_entries
}

fn scan_single_rpf(path: &PathBuf, progress_state: Option<&Arc<Mutex<ScanState>>>) -> std::io::Result<(String, Vec<RpfEntry>)> {
    if path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("ttpf")).unwrap_or(false) {
        let rname = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let (info, descs) = scan_ttpf_descriptors(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if let Some(st) = progress_state {
            st.lock().unwrap().header_info = info.clone();
        }
        let entries: Vec<RpfEntry> = descs.into_iter().map(|d| {
            let fnm = tex::dxgi_info(d.dxgi).map(|x| x.0).unwrap_or("UNK");
            RpfEntry {
                source_rpf: path.clone(),
                rpf_name: rname.clone(),
                name: format!("tex_{:05}_{}x{}_{fnm}{}.ttpf", d.idx, d.w, d.h, if d.has_mips { "_mip" } else { "" }),
                kind: "TTPF",
                offset: d.byte_off,
                comp_size: d.alloc_sz,
                raw_size: d.alloc_sz,
                tracks: d.idx as u32,
                chunks: Vec::new(),
            }
        }).collect();
        return Ok((info, entries));
    }

    let mut f = File::open(path)?;
    let sz = f.metadata()?.len();
    let rname = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let mut hdr = [0u8; 64];
    let n = f.read(&mut hdr)?;
    let magic = String::from_utf8_lossy(&hdr[..n.min(4)]).into_owned();
    let (data_off, hdr_info) = if n >= 40 && &hdr[..4] == b"9FPR" {
        let doff = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let nlen = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]);
        let ecnt = u32::from_le_bytes([hdr[36], hdr[37], hdr[38], hdr[39]]);
        let start = if doff >= 0x40 && (doff as u64) < sz { doff } else { 0x40 };
        (start, format!("{magic}  ·  {sz} B  ·  {ecnt} entries  ·  data@0x{start:x}  ·  names={nlen}B"))
    } else {
        // files shorter than the magic have nothing to show
        let hdr_hex: String = hdr.get(4..n.min(16)).unwrap_or(&[]).iter().map(|b| format!("{b:02x}")).collect();
        (0x40usize, format!("{magic}  ·  {sz} B  ·  {hdr_hex}"))
    };
    if let Some(st) = progress_state {
        st.lock().unwrap().header_info = hdr_info.clone();
    }

    // <=512mb: one sequential read serves both passes (was read twice, painful over smb)
    let whole: Option<Vec<u8>> = if sz <= 512 * 1024 * 1024 {
        f.seek(SeekFrom::Start(0))?;
        let mut d = Vec::with_capacity(sz as usize);
        f.read_to_end(&mut d)?;
        Some(d)
    } else {
        None
    };
    let mut found = Vec::new();
    let mut off = 0u64;
    let mut probe_dec = Decompress::new(true);
    let mut probe_out = vec![0u8; 32768];
    let mut bk2_idx = 0usize;
    let mut awc_idx = 0usize;
    let mut last_ui_mb = 0u64;
    let scan_limit = if rname.eq_ignore_ascii_case("levels.rpf") && sz > VEH_ZONE_OFF { VEH_ZONE_OFF } else { sz };
    // a page probe reads the same bytes whatever offset the whole-file slice starts at, so one pass covers every page
    let mut whole_pages = whole.as_ref().map(|w| probe_pages(w, w.len(), 0, path, &rname)).unwrap_or_default();
    let reader = if whole.is_none() { Some(spawn_chunk_reader(path)?) } else { None };
    let mut chunk: Vec<u8>;
    if let Some((req, _)) = &reader {
        let _ = req.send(0);
    }

    while off < scan_limit {
        if let Some(st) = progress_state {
            let cur_mb = off / (64 * 1024 * 1024);
            if cur_mb != last_ui_mb {
                last_ui_mb = cur_mb;
                st.lock().unwrap().status = format!("Scanning {:.2} / {:.2} GB ({} found)...", off as f64 / 1e9, sz as f64 / 1e9, found.len());
            }
        }
        let read_n;
        let mut chunk_pages = Vec::new();
        let buf: &[u8] = if let Some(w) = &whole {
            let s = &w[(off as usize).min(w.len())..];
            read_n = s.len();
            s
        } else {
            let (req, chunks) = reader.as_ref().expect("reader exists without whole");
            loop {
                let (at, res) = chunks.recv().map_err(|_| std::io::Error::other("reader thread stopped"))?;
                // a read queued before a jump is stale, the jump target was requested right after it
                if at == off {
                    chunk = res?;
                    break;
                }
            }
            read_n = chunk.len();
            if read_n > 0 {
                let _ = req.send(off + read_n as u64);
                chunk_pages = probe_pages(&chunk, read_n, off, path, &rname);
            }
            &chunk
        };
        if read_n == 0 { break; }
        let mut pos = 0usize;
        let mut jumped = false;
        while pos + 44 <= read_n {
            let abs = off as usize + pos;
            let pre = if whole.is_some() && abs.is_multiple_of(0x1000) {
                whole_pages.get_mut(abs / 0x1000).map(Option::take)
            } else if whole.is_none() && pos.is_multiple_of(0x1000) {
                chunk_pages.get_mut(pos / 0x1000).map(Option::take)
            } else {
                None
            };
            let probed = match pre {
                Some(p) => p,
                None => probe_rage9_page(buf, pos, read_n, off, path, &rname, &mut probe_dec, &mut probe_out),
            };
            if let Some(entry) = probed {
                found.push(entry);
                pos += 0x1000;
                continue;
            }
            let tag = &buf[pos..pos + 4];
            if tag == b"ADAT" {
                awc_idx += 1;
                let sc = u32::from_le_bytes([buf[pos + 8], buf[pos + 9], buf[pos + 10], buf[pos + 11]]);
                found.push(RpfEntry {
                    source_rpf: path.clone(),
                    rpf_name: rname.clone(),
                    name: format!("sound_{awc_idx:04}_{sc}st.awc"),
                    kind: "AWC",
                    offset: off + pos as u64,
                    comp_size: 0,
                    raw_size: 0,
                    tracks: sc,
                    chunks: Vec::new(),
                });
            } else if &tag[..3] == b"KB2" && tag[3].is_ascii_lowercase() {
                let fsize = u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as u64;
                let frames = u32::from_le_bytes([buf[pos + 8], buf[pos + 9], buf[pos + 10], buf[pos + 11]]);
                let frames2 = u32::from_le_bytes([buf[pos + 16], buf[pos + 17], buf[pos + 18], buf[pos + 19]]);
                let w = u32::from_le_bytes([buf[pos + 20], buf[pos + 21], buf[pos + 22], buf[pos + 23]]);
                let h = u32::from_le_bytes([buf[pos + 24], buf[pos + 25], buf[pos + 26], buf[pos + 27]]);
                let fn_ = u32::from_le_bytes([buf[pos + 28], buf[pos + 29], buf[pos + 30], buf[pos + 31]]);
                let fd_ = u32::from_le_bytes([buf[pos + 32], buf[pos + 33], buf[pos + 34], buf[pos + 35]]);
                let tracks = u32::from_le_bytes([buf[pos + 40], buf[pos + 41], buf[pos + 42], buf[pos + 43]]);
                let tot = fsize + 8;
                if (1..=500_000).contains(&frames) && frames == frames2 && (64..=8192).contains(&w) && (64..=8192).contains(&h) && fn_ > 0 && fd_ > 0 && tot >= 1024 && tot <= sz - (off + pos as u64) {
                    bk2_idx += 1;
                    let abs_off = off + pos as u64;
                    found.push(RpfEntry {
                        source_rpf: path.clone(),
                        rpf_name: rname.clone(),
                        name: format!("video_{bk2_idx:03}_{w}x{h}_{frames}f.bk2"),
                        kind: "BK2",
                        offset: abs_off,
                        comp_size: tot,
                        raw_size: tot,
                        tracks,
                        chunks: Vec::new(),
                    });
                    let next_page = (abs_off + tot + 0xfff) & !0xfff;
                    if next_page > off + read_n as u64 {
                        off = next_page;
                        jumped = true;
                        if let Some((req, _)) = &reader {
                            let _ = req.send(off);
                        }
                        break;
                    } else {
                        pos = (next_page - off) as usize;
                        continue;
                    }
                }
            }
            pos += 0x1000;
        }
        if !jumped {
            off += read_n as u64;
        }
    }

    let awc_positions: Vec<usize> = found.iter().enumerate().filter(|(_, e)| e.kind == "AWC").map(|(i, _)| i).collect();
    for (k, &i) in awc_positions.iter().enumerate() {
        let nxt = if k + 1 < awc_positions.len() { found[awc_positions[k + 1]].offset } else { sz };
        let csz = (nxt - found[i].offset).min(32 * 1024 * 1024);
        found[i].comp_size = csz;
        found[i].raw_size = csz;
    }

    if let Some(data) = whole {
        let mut i = data_off.min(data.len());
        let mut stitch_chunks: Vec<(u64, u64)> = Vec::new();
        let mut stitch_raw = 0u64;
        let mut stitch_head: Vec<u8> = Vec::new();
        let mut z_idx = 0usize;
        let mut decomp = Decompress::new(true);
        let mut out = vec![0u8; 131072];

        while i + 8 < data.len() {
            if is_zlib_head(&data[i..]) {
                decomp.reset(true);
                let end = (i + 1024 * 1024).min(data.len());
                if let Ok(Status::StreamEnd) = decomp.decompress(&data[i..end], &mut out, FlushDecompress::Finish) {
                    let out_len = decomp.total_out() as usize;
                    let clen = decomp.total_in() as usize;
                    if out_len >= 32 && clen > 0 {
                        if stitch_chunks.is_empty() {
                            stitch_head = out[..out_len.min(300)].to_vec();
                        }
                        stitch_chunks.push((i as u64, clen as u64));
                        stitch_raw += out_len as u64;

                        // rpf9 splits xml and config streams on exact 64kb pages
                        if out_len != 65536 {
                            z_idx += 1;
                            let trimmed = stitch_head.iter().position(|&b| !matches!(b, 0xef | 0xbb | 0xbf | b' ' | b'\t' | b'\r' | b'\n')).unwrap_or(0);
                            let head = &stitch_head[trimmed..stitch_head.len().min(trimmed + 32)];
                            let (kind, name) = if head.starts_with(b"<?xml") || head.starts_with(b"<") {
                                let tag = extract_xml_tag(&stitch_head[trimmed..]).unwrap_or_else(|| "config".into());
                                ("XML", format!("{z_idx:04}_{tag}.xml"))
                            } else if stitch_head.starts_with(b"RBF0") {
                                let slug: String = stitch_head[12..].iter()
                                    .skip_while(|&&b| !b.is_ascii_alphabetic())
                                    .take_while(|&&b| b.is_ascii_alphanumeric() || b == b'_')
                                    .map(|&b| b as char)
                                    .take(36)
                                    .collect();
                                let nm = if slug.len() >= 4 { format!("{z_idx:04}_{slug}.rbf") } else { format!("{z_idx:04}_ui.rbf") };
                                ("RBF0", nm)
                            } else if stitch_head.starts_with(b"GFX") {
                                ("GFX", format!("{z_idx:04}_scaleform.gfx"))
                            } else if let Some(slug) = tex::probe_gxt_snippet(&stitch_head) {
                                ("GXT", format!("{z_idx:04}_{slug}.gxt"))
                            } else if stitch_head.starts_with(b"MHIU") {
                                ("HMAP", format!("{z_idx:04}_world_heightmap_2048.mhiu"))
                            } else if stitch_head.starts_with(b"PSIN") {
                                ("PSIN", format!("{z_idx:04}_resource.psin"))
                            } else if let Some(ver) = dat::probe_dat(&stitch_head) {
                                ("DAT", format!("{z_idx:04}_aud_dat{ver}.rel"))
                            } else if stitch_head.starts_with(b"AMB_STREAM_") {
                                ("XML", format!("{z_idx:04}_audio_packlist.txt"))
                            } else if stitch_head.starts_with(b"driverSettings_") {
                                ("XML", format!("{z_idx:04}_audio_driver_settings.txt"))
                            } else if stitch_head.starts_with(b"BaseCommon:") {
                                ("XML", format!("{z_idx:04}_audio_mounts.txt"))
                            } else if rsc9::is_qqqq_table(&stitch_head) {
                                ("RSC9", format!("{z_idx:04}_rsc9_table.bin"))
                            } else {
                                ("ZLIB", format!("{z_idx:04}_data.bin"))
                            };
                            found.push(RpfEntry {
                                source_rpf: path.clone(),
                                rpf_name: rname.clone(),
                                name,
                                kind,
                                offset: stitch_chunks[0].0,
                                comp_size: clen as u64,
                                raw_size: stitch_raw,
                                tracks: 0,
                                chunks: std::mem::take(&mut stitch_chunks),
                            });
                            stitch_raw = 0;
                        }
                        i = (i + clen + 15) & !15;
                        continue;
                    }
                }
            }
                i += 16;
        }
    }

    if rname.eq_ignore_ascii_case("audio.rpf") {
        enrich_audio_rpf_entries(path, sz, &mut found);
    }

    Ok((hdr_info, found))
}

// lowercase search text per entry, index matches entries
fn build_hay(entries: &[RpfEntry]) -> Vec<String> {
    par_map(entries, |e| {
        let shown = display_entry_name(e, false);
        let mut h = String::with_capacity(e.name.len() + shown.len() + e.rpf_name.len() + e.kind.len() + 3);
        for (i, part) in [e.name.as_str(), &shown, &e.rpf_name, e.kind].into_iter().enumerate() {
            if i > 0 {
                h.push('|');
            }
            h.push_str(part);
        }
        if h.is_ascii() {
            h.make_ascii_lowercase();
            h
        } else {
            h.to_lowercase()
        }
    })
}

// indices in 0..n that pass keep, in order, checked on all cores
fn par_filter(n: usize, keep: impl Fn(usize) -> bool + Sync) -> Vec<usize> {
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let per = n.div_ceil(cores).max(16384);
    if per >= n {
        return (0..n).filter(|&i| keep(i)).collect();
    }
    let keep = &keep;
    std::thread::scope(|s| {
        let parts: Vec<_> = (0..n)
            .step_by(per)
            .map(|a| s.spawn(move || (a..(a + per).min(n)).filter(|&i| keep(i)).collect::<Vec<_>>()))
            .collect();
        parts.into_iter().flat_map(|p| p.join().expect("par_filter worker panicked")).collect()
    })
}

// order keeping map over all cores for the big per entry passes
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = items.len().div_ceil(cores).max(4096);
    if per >= items.len() {
        return items.iter().map(f).collect();
    }
    let f = &f;
    std::thread::scope(|s| {
        let parts: Vec<_> = items.chunks(per).map(|c| s.spawn(move || c.iter().map(f).collect::<Vec<R>>())).collect();
        let mut out = Vec::with_capacity(items.len());
        for p in parts {
            out.extend(p.join().expect("par_map worker panicked"));
        }
        out
    })
}

fn display_entry_name(e: &RpfEntry, audio_named: bool) -> std::borrow::Cow<'_, str> {
    if !audio_named && e.kind == "AWC" && e.name.starts_with("sound_") && e.name.matches('_').count() > 2 {
        if let Some(tail) = e.name.rsplit('_').next() {
            if e.name.len() >= 10 {
                return std::borrow::Cow::Owned(format!("{}_{}", &e.name[..10], tail));
            }
        }
    }
    std::borrow::Cow::Borrowed(&e.name)
}

fn enrich_audio_rpf_entries(audio_rpf: &PathBuf, total_sz: u64, entries: &mut Vec<RpfEntry>) -> HashMap<u64, dat::ResolvedAwcMeta> {
    let mut meta_map = HashMap::new();
    let awc_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.kind == "AWC")
        .map(|(i, _)| i)
        .collect();
    let first_awc_off = awc_indices
        .first()
        .map(|&i| entries[i].offset)
        .unwrap_or(19_509_248);
    let Some(cat) = dat::parse_audio_rpf_catalog(audio_rpf, first_awc_off) else {
        return meta_map;
    };
    let rname = audio_rpf
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    if !entries.iter().any(|e| e.kind == "DAT") {
        for (name, kind, off, raw_size, chunks) in &cat.metadata_chunks {
            let comp_size: u64 = chunks.iter().map(|c| c.1).sum();
            entries.push(RpfEntry {
                source_rpf: audio_rpf.clone(),
                rpf_name: rname.clone(),
                name: name.clone(),
                kind,
                offset: *off,
                comp_size,
                raw_size: *raw_size,
                tracks: 0,
                chunks: chunks.clone(),
            });
        }
    }
    if !awc_indices.is_empty() && !cat.banks.is_empty() {
        let awc_offs_tracks: Vec<(u64, u32)> = awc_indices
            .iter()
            .map(|&i| (entries[i].offset, entries[i].tracks))
            .collect();
        let resolved = dat::resolve_awc_names_from_catalog(&awc_offs_tracks, total_sz, &cat);
        for (k, &entry_idx) in awc_indices.iter().enumerate() {
            if let Some(Some(meta)) = resolved.get(k) {
                entries[entry_idx].name = meta.name.clone();
                if meta.exact_size > 0 {
                    entries[entry_idx].comp_size = meta.exact_size;
                    entries[entry_idx].raw_size = meta.exact_size;
                }
                meta_map.insert(entries[entry_idx].offset, meta.clone());
            }
        }
    }
    meta_map
}

impl eframe::App for Rpf9App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // a file or folder dropped onto the window opens like open file
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone())) {
            self.open_any_path(path);
        }
        self.poll_sgd5_cache();
        if let Some(pick) = self.lod_pick.take() {
            self.apply_lod_pick(pick);
        }
        if std::mem::take(&mut self.tex_pick) {
            self.load_model_textures();
        }
        let (media_loading, media_status) = {
            let media = Arc::clone(&self.media);
            let mut m = media.lock().unwrap();
            if let Some((generation, database)) = m.loaded_vehicle_configurations.take() {
                if generation == self.vehicle_configuration_generation
                    && generation == *self.watch_gen.lock().unwrap()
                {
                    self.vehicle_configuration_db = database;
                    self.vehicle_configuration_loaded = true;
                    self.vehicle_configuration_loading = false;
                    self.refresh_vehicle_variations();
                }
            }
            if let Some(mut mesh) = m.loaded_mesh.take() {
                self.textures.clear();
                self.tex_handle = None;
                self.tex_from_model = false;
                if let Some([yaw, pitch, zoom, pan_x, pan_y]) = self.lod_keep_cam.take() {
                    (mesh.yaw, mesh.pitch, mesh.zoom, mesh.pan_x, mesh.pan_y) = (yaw, pitch, zoom, pan_x, pan_y);
                }
                if let Some((lbl, _)) = self.lod_siblings.iter().find(|(_, i)| Some(*i) == self.selected) {
                    mesh.lod_label = format!("{lbl} detail file · {}", mesh.lod_label);
                }
                let is_vehicle = self
                    .model_entry
                    .and_then(|index| self.state.lock().unwrap().entries.get(index).map(|entry| entry.kind == "VEH"))
                    .unwrap_or(false);
                if is_vehicle {
                    self.vehicle_base_mesh = Some(mesh.clone());
                }
                self.mesh3d = Some(mesh);
                if self.vehicle_base_mesh.is_some() {
                    self.rebuild_vehicle_parts();
                }
                self.show_3d = true;
            }
            let loaded_parts: Vec<_> = std::mem::take(&mut m.loaded_vehicle_parts);
            let had_loaded_parts = !loaded_parts.is_empty();
            for (owner, entry, mesh) in loaded_parts {
                if Some(owner) != self.model_entry {
                    continue;
                }
                self.vehicle_part_meshes.insert(entry, mesh);
                if let Some(choice) = self.vehicle_parts.iter_mut().find(|p| p.entry == entry) {
                    choice.loading = false;
                }
            }
            if had_loaded_parts {
                self.rebuild_vehicle_parts();
            }
            if let Some((texs, note)) = m.loaded_model_textures.take() {
                self.textures = texs;
                self.tex_sel = 0;
                self.tex_handle = None;
                self.tex_zoom = 1.0;
                self.tex_from_model = true;
                self.state.lock().unwrap().status = note;
            }
            if let Some(texs) = m.loaded_textures.take() {
                self.mesh3d = None;
                self.show_3d = false;
                self.tex_from_model = false;
                self.textures = texs;
                self.tex_sel = 0;
                self.tex_handle = None;
                self.tex_zoom = 1.0;
            }
            if let Some((tracks, awc_bytes)) = m.loaded_tracks.take() {
                self.audio_player.set_tracks(tracks, awc_bytes);
                self.audio_player.bank_ctx = m.loaded_bank_ctx.take();
            }
            (m.loading, m.status.clone())
        };
        if let Some(q) = self.audio_player.requested_search.take() {
            self.category = Category::Audio;
            self.filter = q;
        }

        egui::TopBottomPanel::top("top").frame(egui::Frame::new().fill(Color32::from_rgb(24, 14, 36)).inner_margin(10.0)).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(tex) = &self.logo_tex {
                    ui.image((tex.id(), egui::vec2(28.0, 28.0)));
                }
                ui.add_space(6.0);
                ui.label(RichText::new("PERCYSTAR").size(17.0).strong().color(Color32::from_rgb(255, 168, 28)));
                ui.label(RichText::new("RPF9").size(13.0).strong().color(Color32::from_rgb(255, 72, 165)));
                ui.label(RichText::new(format!("v{APP_VERSION}")).size(11.5).color(Color32::from_rgb(165, 145, 195)));
                if ui
                    .add(egui::Button::new(RichText::new("♥ Sponsor").size(11.0).color(Color32::from_rgb(155, 135, 185))).frame(false))
                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                    .on_hover_text("Optional: support ongoing R&D on GitHub Sponsors")
                    .clicked()
                {
                    ctx.open_url(egui::OpenUrl::new_tab("https://github.com/sponsors/rpatty-dev"));
                }
                ui.add_space(10.0);
                // Top-level mode tabs: Explorer | Model Viewer | World Map | Shaders (RDNA2)
                for (tab_id, label) in [
                    (AppTab::Explorer, "Explorer"),
                    (AppTab::ModelViewer, "Model Viewer"),
                    (AppTab::WorldMap, "World Map"),
                    (AppTab::Shaders, "Shaders (RDNA2)"),
                ] {
                    let active = self.tab == tab_id;
                    let btn = egui::Button::new(
                        RichText::new(label)
                            .size(12.5)
                            .strong()
                            .color(if active { Color32::BLACK } else { Color32::from_rgb(225, 215, 240) }),
                    )
                    .fill(if active { Color32::from_rgb(255, 168, 28) } else { Color32::from_rgb(42, 26, 62) });
                    if ui.add(btn).clicked() && self.tab != tab_id {
                        self.tab = tab_id;
                        if tab_id == AppTab::ModelViewer {
                            self.stop_bink_player();
                            self.show_3d = true;
                        } else if tab_id == AppTab::WorldMap {
                            self.stop_bink_player();
                            if self.world_map.data.is_none() {
                                let common = self.target_path.as_ref().and_then(|p| {
                                    if p.is_dir() { Some(p.join("common.rpf")) }
                                    else if p.file_name().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("common.rpf")).unwrap_or(false) { Some(p.clone()) }
                                    else { p.parent().map(|d| d.join("common.rpf")) }
                                }).filter(|p| p.exists());
                                if let Some(cp) = common {
                                    if let Ok(d) = worldmap::load_mhiu_from_rpf(&cp) {
                                        self.world_map.status = format!("Loaded {}x{} MHIU", d.w, d.h);
                                        self.world_map.data = Some(d);
                                    }
                                }
                            }
                        } else if tab_id == AppTab::Shaders {
                            self.stop_bink_player();
                            if self.sh_cache.is_none() {
                                if let Some(lp) = self.find_local_shaders_dir() {
                                    self.load_ps09_path(&lp);
                                }
                            }
                        }
                    }
                }
                let upd_avail = self.update.lock().unwrap().available.clone();
                if let Some((ref new_ver, _)) = upd_avail {
                    let pill = egui::Button::new(
                        RichText::new(format!("★ Update Available: v{new_ver}")).size(12.0).strong().color(Color32::BLACK),
                    )
                    .fill(Color32::from_rgb(255, 168, 28));
                    if ui.add(pill).clicked() {
                        self.update.lock().unwrap().show_modal = true;
                    }
                }
                ui.add_space(10.0);
                if ui.button("Open Folder...").clicked() {
                    if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                        self.start_folder_scan(folder);
                    }
                }
                if ui.button("Open File...").on_hover_text("Open an .rpf/.ttpf archive, .obj model, or .ps09 shader cache").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("Supported files", &["rpf", "ttpf", "obj", "ps09"])
                        .add_filter("RPF / TTPF Archives", &["rpf", "ttpf"])
                        .add_filter("OBJ models / PS09 shaders", &["obj", "ps09"])
                        .pick_file()
                    {
                        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("obj")) {
                            self.open_obj_path(&p);
                        } else {
                            self.open_any_path(p);
                        }
                    }
                }
                if ui.button("Open .OBJ").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("OBJ 3D Model", &["obj"]).pick_file() {
                        self.open_obj_path(&p);
                    }
                }
                if ui.button("Import Catalog").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Models Catalog CSV", &["csv"]).pick_file() {
                        self.import_catalog_csv(&p);
                    }
                }
                if self.is_folder && ui.button("Rescan").clicked() {
                    if let Some(folder) = self.target_path.clone() {
                        self.start_folder_scan(folder);
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let exporting = self.export_busy.load(Ordering::Relaxed);
                    if exporting {
                        if ui.button("Cancel Export").clicked() {
                            self.export_cancel.store(true, Ordering::SeqCst);
                        }
                        ctx.request_repaint_after(Duration::from_millis(200));
                    } else if ui.button("Export All").clicked() {
                        let n = self.state.lock().unwrap().entries.len();
                        let all: Vec<usize> = (0..n).collect();
                        self.export_entries(&all);
                    }
                    if !exporting && ui.button("Export Selected").clicked() {
                        let targets = self.ordered_selected_indices();
                        if !targets.is_empty() {
                            self.export_entries(&targets);
                        }
                    }
                    let copy_shortcut = !ctx.wants_keyboard_input() && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::C));
                    if ui.button("Copy Path").on_hover_text("Copy selected file path(s) — Ctrl+C or Ctrl+Click/Shift+Click for multi-select").clicked() || copy_shortcut {
                        let targets = self.ordered_selected_indices();
                        if !targets.is_empty() {
                            let copied = {
                                let st = self.state.lock().unwrap();
                                let refs: Vec<&RpfEntry> = targets.iter().filter_map(|&i| st.entries.get(i)).collect();
                                if refs.is_empty() { None } else { Some((refs.len(), format_multi_copy(&refs, 1))) }
                            };
                            if let Some((n, s)) = copied {
                                ui.ctx().copy_text(s.clone());
                                self.state.lock().unwrap().status = if n > 1 {
                                    format!("Copied {n} paths")
                                } else {
                                    format!("Copied: {s}")
                                };
                            }
                        }
                    }
                    let r = ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(250.0).hint_text("Search (Ctrl+F) · words AND · -exclude"));
                    if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F)) {
                        r.request_focus();
                    }
                    if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        self.filter.clear();
                    }
                });
            });
        });

        egui::TopBottomPanel::bottom("bottom").frame(egui::Frame::new().fill(Color32::from_rgb(14, 9, 22)).inner_margin(egui::Margin::symmetric(12, 5))).show(ctx, |ui| {
            let (hdr, status, scanning) = {
                let st = self.state.lock().unwrap();
                (st.header_info.clone(), st.status.clone(), st.scanning)
            };
            ui.horizontal(|ui| {
                if scanning { draw_percystar_spinner(ui, 18.0); }
                ui.label(RichText::new(status).size(12.0).color(Color32::from_rgb(185, 190, 204)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(hdr).size(11.5).color(Color32::from_rgb(115, 122, 136)));
                });
            });
        });

        if self.tab == AppTab::ModelViewer {
            self.render_model_viewer_tab(ctx, media_loading, &media_status);
        } else if self.tab == AppTab::WorldMap {
            let nav_tiles: Vec<(usize, f32, f32)> = {
                let st = self.state.lock().unwrap();
                st.entries.iter().enumerate().filter_map(|(i, e)| {
                    if e.kind == "NAVM" {
                        worldmap::parse_nav_coords(&e.name).map(|(wx, wy)| (i, wx, wy))
                    } else { None }
                }).collect()
            };
            let common = self.target_path.as_ref().and_then(|p| {
                if p.is_dir() { Some(p.join("common.rpf")) }
                else { p.parent().map(|d| d.join("common.rpf")) }
            }).filter(|p| p.exists());
            let levels = self.target_path.as_ref().and_then(|p| {
                if p.is_dir() { Some(p.join("levels.rpf")) }
                else { p.parent().map(|d| d.join("levels.rpf")) }
            }).filter(|p| p.exists());
            let mut clicked_nav = None;
            egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::from_rgb(13, 9, 21)).inner_margin(10.0)).show(ctx, |ui| {
                clicked_nav = worldmap::render_world_map_ui(ui, ctx, &mut self.world_map, &nav_tiles, common.as_deref(), levels.as_deref());
            });
            if let Some(idx) = clicked_nav {
                self.tab = AppTab::Explorer;
                self.select_entry(idx);
            }
        } else if self.tab == AppTab::Shaders {
            self.render_shader_tab(ctx);
        } else {
        egui::SidePanel::left("tree").default_width(195.0).frame(egui::Frame::new().fill(Color32::from_rgb(18, 12, 28)).inner_margin(8.0)).show(ctx, |ui| {
            self.refresh_view();
            let c = self.counts;
            ui.label(RichText::new("FILTER BY TYPE").size(11.0).color(Color32::from_rgb(145, 135, 175)));
            ui.add_space(4.0);
            let has_entries = c[0] > 0;
            for (cat, label, count) in [
                (Category::All, "All Files", c[0]),
                (Category::Vehicles, "Vehicles", c[1]),
                (Category::Models, "3D Models", c[2] + c[1]),
                (Category::Textures, "Textures", c[3]),
                (Category::Ui, "UI & Text", c[4]),
                (Category::NavMesh, "NavMesh 3D", c[5]),
                (Category::Videos, "Videos", c[6]),
                (Category::Audio, "Audio", c[7]),
                (Category::Xml, "Configs (XML)", c[8]),
                (Category::Resources, "Anim / PSO / RSC9", c[9]),
                (Category::Binary, "Binary", c[10]),
            ] {
                if has_entries && cat != Category::All && count == 0 {
                    continue;
                }
                if ui.selectable_label(self.category == cat, format!("{label} ({count})")).clicked() {
                    self.category = cat;
                }
            }
            if self.cached_rpf_names.len() > 1 {
                ui.add_space(10.0);
                ui.label(RichText::new("ARCHIVES").size(11.0).color(Color32::from_rgb(110, 118, 134)));
                ui.add_space(4.0);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if ui.selectable_label(self.selected_rpf == "All RPFs", "All RPFs").clicked() {
                        self.selected_rpf = "All RPFs".into();
                    }
                    for rn in &self.cached_rpf_names {
                        if ui.selectable_label(&self.selected_rpf == rn, rn).clicked() {
                            self.selected_rpf = rn.clone();
                        }
                    }
                });
            }
        });

        egui::SidePanel::left("filelist").default_width(390.0).frame(egui::Frame::new().fill(Color32::from_rgb(22, 15, 34)).inner_margin(8.0)).show(ctx, |ui| {
            if self.category == Category::Models {
                ui.horizontal(|ui| {
                    let btn = egui::Button::new(
                        RichText::new(if self.vehicles_only { "★ Vehicles Only: ON" } else { "Vehicles Only" })
                            .strong()
                            .color(if self.vehicles_only { Color32::BLACK } else { Color32::from_rgb(255, 72, 165) }),
                    )
                    .fill(if self.vehicles_only { Color32::from_rgb(255, 168, 28) } else { Color32::from_rgb(38, 24, 56) });
                    if ui.add(btn).clicked() {
                        self.vehicles_only = !self.vehicles_only;
                    }
                });
                ui.add_space(4.0);
            }
            if self.category == Category::Audio {
                ui.horizontal(|ui| {
                    let btn = egui::Button::new(
                        RichText::new(if self.audio_named { "Audio Names: ON" } else { "Audio Names: OFF (Raw)" })
                            .strong()
                            .color(if self.audio_named { Color32::BLACK } else { Color32::from_rgb(255, 72, 165) }),
                    )
                    .fill(if self.audio_named { Color32::from_rgb(90, 235, 165) } else { Color32::from_rgb(38, 24, 56) });
                    if ui.add(btn).clicked() {
                        self.audio_named = !self.audio_named;
                    }
                });
                ui.add_space(4.0);
            }
            self.refresh_view();
            if !ctx.wants_keyboard_input() && !self.view.is_empty() {
                let cur_pos = self.selected.and_then(|sel| self.view.iter().position(|&i| i == sel));
                let allow_wasd = !self.show_3d;
                let step_to: Option<usize> = ctx.input(|i| {
                    if i.key_pressed(egui::Key::ArrowDown) || (allow_wasd && i.key_pressed(egui::Key::S)) {
                        Some(match cur_pos {
                            Some(p) => (p + 1).min(self.view.len() - 1),
                            None => 0,
                        })
                    } else if i.key_pressed(egui::Key::ArrowUp) || (allow_wasd && i.key_pressed(egui::Key::W)) {
                        Some(match cur_pos {
                            Some(p) => p.saturating_sub(1),
                            None => self.view.len() - 1,
                        })
                    } else if i.key_pressed(egui::Key::PageDown) {
                        Some(cur_pos.map_or(0, |p| (p + 15).min(self.view.len() - 1)))
                    } else if i.key_pressed(egui::Key::PageUp) {
                        Some(cur_pos.map_or(0, |p| p.saturating_sub(15)))
                    } else if i.key_pressed(egui::Key::Home) {
                        Some(0)
                    } else if i.key_pressed(egui::Key::End) {
                        Some(self.view.len() - 1)
                    } else {
                        None
                    }
                });
                if let Some(pos) = step_to {
                    let target_idx = self.view[pos];
                    if self.selected != Some(target_idx) {
                        self.select_entry(target_idx);
                        self.explorer_scroll_to_sel = true;
                    }
                }
            }
            if !ctx.wants_keyboard_input() && !self.view.is_empty() && ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::A)) {
                self.multi_selected = self.view.iter().copied().collect();
                self.state.lock().unwrap().status = format!("{} entries selected (Ctrl+C to copy)", self.multi_selected.len());
            }
            ui.horizontal(|ui| {
                let n_sel = self.multi_selected.len();
                let res_lbl = if n_sel > 1 {
                    format!("{} results ({n_sel} selected)", self.view.len())
                } else {
                    format!("{} results", self.view.len())
                };
                ui.label(RichText::new(res_lbl).size(11.0).color(Color32::from_rgb(110, 118, 134)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    sort_combo(ui, "explorer_sort_combo", &mut self.sort_mode);
                });
            });
            let mut clicked: Option<(usize, bool, bool)> = None;
            let mut copied_msg: Option<String> = None;
            let mut row_rects: Vec<(usize, f32, f32)> = Vec::new();
            let scroll_to_sel = std::mem::take(&mut self.explorer_scroll_to_sel);
            // rows must be exactly row_h tall or show_rows miscomputes total height
            const ROW_H: f32 = 20.0;
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            let (p_down, p_pos, ctrl_held, esc_pressed) = ctx.input(|i| {
                (
                    i.pointer.primary_down(),
                    i.pointer.interact_pos(),
                    i.modifiers.command,
                    i.key_pressed(egui::Key::Escape),
                )
            });
            if esc_pressed || !p_down {
                self.marquee_start = None;
            }
            let marquee_active = p_down
                && self.marquee_start.is_some()
                && p_pos.is_some_and(|p| self.marquee_start.unwrap().distance(p) > 5.0);
            let scroll_resp = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .enable_scrolling(!marquee_active)
                .show_rows(ui, ROW_H, self.view.len(), |ui, range| {
                let rows: Vec<(usize, RpfEntry)> = {
                    let st = self.state.lock().unwrap();
                    self.view[range].iter().filter_map(|&i| st.entries.get(i).map(|e| (i, e.clone()))).collect()
                };
                for (idx, e) in &rows {
                    let w = ui.available_width();
                    let row_alloc = ui.allocate_ui_with_layout(egui::vec2(w, ROW_H), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.set_height(ROW_H);
                        ui.label(RichText::new(format!("{:<4}", e.kind)).strong().size(11.5).color(kind_color(e.kind)));
                        let sz_s = format_size(e.raw_size);
                        let dname = display_entry_name(e, self.audio_named);
                        let label = if self.is_folder {
                            format!("{} / {}  ({})", e.rpf_name, dname, sz_s)
                        } else {
                            format!("{}  ({})", dname, sz_s)
                        };
                        let is_sel = self.selected == Some(*idx) || self.multi_selected.contains(idx);
                        let resp = ui.selectable_label(is_sel, RichText::new(label).size(12.5));
                        if is_sel && scroll_to_sel && self.selected == Some(*idx) {
                            resp.scroll_to_me(Some(egui::Align::Center));
                        }
                        if resp.clicked() {
                            let (ctrl, shift) = ui.input(|i| (i.modifiers.command, i.modifiers.shift));
                            clicked = Some((*idx, ctrl, shift));
                        }
                        if self.multi_selected.len() > 1 && self.multi_selected.contains(idx) {
                            let targets = self.ordered_selected_indices();
                            resp.context_menu(|ui| {
                                let n = targets.len();
                                if ui.button(format!("Copy {n} File Names")).clicked() {
                                    let st = self.state.lock().unwrap();
                                    let refs: Vec<&RpfEntry> = targets.iter().filter_map(|&i| st.entries.get(i)).collect();
                                    ui.ctx().copy_text(format_multi_copy(&refs, 0));
                                    copied_msg = Some(format!("Copied {n} file names"));
                                    ui.close_menu();
                                }
                                if ui.button(format!("Copy {n} Archive / File Paths")).clicked() {
                                    let st = self.state.lock().unwrap();
                                    let refs: Vec<&RpfEntry> = targets.iter().filter_map(|&i| st.entries.get(i)).collect();
                                    ui.ctx().copy_text(format_multi_copy(&refs, 1));
                                    copied_msg = Some(format!("Copied {n} paths"));
                                    ui.close_menu();
                                }
                                if ui.button(format!("Copy {n} Offset Locations")).clicked() {
                                    let st = self.state.lock().unwrap();
                                    let refs: Vec<&RpfEntry> = targets.iter().filter_map(|&i| st.entries.get(i)).collect();
                                    ui.ctx().copy_text(format_multi_copy(&refs, 2));
                                    copied_msg = Some(format!("Copied {n} locations"));
                                    ui.close_menu();
                                }
                            });
                        } else if let Some(msg) = entry_context_menu(&resp, e) {
                            copied_msg = Some(msg);
                        }
                    });
                    row_rects.push((*idx, row_alloc.response.rect.min.y, row_alloc.response.rect.max.y));
                }
            });
            // Marquee drag selection in Explorer list (#38, #41 scroll-locked while active)
            let list_rect = scroll_resp.inner_rect;
            if p_down && !esc_pressed {
                if let Some(pos) = p_pos {
                    if self.marquee_start.is_none() && list_rect.contains(pos) {
                        self.marquee_start = Some(pos);
                    }
                    if let Some(start) = self.marquee_start {
                        if start.distance(pos) > 5.0 {
                            let drag_rect = egui::Rect::from_two_pos(start, pos).intersect(list_rect);
                            ui.painter().rect(
                                drag_rect,
                                2.0,
                                Color32::from_rgba_unmultiplied(120, 210, 255, 40),
                                egui::Stroke::new(1.0_f32, Color32::from_rgb(120, 210, 255)),
                                egui::StrokeKind::Inside,
                            );
                            if !ctrl_held {
                                self.multi_selected.clear();
                            }
                            for &(idx, y0, y1) in &row_rects {
                                if y1 >= drag_rect.min.y && y0 <= drag_rect.max.y {
                                    self.multi_selected.insert(idx);
                                }
                            }
                        }
                    }
                }
            } else {
                self.marquee_start = None;
            }
            if let Some(msg) = copied_msg {
                self.state.lock().unwrap().status = msg;
            }
            if let Some((i, ctrl, shift)) = clicked {
                apply_multi_select(&self.view, &mut self.multi_selected, &mut self.multi_anchor, i, ctrl, shift);
                if !ctrl && !shift {
                    self.select_entry(i);
                } else {
                    self.selected = Some(i);
                }
            }
        });

        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::from_rgb(14, 10, 22)).inner_margin(12.0)).show(ctx, |ui| {
            if self.parent_hwnd == 0 {
                let h = active_window();
                if h != 0 { self.parent_hwnd = h; }
            }
            let (bink_pid, bink_hwnd_opt, bink_clip_opt) = {
                let m = self.media.lock().unwrap();
                (m.bink_child_pid, m.bink_hwnd, m.bink_clip.clone())
            };

            if let Some(pid) = bink_pid {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&media_status).strong().color(Color32::from_rgb(252, 175, 23)));
                    if ui.button("Stop").clicked() {
                        self.stop_bink_player();
                    }
                    if ui.button("Pop-out").clicked() {
                        if let Some(cp) = bink_clip_opt {
                            self.stop_bink_player();
                            let _ = Command::new(find_binkplay()).arg(&cp).spawn();
                        }
                    }
                });
                ui.add_space(6.0);
                let rect = ui.available_rect_before_wrap();
                let ppp = ctx.pixels_per_point();
                let (rx, ry, rw, rh) = ((rect.min.x * ppp) as i32, (rect.min.y * ppp) as i32, (rect.width() * ppp) as i32, (rect.height() * ppp) as i32);
                if let Some(bh) = bink_hwnd_opt {
                    let cur = [rx, ry, rw, rh];
                    if cur != self.last_dock_rect {
                        self.last_dock_rect = cur;
                        move_window(bh, rx, ry, rw.max(64), rh.max(64));
                    }
                } else if self.parent_hwnd != 0 {
                    if let Some(bh) = find_hwnd_for_pid(pid) {
                        dock_child_hwnd(bh, self.parent_hwnd, rx, ry, rw, rh);
                        self.media.lock().unwrap().bink_hwnd = Some(bh);
                        self.last_dock_rect = [rx, ry, rw, rh];
                    } else {
                        ctx.request_repaint();
                    }
                }
            } else if media_loading {
                ui.horizontal(|ui| {
                    draw_percystar_spinner(ui, 24.0);
                    ui.label(RichText::new(&media_status).color(Color32::from_rgb(252, 175, 23)));
                });
            } else if !self.audio_player.tracks.is_empty() {
                audio_player::render_awc_audio_view(ui, ctx, &mut self.audio_player, &media_status);
            } else if !media_status.is_empty() {
                let mut lines = media_status.lines();
                let headline = lines.next().unwrap_or(&media_status);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(headline).strong().color(Color32::from_rgb(255, 140, 120)));
                    if ui.small_button("Copy").clicked() {
                        ctx.copy_text(media_status.clone());
                    }
                });
                let rest: Vec<&str> = lines.collect();
                if !rest.is_empty() {
                    ui.add_space(6.0);
                    egui::Frame::new()
                        .fill(Color32::from_rgb(30, 18, 44))
                        .stroke(egui::Stroke::new(1.0_f32, Color32::from_rgb(110, 58, 78)))
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
                                for line in rest {
                                    ui.label(RichText::new(line).monospace().size(11.5).color(Color32::from_rgb(235, 210, 215)));
                                }
                            });
                        });
                }
            } else if !self.textures.is_empty() {
                self.render_texture_viewer(ui);
            } else if self.mesh3d.is_some() {
                self.vehicle_variations_ui(ui);
                self.vehicle_parts_ui(ui);
                self.vehicle_data_ui(ui);
                let mesh = self.mesh3d.as_mut().unwrap();
                let mut toggle_3d = self.show_3d;
                ui.horizontal(|ui| {
                    if ui.selectable_label(!toggle_3d, "Info").clicked() { toggle_3d = false; }
                    if ui.selectable_label(toggle_3d, if self.anim_dur > 0.0 { "3D Dummy" } else { "3D View" }).clicked() { toggle_3d = true; }
                    if self.anim_dur > 0.0 {
                        ui.separator();
                        if ui.button(if self.anim_playing { "Pause" } else { "Play" }).clicked() {
                            self.anim_playing = !self.anim_playing;
                        }
                        let mut t = self.anim_time;
                        if ui.add(egui::Slider::new(&mut t, 0.0..=self.anim_dur).suffix("s").fixed_decimals(2)).changed() {
                            self.anim_time = t;
                            let next = anim::build_stickman_mesh(&self.anim_name, t, self.anim_dur, self.anim_tracks, self.anim_seed);
                            mesh.verts = next.verts;
                            mesh.invalidate_normals();
                        }
                    }
                    if let Some(pick) = lod_switch_ui(ui, mesh, &self.lod_siblings, self.selected) {
                        self.lod_pick = Some(pick);
                    }
                    if self.model_entry.is_some() && ui.button("Texture Browser").on_hover_text(MODEL_TEX_TIP).clicked() {
                        self.tex_pick = true;
                    }
                });
                if self.anim_dur > 0.0 && self.anim_playing && toggle_3d {
                    let dt = ctx.input(|i| i.unstable_dt).clamp(0.001, 0.05);
                    self.anim_time = (self.anim_time + dt) % self.anim_dur.max(0.1);
                    let next = anim::build_stickman_mesh(&self.anim_name, self.anim_time, self.anim_dur, self.anim_tracks, self.anim_seed);
                    mesh.verts = next.verts;
                    mesh.invalidate_normals();
                    ctx.request_repaint();
                }
                self.show_3d = toggle_3d;
                ui.separator();
                if self.show_3d {
                    render_mesh3d_viewport(ui, mesh);
                } else {
                    egui::ScrollArea::both().show(ui, |ui| {
                        ui.add(egui::TextEdit::multiline(&mut self.preview).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY));
                    });
                }
            } else {
                if self.psin_selected {
                    self.psin_tools_ui(ui);
                }
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.add(egui::TextEdit::multiline(&mut self.preview).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY));
                });
            }
        });
        } // end Explorer tab

        let (show_upd, upd_avail, upd_inst, upd_status) = {
            let g = self.update.lock().unwrap();
            (g.show_modal, g.available.clone(), g.installing, g.status.clone())
        };
        if show_upd {
            if let Some((ref new_ver, ref rel_url)) = upd_avail {
                let mut open = true;
                egui::Window::new("Update Available")
                    .open(&mut open)
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                    .show(ctx, |ui| {
                        ui.label(RichText::new(format!("PercyStar v{new_ver} is ready!")).size(15.0).strong().color(Color32::from_rgb(255, 168, 28)));
                        ui.label(RichText::new(format!("Installed: v{APP_VERSION}  ->  Latest: v{new_ver}")).color(Color32::from_rgb(200, 190, 220)));
                        ui.add_space(8.0);
                        if !upd_status.is_empty() {
                            ui.label(RichText::new(&upd_status).color(Color32::from_rgb(255, 72, 165)));
                            ui.add_space(4.0);
                        }
                        ui.horizontal(|ui| {
                            let btn = egui::Button::new(
                                RichText::new(if upd_inst { "Updating..." } else { "Download & Install" }).strong().color(Color32::BLACK),
                            )
                            .fill(Color32::from_rgb(255, 168, 28));
                            // releases ship a Win64 zip only; other platforms just get "View Release"
                            if cfg!(windows) && ui.add_enabled(!upd_inst, btn).clicked() {
                                spawn_install_update(new_ver.clone(), Arc::clone(&self.update), ctx.clone());
                            }
                            if ui.button("View Release").clicked() {
                                open_url(rel_url);
                            }
                            if ui.add_enabled(!upd_inst, egui::Button::new("Later")).clicked() {
                                self.update.lock().unwrap().show_modal = false;
                            }
                        });
                    });
                if !open {
                    self.update.lock().unwrap().show_modal = false;
                }
            }
        }
    }
}


fn export_models_cli(rpf_path: &PathBuf, out_dir: &PathBuf, filter: &str, vehicles_only: bool) {
    let _ = fs::create_dir_all(out_dir);
    let q = filter.to_ascii_lowercase();
    eprintln!("Scanning {}...", rpf_path.display());
    let entries = if vehicles_only {
        scan_levels_vehicle_zone(rpf_path, None)
    } else {
        match scan_single_rpf(rpf_path, None) {
            Ok((_, e)) => e,
            Err(err) => {
                eprintln!("Scan failed: {err}");
                return;
            }
        }
    };
    let mut best: HashMap<String, (u64, String, bool)> = HashMap::new();
    for e in entries {
        if vehicles_only {
            if e.kind != "VEH" || e.name.to_ascii_lowercase().contains("_mid.") { continue; }
            let stem = e.name.trim_end_matches(".#ft");
            let is_hi = stem.ends_with("_hi");
            let base = stem.trim_end_matches("_hi").to_ascii_lowercase();
            match best.get(&base) {
                Some((_, _, prev_hi)) if *prev_hi && !is_hi => {}
                _ => { best.insert(base, (e.offset, e.name, is_hi)); }
            }
        } else {
            if e.kind != "MODEL" && e.kind != "VEH" { continue; }
            let lo = e.name.to_ascii_lowercase();
            if !q.is_empty() && !q.split(',').any(|tok| !tok.is_empty() && lo.contains(tok)) { continue; }
            best.entry(lo).or_insert((e.offset, e.name, false));
        }
    }
    let mut list: Vec<(String, u64, String)> = best.into_iter().map(|(k, (o, n, _))| (k, o, n)).collect();
    list.sort_by(|a, b| a.0.cmp(&b.0));
    eprintln!("Found {} models -> exporting .OBJs...", list.len());
    let mut ok = 0usize;
    for (idx, (key, off, name)) in list.iter().enumerate() {
        if let Ok(mesh) = decode_rage9_model_from_rpf(rpf_path, *off, name) {
            let fname = if vehicles_only { format!("{key}.obj") } else { format!("{}.obj", name.replace('#', "").replace('.', "_")) };
            if mesh.export_obj(&out_dir.join(safe_file_name(&fname))).is_ok() {
                ok += 1;
                if ok <= 25 || ok.is_multiple_of(50) {
                    eprintln!("  [{}/{}] {} ({} v, {} f)", idx + 1, list.len(), key, mesh.verts.len(), mesh.tris.len());
                }
            }
        }
    }
    eprintln!("DONE: Exported {ok}/{} models to {}", list.len(), out_dir.display());
}

fn dump_textures_cli(src_path: &PathBuf, out_dir: &PathBuf, limit: usize) {
    let _ = fs::create_dir_all(out_dir);
    let is_ttpf = src_path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("ttpf")).unwrap_or(false);
    let (mut ok, mut bcpk, mut fail) = (0usize, 0usize, 0usize);
    if is_ttpf {
        let ix = match tex::ttpf_index(src_path) {
            Ok(i) => i,
            Err(e) => { eprintln!("TTPF index error: {e}"); return; }
        };
        let n = if limit == 0 { ix.descs.len() } else { limit.min(ix.descs.len()) };
        eprintln!("Dumping up to {n}/{} textures from {}...", ix.descs.len(), src_path.display());
        for d in ix.descs.iter().take(n) {
            match decode_ttpf_texture(src_path, d.idx) {
                Ok(tx) => {
                    if export_texture_png(&tx, &out_dir.join(format!("{}.png", safe_file_name(&tx.name)))).is_ok() {
                        ok += 1;
                    }
                }
                Err(e) if e.contains("BCPack") => bcpk += 1,
                Err(_) => fail += 1,
            }
            if (ok + bcpk + fail) % 250 == 0 {
                eprintln!("  progress: ok={ok} bcpack={bcpk} fail={fail}");
            }
        }
    } else {
        let entries = match scan_single_rpf(src_path, None) {
            Ok((_, e)) => e,
            Err(err) => { eprintln!("Scan failed: {err}"); return; }
        };
        let txds: Vec<&RpfEntry> = entries.iter().filter(|e| e.kind == "TXD").collect();
        eprintln!("Found {} TXDs in {} -> dumping PNGs...", txds.len(), src_path.display());
        let mut seen = std::collections::HashSet::new();
        'outer: for e in txds {
            match decode_txd_from_rpf(src_path, e.offset) {
                Ok(vec_t) => {
                    for tx in &vec_t {
                        let safe: String = tx.name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
                        let fname = if seen.insert(safe.clone()) { format!("{safe}.png") } else { format!("{safe}_{:x}.png", e.offset) };
                        if export_texture_png(tx, &out_dir.join(fname)).is_ok() {
                            ok += 1;
                            if limit > 0 && ok >= limit { break 'outer; }
                        }
                    }
                }
                Err(_) => fail += 1,
            }
        }
    }
    eprintln!("DONE: Exported {ok} PNGs (bcpack={bcpk}, fail={fail}) to {}", out_dir.display());
}

fn load_ps09_from_path(path: &Path) -> Result<Ps09Cache, String> {
    if path.is_dir() {
        let mut bins: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("bin")).unwrap_or(false))
            .collect();
        bins.sort();
        if bins.is_empty() {
            return Err("No .bin chunks found in directory".into());
        }
        let mut raw = Vec::with_capacity(bins.len() * 65536);
        for b in bins {
            raw.extend_from_slice(&fs::read(&b).map_err(|e| e.to_string())?);
        }
        Ps09Cache::parse(raw).ok_or_else(|| "Directory chunks did not start with PS\\x00\\x09".into())
    } else {
        let data = fs::read(path).map_err(|e| e.to_string())?;
        if data.starts_with(b"PS\x00\x09") {
            return Ps09Cache::parse(data).ok_or_else(|| "Invalid PS\\x00\\x09 stream".into());
        }
        // Decompress all 64KB zlib chunks starting from the first PS\x00\x09 stream inside shaders.rpf
        let mut raw = Vec::new();
        let mut pos = 0usize;
        let mut found = false;
        while pos + 2 < data.len() {
            if data[pos] == 0x78 && matches!(data[pos + 1], 0x9c | 0xda | 0x01) {
                let mut dec = Decompress::new(true);
                let mut out = vec![0u8; 65536];
                if let Ok(Status::StreamEnd | Status::Ok) = dec.decompress(&data[pos..], &mut out, FlushDecompress::Finish) {
                    let produced = dec.total_out() as usize;
                    if produced >= 64 {
                        if !found && out[..produced].starts_with(b"PS\x00\x09") {
                            found = true;
                        }
                        if found {
                            raw.extend_from_slice(&out[..produced]);
                            pos += (dec.total_in() as usize).max(2);
                            continue;
                        }
                    }
                }
            }
            pos += 1;
        }
        if !found {
            return Err("No PS\\x00\\x09 zlib stream found in file".into());
        }
        Ps09Cache::parse(raw).ok_or_else(|| "Decompressed PS\\x00\\x09 stream failed validation".into())
    }
}

fn disasm_shaders_cli(src: &Path, out_dir: &Path, limit: usize) {
    let cache = match load_ps09_from_path(src) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load PS09 from {}: {e}", src.display());
            return;
        }
    };
    let _ = fs::create_dir_all(out_dir);
    println!(
        "Loaded PS\\x00\\x09 ({}, drv {}): {} shaders, {} Gfx PSOs, {} Compute PSOs",
        cache.header.id,
        cache.header.driver,
        cache.shaders.len(),
        cache.header.graphics_pso_count,
        cache.header.compute_pso_count
    );
    let mut csv = String::from("index,stage,pso_hash,code_off,code_size,insns,bindings,has_rt,has_ngg,numthreads\n");
    let (mut cs, mut ps, mut vs, mut ngg, mut rt) = (0usize, 0, 0, 0, 0);
    for s in &cache.shaders {
        match s.stage {
            ShaderStage::Compute => cs += 1,
            ShaderStage::Pixel => ps += 1,
            ShaderStage::Vertex => vs += 1,
            ShaderStage::MeshNgg => ngg += 1,
            ShaderStage::RayTracing => rt += 1,
            _ => {}
        }
        let nt = s.num_threads.map(|[x, y, z]| format!("{x}x{y}x{z}")).unwrap_or_default();
        csv.push_str(&format!(
            "{},{},0x{:016x},0x{:08x},{},{},{},{},{},{}\n",
            s.index, s.stage.tag(), s.pso_hash, s.code_offset, s.code_size,
            s.insn_count, s.bindings.len(), s.has_rt as u8, s.has_ngg as u8, nt
        ));
    }
    let _ = fs::write(out_dir.join("shaders_index.csv"), csv);
    println!("Stage totals: PS={ps} VS={vs} CS={cs} NGG/Mesh={ngg} RT_BVH64={rt}");
    let dump_n = if limit == 0 { 25.min(cache.shaders.len()) } else { limit.min(cache.shaders.len()) };
    for s in cache.shaders.iter().take(dump_n) {
        let code = cache.shader_bytes(s);
        let stem = format!("sh_{:05}_{}", s.index, s.stage.tag());
        let _ = fs::write(out_dir.join(format!("{stem}.asm")), format_gfx103_disasm(s, code));
        let _ = fs::write(out_dir.join(format!("{stem}.hlsl")), lift_to_hlsl(s, code));
        let _ = fs::write(out_dir.join(format!("{stem}.spv")), lift_to_spirv(s, code));
    }
    println!("Wrote shaders_index.csv + {dump_n} shaders (.asm/.hlsl/.spv) to {}", out_dir.display());
}

fn extract_sgd5_cli(src: &Path, out_dir: &Path, limit: usize) {
    match sgd5::extract_sgd5(src, out_dir, limit) {
        Ok(extraction) => {
            let mut stage_counts = std::collections::BTreeMap::new();
            for file in &extraction.files {
                for shader in &file.shaders {
                    *stage_counts.entry(shader.stage).or_insert(0usize) += 1;
                }
            }
            let counts = stage_counts
                .iter()
                .map(|(stage, count)| format!("{stage}={count}"))
                .collect::<Vec<_>>()
                .join(" ");
            let shader_count = extraction
                .files
                .iter()
                .map(|file| file.shaders.len())
                .sum::<usize>();
            let effect_count = extraction
                .files
                .iter()
                .map(|file| file.effect_count as usize)
                .sum::<usize>();
            println!(
                "Loaded SGD5: {} groups, {shader_count} shaders ({counts}), {effect_count} effects",
                extraction.files.len()
            );
            println!(
                "Extracted {} DXBC shaders and {} manifests to {}",
                extraction.extracted_count,
                extraction.manifests.len(),
                out_dir.display()
            );
        }
        Err(error) => {
            eprintln!("Failed to extract SGD5 from {}: {error}", src.display());
        }
    }
}

fn decode_audio_cli(rpf_path: &Path, out_dir: &Path, limit: usize) {
    let _ = fs::create_dir_all(out_dir);
    let Ok(mut f) = File::open(rpf_path) else {
        eprintln!("Failed to open {}", rpf_path.display());
        return;
    };
    let sz = f.metadata().map(|m| m.len()).unwrap_or(0).min(2 * 1024 * 1024 * 1024);
    let mut off = 0u64;
    let mut buf = vec![0u8; 16 * 1024 * 1024];
    let mut banks = Vec::new();
    while off < sz {
        let Ok(n) = f.read(&mut buf) else { break };
        if n == 0 { break; }
        let mut pos = 0usize;
        while pos + 16 <= n {
            if &buf[pos..pos + 4] == b"ADAT" {
                let sc = u32::from_le_bytes([buf[pos + 8], buf[pos + 9], buf[pos + 10], buf[pos + 11]]);
                let hdr_sz = u32::from_le_bytes([buf[pos + 12], buf[pos + 13], buf[pos + 14], buf[pos + 15]]);
                if (1..=256).contains(&sc) && (16..=65536).contains(&hdr_sz) {
                    banks.push((off + pos as u64, sc));
                    if limit > 0 && banks.len() >= limit { break; }
                }
            }
            pos += 16;
        }
        if limit > 0 && banks.len() >= limit { break; }
        off += n.saturating_sub(64) as u64;
        let _ = f.seek(SeekFrom::Start(off));
    }
    let mut total_wavs = 0usize;
    let mut bank_data = vec![0u8; 256 * 1024];
    for (i, &(bank_off, sc)) in banks.iter().enumerate() {
        let _ = f.seek(SeekFrom::Start(bank_off));
        let n = f.read(&mut bank_data).unwrap_or(0);
        let stem = format!("bank_{i:03}_off_{bank_off:08x}_{sc}st");
        let res = awc::decode_awc_bytes_to_wavs_detailed(&bank_data[..n], out_dir, &stem);
        total_wavs += res.wavs.len();
    }
    println!("Exported {total_wavs} WAV tracks from {} banks -> {}", banks.len(), out_dir.display());
}

fn decode_navmesh_cli(rpf_path: &Path, out_dir: &Path, limit: usize) {
    let _ = fs::create_dir_all(out_dir);
    let Ok(mut f) = File::open(rpf_path) else {
        eprintln!("Failed to open {}", rpf_path.display());
        return;
    };
    let base_off = 786_010_112u64;
    let _ = f.seek(SeekFrom::Start(base_off));
    let mut buf = vec![0u8; 32 * 1024 * 1024];
    let n = f.read(&mut buf).unwrap_or(0);
    let mut dec = Decompress::new(true);
    let mut out = vec![0u8; 32768];
    let mut ok = 0usize;
    let mut pos = 0usize;
    while pos + 32 <= n {
        if buf[pos] == 0x78 && matches!(buf[pos + 1], 0x01 | 0x5e | 0x9c | 0xda) {
            dec.reset(true);
            let _ = dec.decompress(&buf[pos..(pos + 32768).min(n)], &mut out, FlushDecompress::None);
            if dec.total_out() >= 0xD0 && u32::from_le_bytes([out[0x14], out[0x15], out[0x16], out[0x17]]) == 0x0001_0015 {
                let midx = u32::from_le_bytes([out[0xC8], out[0xC9], out[0xCA], out[0xCB]]);
                let nm = if midx < 512 * 512 {
                    format!("nav_{:03}_{:03}", midx % 512, midx / 512)
                } else {
                    format!("nav_dyn_{midx:06x}")
                };
                if let Ok(mesh) = tex::decode_navmesh_from_rpf(rpf_path, base_off + pos as u64, &nm) {
                    if mesh.export_obj(&out_dir.join(format!("{nm}.obj"))).is_ok() {
                        ok += 1;
                        if limit > 0 && ok >= limit { break; }
                    }
                }
            }
        }
        pos += 16;
    }
    println!("Exported {ok} CNavMesh .obj tiles -> {}", out_dir.display());
}

fn disassemble_dxbc_cli(src: &Path, output: Option<&PathBuf>) {
    match fs::read(src) {
        Ok(data) => {
            let text = dxbc::disassemble(&data);
            match output {
                Some(path) => {
                    if let Err(error) = fs::write(path, &text) {
                        eprintln!("Failed to write {}: {error}", path.display());
                    }
                }
                None => println!("{text}"),
            }
        }
        Err(error) => eprintln!("Failed to read {}: {error}", src.display()),
    }
}

fn main() -> eframe::Result<()> {
    // remove leftover old exe from previous self update
    if let Ok(cur) = std::env::current_exe() {
        let _ = fs::remove_file(cur.with_extension("exe.old"));
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--gpu-world" {
        if args.get(2).is_some_and(|a| a == "--smoke-scene") {
            if let Err(error) = world_gpu::run_gpu_world_smoke(args.get(3).map(String::as_str).unwrap_or("terrain")) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            return Ok(());
        }
        let mut common_p: Option<PathBuf> = None;
        let mut levels_p: Option<PathBuf> = None;
        let mut idx = 2usize;
        while idx + 1 < args.len() {
            if args[idx] == "--common" {
                common_p = Some(PathBuf::from(&args[idx + 1]));
                idx += 2;
            } else if args[idx] == "--levels" {
                levels_p = Some(PathBuf::from(&args[idx + 1]));
                idx += 2;
            } else {
                idx += 1;
            }
        }
        if let Err(error) = world_gpu::run_gpu_world_cli(common_p.as_deref(), levels_p.as_deref()) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if args.len() >= 3 && args[1] == "--bench" {
        bench::run(Path::new(&args[2]), args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3), args.get(4).map_or("", String::as_str));
        return Ok(());
    }
    if args.len() >= 3 && args[1] == "--probe-maps" {
        mapdata::probe_maps_cli(&PathBuf::from(&args[2]));
        return Ok(());
    }
    if args.len() >= 3 && args[1] == "--export-psin" {
        let tsv = args.iter().any(|a| a == "--tsv");
        match fs::read(&args[2]) {
            Ok(data) => match psin::parse_psin_doc(&data) {
                Ok(doc) => print!("{}", doc.export_placements_csv(None, tsv)),
                Err(e) => eprintln!("PSIN error: {e}"),
            },
            Err(e) => eprintln!("Read error: {e}"),
        }
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--decode-bk2" {
        let bk2 = PathBuf::from(&args[2]);
        let out_dir = PathBuf::from(&args[3]);
        let max_f = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);
        // never from the folder of the input clip or by bare name both let a planted dll run
        let dll = std::env::var_os("BINK2_DLL")
            .map(PathBuf::from)
            .or_else(|| std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("bink2w64.dll"))))
            .filter(|p| p.is_absolute() && p.is_file());
        let Some(dll) = dll else {
            eprintln!("BK2 decode error: bink2w64.dll not found next to PercyStar (or set BINK2_DLL to its full path)");
            return Ok(());
        };
        match bink2::decode_bk2_to_pngs(&dll, &bk2, &out_dir, max_f) {
            Ok((info, n)) => println!("Decoded {n} frames ({}x{}) to {}", info.width, info.height, out_dir.display()),
            Err(e) => eprintln!("BK2 decode error: {e}"),
        }
        return Ok(());
    }
    if args.len() >= 3 && args[1] == "--inspect-exe" {
        let exe_path = PathBuf::from(&args[2]);
        match fs::read(&exe_path) {
            Ok(data) => {
                println!("{}", boot::inspect_pe64_for_bringup(&data));
                if let Some(dir) = exe_path.parent() {
                    if let Ok(cfg) = fs::read_to_string(dir.join("MicrosoftGame.Config")) {
                        let lay = fs::read_dir(dir).ok().and_then(|rd| {
                            rd.filter_map(|e| e.ok()).find_map(|e| {
                                let n = e.file_name().to_string_lossy().into_owned();
                                if n.starts_with("layout_") && n.ends_with(".xml") {
                                    fs::read_to_string(e.path()).ok()
                                } else {
                                    None
                                }
                            })
                        });
                        println!("{}", boot::summarize_gdk_manifest(&cfg, lay.as_deref()));
                    }
                }
            }
            Err(e) => eprintln!("Failed to read {}: {e}", exe_path.display()),
        }
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--disasm-shaders" {
        let lim = args.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(25);
        disasm_shaders_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), lim);
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--extract-sgd5" {
        let lim = args
            .get(4)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        extract_sgd5_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), lim);
        return Ok(());
    }
    if args.len() >= 3 && args[1] == "--disassemble-dxbc" {
        disassemble_dxbc_cli(
            &PathBuf::from(&args[2]),
            args.get(3).map(PathBuf::from).as_ref(),
        );
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--dump-textures" {
        let lim = args.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        dump_textures_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), lim);
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--extract-vehicles" {
        export_models_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), "", true);
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--extract-models" {
        let filt = args.get(4).map(|s| s.as_str()).unwrap_or("");
        export_models_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), filt, false);
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--decode-audio" {
        let lim = args.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        decode_audio_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), lim);
        return Ok(());
    }
    if args.len() >= 4 && args[1] == "--decode-navmesh" {
        let lim = args.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        decode_navmesh_cli(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), lim);
        return Ok(());
    }

    // decod embedded logo once and share bytes across window icon and topbar
    let logo_rgba = image::load_from_memory(include_bytes!("../assets/logo.png"))
        .ok()
        .map(|img| {
            let rgba = img.to_rgba8();
            (rgba.as_raw().to_vec(), rgba.width(), rgba.height())
        });
    let mut vb = egui::ViewportBuilder::default().with_inner_size([1320.0, 800.0]);
    if let Some((ref raw, w, h)) = logo_rgba {
        vb = vb.with_icon(egui::IconData { rgba: raw.clone(), width: w, height: h });
    }
    let opts = eframe::NativeOptions { viewport: vb, ..Default::default() };
    let title = format!("PercyStar v{APP_VERSION}");
    // percystar <path> opens a folder archive model or shader cache directly
    let startup = args.get(1).filter(|a| !a.starts_with("--")).map(PathBuf::from).filter(|p| p.exists());
    eframe::run_native(&title, opts, Box::new(move |cc| Ok(Box::new(Rpf9App::new(cc, logo_rgba, startup)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    // name recovery over the small archives of a real dump, prints what it resolves
    #[test]
    fn name_recovery_on_real_archives() {
        let Ok(levels) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        let Some(root) = Path::new(&levels).parent() else { return };
        let mut entries = Vec::new();
        for rpf in ["common.rpf", "script.rpf", "data.rpf"] {
            let path = root.join(rpf);
            if !path.exists() {
                return;
            }
            let (_, found) = scan_single_rpf(&path, None).expect("scan");
            entries.extend(found.into_iter().filter(|e| matches!(e.kind, "PSIN" | "XML" | "RBF0" | "DAT" | "GFX" | "ZLIB")));
        }
        psin::seed_names();
        // extra word lists make the run match what the app does with a filled names folder
        if let Ok(dir) = std::env::var("PERCYSTAR_NAMES_DIR") {
            names::load_dir(Path::new(&dir));
        }
        let (targets, tokens) = gather_names(&entries, |_| {});
        let started = Instant::now();
        let found = names::recover(&tokens, &targets);
        let all = targets.all();
        let got: HashSet<u32> = found.exact.iter().chain(&found.built).map(|f| f.0).collect();
        eprintln!(
            "recover took {:.1}s: structs {}/{}, fields {}/{}, enum values {}/{}",
            started.elapsed().as_secs_f32(),
            targets.structs.iter().filter(|h| got.contains(h)).count(),
            targets.structs.len(),
            targets.fields.keys().filter(|h| got.contains(h)).count(),
            targets.fields.len(),
            targets.enums.values().flatten().filter(|h| got.contains(h)).count(),
            targets.enums.values().map(|v| v.len()).sum::<usize>()
        );
        if let Ok(out) = std::env::var("PERCYSTAR_NAMES_OUT") {
            let _ = names::save_list(Path::new(&out), names::RECOVERED_FILE, &found.exact);
            let _ = names::save_list(Path::new(&out), names::BUILT_FILE, &found.built);
        }
        eprintln!(
            "{} entries, {} words, {} hashed names: {} exact, {} built (noise about {:.0}); structs {} fields {} enum values {}",
            entries.len(),
            tokens.len(),
            all.len(),
            found.exact.len(),
            found.built.len(),
            found.built_noise,
            targets.structs.len(),
            targets.fields.len(),
            targets.enums.values().map(|v| v.len()).sum::<usize>()
        );
        let has = |n: &str| found.exact.iter().any(|f| f.1 == n);
        assert!(has("fMass") && has("fBrakeForce"), "handling names not found");
        assert!(found.exact.len() > 500 && found.built.len() > 300);
        assert!(found.built_noise < found.built.len() as f64 * 0.1);
    }

    fn wait_idle(media: &Arc<Mutex<MediaState>>) {
        for _ in 0..400 {
            if !media.lock().unwrap_or_else(|p| p.into_inner()).loading {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("media job never finished");
    }

    #[test]
    fn style_stays_dark_when_the_system_theme_is_light() {
        let ctx = egui::Context::default();
        setup_style(&ctx);
        let input = egui::RawInput { system_theme: Some(egui::Theme::Light), ..Default::default() };
        let _ = ctx.run(input, |ctx| {
            let visuals = ctx.style().visuals.clone();
            assert!(visuals.dark_mode);
            assert_eq!(visuals.panel_fill, Color32::from_rgb(18, 12, 28));
        });
    }

    #[test]
    fn media_jobs_drop_stale_results_and_survive_panics() {
        let media = Arc::new(Mutex::new(MediaState::default()));
        let (release, held) = std::sync::mpsc::channel::<()>();
        spawn_media_job(&media, "slow".into(), true, move |_| { let _ = held.recv(); Ok(1u32) }, |m, _| m.status = "old".into());
        spawn_media_job(&media, "fast".into(), true, |_| Ok(2u32), |m, _| m.status = "new".into());
        wait_idle(&media);
        release.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(media.lock().unwrap().status, "new");

        spawn_media_job(&media, "boom".into(), true, |_| -> Result<u32, String> { panic!("decoder bug") }, |m, res| m.status = res.unwrap_err());
        wait_idle(&media);
        assert_eq!(media.lock().unwrap().status, "decoder panicked");
    }

    #[test]
    fn model_sidebar_width_stays_stable_across_frames() {
        let ctx = egui::Context::default();
        let mut filter = String::new();
        let mut widths = Vec::new();
        for frame in 0..64 {
            if frame == 16 { filter = "long_model_search_".repeat(100); }
            let _ = ctx.run(egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1320.0, 800.0))),
                ..Default::default()
            }, |ctx| {
                let panel = model_viewer_sidebar(ctx).show(ctx, |ui| {
                    model_search_row(ui, &mut filter);
                });
                widths.push(panel.response.rect.width());
                egui::CentralPanel::default().show(ctx, |_| {});
            });
        }
        assert!(widths.iter().all(|w| (w - widths[0]).abs() <= 1.0),
            "Sidebar expands without input: first={} last={}", widths[0], widths.last().unwrap());
    }

    #[test]
    fn model_sidebar_recovers_from_an_oversized_saved_width() {
        let ctx = egui::Context::default();
        ctx.data_mut(|data| data.insert_persisted(
            egui::Id::new("mv_sidebar"),
            egui::containers::panel::PanelState {
                rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(2000.0, 800.0)),
            },
        ));
        let mut filter = "long_search_".repeat(100);
        let mut hide_lods = true;
        let mut sort = SortMode::Relevance;
        for width in [1320.0_f32, 640.0, 960.0] {
            for _ in 0..8 {
                let _ = ctx.run(egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 800.0))),
                    ..Default::default()
                }, |ctx| {
                    let panel = model_viewer_sidebar(ctx).show(ctx, |ui| {
                        model_search_row(ui, &mut filter);
                        let _ = model_browser_options(ui, &mut hide_lods, &mut sort, true);
                    });
                    assert!(panel.response.rect.width() <= (width * 0.45).min(480.0) + 1.0,
                        "Sidebar {} exceeds limit for window {width}", panel.response.rect.width());
                    egui::CentralPanel::default().show(ctx, |ui| {
                        assert!(ui.available_width() > width * 0.5);
                    });
                });
            }
        }
    }

    #[test]
    fn lod_sibling_names_cover_hi_mid_and_base() {
        let want = ["cheetah2_hi.#ft".to_string(), "cheetah2_mid.#ft".to_string(), "cheetah2.#ft".to_string()];
        for n in ["cheetah2.#ft", "cheetah2_hi.#ft", "cheetah2_MID.#ft"] {
            assert_eq!(lod_sibling_names(n).unwrap(), want, "{n}");
        }
        assert_eq!(lod_sibling_names("a_b_bonnet_01_mid.#ft").unwrap()[2], "a_b_bonnet_01.#ft");
        assert!(lod_sibling_names("noext").is_none());
        assert!(lod_sibling_names("_hi.#ft").is_none());
    }

    #[test]
    fn model_viewer_categories_classify_correctly() {
        assert_eq!(classify_model_cat("VEH", "banshee_hi.#ft"), ModelCat::Vehicles);
        assert_eq!(classify_model_cat("MODEL", "sentinelclassic2_skirt_split_l_histr_01.#ft"), ModelCat::Vehicles);
        assert_eq!(classify_model_cat("MODEL", "cmn_spoiler_01.#ft"), ModelCat::Vehicles);
        assert_eq!(classify_model_cat("MODEL", "toros_bumper_f_01.#ft"), ModelCat::Vehicles);
        assert_eq!(classify_model_cat("MODEL", "a_td_plane.#dr"), ModelCat::AircraftBoats);
        assert_eq!(classify_model_cat("MODEL", "airgator_engine_01.#ft"), ModelCat::AircraftBoats);
        assert_eq!(classify_model_cat("MODEL", "cmn_boat_propeller_01.#ft"), ModelCat::AircraftBoats);
        assert_eq!(classify_model_cat("MODEL", "polmav_rotor_01.#ft"), ModelCat::AircraftBoats);
        assert_eq!(classify_model_cat("MODEL", "w_ar_carbinerifle.#dr"), ModelCat::Weapons);
        assert_eq!(classify_model_cat("MODEL", "w_preset_rifle_assault_01.#dr"), ModelCat::Weapons);
        assert_eq!(classify_model_cat("MODEL", "crr_alligator_american_m_00.#dr"), ModelCat::Animals);
        assert_eq!(classify_model_cat("MODEL", "crm_dog_malinoisbelgian_m_00_body_10_coa_helis.#dr"), ModelCat::Animals);
        assert_eq!(classify_model_cat("MODEL", "cs_donnie_head_mr1_cau_001_m_full.#dr"), ModelCat::Characters);
        assert_eq!(classify_model_cat("MODEL", "beardbandholz_mr1_000_straight_l05_n_full"), ModelCat::Characters);
        assert_eq!(classify_model_cat("MODEL", "hc_kingfishers_head_mr1_000.#dr"), ModelCat::Characters);
        assert_eq!(classify_model_cat("MODEL", "bodicecorset_fb1_002.#dr"), ModelCat::Clothing);
        assert_eq!(classify_model_cat("MODEL", "skirtpencil_fb1_019_slim_hw_high_full.#dr"), ModelCat::Clothing);
        assert_eq!(classify_model_cat("MODEL", "gmt_skirt_mini_f_hang_01a.#dr"), ModelCat::Clothing);
        assert_eq!(classify_model_cat("MODEL", "shoesboat_fr1_000_low_fh_full.#dr"), ModelCat::Clothing);
        assert_eq!(classify_model_cat("MODEL", "prop_am_table_dining_13b.#dr"), ModelCat::Props);
        assert_eq!(classify_model_cat("MODEL", "prop_am_boatlift_02.#dr"), ModelCat::Props);
        assert_eq!(classify_model_cat("MODEL", "a_bfyb01_sea_tr_02_bridge_01.#dr"), ModelCat::World);
        assert_eq!(classify_model_cat("MODEL", "int_grg_14_painted_wall_skirting.#dr"), ModelCat::World);
        assert_eq!(classify_model_cat("NAVM", "nav_160_179.ynv"), ModelCat::NavMesh);
    }

    #[test]
    fn semver_comparison_works() {
        assert_eq!(parse_semver("1.0.3"), Some((1, 0, 3)));
        assert_eq!(parse_semver("v1.0.4"), Some((1, 0, 4)));
        assert!(parse_semver("v1.0.4") > parse_semver("1.0.3"));
        assert!(parse_semver("1.1.0") > parse_semver("1.0.99"));
        assert!(parse_semver("v1.0.3") == parse_semver("1.0.3"));
        assert!(parse_semver("bad").is_none());
        // tag suffixes never reach a url the release page is rebuilt from the numbers
        let (a, b, c) = parse_semver("v9.9.9-x&calc.exe").unwrap();
        assert_eq!(release_page_url(&format!("{a}.{b}.{c}")), format!("{REPO_RELEASES_URL}/tag/v9.9.9"));
    }

    #[test]
    fn exported_names_cannot_leave_the_output_folder() {
        let out = Path::new("/out");
        for bad in ["..\\..\\evil", "../../evil", "/etc/passwd", "C:\\Windows\\x", "..", ".", "", "a\u{0}b", "con?.obj", "trail. "] {
            let name = safe_file_name(bad);
            let joined = out.join(&name);
            assert_eq!(joined.parent(), Some(out), "{bad:?} -> {name:?}");
            assert!(!name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':']), "{bad:?} -> {name:?}");
        }
        // ordinary archive names are untouched
        for ok in ["banshee_hi.ft", "nav_256_256_(0,0).ynv", "cheetah2.#ft", "a b-c_d.png"] {
            assert_eq!(safe_file_name(ok), ok);
        }
    }

    #[test]
    fn non_ascii_model_names_classify_without_panicking() {
        let _ = classify_model_cat("MODEL", "\u{e9}t\u{e9}_car.#dr");
        assert!(!match_known_stem("\u{e9}abcd", &["abcd"]));
        assert!(match_known_stem("xabcd", &["abcd"]));
    }

    /// PERCYSTAR_PACKS_RPF=<packs.rpf> cargo test packs_vehicle_table_is_schema_not_animation -- --nocapture
    #[test]
    fn packs_vehicle_table_is_schema_not_animation() {
        let Ok(path) = std::env::var("PERCYSTAR_PACKS_RPF") else { return };
        let path = PathBuf::from(path);
        if !path.exists() {
            return;
        }
        let mut file = File::open(path).expect("open packs.rpf");
        let (bytes, _, _) = read_virtual_from_file(&mut file, 0x0bae_c010).expect("read vehicle table");
        assert!(meta7a::has_schema_enums(&bytes), "vehicle tables must not be classified as clip dictionaries");
        let configurations = vehicle_variations::parse_configurations(&bytes);
        let stanier = configurations.get(&psin::joaat("stanierg2")).expect("stanierg2 configurations");
        assert!(stanier.len() >= 20, "expected the authored stanierg2 rows, got {}", stanier.len());
    }

    /// PERCYSTAR_PACKS_RPF=<packs.rpf> cargo test authored_stanierg2_configurations -- --ignored --nocapture
    #[test]
    #[ignore]
    fn authored_stanierg2_configurations() {
        let Ok(path) = std::env::var("PERCYSTAR_PACKS_RPF") else { return };
        let path = PathBuf::from(path);
        let (_, entries) = scan_single_rpf(&path, None).expect("scan packs.rpf");
        let mut metadata: Vec<_> = entries.iter().filter(|entry| entry.kind == "META7A").collect();
        // Known table resource in this supported build; also makes the fixture
        // useful when an imported/partial catalog did not index metadata rows.
        let fallback;
        if metadata.is_empty() {
            // Known packs.rpf vehicle-config resource for this supported build.
            fallback = RpfEntry {
                source_rpf: path.clone(),
                rpf_name: "packs.rpf".into(),
                name: "vehicle_tables.#7a".into(),
                kind: "META7A",
                offset: 0x0bae_c010,
                comp_size: 0,
                raw_size: 0,
                tracks: 0,
                chunks: Vec::new(),
            };
            metadata.push(&fallback);
        }
        assert!(!metadata.is_empty(), "no 0x7a metadata resources indexed");
        let mut database = vehicle_variations::VehicleConfigurationDb::new();
        let mut failures = 0usize;
        for entry in &metadata {
            let mut file = File::open(&entry.source_rpf).expect("open packs.rpf");
            match read_virtual_from_file(&mut file, entry.offset) {
                Ok((bytes, _, _)) => vehicle_variations::merge_configurations(
                    &mut database,
                    vehicle_variations::parse_configurations(&bytes),
                ),
                Err(_) => failures += 1,
            }
        }
        assert!(failures < metadata.len(), "failed to decode every packs.rpf 0x7a resource");
        let configurations = database
            .get(&psin::joaat("stanierg2"))
            .unwrap_or_else(|| panic!("stanierg2 configurations missing after {} meta resources", metadata.len()));
        assert!(
            configurations.len() >= 20,
            "expected ~23 stanierg2 configurations, got {}",
            configurations.len()
        );
        assert!(
            configurations.iter().any(|configuration| {
                !configuration.parts.is_empty()
                    && configuration.parts.contains(&psin::joaat("cmn_siren_roofbar1_02"))
            }),
            "stanierg2 police configuration should reference the shared roofbar with a resolved part list"
        );
    }

    #[test]
    fn sgd5_shader_lifts_through_gfx103_paths() {
        let shader = sgd5::Sgd5Shader {
            stage: "VS",
            index: 7,
            name: "synthetic_vertex".into(),
            wave_size: 32,
            compressed_size: 8,
            uncompressed_size: 8,
            sga_hash: 0x0123_4567_89ab_cdef,
            register_count: 8,
            cbuffer_count: 1,
            texture_count: 0,
            binary_offset: 0,
            reflection_records: Vec::new(),
        };
        let mut data = vec![0u8; 0x20];
        data.extend_from_slice(&0xBFA0_0003u32.to_le_bytes());
        data.extend_from_slice(&0xBF81_0000u32.to_le_bytes());
        let fssm = dxbc::FssmShader {
            code_offset: 0x20,
            code_size: 8,
            data,
        };
        let entry = sgd5_ps09_entry(
            &sgd5::Sgd5File {
                data: Vec::new(),
                group_hash: 0,
                descriptors: Vec::new(),
                shared_reflection: sgd5::Sgd5SharedReflection {
                    size: 0,
                    referenced_count: 0,
                    extra_count: 0,
                    reserved: 0,
                    records: Vec::new(),
                },
                shaders: vec![shader.clone()],
                declared_effect_count: 0,
                effect_count: 0,
                trailer_flags: 0,
                trailer_reserved: [0; 6],
                effects: Vec::new(),
                hash_names: Default::default(),
                effect_shader_map: Default::default(),
            },
            &shader,
            &fssm,
        );

        assert_eq!(entry.stage, ShaderStage::Vertex);
        assert_eq!(entry.sga_hash, Some(shader.sga_hash));
        assert_eq!(entry.insn_count, 2);
        assert!(lift_to_hlsl(&entry, fssm.code()).contains("VSMain"));
        assert_eq!(
            &lift_to_spirv(&entry, fssm.code())[..4],
            &0x0723_0203u32.to_le_bytes()
        );
    }

    #[test]
    fn deswizzle_and_ui_decoders_work() {
        // 4KB SW_4KB_D bijective permutation test on 64x64 BC7 (16x16 blocks, 1 tile)
        let mut raw = vec![0u8; 4096];
        for blk in 0..256 {
            raw[blk * 16] = blk as u8;
        }
        let lin = tex::deswizzle_4kb_d(&raw, 64, 64, true, 16);
        assert_eq!(lin.len(), 4096);
        let mut seen = [false; 256];
        for blk in 0..256 {
            seen[lin[blk * 16] as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "SW_4KB_D 128bpp must be a full bijection");

        // 64KB TTPF bijective permutation test on 256x256 BC7 (64x64 blocks, 1 tile)
        let mut raw64 = vec![0u8; 65536];
        for blk in 0..4096 {
            raw64[blk * 16] = (blk & 0xFF) as u8;
            raw64[blk * 16 + 1] = ((blk >> 8) & 0xFF) as u8;
        }
        let lin64 = tex::deswizzle_64kb_ttpf(&raw64, 256, 256, true, 16);
        assert_eq!(lin64.len(), 65536);
        let mut seen64 = vec![false; 4096];
        for blk in 0..4096 {
            let v = (lin64[blk * 16] as usize) | ((lin64[blk * 16 + 1] as usize) << 8);
            seen64[v] = true;
        }
        assert!(seen64.iter().all(|&b| b), "TTPF 64KB 128bpp must be a full bijection");

        // RBF0 GTA6 header (0xFB 0xFF 0x01 0x00) decode test
        let mut rbf = Vec::new();
        rbf.extend_from_slice(b"RBF0");
        rbf.extend_from_slice(&[0xFB, 0xFF, 0x01, 0x00]); // GTA6 header
        rbf.extend_from_slice(&[0x00, 0x00, 0x04, 0x00]); // idx=0, struct, name len=4
        rbf.extend_from_slice(b"Root");
        rbf.extend_from_slice(&[0u8; 6]); // struct header
        rbf.extend_from_slice(&[0x01, 0x10, 0x02, 0x00]); // idx=1, u32 attr, len=2
        rbf.extend_from_slice(b"id");
        rbf.extend_from_slice(&42u32.to_le_bytes());
        rbf.extend_from_slice(&[0xFF, 0xFF]); // close Root
        let xml = decode_rbf0(&rbf).expect("RBF0 should decode");
        assert!(xml.contains("<Root id=\"42\" />"), "got: {xml}");

        // scaleform gfx inspector test
        let mut gfx = Vec::new();
        gfx.extend_from_slice(b"GFX\x2a");
        gfx.extend_from_slice(&128u32.to_le_bytes());
        gfx.extend_from_slice(b"\0com.rockstargames.gtav.HudMap\0");
        let dump = inspect_gfx(&gfx);
        assert!(dump.contains("Version: 42"));
        assert!(dump.contains("com.rockstargames.gtav.HudMap"));

        // unicode 13 language gxt text table probe and decode test
        let mut gxt = vec![0u8; 0x60];
        gxt[0x10..0x15].copy_from_slice(b"0DRPz");
        gxt.extend_from_slice("彩色字幕\0Привет мир\0한국어 자막\0日本語テキスト\0".as_bytes());
        assert!(tex::probe_gxt_snippet(&gxt).is_some());
        let txt = decode_0drpz_text(&gxt).expect("unicode gxt should decode");
        assert!(txt.contains("彩色字幕") && txt.contains("Привет мир") && txt.contains("한국어 자막") && txt.contains("日本語テキスト"));

        // small table: two strings in a pool past the bitmap end, reached only through a {u32, len, ptr} record
        let mut small = vec![0u8; 0xC0];
        small[0x08..0x10].copy_from_slice(&0x5000_0080u64.to_le_bytes());
        small[0x10..0x15].copy_from_slice(b"0DRPz");
        small[0x60..0x64].copy_from_slice(b"O\x80\x80\x80");
        let pool = "Open the door\0テスト文字列\0".as_bytes();
        small[0xA0..0xA4].copy_from_slice(&0x10u32.to_le_bytes());
        small[0xA4..0xA8].copy_from_slice(&(pool.len() as u32).to_le_bytes());
        small[0xA8..0xB0].copy_from_slice(&0x5000_00C0u64.to_le_bytes());
        small.extend_from_slice(pool);
        let txt = decode_0drpz_text(&small).expect("small pooled gxt should decode");
        assert!(txt.contains("(2 strings)") && txt.contains("Open the door") && txt.contains("テスト文字列"));
    }

    /// Writes sample PNGs from a big .ttpf so decode quality can be eyeballed:
    /// PERCY_TTPF=<file> PERCY_OUT=<dir> cargo test --release dump_ttpf_samples -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_ttpf_samples() {
        let (Ok(src), Ok(out)) = (std::env::var("PERCY_TTPF"), std::env::var("PERCY_OUT")) else { return };
        let src = PathBuf::from(src);
        let t = Instant::now();
        let ix = tex::ttpf_index(&src).expect("index");
        eprintln!("index: {} textures in {:.1}s", ix.descs.len(), t.elapsed().as_secs_f32());
        let _ = fs::create_dir_all(&out);
        let (mut ok, mut bcpack, mut fail) = (0, 0, 0);
        let step = (ix.descs.len() / 60).max(1);
        for d in ix.descs.iter().step_by(step) {
            let t1 = Instant::now();
            match decode_ttpf_texture(&src, d.idx) {
                Ok(tx) => {
                    ok += 1;
                    let _ = export_texture_png(&tx, &Path::new(&out).join(format!("{}.png", tx.name)));
                    if ok <= 5 { eprintln!("  {} in {} ms", tx.name, t1.elapsed().as_millis()); }
                }
                Err(e) if e.contains("BCPack") => bcpack += 1,
                Err(e) => { fail += 1; eprintln!("  #{} fail: {e}", d.idx); }
            }
        }
        eprintln!("ok={ok} bcpack={bcpack} fail={fail}");
        // raw virtual dumps (whole allocation) for offline layout analysis
        if let Ok(ids) = std::env::var("PERCY_RAW") {
            for id in ids.split(',').filter_map(|x| x.parse::<usize>().ok()) {
                let d = &ix.descs[id];
                if let Ok(raw) = tex::ttpf_read_virtual(&src, &ix, d.byte_off, d.alloc_sz as usize) {
                    let nm = format!("raw_{}_{}x{}_f{:02x}_m{}_a{}.bin", id, d.w, d.h, d.fmt_byte, d.has_mips as u8, d.alloc_sz);
                    let _ = fs::write(Path::new(&out).join(nm), raw);
                }
            }
        }
    }

    /// PERCY_RPF=<textures.rpf> PERCY_OUT=<dir>: dump every TXD texture as PNG
    #[test]
    #[ignore]
    fn dump_txd_samples() {
        let (Ok(src), Ok(out)) = (std::env::var("PERCY_RPF"), std::env::var("PERCY_OUT")) else { return };
        let src = PathBuf::from(src);
        let _ = fs::create_dir_all(&out);
        let t = Instant::now();
        let (_, entries) = scan_single_rpf(&src, None).unwrap();
        let txds: Vec<_> = entries.iter().filter(|e| e.kind == "TXD").collect();
        eprintln!("{} TXD in {:.1}s", txds.len(), t.elapsed().as_secs_f32());
        let (mut ok, mut fail) = (0, 0);
        for e in txds.iter().take(40) {
            let t1 = Instant::now();
            match decode_txd_from_rpf(&src, e.offset) {
                Ok(v) => for (k, tx) in v.iter().enumerate().take(3) {
                    ok += 1;
                    let _ = export_texture_png(tx, &Path::new(&out).join(format!("{}_{k}_{}.png", e.offset, tx.name.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_"))));
                    if ok < 4 { eprintln!("  {} {}x{} {} in {} ms", tx.name, tx.w, tx.h, tx.fmt_name, t1.elapsed().as_millis()); }
                },
                Err(er) => { fail += 1; eprintln!("  fail {}: {er}", e.name); }
            }
        }
        eprintln!("ok={ok} fail={fail}");
    }

    /// Brute-force layout finder: PERCY_TTPF, PERCY_OUT, PERCY_IDS=1,2,3
    #[test]
    #[ignore]
    fn ttpf_layout_search() {
        let (Ok(src), Ok(out), Ok(ids)) = (std::env::var("PERCY_TTPF"), std::env::var("PERCY_OUT"), std::env::var("PERCY_IDS")) else { return };
        let src = PathBuf::from(src);
        let ix = tex::ttpf_index(&src).unwrap();
        let _ = fs::create_dir_all(&out);
        for id in ids.split(',').filter_map(|x| x.parse::<usize>().ok()) {
            let d = &ix.descs[id];
            let Ok(raw) = tex::ttpf_read_virtual(&src, &ix, d.byte_off, d.alloc_sz as usize) else { eprintln!("{id}: bcpack"); continue };
            let mut res = Vec::new();
            for (dxgi, bpe, is_bc) in [(98u8, 16usize, true), (71, 8, true), (77, 16, true), (80, 8, true), (83, 16, true), (95, 16, true), (28, 4, false), (61, 1, false), (49, 2, false)] {
                let (bw, bh) = if is_bc { (d.w.div_ceil(4), d.h.div_ceil(4)) } else { (d.w, d.h) };
                for t64 in [false, true] {
                    if t64 && !is_bc { continue; }
                    let pad = if t64 {
                        let (tw, th) = if bpe == 16 { (64, 64) } else { (128, 64) };
                        bw.div_ceil(tw) * tw * bh.div_ceil(th) * th * bpe
                    } else { tex::padded_4kb_size(d.w, d.h, is_bc, bpe) };
                    let mut offs = vec![0usize];
                    if raw.len() > pad { offs.push(raw.len() - pad); }
                    let mut o = 0x1000; while o + pad <= raw.len() && offs.len() < 40 { offs.push(o); o += 0x1000; }
                    offs.sort(); offs.dedup();
                    for off in offs {
                        if off + pad > raw.len() && off != 0 { continue; }
                        let sl = &raw[off..raw.len().min(off + pad)];
                        let lin = if t64 { tex::deswizzle_64kb_ttpf(sl, d.w, d.h, is_bc, bpe) } else { tex::deswizzle_4kb_d(sl, d.w, d.h, is_bc, bpe) };
                        let Some(px) = tex::decode_linear_to_rgba(&lin, d.w, d.h, dxgi) else { continue };
                        let mut diff = 0u64;
                        for y in 0..d.h { for x in 1..d.w {
                            let (a, b) = (px[y * d.w + x - 1], px[y * d.w + x]);
                            diff += (a.r() as i32 - b.r() as i32).unsigned_abs() as u64 + (a.g() as i32 - b.g() as i32).unsigned_abs() as u64;
                        }}
                        for y in 1..d.h { for x in 0..d.w {
                            let (a, b) = (px[(y - 1) * d.w + x], px[y * d.w + x]);
                            diff += (a.r() as i32 - b.r() as i32).unsigned_abs() as u64 + (a.g() as i32 - b.g() as i32).unsigned_abs() as u64;
                        }}
                        res.push((diff as f64 / (d.w * d.h) as f64, dxgi, t64, off, px));
                    }
                }
            }
            res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            eprintln!("tex {id} {}x{} f{:02x} mips={} alloc={:#x}", d.w, d.h, d.fmt_byte, d.has_mips, d.alloc_sz);
            for (k, (sc, dx, t64, off, px)) in res.iter().take(4).enumerate() {
                eprintln!("   score {sc:.1} dxgi {dx} t64 {t64} off {off:#x}");
                let t = tex::DecodedTexture { name: String::new(), w: d.w, h: d.h, mips: 1, dxgi: *dx, fmt_name: "", pixels: px.clone(), raw_dds_payload: vec![] };
                let _ = export_texture_png(&t, &Path::new(&out).join(format!("{id}_{k}.png")));
            }
        }
    }

    #[test]
    fn real_gta6_archives_smoke_test() {
        let base = Path::new(r"D:\Grand Theft Auto VI Build");
        if !base.exists() { return; }

        // 1. TTPF native scan + decode
        let ttpf = base.join("data").join("ttdb").join("ui").join("platform.ttpf");
        if ttpf.exists() {
            let (info, descs) = scan_ttpf_descriptors(&ttpf).expect("scan platform.ttpf");
            assert!(!descs.is_empty(), "platform.ttpf descriptors empty: {info}");
            let mut decoded_ok = 0usize;
            for d in descs.iter().take(25) {
                if let Ok(t) = decode_ttpf_texture(&ttpf, d.idx) {
                    assert_eq!(t.pixels.len(), t.w * t.h);
                    decoded_ok += 1;
                    if decoded_ok >= 3 { break; }
                }
            }
            assert!(decoded_ok >= 1, "expected at least 1 decoded texture from platform.ttpf");
        }

        // 1b. Previously unsupported TTPF formats: DXGI 28 (RGBA8), 54 (R16F), 61 (R8), 65 (A8), 87 (BGRA8), 96 (BC6H_SF16)
        for (rel, idx, exp_dxgi) in [
            ("data\\ttdb\\graphics\\ttdb_graphics.ttpf", 1usize, 28u8),
            ("data\\ttdb\\graphics\\ttdb_graphics.ttpf", 563usize, 87u8),
            ("data\\ttdb\\maps\\ttdb_maps_0_7.ttpf", 605usize, 54u8),
            ("data\\ttdb\\interiors\\ttdb_interiors_0_3.ttpf", 1984usize, 65u8),
            ("data\\ttdb\\vfx\\ttdb_vfx.ttpf", 608usize, 61u8),
            ("data\\ttdb\\lighting\\lighting.ttpf", 334usize, 96u8),
        ] {
            let p = base.join(rel);
            if p.exists() {
                let t = decode_ttpf_texture(&p, idx).unwrap_or_else(|e| panic!("{rel} #{idx}: {e}"));
                assert_eq!(t.dxgi, exp_dxgi);
                assert_eq!(t.pixels.len(), t.w * t.h);
            }
        }

        // 2. TXD native scan + decode from textures.rpf
        let tex_rpf = base.join("textures.rpf");
        if tex_rpf.exists() {
            let (_, entries) = scan_single_rpf(&tex_rpf, None).expect("scan textures.rpf");
            let txds: Vec<&RpfEntry> = entries.iter().filter(|e| e.kind == "TXD").collect();
            assert!(!txds.is_empty(), "textures.rpf should yield TXD entries");
            let t0 = decode_txd_from_rpf(&tex_rpf, txds[0].offset).expect("decode first TXD");
            assert!(!t0.is_empty());
            assert_eq!(t0[0].pixels.len(), t0[0].w * t0[0].h);
        }

        // 3. CNavMesh 3D decode from navmeshes.rpf (first CNavMesh block starts at ~750MB after path regions)
        let nav_rpf = base.join("navmeshes.rpf");
        if nav_rpf.exists() {
            let mut f = File::open(&nav_rpf).unwrap();
            let base_off = 786_010_112u64; // page holding first CNavMesh tile (786010128 - 0x10)
            f.seek(SeekFrom::Start(base_off)).unwrap();
            let mut buf = vec![0u8; 256 * 1024];
            let n = f.read(&mut buf).unwrap();
            let mut dec = Decompress::new(true);
            let mut out = vec![0u8; 32768];
            let e = probe_rage9_page(&buf, 0, n, base_off, &nav_rpf, "navmeshes.rpf", &mut dec, &mut out)
                .expect("should detect NAVM in navmeshes.rpf");
            assert_eq!(e.kind, "NAVM");
            assert!(e.name.starts_with("nav_160_179_"), "got {}", e.name);
            let mesh = decode_navmesh_from_rpf(&nav_rpf, e.offset, &e.name).expect("decode NAVM mesh");
            assert!(!mesh.verts.is_empty() && !mesh.tris.is_empty());
        }

        // 4. packs.rpf catalog model decode (verifies +0xA8 name + SW_4KB_D embedded tex)
        let packs_rpf = base.join("packs.rpf");
        if packs_rpf.exists() {
            let mesh = decode_rage9_model_from_rpf(&packs_rpf, 0x47ab1010, "fallback").expect("decode packs.rpf model");
            assert!(!mesh.verts.is_empty() && !mesh.tris.is_empty());
        }
    }

    #[test]
    fn multi_select_click_ctrl_shift_works() {
        let view = [10usize, 20, 30, 40, 50];
        let mut multi = HashSet::new();
        let mut anchor = None;

        // Plain click selects one and sets anchor
        apply_multi_select(&view, &mut multi, &mut anchor, 20, false, false);
        assert_eq!(multi, HashSet::from([20]));
        assert_eq!(anchor, Some(20));

        // Shift+click selects contiguous range in view order (20..=40)
        apply_multi_select(&view, &mut multi, &mut anchor, 40, false, true);
        assert_eq!(multi, HashSet::from([20, 30, 40]));
        assert_eq!(anchor, Some(20));

        // Ctrl+click toggles individual item
        apply_multi_select(&view, &mut multi, &mut anchor, 30, true, false);
        assert_eq!(multi, HashSet::from([20, 40]));
        assert_eq!(anchor, Some(30));

        // Ctrl+click adds another item
        apply_multi_select(&view, &mut multi, &mut anchor, 10, true, false);
        assert_eq!(multi, HashSet::from([10, 20, 40]));

        // Plain click resets to single item
        apply_multi_select(&view, &mut multi, &mut anchor, 50, false, false);
        assert_eq!(multi, HashSet::from([50]));
        assert_eq!(anchor, Some(50));
    }

    #[test]
    fn multi_copy_formats_multiple_entries() {
        let e1 = RpfEntry {
            source_rpf: PathBuf::from("packs.rpf"),
            rpf_name: "packs.rpf".into(),
            name: "alpha.#ft".into(),
            kind: "VEH",
            offset: 0x1000,
            comp_size: 64,
            raw_size: 128,
            tracks: 0,
            chunks: vec![],
        };
        let e2 = RpfEntry {
            source_rpf: PathBuf::from("textures.rpf"),
            rpf_name: "textures.rpf".into(),
            name: "beta.#td".into(),
            kind: "TXD",
            offset: 0x20a0,
            comp_size: 128,
            raw_size: 256,
            tracks: 0,
            chunks: vec![],
        };
        let list = [&e1, &e2];
        assert_eq!(format_multi_copy(&list, 0), "alpha.#ft\nbeta.#td");
        assert_eq!(
            format_multi_copy(&list, 1),
            "packs.rpf / alpha.#ft\ntextures.rpf / beta.#td"
        );
        assert_eq!(
            format_multi_copy(&list, 2),
            "packs.rpf @ 0x00001000 (alpha.#ft)\ntextures.rpf @ 0x000020a0 (beta.#td)"
        );
    }
}

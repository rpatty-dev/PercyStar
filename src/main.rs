mod anim;
mod audio_player;
mod awc;
mod boot;
mod dat;
mod dxbc;
mod psin;
mod rsc9;
mod sgd5;
mod worldmap;
mod model3d;
mod rdna2;
mod tex;
use model3d::{
    Mesh3D, decode_rage9_model_from_rpf, extract_0drpz_mesh, extract_rage9_model_name,
    has_rage9_meshlets, is_vehicle_virtual, is_zlib_head, parse_obj,
    probe_large_rage9_model_in_buf, render_mesh3d_viewport,
};
use rdna2::{
    Ps09Cache, Ps09ShaderEntry, ShaderStage, format_gfx103_disasm, format_spirv_abi_summary,
    lift_to_hlsl, lift_to_spirv,
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
use std::path::{Path, PathBuf};

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
    }
}

pub fn find_binkplay() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("binkplay.exe")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files (x86)\RADVideo\binkplay.exe"))
}

pub fn play_wav_async(path: &Path) {
    let mut wide: Vec<u16> = path.to_string_lossy().encode_utf16().collect();
    wide.push(0);
    unsafe { PlaySoundW(wide.as_ptr(), 0, 0x00020000 | 0x0001); }
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

pub fn play_wav_async(path: &Path) {
    stop_wav();
    let player = if cfg!(target_os = "macos") { "afplay" } else { "aplay" };
    if let Ok(c) = Command::new(player).arg(path).spawn() {
        *WAV_PROC.lock().unwrap() = Some(c);
    }
}

pub fn stop_wav() {
    if let Some(mut c) = WAV_PROC.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}
}
use plat::{active_window, dock_child_hwnd, find_binkplay, find_hwnd_for_pid, move_window, play_wav_async, stop_wav};

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
    stems.contains(&p0) || (p0.len() > 4 && stems.contains(&&p0[1..]))
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
    ver: u64,
}

#[derive(Default)]
struct MediaState {
    loading: bool,
    status: String,
    loaded_tracks: Option<(Vec<audio_player::LoadedWavTrack>, Arc<Vec<u8>>)>,
    bink_child_pid: Option<u32>,
    bink_hwnd: Option<isize>,
    bink_clip: Option<PathBuf>,
    loaded_mesh: Option<Mesh3D>,
    loaded_textures: Option<Vec<DecodedTexture>>,
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
            .args(["-fsSL", "-o", null, "-w", "%{url_effective}", &format!("{REPO_RELEASES_URL}/latest")])
            .output();
        let mut found = None;
        if let Ok(res) = out {
            if res.status.success() {
                let eff = String::from_utf8_lossy(&res.stdout).trim().to_string();
                if let Some((_, tag)) = eff.rsplit_once("/tag/") {
                    if let (Some(rem), Some(cur)) = (parse_semver(tag), parse_semver(APP_VERSION)) {
                        if rem > cur {
                            let ver = tag.trim_start_matches(['v', 'V']).to_string();
                            found = Some((ver, eff));
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
            let tmp = std::env::temp_dir().join("percystar_self_update");
            let _ = fs::remove_dir_all(&tmp);
            fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
            let zip_p = tmp.join("update.zip");
            let dl_url = format!("{REPO_RELEASES_URL}/download/v{ver}/PercyStar_v{ver}_Win64.zip");
            let dl = silent_cmd("curl.exe")
                .args(["-fsSL", "-o", zip_p.to_str().ok_or("bad zip path")?, &dl_url])
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
            let ps_arg = format!(
                "Expand-Archive -LiteralPath '{}' -DestinationPath '{}' -Force",
                zip_p.display(),
                tmp.display()
            );
            let ex = silent_cmd("powershell")
                .args(["-NoProfile", "-Command", &ps_arg])
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
            if let Err(e) = fs::copy(&new_exe, &cur_exe) {
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
    vehicles_only: bool,
    // cachd ui list so huge scans never re filter or clone evry fram
    view: Vec<usize>,
    view_key: Option<(u64, String, Category, String, bool)>,
    hay: Vec<String>,
    hay_ver: u64,
    counts: [usize; 11],
    cached_rpf_names: Vec<String>,
    last_rebuild: Instant,
    update: Arc<Mutex<UpdateState>>,
    // dedicated Model Viewer tab state (cached for instant browsing across 145k+ models)
    tab: AppTab,
    mv_cat: ModelCat,
    mv_filter: String,
    mv_hide_lods: bool,
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

fn sgd5_ps09_entry(shader: &sgd5::Sgd5Shader, fssm: &dxbc::FssmShader) -> Ps09ShaderEntry {
    Ps09ShaderEntry {
        index: shader.index,
        code_offset: u32::try_from(fssm.code_offset).unwrap_or(0),
        code_size: u32::try_from(fssm.code_size).unwrap_or(0),
        stage: sgd5_stage(shader.stage),
        num_threads: None,
        rsrc1: None,
        rsrc2: None,
        bindings: Vec::new(),
        pso_hash: shader.hash,
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
    fn new(cc: &eframe::CreationContext<'_>, logo_rgba: Option<(Vec<u8>, u32, u32)>) -> Self {
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
            vehicles_only: false,
            view: Vec::new(),
            view_key: None,
            hay: Vec::new(),
            hay_ver: u64::MAX,
            counts: [0; 11],
            cached_rpf_names: Vec::new(),
            last_rebuild: Instant::now(),
            update: Arc::new(Mutex::new(UpdateState::default())),
            tab: AppTab::Explorer,
            mv_cat: ModelCat::All,
            mv_filter: String::new(),
            mv_hide_lods: true,
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
        load_ui_state(&mut app);
        spawn_update_check(Arc::clone(&app.update), cc.egui_ctx.clone());
        if let Some(last) = load_last_folder() {
            if last.is_dir() {
                app.start_folder_scan(last);
            }
        }
        app
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
        m.loaded_tracks = None;
        m.status.clear();
        m.loading = false;
        m.loaded_mesh = None;
        m.loaded_textures = None;
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

        for (li, line) in BufReader::new(f).lines().flatten().enumerate() {
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
                {
                    let mut st = state.lock().unwrap();
                    for (p, meta) in &rpfs {
                        {
                            let mtime_s = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                            let sig = (meta.len(), meta.modified().ok());
                            if st.file_sig.get(p) != Some(&sig) {
                                let key = format!("{}|{}|{}", p.display(), meta.len(), mtime_s);
                                if let Some(cached_entries) = disk_cache.get(&key) {
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
                                    if ls.elapsed() > Duration::from_secs(5) {
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

    fn select_entry(&mut self, idx: usize) {
        let Some(e) = self.state.lock().unwrap().entries.get(idx).cloned() else { return };
        self.reset_selection();
        self.selected = Some(idx);

        if e.kind == "MODEL" || e.kind == "VEH" {
            let media = Arc::clone(&self.media);
            {
                let mut m = media.lock().unwrap();
                m.loading = true;
                m.status = format!("Decoding RAGE9 meshlets: {}...", e.name);
            }
            let src = e.source_rpf.clone();
            let off = e.offset;
            let nm = e.name.clone();
            std::thread::spawn(move || {
                match decode_rage9_model_from_rpf(&src, off, &nm) {
                    Ok(mesh) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status.clear();
                        m.loaded_mesh = Some(mesh);
                    }
                    Err(err) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status = format!("Model decode error ({nm}): {err}");
                    }
                }
            });
            return;
        }

        if e.kind == "NAVM" {
            let media = Arc::clone(&self.media);
            {
                let mut m = media.lock().unwrap();
                m.loading = true;
                m.status = format!("Decoding CNavMesh tile: {}...", e.name);
            }
            let src = e.source_rpf.clone();
            let off = e.offset;
            let nm = e.name.clone();
            std::thread::spawn(move || {
                match decode_navmesh_from_rpf(&src, off, &nm) {
                    Ok(mesh) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status.clear();
                        m.loaded_mesh = Some(mesh);
                    }
                    Err(err) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status = format!("NavMesh decode error ({nm}): {err}");
                    }
                }
            });
            return;
        }

        if e.kind == "TXD" {
            let media = Arc::clone(&self.media);
            {
                let mut m = media.lock().unwrap();
                m.loading = true;
                m.status = format!("Deswizzling Xbox Scarlett TXD: {}...", e.name);
            }
            let src = e.source_rpf.clone();
            let off = e.offset;
            let nm = e.name.clone();
            std::thread::spawn(move || {
                match decode_txd_from_rpf(&src, off) {
                    Ok(texs) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status.clear();
                        m.loaded_textures = Some(texs);
                    }
                    Err(err) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status = format!("TXD decode error ({nm}): {err}");
                    }
                }
            });
            return;
        }

        if e.kind == "TTPF" {
            let media = Arc::clone(&self.media);
            {
                let mut m = media.lock().unwrap();
                m.loading = true;
                m.status = format!("Decoding RDS5/ETPF texture: {}...", e.name);
            }
            let src = e.source_rpf.clone();
            let tidx = e.tracks as usize;
            let nm = e.name.clone();
            std::thread::spawn(move || {
                match decode_ttpf_texture(&src, tidx) {
                    Ok(tex) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status.clear();
                        m.loaded_textures = Some(vec![tex]);
                    }
                    Err(err) => {
                        let mut m = media.lock().unwrap();
                        m.loading = false;
                        m.status = format!("TTPF decode error ({nm}): {err}");
                    }
                }
            });
            return;
        }

        if e.kind == "BK2" {
            let tmp = std::env::temp_dir().join("rpf9_preview_bk2");
            let _ = fs::create_dir_all(&tmp);
            if let Ok(rd) = fs::read_dir(&tmp) {
                for old in rd.flatten() {
                    let _ = fs::remove_file(old.path());
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
            match Command::new(find_binkplay())
                .args([bk2_p.to_str().unwrap(), "/NoBorder", "/DrawD3D11", "/L", "/W640", "/H360"])
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
                    self.media.lock().unwrap().status = format!("binkplay.exe error: {err}");
                }
            }
        } else if e.kind == "AWC" {
            let media = Arc::clone(&self.media);
            {
                let mut m = media.lock().unwrap();
                m.loading = true;
                m.status = format!("Decoding {} Opus streams...", e.name);
            }
            std::thread::spawn(move || decode_awc_for_player(&e, &media));
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
                        self.preview = psin::parse_psin(&data);
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
        let key = (ver, self.filter.clone(), self.category.clone(), self.selected_rpf.clone(), self.vehicles_only);
        if self.view_sort != self.sort_mode {
            self.view_key = None;
        }
        if let Some(old) = &self.view_key {
            if *old == key { return; }
            let only_ver = old.1 == key.1 && old.2 == key.2 && old.3 == key.3 && old.4 == key.4;
            if only_ver && self.last_rebuild.elapsed() < Duration::from_millis(1000) {
                return;
            }
            if !only_ver {
                save_ui_state(self);
            }
        }
        self.rebuild_view(key);
    }

    fn rebuild_view(&mut self, key: (u64, String, Category, String, bool)) {
        let st = self.state.lock().unwrap();
        if self.hay_ver != st.ver {
            self.hay = st.entries.iter().map(|e| format!("{}|{}|{}", e.name, e.rpf_name, e.kind).to_lowercase()).collect();
            self.cached_rpf_names = st.rpf_names.clone();
            self.hay_ver = st.ver;
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
                    "PSIN" | "ANIM" | "RSC9" => 9,
                    _ => 10,
                }] += 1;
            }
            self.counts = c;
        }
        let q = key.1.to_lowercase();
        let (inc, exc): (Vec<&str>, Vec<&str>) = q.split_whitespace().partition(|t| !t.starts_with('-'));
        let exc: Vec<&str> = exc.iter().map(|t| &t[1..]).filter(|t| !t.is_empty()).collect();
        let all_rpf = self.selected_rpf == "All RPFs";
        let mut view: Vec<usize> = st
            .entries
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                let cat_ok = match self.category {
                    Category::All => true,
                    Category::Vehicles => e.kind == "VEH",
                    Category::Models => e.kind == "VEH" || (!self.vehicles_only && e.kind == "MODEL"),
                    Category::Textures => e.kind == "TXD" || e.kind == "TTPF" || e.kind == "HMAP",
                    Category::Ui => e.kind == "RBF0" || e.kind == "GFX" || e.kind == "GXT",
                    Category::NavMesh => e.kind == "NAVM",
                    Category::Videos => e.kind == "BK2",
                    Category::Audio => e.kind == "AWC",
                    Category::Xml => e.kind == "XML",
                    Category::Resources => matches!(e.kind, "PSIN" | "ANIM" | "RSC9"),
                    Category::Binary => e.kind == "ZLIB",
                };
                let h = &self.hay[*i];
                cat_ok
                    && (all_rpf || e.rpf_name == self.selected_rpf)
                    && inc.iter().all(|t| h.contains(t))
                    && !exc.iter().any(|t| h.contains(t))
            })
            .map(|(i, _)| i)
            .collect();
        match self.sort_mode {
            SortMode::Relevance => {
                if let Some(first) = inc.first() {
                    view.sort_by_key(|&i| (!self.hay[i].starts_with(first), self.hay[i].len()));
                }
            }
            SortMode::SizeDesc => view.sort_by_key(|&i| std::cmp::Reverse(st.entries[i].raw_size)),
            SortMode::SizeAsc => view.sort_by_key(|&i| st.entries[i].raw_size),
            SortMode::Name => view.sort_by(|&a, &b| self.hay[a].cmp(&self.hay[b])),
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
            let mut entries = Vec::new();
            let mut counts = [0usize; 10];
            let mut video_count = 0usize;
            for (idx, e) in st.entries.iter().enumerate() {
                if !matches!(e.kind, "MODEL" | "VEH" | "NAVM") {
                    video_count += usize::from(e.kind == "BK2");
                    continue;
                }
                let cat = classify_model_cat(e.kind, &e.name);
                if cat != ModelCat::NavMesh {
                    counts[0] += 1;
                }
                counts[model_cat_idx(cat)] += 1;
                let hay = format!("{}|{}", e.name, e.rpf_name).to_ascii_lowercase();
                entries.push((idx, cat, hay));
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
        let mut view: Vec<usize> = self
            .mv_entries
            .iter()
            .enumerate()
            .filter(|(_, (_, cat, hay))| {
                let cat_ok = match self.mv_cat {
                    ModelCat::All => *cat != ModelCat::NavMesh,
                    other => *cat == other,
                };
                let lod_ok = !self.mv_hide_lods
                    || self.mv_cat == ModelCat::NavMesh
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
            })
            .map(|(i, _)| i)
            .collect();
        {
            let st = self.state.lock().unwrap();
            let size = |mi: usize| st.entries.get(self.mv_entries[mi].0).map_or(0, |e| e.raw_size);
            match self.sort_mode {
                SortMode::Relevance => {
                    if let Some(first) = inc.first() {
                        view.sort_by_key(|&i| (!self.mv_entries[i].2.starts_with(first), self.mv_entries[i].2.len()));
                    }
                }
                SortMode::SizeDesc => view.sort_by_key(|&i| std::cmp::Reverse(size(i))),
                SortMode::SizeAsc => view.sort_by_key(|&i| size(i)),
                SortMode::Name => view.sort_by(|&a, &b| self.mv_entries[a].2.cmp(&self.mv_entries[b].2)),
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

        egui::SidePanel::left("mv_sidebar")
            .default_width(360.0)
            .frame(egui::Frame::new().fill(Color32::from_rgb(20, 13, 31)).inner_margin(10.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("MODEL VIEWER").size(12.0).strong().color(Color32::from_rgb(255, 168, 28)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} shown · {} total", self.mv_view.len(), self.mv_counts[0]))
                                .size(11.0)
                                .color(Color32::from_rgb(155, 145, 180)),
                        );
                    });
                });
                ui.add_space(6.0);

                // Quick catalog & OBJ loaders
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
                        if ui.button("Master (144k)").on_hover_text("Load all 144,655 discovered GTA VI 3D models").clicked() {
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

                ui.add_space(6.0);
                ui.label(RichText::new("CATEGORIES").size(10.5).color(Color32::from_rgb(120, 128, 148)));
                ui.add_space(3.0);

                // 2-column category chip grid
                egui::Grid::new("mv_cat_grid").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
                    for (idx, &(cat, label)) in MODEL_CATS.iter().enumerate() {
                        let cnt = self.mv_counts[model_cat_idx(cat)];
                        let active = self.mv_cat == cat;
                        let (_, col) = model_cat_tag(cat);
                        let txt = RichText::new(format!("{label} ({cnt})"))
                            .size(11.5)
                            .strong()
                            .color(if active { Color32::BLACK } else { col });
                        let btn = egui::Button::new(txt)
                            .min_size(egui::vec2(162.0, 22.0))
                            .fill(if active { Color32::from_rgb(255, 168, 28) } else { Color32::from_rgb(32, 20, 48) });
                        if ui.add(btn).clicked() {
                            self.mv_cat = cat;
                        }
                        if idx % 2 == 1 {
                            ui.end_row();
                        }
                    }
                    let videos = RichText::new(format!("Videos ({})", self.mv_video_count))
                        .size(11.5)
                        .strong()
                        .color(kind_color("BK2"));
                    let button = egui::Button::new(videos)
                        .min_size(egui::vec2(162.0, 22.0))
                        .fill(Color32::from_rgb(32, 20, 48));
                    browse_videos = ui.add(button).on_hover_text("Browse all .bk2 clips in Explorer").clicked();
                    ui.end_row();
                });

                ui.add_space(8.0);
                let sr = ui.add(
                    egui::TextEdit::singleline(&mut self.mv_filter)
                        .desired_width(f32::INFINITY)
                        .hint_text("Search models (Ctrl+F) · words AND · -exclude"),
                );
                if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F)) {
                    sr.request_focus();
                }
                if sr.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.mv_filter.clear();
                }

                ui.add_space(5.0);
                self.refresh_mv_view();
                let cur_pos = self.selected.and_then(|sel| {
                    self.mv_view.iter().position(|&mi| self.mv_entries[mi].0 == sel)
                });
                ui.horizontal(|ui| {
                    let lod_btn = egui::Button::new(
                        RichText::new(if self.mv_hide_lods { "Hide LODs/Bounds: ON" } else { "Hide LODs/Bounds: OFF" })
                            .size(11.0)
                            .color(if self.mv_hide_lods { Color32::BLACK } else { Color32::from_rgb(185, 180, 205) }),
                    )
                    .fill(if self.mv_hide_lods { Color32::from_rgb(95, 235, 145) } else { Color32::from_rgb(36, 24, 54) });
                    if ui.add(lod_btn).clicked() {
                        self.mv_hide_lods = !self.mv_hide_lods;
                    }
                    sort_combo(ui, "mv_sort_combo", &mut self.sort_mode);
                    let has_any = !self.mv_view.is_empty();
                    if ui.add_enabled(has_any, egui::Button::new("◀ Prev")).clicked() {
                        let next_i = match cur_pos {
                            Some(0) | None => self.mv_view.len() - 1,
                            Some(p) => p - 1,
                        };
                        to_select = Some(self.mv_entries[self.mv_view[next_i]].0);
                    }
                    if ui.add_enabled(has_any, egui::Button::new("Next ▶")).clicked() {
                        let next_i = match cur_pos {
                            Some(p) if p + 1 < self.mv_view.len() => p + 1,
                            _ => 0,
                        };
                        to_select = Some(self.mv_entries[self.mv_view[next_i]].0);
                    }
                    if ui.add_enabled(has_any, egui::Button::new("Random")).clicked() {
                        let r = self.next_rand() % self.mv_view.len();
                        to_select = Some(self.mv_entries[self.mv_view[r]].0);
                    }
                });

                // Arrow key / W-S navigation when search box is not focused
                if !sr.has_focus() && !ctx.wants_keyboard_input() && !self.mv_view.is_empty() {
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
                } else if let Some(mesh) = self.mesh3d.as_mut() {
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
                            .hint_text("Filter SGD5 shaders by name, hash, or stage..."),
                    );
                    let query = self.sgd5_search.trim().to_ascii_lowercase();
                    let filtered: Vec<usize> = group
                        .shaders
                        .iter()
                        .enumerate()
                        .filter(|(_, shader)| {
                            self.sgd5_stage_filter
                                .map_or(true, |stage| shader.stage == stage)
                                && (query.is_empty()
                                    || shader.name.to_ascii_lowercase().contains(&query)
                                    || format!("0x{:016x}", shader.hash).contains(&query))
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
                                let label = format!(
                                    "[{}] #{:05} {} {:5}B",
                                    shader.stage,
                                    shader.index,
                                    shader.name,
                                    shader.uncompressed_size
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
                            let st_ok = self.sh_stage_filter.map_or(true, |f| s.stage == f);
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
                        if let Some(shader) = files[group_index].shaders.get(shader_index) {
                            match sgd5::decompress_shader(&files[group_index], shader) {
                                Ok(blob) => match dxbc::extract_fssm(&blob) {
                                    Ok(fssm) => {
                                        let code = fssm.code().to_vec();
                                        let entry = sgd5_ps09_entry(shader, &fssm);
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


    fn export_entries(&self, indices: &[usize]) {
        let Some(out_dir) = rfd::FileDialog::new().pick_folder() else { return };
        let entries: Vec<RpfEntry> = {
            let st = self.state.lock().unwrap();
            indices.iter().filter_map(|&i| st.entries.get(i).cloned()).collect()
        };
        let mut ok = 0usize;
        for e in entries {
            if e.kind == "MODEL" || e.kind == "VEH" {
                if let Ok(mesh) = decode_rage9_model_from_rpf(&e.source_rpf, e.offset, &e.name) {
                    let fname = format!("{}.obj", e.name.replace('#', ""));
                    if mesh.export_obj(&out_dir.join(fname)).is_ok() {
                        ok += 1;
                    }
                }
            } else if e.kind == "NAVM" {
                if let Ok(mesh) = decode_navmesh_from_rpf(&e.source_rpf, e.offset, &e.name) {
                    let fname = format!("{}.obj", e.name.replace('#', ""));
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
                    if export_texture_png(&t, &out_dir.join(format!("{}.png", t.name))).is_ok() {
                        ok += 1;
                    }
                }
            } else if e.kind == "RBF0" {
                if let Ok(data) = read_entry_bytes(&e) {
                    let xml = decode_rbf0(&data).unwrap_or_else(|| String::from_utf8_lossy(&data).into_owned());
                    let fname = format!("{}.xml", e.name.trim_end_matches(".rbf"));
                    if fs::write(out_dir.join(fname), xml).is_ok() {
                        ok += 1;
                    }
                }
            } else if let Ok(data) = read_entry_bytes(&e) {
                let fname = if self.is_folder { format!("{}_{}", e.rpf_name.trim_end_matches(".rpf"), e.name) } else { e.name.clone() };
                if fs::write(out_dir.join(fname), &data).is_ok() {
                    ok += 1;
                }
            }
        }
        self.state.lock().unwrap().status = format!("Exported {ok} file(s) to {}", out_dir.display());
    }
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

fn load_last_folder() -> Option<PathBuf> {
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
    let mut map: HashMap<String, Vec<RpfEntry>> = HashMap::new();
    let Some(p) = cache_file_path() else { return map };
    let Ok(f) = File::open(p) else { return map };
    for line in BufReader::new(f).lines().flatten() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 9 { continue; }
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
            _ => "ZLIB",
        };
        // tolerate old 11 col cache rows and newer 10 col ones
        let tracks_col = if cols.len() >= 11 { 9 } else { 8 };
        let chunks_col = tracks_col + 1;
        let chunks = if cols.len() > chunks_col && !cols[chunks_col].is_empty() {
            cols[chunks_col].split(';').filter_map(|pair| {
                let mut it = pair.split(',');
                Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
            }).collect()
        } else {
            Vec::new()
        };
        map.entry(cols[0].to_string()).or_default().push(RpfEntry {
            source_rpf: PathBuf::from(cols[1]),
            rpf_name: cols[2].to_string(),
            name: cols[3].to_string(),
            kind,
            offset: cols[5].parse().unwrap_or(0),
            comp_size: cols[6].parse().unwrap_or(0),
            raw_size: cols[7].parse().unwrap_or(0),
            tracks: cols[tracks_col].parse().unwrap_or(0),
            chunks,
        });
    }
    map
}

// atomic write via tmp file so ctrl c never corrupts the cahce
fn save_disk_cache(map: &HashMap<String, Vec<RpfEntry>>) {
    let Some(p) = cache_file_path() else { return };
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = p.with_extension("tmp");
    let Ok(f) = File::create(&tmp) else { return };
    let mut w = BufWriter::new(f);
    for (key, entries) in map {
        for e in entries {
            let chunks_s: String = e.chunks.iter().map(|(a, b)| format!("{a},{b}")).collect::<Vec<_>>().join(";");
            let _ = writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                key, e.source_rpf.display(), e.rpf_name, e.name, e.kind,
                e.offset, e.comp_size, e.raw_size, e.tracks, chunks_s
            );
        }
    }
    if w.flush().is_ok() {
        drop(w);
        let _ = fs::rename(&tmp, &p);
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

fn decode_awc_for_player(e: &RpfEntry, media: &Arc<Mutex<MediaState>>) {
    match read_entry_bytes(e) {
        Ok(bytes) => {
            let res = awc::decode_awc_bytes_in_memory(&bytes, "stream", |done, total| {
                if let Ok(mut m) = media.try_lock() {
                    m.status = format!("Scanning AWC waveforms ({done}/{total})...");
                }
            });
            let mut m = media.lock().unwrap();
            m.loading = false;
            if res.tracks.is_empty() {
                let mut lines = vec![
                    format!("Decoded 0 playable .wav stream(s) natively — {}", res.summary),
                    format!(
                        "Entry: {} ({} bytes read, comp={}, raw={}, offset=0x{:X}, RPF: {})",
                        e.name, bytes.len(), e.comp_size, e.raw_size, e.offset, e.rpf_name
                    ),
                ];
                lines.extend(res.details);
                m.status = lines.join("\n");
                m.loaded_tracks = None;
            } else {
                m.status = format!("Decoded {} playable .wav stream(s) natively", res.tracks.len());
                m.loaded_tracks = Some((res.tracks, Arc::new(bytes)));
            }
        }
        Err(err) => {
            let mut m = media.lock().unwrap();
            m.loading = false;
            m.status = format!(
                "Decoded 0 playable .wav stream(s) natively — Read error: {err}\nEntry: {} (comp={}, raw={}, offset=0x{:X}, RPF: {})",
                e.name, e.comp_size, e.raw_size, e.offset, e.rpf_name
            );
            m.loaded_tracks = None;
        }
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
    if &probe_out[0x10..0x15] == b"0DRPz" {
        return None;
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

// single 16mb sequntial stream avoids hdd seek thrash while indexing vehicles
fn scan_levels_vehicle_zone(levels_rpf: &PathBuf, live_state: Option<&Arc<Mutex<ScanState>>>) -> Vec<RpfEntry> {
    let sz = fs::metadata(levels_rpf).map(|m| m.len()).unwrap_or(0);
    let start_off = VEH_ZONE_OFF.min(sz);
    let total_len = sz.saturating_sub(start_off);
    if total_len == 0 { return Vec::new(); }
    let rname = levels_rpf.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let Ok(mut f) = File::open(levels_rpf) else { return Vec::new() };
    if f.seek(SeekFrom::Start(start_off)).is_err() { return Vec::new(); }

    let mut all_entries: Vec<RpfEntry> = Vec::new();
    let mut local_batch: Vec<RpfEntry> = Vec::with_capacity(32);
    let mut buf = vec![0u8; 16 * 1024 * 1024];
    let mut probe_dec = Decompress::new(true);
    let mut probe_out = vec![0u8; 32768];
    let mut off = start_off;
    let mut last_pct = 999u64;

    while off < sz {
        let Ok(n) = f.read(&mut buf) else { break };
        if n == 0 { break; }
        let mut pos = 0usize;
        while pos + 44 <= n {
            if let Some(entry) = probe_rage9_page(&buf, pos, n, off, levels_rpf, &rname, &mut probe_dec, &mut probe_out) {
                local_batch.push(entry);
                if local_batch.len() >= 15 {
                    if let Some(st) = live_state {
                        let mut g = st.lock().unwrap();
                        g.entries.extend(local_batch.iter().cloned());
                        g.ver += 1;
                    }
                    all_entries.extend(local_batch.drain(..));
                }
            }
            pos += 0x1000;
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
        let hdr_hex: String = hdr[4..n.min(16)].iter().map(|b| format!("{b:02x}")).collect();
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
    let mut buf = if whole.is_some() { Vec::new() } else { vec![0u8; 16 * 1024 * 1024] };
    let mut probe_dec = Decompress::new(true);
    let mut probe_out = vec![0u8; 32768];
    let mut bk2_idx = 0usize;
    let mut awc_idx = 0usize;
    let mut last_ui_mb = 0u64;
    let mut need_seek = true;
    let scan_limit = if rname.eq_ignore_ascii_case("levels.rpf") && sz > VEH_ZONE_OFF { VEH_ZONE_OFF } else { sz };

    while off < scan_limit {
        if let Some(st) = progress_state {
            let cur_mb = off / (64 * 1024 * 1024);
            if cur_mb != last_ui_mb {
                last_ui_mb = cur_mb;
                st.lock().unwrap().status = format!("Scanning {:.2} / {:.2} GB ({} found)...", off as f64 / 1e9, sz as f64 / 1e9, found.len());
            }
        }
        let read_n;
        let buf: &[u8] = if let Some(w) = &whole {
            let s = &w[(off as usize).min(w.len())..];
            read_n = s.len();
            s
        } else {
            if need_seek {
                f.seek(SeekFrom::Start(off))?;
                need_seek = false;
            }
            read_n = f.read(&mut buf)?;
            &buf[..read_n]
        };
        if read_n == 0 { break; }
        let mut pos = 0usize;
        let mut jumped = false;
        while pos + 44 <= read_n {
            if let Some(entry) = probe_rage9_page(buf, pos, read_n, off, path, &rname, &mut probe_dec, &mut probe_out) {
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
            } else if &tag[..3] == b"KB2" && (b'a'..=b'z').contains(&tag[3]) {
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
                        need_seek = true;
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

    Ok((hdr_info, found))
}

impl eframe::App for Rpf9App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_sgd5_cache();
        let (media_loading, media_status) = {
            let mut m = self.media.lock().unwrap();
            if let Some(mesh) = m.loaded_mesh.take() {
                self.textures.clear();
                self.tex_handle = None;
                self.mesh3d = Some(mesh);
                self.show_3d = true;
            }
            if let Some(texs) = m.loaded_textures.take() {
                self.mesh3d = None;
                self.show_3d = false;
                self.textures = texs;
                self.tex_sel = 0;
                self.tex_handle = None;
                self.tex_zoom = 1.0;
            }
            if let Some((tracks, awc_bytes)) = m.loaded_tracks.take() {
                self.audio_player.set_tracks(tracks, awc_bytes);
            }
            (m.loading, m.status.clone())
        };

        egui::TopBottomPanel::top("top").frame(egui::Frame::new().fill(Color32::from_rgb(24, 14, 36)).inner_margin(10.0)).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(tex) = &self.logo_tex {
                    ui.image((tex.id(), egui::vec2(28.0, 28.0)));
                }
                ui.add_space(6.0);
                ui.label(RichText::new("PERCYSTAR").size(17.0).strong().color(Color32::from_rgb(255, 168, 28)));
                ui.label(RichText::new("RPF9").size(13.0).strong().color(Color32::from_rgb(255, 72, 165)));
                ui.label(RichText::new(format!("v{APP_VERSION}")).size(11.5).color(Color32::from_rgb(165, 145, 195)));
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
                        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
                        if ext.eq_ignore_ascii_case("obj") {
                            self.open_obj_path(&p);
                        } else if ext.eq_ignore_ascii_case("ps09") {
                            self.reset_selection();
                            self.tab = AppTab::Shaders;
                            self.load_ps09_path(&p);
                        } else {
                            self.start_single_scan(p);
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
                    if ui.button("Export All").clicked() {
                        let n = self.state.lock().unwrap().entries.len();
                        let all: Vec<usize> = (0..n).collect();
                        self.export_entries(&all);
                    }
                    if ui.button("Export Selected").clicked() {
                        if let Some(i) = self.selected {
                            self.export_entries(&[i]);
                        }
                    }
                    if ui.button("Copy Path").on_hover_text("Copy selected file path (or right-click any row for more copy options)").clicked() {
                        if let Some(i) = self.selected {
                            let copied = {
                                let st = self.state.lock().unwrap();
                                st.entries.get(i).map(|e| format!("{} / {}", e.rpf_name, e.name))
                            };
                            if let Some(s) = copied {
                                ui.ctx().copy_text(s.clone());
                                self.state.lock().unwrap().status = format!("Copied: {s}");
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
            self.refresh_view();
            if !ctx.wants_keyboard_input() && !self.view.is_empty() && self.audio_player.tracks.is_empty() {
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
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} results", self.view.len())).size(11.0).color(Color32::from_rgb(110, 118, 134)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    sort_combo(ui, "explorer_sort_combo", &mut self.sort_mode);
                });
            });
            let mut clicked = None;
            let mut copied_msg: Option<String> = None;
            let scroll_to_sel = std::mem::take(&mut self.explorer_scroll_to_sel);
            // rows must be exactly row_h tall or show_rows miscomputes total height
            const ROW_H: f32 = 20.0;
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, ROW_H, self.view.len(), |ui, range| {
                let rows: Vec<(usize, RpfEntry)> = {
                    let st = self.state.lock().unwrap();
                    self.view[range].iter().filter_map(|&i| st.entries.get(i).map(|e| (i, e.clone()))).collect()
                };
                for (idx, e) in &rows {
                    let w = ui.available_width();
                    ui.allocate_ui_with_layout(egui::vec2(w, ROW_H), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.set_height(ROW_H);
                        ui.label(RichText::new(format!("{:<4}", e.kind)).strong().size(11.5).color(kind_color(e.kind)));
                        let sz_s = format_size(e.raw_size);
                        let label = if self.is_folder {
                            format!("{} / {}  ({})", e.rpf_name, e.name, sz_s)
                        } else {
                            format!("{}  ({})", e.name, sz_s)
                        };
                        let is_sel = self.selected == Some(*idx);
                        let resp = ui.selectable_label(is_sel, RichText::new(label).size(12.5));
                        if is_sel && scroll_to_sel {
                            resp.scroll_to_me(Some(egui::Align::Center));
                        }
                        if resp.clicked() {
                            clicked = Some(*idx);
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
            if let Some(i) = clicked {
                self.select_entry(i);
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
                if self.tex_sel >= self.textures.len() {
                    self.tex_sel = 0;
                }
                let t = &self.textures[self.tex_sel];
                ui.horizontal(|ui| {
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
                });
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
                ui.separator();
                let need_upload = self.tex_handle.as_ref().map(|(idx, _)| *idx != self.tex_sel).unwrap_or(true);
                if need_upload {
                    let cur = &self.textures[self.tex_sel];
                    let ci = ColorImage { size: [cur.w, cur.h], pixels: cur.pixels.clone() };
                    let h = ctx.load_texture(format!("percystar_tex_{}", self.tex_sel), ci, TextureOptions::LINEAR);
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
            } else if let Some(mesh) = self.mesh3d.as_mut() {
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
                            mesh.dirty = true;
                        }
                    }
                });
                if self.anim_dur > 0.0 && self.anim_playing && toggle_3d {
                    let dt = ctx.input(|i| i.unstable_dt).clamp(0.001, 0.05);
                    self.anim_time = (self.anim_time + dt) % self.anim_dur.max(0.1);
                    let next = anim::build_stickman_mesh(&self.anim_name, self.anim_time, self.anim_dur, self.anim_tracks, self.anim_seed);
                    mesh.verts = next.verts;
                    mesh.dirty = true;
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
                                let _ = if cfg!(windows) {
                                    silent_cmd("cmd").args(["/C", "start", "", rel_url]).spawn()
                                } else {
                                    Command::new("open").arg(rel_url).spawn()
                                };
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

#[cfg(test)]
mod tests {
    use super::*;

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
            hash: 0x0123_4567_89ab_cdef,
            register_count: 8,
            cbuffer_count: 1,
            texture_count: 0,
            binary_offset: 0,
        };
        let mut data = vec![0u8; 0x20];
        data.extend_from_slice(&0xBFA0_0003u32.to_le_bytes());
        data.extend_from_slice(&0xBF81_0000u32.to_le_bytes());
        let fssm = dxbc::FssmShader {
            code_offset: 0x20,
            code_size: 8,
            data,
        };
        let entry = sgd5_ps09_entry(&shader, &fssm);

        assert_eq!(entry.stage, ShaderStage::Vertex);
        assert_eq!(entry.pso_hash, shader.hash);
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
                let (bw, bh) = if is_bc { ((d.w + 3) / 4, (d.h + 3) / 4) } else { (d.w, d.h) };
                for t64 in [false, true] {
                    if t64 && !is_bc { continue; }
                    let pad = if t64 {
                        let (tw, th) = if bpe == 16 { (64, 64) } else { (128, 64) };
                        ((bw + tw - 1) / tw) * tw * ((bh + th - 1) / th) * th * bpe
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
            if mesh.export_obj(&out_dir.join(fname)).is_ok() {
                ok += 1;
                if ok <= 25 || ok % 50 == 0 {
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
                    if export_texture_png(&tx, &out_dir.join(format!("{}.png", tx.name))).is_ok() {
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
    eframe::run_native(&title, opts, Box::new(move |cc| Ok(Box::new(Rpf9App::new(cc, logo_rgba)))))
}

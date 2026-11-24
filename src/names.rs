// names behind joaat hashes
// sources are a small built in list, user word lists and names recovered from the open dump
// schema names (structs, fields, enum values) are hashed as written, data values in lower case
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{OnceLock, RwLock};

// files inside the names folder that hold what recover found
pub const RECOVERED_FILE: &str = "recovered.txt";
pub const BUILT_FILE: &str = "built.txt";

#[derive(Default)]
struct Dict {
    // hash of the name as written
    exact: HashMap<u32, String>,
    // hash of the lower case name
    lower: HashMap<u32, String>,
    // names assembled from words instead of found written out
    built: HashSet<u32>,
}

static DICT: OnceLock<RwLock<Dict>> = OnceLock::new();

fn dict() -> &'static RwLock<Dict> {
    DICT.get_or_init(|| RwLock::new(Dict::default()))
}

#[inline]
fn feed(mut h: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        h = h.wrapping_add(b as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h
}

#[inline]
fn finish(mut h: u32) -> u32 {
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h.wrapping_add(h << 15)
}

// joaat without the lower casing, the form schema names use
pub fn joaat_cs(s: &str) -> u32 {
    finish(feed(0, s.as_bytes()))
}

// name of a struct, field, enum or enum value
pub fn lookup(hash: u32) -> Option<String> {
    let d = dict().read().ok()?;
    d.exact.get(&hash).or_else(|| d.lower.get(&hash)).cloned()
}

// name behind a hash stored as data such as a model or bone name
pub fn lookup_value(hash: u32) -> Option<String> {
    let d = dict().read().ok()?;
    d.lower.get(&hash).or_else(|| d.exact.get(&hash)).cloned()
}

// true when the name of this hash was assembled from words and may be a chance match
pub fn is_built(hash: u32) -> bool {
    dict().read().is_ok_and(|d| d.built.contains(&hash))
}

// adds assembled names, ones already known from elsewhere keep their standing
// returns how many were new; the check and the insert share one lock so the count
// stays correct while other threads (or parallel tests) add names at the same time
pub fn add_built<I, S>(names: I) -> usize
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let Ok(mut d) = dict().write() else { return 0 };
    let mut added = 0;
    for n in names {
        let n = n.as_ref().trim();
        if !is_name(n) {
            continue;
        }
        let hash = joaat_cs(n);
        if d.exact.contains_key(&hash) || d.lower.contains_key(&crate::psin::joaat(n)) {
            continue;
        }
        d.exact.insert(hash, n.to_string());
        d.lower.entry(crate::psin::joaat(n)).or_insert_with(|| n.to_string());
        d.built.insert(hash);
        added += 1;
    }
    added
}

pub fn len() -> usize {
    dict().read().map(|d| d.exact.len()).unwrap_or(0)
}

// names earlier games of the engine used, many are unchanged
const SEED: &str = "fMass fInitialDragCoeff fDownforceModifier fPercentSubmerged vecCentreOfMassOffset vecInertiaMultiplier fDriveBiasFront nInitialDriveGears fInitialDriveForce fDriveInertia fClutchChangeRateScaleUpShift fClutchChangeRateScaleDownShift fInitialDriveMaxFlatVel fBrakeForce fBrakeBiasFront fHandBrakeForce fSteeringLock fTractionCurveMax fTractionCurveMin fTractionCurveLateral fTractionSpringDeltaMax fLowSpeedTractionLossMult fCamberStiffnesss fTractionBiasFront fTractionLossMult fSuspensionForce fSuspensionCompDamp fSuspensionReboundDamp fSuspensionUpperLimit fSuspensionLowerLimit fSuspensionRaise fSuspensionBiasFront fAntiRollBarForce fAntiRollBarBiasFront fRollCentreHeightFront fRollCentreHeightRear fCollisionDamageMult fWeaponDamageMult fDeformationDamageMult fEngineDamageMult fPetrolTankVolume fOilVolume fSeatOffsetDistX fSeatOffsetDistY fSeatOffsetDistZ nMonetaryValue strModelFlags strHandlingFlags strDamageFlags AIHandling SubHandlingData handlingName HandlingData CHandlingData CHandlingDataMgr CCarHandlingData CBikeHandlingData CBoatHandlingData CFlyingHandlingData modelName txdName handlingId gameName vehicleMakeName expressionDictName expressionName animConvRoofDictName animConvRoofName ptfxAssetName audioNameHash layout coverBoundOffsets explosionInfo scenarioLayout cameraName aimCameraName bonnetCameraName povCameraName type plateType dashboardType vehicleClass wheelType trailers additionalTrailers drivers extraIncludes doorsWithCollisionWhenClosed driveableDoors bumpersNeedToCollideWithMap needsRopeTexture flags rewards cinematicPartCamera NmBraceOverrideSet buoyancySphereOffset buoyancySphereSizeScale pOverrideRagdollThreshold firstPersonDrivebyData InitDatas txdRelationships residentTxd residentAnims name Name Items items value Value key Key hash Hash id Id model Model vehicle Vehicle vehicleName kits Kits colors Colors variationData lodDistances diffuseTint dirtLevelMin dirtLevelMax swankness frequency maxNum mass Mass topSpeed acceleration braking traction drag gears driveForce maxFlatVel brakeForce steeringLock suspension engine Engine transmission wheels Wheels tyres Tyres aero Aero fBackEndPopUpCarImpulseMult fBackEndPopUpBuildingImpulseMult fBackEndPopUpMaxDeltaSpeed fToeFront fToeRear fCamberFront fCamberRear fCastor fEngineResistance fMaxDriveBiasTransfer fJumpForceScale fThrust fThrustFallOff fThrustVectoring fYawMult fYawStabilise fSideSlipMult fRollMult fRollStabilise fPitchMult fPitchStabilise fFormLiftMult fAttackLiftMult fAttackDiveMult fGearDownDragV fGearDownLiftMult fWindMult fMoveRes fTurnRes fSpeedRes fGearDoorFrontOpen fGearDoorRearOpen fTurublenceMagnitudeMax fTurublenceForceMulti fTurublenceRollTorqueMulti fTurublencePitchTorqueMulti fBodyDamageControlEffectMult fInputSensitivityForDifficulty fOnGroundYawBoostSpeedPeak fOnGroundYawBoostSpeedCap fEngineOffGlideMulti fBoxFrontMult fBoxRearMult fBoxSideMult fSampleTop fSampleBottom fAquaplaneForce fAquaplanePushWaterMult fAquaplanePushWaterCap fAquaplanePushWaterApply fRudderForce fRudderOffsetSubmerge fRudderOffsetForce fRudderOffsetForceZMult fWaveAudioMult fLook_L_R_CamHeight fDragCoefficient fKeelSphereSize fPropRadius fLowLodAngOffset fLowLodDraughtOffset fImpellerOffset fImpellerForceMult fDinghySphereBuoyConst fProwRaiseMult fLeanFwdCOMMult fLeanFwdForceMult fLeanBakCOMMult fLeanBakForceMult fMaxBankAngle fFullAnimAngle fDesLeanReturnFrac fStickLeanMult fBrakingStabilityMult fInAirSteerMult fWheelieBalancePoint fStoppieBalancePoint fWheelieSteerMult fRearBalanceMult fFrontBalanceMult fBikeGroundSideFrictionMult fBikeWheelGroundSideFrictionMult fBikeOnStandLeanAngle fBikeOnStandSteerAngle fJumpForce archetypeName position rotation scaleXY scaleZ parentIndex lodDist childLodDist lodLevel numChildren priorityLevel extensions ambientOcclusionMultiplier artificialAmbientOcclusion tintValue entities containerLods boxOccluders occludeModels physicsDictionaries instancedData timeCycleModifiers carGenerators LODLightsSOA DistantLODLightsSOA block streamingExtentsMin streamingExtentsMax entitiesExtentsMin entitiesExtentsMax contentFlags parent bbMin bbMax bsCentre bsRadius hdTextureDist textureDictionary clipDictionary drawableDictionary physicsDictionary assetType assetName specialAttribute archetypes dependencies compositeEntityTypes";

pub fn seed() {
    add(SEED.split(' '));
}

// identifiers only so a word list with stray text cannot fill the table with junk
pub fn is_name(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && b.len() <= 96 && !b[0].is_ascii_digit() && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

// adds names keeping the first one seen for a hash, returns how many were new
pub fn add<I, S>(names: I) -> usize
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let Ok(mut d) = dict().write() else { return 0 };
    let mut added = 0;
    for n in names {
        let n = n.as_ref().trim();
        if !is_name(n) {
            continue;
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = d.exact.entry(joaat_cs(n)) {
            slot.insert(n.to_string());
            added += 1;
        }
        d.lower.entry(crate::psin::joaat(n)).or_insert_with(|| n.to_string());
    }
    added
}

// loads every list in the folder: txt files hold one name per line
// tsv files are name tables with a header row (hash, name, kind, method), as published by name recovery projects
pub fn load_dir(dir: &Path) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    let mut added = 0;
    for entry in rd.flatten() {
        let p = entry.path();
        let ext = p.extension().map(|x| x.to_ascii_lowercase());
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        if ext.as_deref() == Some(std::ffi::OsStr::new("txt")) {
            if p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(BUILT_FILE)) {
                added += add_built(text.lines());
            } else {
                added += add(text.lines());
            }
        } else if ext.as_deref() == Some(std::ffi::OsStr::new("tsv")) {
            added += load_table(&text);
        }
    }
    added
}

// names of a tab separated table, the ones not found written out anywhere are kept as built
fn load_table(text: &str) -> usize {
    let (mut sure, mut built) = (Vec::new(), Vec::new());
    for line in text.lines().skip(1) {
        let mut cols = line.split('\t');
        let (Some(_), Some(name)) = (cols.next(), cols.next()) else { continue };
        let method = cols.nth(1).unwrap_or("");
        if matches!(method.trim(), "name list" | "game text" | "") { sure.push(name) } else { built.push(name) }
    }
    add(sure.iter()) + add_built(built.iter())
}

// merges names into one of the folder's lists
pub fn save_list(dir: &Path, file: &str, found: &[(u32, String)]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(file);
    let mut all: Vec<String> = std::fs::read_to_string(&path).unwrap_or_default().lines().map(str::to_string).collect();
    all.extend(found.iter().map(|(_, n)| n.clone()));
    all.retain(|n| is_name(n));
    all.sort_unstable();
    all.dedup();
    std::fs::write(path, all.join("\n"))
}

// collects identifier like words from text, runs of one repeated letter are noise from binary data
pub fn harvest_tokens(bytes: &[u8], out: &mut HashSet<String>) {
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if !(c.is_ascii_alphabetic() || c == b'_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let tok = &bytes[start..i];
        if (3..=64).contains(&tok.len()) && !tok.windows(3).any(|w| w[0] == w[1] && w[1] == w[2]) {
            // checked ascii above
            out.insert(String::from_utf8_lossy(tok).into_owned());
        }
    }
}

// spellings a token may take as a field name: without a type prefix and split at namespace marks
pub fn variants(name: &str, out: &mut Vec<String>) {
    out.clear();
    out.push(name.to_string());
    for part in name.split("__") {
        if part.len() >= 3 && part != name {
            out.push(part.to_string());
        }
    }
    let base = out.len();
    for k in 0..base {
        let s = out[k].clone();
        let stripped = s.strip_prefix("m_").unwrap_or(&s);
        if stripped.len() != s.len() && stripped.len() >= 3 {
            out.push(stripped.to_string());
        }
        let b = stripped.as_bytes();
        if b.len() >= 4 && b"fibnvsu".contains(&b[0]) && b[1].is_ascii_uppercase() {
            out.push(stripped[1..].to_string());
        }
    }
}

// the hashed names of a set of metadata files and what is known about each
#[derive(Default)]
pub struct NameTargets {
    pub structs: HashSet<u32>,
    // field name to the set of value types it is declared with, one bit per type id
    pub fields: HashMap<u32, u128>,
    // enum to the hashes of its value names
    pub enums: HashMap<u32, HashSet<u32>>,
    // struct to the names of its fields
    pub members: HashMap<u32, HashSet<u32>>,
}

impl NameTargets {
    pub fn all(&self) -> HashSet<u32> {
        let mut all: HashSet<u32> = self.structs.iter().copied().collect();
        all.extend(self.fields.keys());
        all.extend(self.enums.keys());
        all.extend(self.enums.values().flatten());
        all
    }
}

pub struct Recovered {
    // names found written out somewhere
    pub exact: Vec<(u32, String)>,
    // names assembled from words of the exact ones
    pub built: Vec<(u32, String)>,
    // how many of the built names are expected to be chance matches
    pub built_noise: f64,
}

// random hashes are probed alongside the wanted ones, how often they hit measures the chance matches of a search
const DECOYS_PER_TARGET: usize = 4;
// a search is kept only when it finds this many times more names than its decoys predict by chance
const SIGNAL_RATIO: f64 = 6.0;
// vocabulary size of the word searches
const WORDS: usize = 3000;
// rarer words only join a search next to a common one
const RARE_WORDS: usize = 9000;
const ROUNDS: usize = 3;

fn decoys_for(wanted: &HashSet<u32>, avoid: &HashSet<u32>) -> HashSet<u32> {
    let mut out = HashSet::with_capacity(wanted.len() * DECOYS_PER_TARGET);
    let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ wanted.len() as u64;
    while out.len() < wanted.len() * DECOYS_PER_TARGET {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let h = (x >> 16) as u32;
        if !avoid.contains(&h) {
            out.insert(h);
        }
    }
    out
}

// membership test for many probes: a bitmap rejects nearly all misses before the sets are asked
struct Probe<'a> {
    bits: Vec<u64>,
    wanted: &'a HashSet<u32>,
    decoys: &'a HashSet<u32>,
}

impl<'a> Probe<'a> {
    fn new(wanted: &'a HashSet<u32>, decoys: &'a HashSet<u32>) -> Self {
        let mut bits = vec![0u64; 1 << 18];
        for h in wanted.iter().chain(decoys) {
            bits[(h >> 14) as usize] |= 1 << ((h >> 8) & 63);
        }
        Self { bits, wanted, decoys }
    }

    // 1 for a wanted hash, 2 for a decoy
    #[inline]
    fn class(&self, h: u32) -> u8 {
        if self.bits[(h >> 14) as usize] & (1 << ((h >> 8) & 63)) == 0 {
            0
        } else if self.wanted.contains(&h) {
            1
        } else {
            self.decoys.contains(&h) as u8 * 2
        }
    }
}

// keeps the smaller spelling so results do not depend on thread timing
fn keep(map: &mut HashMap<u32, String>, h: u32, name: String) {
    match map.get(&h) {
        Some(old) if *old <= name => {}
        _ => {
            map.insert(h, name);
        }
    }
}

// first + sep + second for every pair: the wanted names it spells and how many decoys it hit
fn cross(firsts: &[String], seconds: &[String], sep: &str, wanted: &HashSet<u32>, decoys: &HashSet<u32>) -> (HashMap<u32, String>, usize) {
    let mut out = HashMap::new();
    let mut misses = 0usize;
    if firsts.is_empty() || seconds.is_empty() || wanted.is_empty() {
        return (out, 0);
    }
    let probe = Probe::new(wanted, decoys);
    let tails: Vec<Vec<u8>> = seconds.iter().map(|s| [sep.as_bytes(), s.as_bytes()].concat()).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let chunk = firsts.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = firsts
            .chunks(chunk)
            .map(|part| {
                let (probe, tails) = (&probe, &tails);
                scope.spawn(move || {
                    let mut found = Vec::new();
                    let mut decoy_hits = 0usize;
                    for first in part {
                        let state = feed(0, first.as_bytes());
                        for (k, tail) in tails.iter().enumerate() {
                            let h = finish(feed(state, tail));
                            match probe.class(h) {
                                1 => found.push((h, format!("{first}{sep}{}", seconds[k]))),
                                2 => decoy_hits += 1,
                                _ => {}
                            }
                        }
                    }
                    (found, decoy_hits)
                })
            })
            .collect();
        for handle in handles {
            let (found, decoy_hits) = handle.join().unwrap_or_default();
            misses += decoy_hits;
            for (h, name) in found {
                keep(&mut out, h, name);
            }
        }
    });
    (out, misses)
}

// the same test for a plain list of candidate names
fn sift(names: impl Iterator<Item = String>, wanted: &HashSet<u32>, decoys: &HashSet<u32>) -> (HashMap<u32, String>, usize) {
    let mut out = HashMap::new();
    let mut misses = 0usize;
    for name in names {
        let h = joaat_cs(&name);
        if wanted.contains(&h) {
            keep(&mut out, h, name);
        } else if decoys.contains(&h) {
            misses += 1;
        }
    }
    (out, misses)
}

// byte ranges of the alphabetic words a name is made of: split at case changes, digits and underscores
fn word_spans(name: &str) -> Vec<(usize, usize)> {
    let b = name.as_bytes();
    let mut spans = Vec::new();
    let mut start = 0;
    for i in 1..=b.len() {
        let split = i == b.len()
            || !b[i].is_ascii_alphabetic()
            || !b[i - 1].is_ascii_alphabetic()
            || (b[i].is_ascii_uppercase() && b[i - 1].is_ascii_lowercase())
            || (b[i].is_ascii_uppercase() && i + 1 < b.len() && b[i + 1].is_ascii_lowercase());
        if split {
            if b[start..i].iter().all(|c| c.is_ascii_alphabetic()) && i > start {
                spans.push((start, i));
            }
            start = i;
        }
    }
    spans
}

fn capitalised(word: &str) -> String {
    let mut w = word.to_ascii_lowercase();
    w[..1].make_ascii_uppercase();
    w
}

fn count_words<'a>(names: impl Iterator<Item = &'a str>, weight: usize, count: &mut HashMap<String, usize>) {
    for name in names {
        for (s, e) in word_spans(name) {
            if e - s >= 2 {
                *count.entry(capitalised(&name[s..e])).or_default() += weight;
            }
        }
    }
}

fn most_used(count: HashMap<String, usize>, cap: usize) -> Vec<String> {
    let mut words: Vec<(String, usize)> = count.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    words.truncate(cap);
    words.into_iter().map(|w| w.0).collect()
}

const fn kinds(list: &[u8]) -> u128 {
    let mut mask = 0u128;
    let mut i = 0;
    while i < list.len() {
        mask |= 1 << list[i];
        i += 1;
    }
    mask
}

const INTS: u128 = kinds(&[0x10, 0x11, 0x12, 0x13, 0x14, 0x15]);
const FLOATS: u128 = kinds(&[0x20, 0x21]);
const VECTORS: u128 = kinds(&[0x30, 0x31, 0x32, 0x33, 0x34]);
const TEXTS: u128 = kinds(&[0x42, 0x44, 0x4a, 0x4b, 0x4d]);

// type prefixes of field names and the value types each one is allowed on
const TYPED_PREFIXES: &[(&str, u128)] = &[
    ("f", FLOATS),
    ("b", kinds(&[0x01])),
    ("i", INTS),
    ("n", INTS),
    ("u", kinds(&[0x11, 0x13, 0x15, 0x17])),
    ("v", VECTORS),
    ("vec", VECTORS),
    ("s", TEXTS),
    ("str", TEXTS),
    ("e", kinds(&[0x60, 0x61, 0x62])),
    ("a", kinds(&[0x50, 0x52, 0x5c])),
    ("p", kinds(&[0x07])),
    ("m_", u128::MAX),
    ("m_f", FLOATS),
    ("m_b", kinds(&[0x01])),
    ("m_i", INTS),
];

// words that come in pairs, a name with one of them often has a twin with the other
const TWINS: &[(&str, &str)] = &[
    ("Min", "Max"),
    ("Start", "End"),
    ("Begin", "End"),
    ("Front", "Rear"),
    ("Left", "Right"),
    ("In", "Out"),
    ("Inner", "Outer"),
    ("Enable", "Disable"),
    ("Lower", "Upper"),
    ("Low", "High"),
    ("Up", "Down"),
    ("Near", "Far"),
    ("Open", "Close"),
    ("On", "Off"),
    ("First", "Last"),
    ("Enter", "Exit"),
    ("Source", "Target"),
    ("Small", "Large"),
    ("Short", "Long"),
    ("Top", "Bottom"),
    ("Width", "Height"),
    ("Day", "Night"),
    ("Male", "Female"),
    ("Primary", "Secondary"),
    ("Forward", "Backward"),
    ("Attack", "Release"),
    ("Yaw", "Pitch"),
    ("Pitch", "Roll"),
    ("Yaw", "Roll"),
    ("X", "Y"),
    ("Y", "Z"),
    ("X", "Z"),
    ("L", "R"),
];

const SUFFIXES: &[&str] = &[
    "Hash", "Name", "Id", "Index", "Count", "Scale", "Mult", "Multiplier", "Min", "Max", "X", "Y", "Z", "Offset", "Enabled", "Flags", "Type", "Info", "Data", "Set", "List", "Time", "Speed",
];

// names one step away from a known one: its twin, its plural, a common suffix, another first letter case
fn relatives(name: &str, out: &mut Vec<String>) {
    for (s, e) in word_spans(name) {
        let word = &name[s..e];
        for &(a, b) in TWINS {
            for (from, to) in [(a, b), (b, a)] {
                if !word.eq_ignore_ascii_case(from) {
                    continue;
                }
                let styled = if word.len() > 1 && word.bytes().all(|c| c.is_ascii_uppercase()) {
                    to.to_ascii_uppercase()
                } else if word.as_bytes()[0].is_ascii_lowercase() {
                    to.to_ascii_lowercase()
                } else {
                    to.to_string()
                };
                out.push(format!("{}{styled}{}", &name[..s], &name[e..]));
            }
        }
    }
    out.push(format!("{name}s"));
    if let Some(single) = name.strip_suffix('s') {
        out.push(single.to_string());
    }
    for suffix in SUFFIXES {
        out.push(format!("{name}{suffix}"));
    }
    let b = name.as_bytes();
    if b.len() > 2 && b"fbinuvse".contains(&b[0]) && b[1].is_ascii_uppercase() {
        out.push(name[1..].to_string());
    }
    let mut flipped = name.to_string();
    if b[0].is_ascii_uppercase() {
        flipped[..1].make_ascii_lowercase();
    } else {
        flipped[..1].make_ascii_uppercase();
    }
    out.push(flipped);
}

struct Search<'a> {
    targets: &'a NameTargets,
    all: &'a HashSet<u32>,
    // names found written out plus the built ones accepted in earlier rounds
    known: HashMap<u32, String>,
    round: HashMap<u32, String>,
    noise: f64,
}

impl Search<'_> {
    fn open<'s>(&self, set: impl Iterator<Item = &'s u32>) -> HashSet<u32> {
        set.filter(|h| !self.known.contains_key(h) && !self.round.contains_key(h)).copied().collect()
    }

    // accepts what a search found unless it is no better than chance
    fn take(&mut self, found: (HashMap<u32, String>, usize)) {
        let (names, decoy_hits) = found;
        let noise = decoy_hits as f64 / DECOYS_PER_TARGET as f64;
        let fresh: Vec<(u32, String)> = names.into_iter().filter(|(h, _)| !self.known.contains_key(h) && !self.round.contains_key(h)).collect();
        if !fresh.is_empty() && fresh.len() as f64 >= SIGNAL_RATIO * noise {
            self.noise += noise;
            self.round.extend(fresh);
        }
    }

    fn run(&mut self, firsts: &[String], seconds: &[String], sep: &str, wanted: &HashSet<u32>) {
        let decoys = decoys_for(wanted, self.all);
        let found = cross(firsts, seconds, sep, wanted, &decoys);
        self.take(found);
    }

    fn one_round(&mut self, extra_words: &HashMap<String, usize>) {
        let mut count = extra_words.clone();
        count_words(self.known.values().map(String::as_str), 5, &mut count);
        let mut ranked = most_used(count, RARE_WORDS);
        let rare: Vec<String> = ranked.split_off(ranked.len().min(WORDS));
        let rare_lower: Vec<String> = rare.iter().map(|w| w.to_ascii_lowercase()).collect();
        let words = ranked;
        let lower: Vec<String> = words.iter().map(|w| w.to_ascii_lowercase()).collect();
        let upper: Vec<String> = words.iter().map(|w| w.to_ascii_uppercase()).collect();
        let mut names: Vec<String> = self.known.values().filter(|n| (3..=40).contains(&n.len())).cloned().collect();
        names.sort_unstable();
        names.dedup();
        let mut capped: Vec<String> = names.iter().map(|n| capitalise_first(n)).collect();
        capped.sort_unstable();
        capped.dedup();

        // two words run together, with and without a capital first
        let any = self.open(self.all.iter());
        self.run(&words, &words, "", &any);
        self.run(&lower, &words, "", &any);
        // a rare word next to a common one
        let common = &words[..words.len().min(800)];
        let any = self.open(self.all.iter());
        self.run(common, &rare, "", &any);
        self.run(&rare, common, "", &any);
        self.run(&rare_lower, common, "", &any);
        self.run(&lower[..common.len()], &rare, "", &any);
        // three of the most used words
        let top = &words[..words.len().min(200)];
        let two: Vec<String> = top.iter().flat_map(|a| top.iter().map(move |b| format!("{a}{b}"))).collect();
        let any = self.open(self.all.iter());
        self.run(&two, top, "", &any);
        // a known name with one more word on either side, only common words keep chance matches rare
        let any = self.open(self.all.iter());
        self.run(&names, &words[..words.len().min(500)], "", &any);
        self.run(&words[..words.len().min(800)], &capped, "", &any);
        self.run(&lower[..lower.len().min(300)], &capped, "", &any);

        // a type prefix only counts when the field really holds that type
        for &(prefix, allowed) in TYPED_PREFIXES {
            let wanted: HashSet<u32> = self.targets.fields.iter().filter(|(_, &k)| k & allowed != 0 && k & !(allowed | 1 << 5) == 0).map(|(h, _)| *h).collect();
            let wanted = self.open(wanted.iter());
            if wanted.is_empty() {
                continue;
            }
            let decoys = decoys_for(&wanted, self.all);
            let firsts: Vec<String> = words.iter().map(|w| format!("{prefix}{w}")).collect();
            let (mut found, mut misses) = cross(&firsts, &words, "", &wanted, &decoys);
            for part in [sift(firsts.iter().cloned(), &wanted, &decoys), sift(capped.iter().map(|n| format!("{prefix}{n}")), &wanted, &decoys)] {
                misses += part.1;
                for (h, name) in part.0 {
                    keep(&mut found, h, name);
                }
            }
            self.take((found, misses));
        }

        // struct names: the leading words several known structs share, then one or two more words
        let struct_names: Vec<String> = self.targets.structs.iter().filter_map(|h| self.known.get(h)).cloned().collect();
        let mut lead: HashMap<String, usize> = HashMap::new();
        for name in &struct_names {
            for (_, end) in word_spans(name).into_iter().take(3) {
                if end < name.len() && end <= 24 {
                    *lead.entry(name[..end].to_string()).or_default() += 1;
                }
            }
        }
        let mut leads: Vec<(String, usize)> = lead.into_iter().filter(|l| l.1 >= 3).collect();
        leads.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        leads.truncate(120);
        let wanted = self.open(self.targets.structs.iter());
        if !wanted.is_empty() {
            let decoys = decoys_for(&wanted, self.all);
            for (prefix, _) in &leads {
                // each lead is judged on its own so a few productive ones are not drowned by the rest
                let firsts: Vec<String> = words.iter().map(|w| format!("{prefix}{w}")).collect();
                let one = sift(firsts.iter().cloned(), &wanted, &decoys);
                self.take(one);
                let two = cross(&firsts, &words[..words.len().min(800)], "", &wanted, &decoys);
                self.take(two);
            }
        }
        // nested types are named Outer__Inner
        let outers: Vec<String> = struct_names.iter().map(|n| format!("{n}__")).collect();
        let any = self.open(self.all.iter());
        self.run(&outers, &words, "", &any);

        // enum values without a prefix: one or two capital words joined by an underscore
        let values: HashSet<u32> = self.targets.enums.values().flatten().copied().collect();
        let wanted = self.open(values.iter());
        self.run(&upper, &upper, "_", &wanted);
        // values of one enum share a prefix, known values give it away
        let top_upper: Vec<String> = upper.iter().take(150).cloned().collect();
        let upper_pairs: Vec<String> = top_upper.iter().flat_map(|a| top_upper.iter().map(move |b| format!("{a}_{b}"))).collect();
        let mut found = HashMap::new();
        let mut misses = 0;
        for values in self.targets.enums.values() {
            let mut prefixes: HashMap<String, usize> = HashMap::new();
            for name in values.iter().filter_map(|h| self.known.get(h)) {
                for (i, _) in name.match_indices('_') {
                    *prefixes.entry(name[..=i].to_string()).or_default() += 1;
                }
            }
            let wanted = self.open(values.iter());
            if prefixes.is_empty() || wanted.is_empty() {
                continue;
            }
            let decoys = decoys_for(&wanted, self.all);
            let mut ranked: Vec<(String, usize)> = prefixes.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())).then_with(|| a.0.cmp(&b.0)));
            for (prefix, _) in ranked.into_iter().take(3) {
                let firsts: Vec<String> = upper.iter().map(|w| format!("{prefix}{w}")).collect();
                let short: Vec<String> = top_upper.iter().map(|w| format!("{prefix}{w}")).collect();
                for part in [sift(firsts.iter().cloned(), &wanted, &decoys), cross(&firsts, &upper, "_", &wanted, &decoys), cross(&short, &upper_pairs, "_", &wanted, &decoys)] {
                    misses += part.1;
                    for (h, name) in part.0 {
                        keep(&mut found, h, name);
                    }
                }
            }
        }
        self.take((found, misses));

        // twins, plurals and common suffixes of known names
        let any = self.open(self.all.iter());
        let decoys = decoys_for(&any, self.all);
        let mut near = Vec::new();
        for name in &names {
            relatives(name, &mut near);
        }
        self.take(sift(near.into_iter(), &any, &decoys));

        // inside one struct the field names reuse the words of their siblings
        let mut families = Vec::new();
        for (owner, members) in &self.targets.members {
            let wanted = self.open(members.iter());
            if wanted.is_empty() {
                continue;
            }
            let mut count = HashMap::new();
            count_words(members.iter().chain(std::iter::once(owner)).filter_map(|h| self.known.get(h)).map(String::as_str), 1, &mut count);
            let local = most_used(count, 40);
            if !local.is_empty() {
                let decoys = decoys_for(&wanted, self.all);
                families.push(Family { wanted, decoys, local });
            }
        }
        let common = &words[..words.len().min(1500)];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let chunk = families.len().div_ceil(threads).max(1);
        let fields = &self.targets.fields;
        let mut found = HashMap::new();
        let mut misses = 0;
        std::thread::scope(|scope| {
            let handles: Vec<_> = families
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        let mut names = Vec::new();
                        let mut decoy_hits = 0;
                        for family in part {
                            let (n, d) = family.search(common, 400, fields);
                            names.extend(n);
                            decoy_hits += d;
                        }
                        (names, decoy_hits)
                    })
                })
                .collect();
            for handle in handles {
                let (names, decoy_hits) = handle.join().unwrap_or_default();
                misses += decoy_hits;
                for (h, name) in names {
                    keep(&mut found, h, name);
                }
            }
        });
        self.take((found, misses));

        // three words where two of them already occur side by side in a known name
        let mut pairs: HashMap<String, usize> = HashMap::new();
        for name in &names {
            let spans = word_spans(name);
            for w in spans.windows(2) {
                *pairs.entry(format!("{}{}", capitalised(&name[w[0].0..w[0].1]), capitalised(&name[w[1].0..w[1].1]))).or_default() += 1;
            }
        }
        let bigrams = most_used(pairs, 2500);
        let few = &words[..words.len().min(600)];
        let any = self.open(self.all.iter());
        self.run(&bigrams, few, "", &any);
        self.run(few, &bigrams, "", &any);
        self.run(&lower[..few.len().min(300)], &bigrams, "", &any);
        // four words as two such pairs
        let any = self.open(self.all.iter());
        self.run(&bigrams, &bigrams, "", &any);
        // nested types with a two word inner name
        self.run(&outers, &bigrams, "", &any);
    }
}

// type prefixes tried inside a struct, with the value types each is allowed on
const FAMILY_PREFIXES: &[(&str, u128)] = &[("f", FLOATS), ("b", kinds(&[0x01])), ("i", INTS), ("n", INTS), ("u", INTS), ("v", VECTORS), ("s", TEXTS), ("e", kinds(&[0x60, 0x61, 0x62])), ("p", kinds(&[0x07]))];

// the unresolved field names of one struct and the words its known names are made of
struct Family {
    wanted: HashSet<u32>,
    decoys: HashSet<u32>,
    local: Vec<String>,
}

impl Family {
    // looks for field names built from the words of this struct plus at most one common word
    fn search(&self, common: &[String], few: usize, fields: &HashMap<u32, u128>) -> (Vec<(u32, String)>, usize) {
        let mut mask = [0u64; 4];
        for h in self.wanted.iter().chain(&self.decoys) {
            mask[(h >> 30) as usize] |= 1 << ((h >> 24) & 63);
        }
        let mut found = Vec::new();
        let mut misses = 0usize;
        let mut check = |state: u32, parts: &[&str], allowed: u128| {
            let h = finish(state);
            if mask[(h >> 30) as usize] & (1 << ((h >> 24) & 63)) == 0 {
                return;
            }
            if self.wanted.contains(&h) {
                // a type prefix must agree with what the field holds
                if allowed == u128::MAX || fields.get(&h).is_some_and(|k| k & allowed != 0 && k & !(allowed | 1 << 5) == 0) {
                    found.push((h, parts.concat()));
                }
            } else if self.decoys.contains(&h) {
                misses += 1;
            }
        };
        let lower: Vec<String> = self.local.iter().map(|w| w.to_ascii_lowercase()).collect();
        let mut starts: Vec<(String, u128)> = vec![(String::new(), u128::MAX)];
        starts.extend(FAMILY_PREFIXES.iter().map(|&(p, k)| (p.to_string(), k)));
        for (prefix, allowed) in &starts {
            let base = feed(0, prefix.as_bytes());
            let p = prefix.as_str();
            for (ai, a) in self.local.iter().enumerate() {
                // lower case first word only without a prefix
                let both = [a.as_str(), lower[ai].as_str()];
                let firsts: &[&str] = if p.is_empty() { &both } else { &both[..1] };
                for &first in firsts {
                    let sa = feed(base, first.as_bytes());
                    if !p.is_empty() {
                        check(sa, &[p, first], *allowed);
                    }
                    for b in &self.local {
                        let sab = feed(sa, b.as_bytes());
                        check(sab, &[p, first, b], *allowed);
                        for g in &common[..few.min(common.len())] {
                            check(feed(sab, g.as_bytes()), &[p, first, b, g], *allowed);
                        }
                    }
                    for (gi, g) in common.iter().enumerate() {
                        let sag = feed(sa, g.as_bytes());
                        check(sag, &[p, first, g], *allowed);
                        if gi < few {
                            for b in &self.local {
                                check(feed(sag, b.as_bytes()), &[p, first, g, b], *allowed);
                            }
                        }
                    }
                }
            }
            for (gi, g) in common.iter().enumerate() {
                let lowered = g.to_ascii_lowercase();
                let both = [g.as_str(), lowered.as_str()];
                let firsts: &[&str] = if p.is_empty() { &both } else { &both[..1] };
                for &first in firsts {
                    let sg = feed(base, first.as_bytes());
                    for a in &self.local {
                        let sga = feed(sg, a.as_bytes());
                        check(sga, &[p, first, a], *allowed);
                        if gi < few {
                            for b in &self.local {
                                check(feed(sga, b.as_bytes()), &[p, first, a, b], *allowed);
                            }
                        }
                    }
                }
            }
        }
        (found, misses)
    }
}

fn capitalise_first(name: &str) -> String {
    let mut n = name.to_string();
    n[..1].make_ascii_uppercase();
    n
}

// finds the names behind the hashed names: first tokens as written, then names assembled from the words of those
pub fn recover(tokens: &HashSet<String>, targets: &NameTargets) -> Recovered {
    let all = targets.all();
    let mut exact: HashMap<u32, String> = HashMap::new();
    let mut loose: HashMap<u32, String> = HashMap::new();
    let mut vars = Vec::new();
    let listed: Vec<String> = dict().read().map(|d| d.exact.iter().filter(|(h, _)| !d.built.contains(h)).map(|(_, n)| n.clone()).collect()).unwrap_or_default();
    for tok in tokens.iter().chain(&listed) {
        variants(tok, &mut vars);
        for (k, v) in vars.iter().enumerate() {
            let h = joaat_cs(v);
            if all.contains(&h) {
                keep(if k == 0 { &mut exact } else { &mut loose }, h, v.clone());
            }
        }
    }
    for (h, v) in loose {
        exact.entry(h).or_insert(v);
    }

    // words of every listed name and of the dump's own identifiers, the known names weigh in each round
    let mut extra_words = HashMap::new();
    count_words(listed.iter().map(String::as_str), 1, &mut extra_words);
    count_words(tokens.iter().map(String::as_str).filter(|t| t.as_bytes().windows(2).any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase())), 1, &mut extra_words);

    let mut search = Search { targets, all: &all, known: exact.clone(), round: HashMap::new(), noise: 0.0 };
    let mut built: HashMap<u32, String> = HashMap::new();
    for _ in 0..ROUNDS {
        search.one_round(&extra_words);
        if search.round.is_empty() {
            break;
        }
        for (h, name) in std::mem::take(&mut search.round) {
            built.insert(h, name.clone());
            search.known.insert(h, name);
        }
    }

    let mut exact: Vec<(u32, String)> = exact.into_iter().collect();
    let mut built: Vec<(u32, String)> = built.into_iter().collect();
    exact.sort_unstable();
    built.sort_unstable();
    Recovered { exact, built, built_noise: search.noise }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_variants() {
        let mut toks = HashSet::new();
        harvest_tokens(b"<fInitialDrag value=\"1\"/> m_bEnabled rage__fwThing aaaa 12ab x", &mut toks);
        assert!(toks.contains("fInitialDrag") && toks.contains("m_bEnabled") && toks.contains("rage__fwThing"));
        assert!(!toks.contains("aaaa") && !toks.contains("x"));
        let mut v = Vec::new();
        variants("m_bEnabled", &mut v);
        assert!(v.contains(&"bEnabled".to_string()) && v.contains(&"Enabled".to_string()));
    }

    #[test]
    fn recover_finds_exact_names_then_builds_pairs() {
        let mut t = NameTargets::default();
        // written out in the tokens
        for n in ["fMass", "HeadingOffset", "MaxRange", "Brake"] {
            t.fields.insert(joaat_cs(n), 1 << 0x21);
        }
        // only reachable by combining words of the names above
        t.fields.insert(joaat_cs("MaxHeading"), 1 << 0x21);
        t.fields.insert(joaat_cs("rangeOffset"), 1 << 0x21);
        t.fields.insert(joaat_cs("fBrakeRange"), 1 << 0x21);
        // a type prefix on a field of another type must not be accepted
        t.fields.insert(joaat_cs("bHeading"), 1 << 0x21);
        t.fields.insert(joaat_cs("bBrake"), 1 << 0x01);
        let values: HashSet<u32> = ["VFX_MASS_HEADING", "VFX_BRAKE", "VFX_RANGE_MAX"].iter().map(|n| joaat_cs(n)).collect();
        t.enums.insert(7, values);
        let tokens: HashSet<String> = ["fMass", "HeadingOffset", "MaxRange", "Brake", "VFX_MASS_HEADING", "unrelated"].iter().map(|s| s.to_string()).collect();
        let r = recover(&tokens, &t);
        let exact: Vec<&str> = r.exact.iter().map(|x| x.1.as_str()).collect();
        let built: Vec<&str> = r.built.iter().map(|x| x.1.as_str()).collect();
        for want in ["fMass", "HeadingOffset", "MaxRange", "Brake", "VFX_MASS_HEADING"] {
            assert!(exact.contains(&want), "{want} missing from {exact:?}");
        }
        for want in ["MaxHeading", "rangeOffset", "fBrakeRange", "bBrake", "VFX_BRAKE", "VFX_RANGE_MAX"] {
            assert!(built.contains(&want), "{want} missing from {built:?}");
        }
        assert!(!built.contains(&"bHeading"));
        assert!(r.exact.iter().chain(&r.built).all(|(h, n)| joaat_cs(n) == *h));
    }

    #[test]
    fn tables_load_with_their_confidence() {
        let n = load_table("hash\tname\tkind\tmethod\n00000001\tzzTableListed\tfield\tname list\n00000002\tzzTableJudged\tfield\tjudged\n");
        assert_eq!(n, 2);
        assert!(lookup(joaat_cs("zzTableListed")).is_some() && !is_built(joaat_cs("zzTableListed")));
        assert!(is_built(joaat_cs("zzTableJudged")));
    }

    #[test]
    fn add_ignores_junk_and_serves_both_spellings() {
        assert!(!is_name("has space") && !is_name("9lives") && is_name("CVehicleThing"));
        let n = add(["zzTestOnlyName", "zzTestOnlyName", "not a name"]);
        assert_eq!(n, 1);
        assert_eq!(lookup(joaat_cs("zzTestOnlyName")).as_deref(), Some("zzTestOnlyName"));
        assert_eq!(lookup_value(crate::psin::joaat("zztestonlyname")).as_deref(), Some("zzTestOnlyName"));
    }
}

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::OnceLock;

// common rage pso struct and field names for joaat hash resolution
const KNOWN_NAMES: &[&str] = &[
    "Name", "Flags", "Type", "Id", "Hash", "Key", "Value", "Index", "Count", "Version",
    "Enabled", "Active", "Visible", "Hidden", "Priority", "State", "Category", "Group",
    "Tag", "Tags", "Label", "Title", "Text", "Description", "Path", "File", "Asset",
    "Position", "Rotation", "Scale", "Offset", "Size", "Min", "Max", "Bounds", "Radius",
    "Color", "Intensity", "Duration", "Delay", "Speed", "Weight", "Probability", "Chance",
    "Items", "Entries", "Data", "Children", "Parent", "Components", "Attributes",
    "Properties", "Params", "Metadata", "Events", "Actions", "Conditions", "Triggers",
    "Script", "ModelName", "ArchetypeName", "TextureName", "AudioName", "AnimName",
    "x", "y", "z", "w", "r", "g", "b", "a",
    "CMapData", "CMapTypes", "CEntityDef", "CBaseArchetypeDef", "CMloArchetypeDef",
    "CMloRoomDef", "CMloPortalDef", "CMloEntitySet", "CMloInstanceDef", "fwEntityDef",
    "rage__fwPropInstanceListDef",
    "entities", "archetypeName", "position", "rotation", "scaleXY", "scaleZ",
    "rooms", "portals", "entitySets", "attachedObjects", "flags", "lodDist", "childLodDist",
    "parentIndex", "numChildren", "roomIndex", "entitySet",
    "CDataFileMgr__ContentDef", "CDataFileMgr__ContentChangeSetDef",
    "CWeaponInfo", "CVehicleModelInfo", "CPedModelInfo", "CScenarioPoint",
    "CVehiclePointsMeta", "CTuningFile", "CDamageDataManager", "GridAtlasInfo",
    "CMovieSubtitleContainer", "CMovieSubtitleEventArg", "CAnimalAttackBehaviours",
    "CBaitingInfo", "TextBlockName", "Tunables", "VehiclePointFiles", "Behaviours",
    "Contexts", "Condition", "contentSets", "includedSets", "includedChangesets",
    "registrationGroup", "associatedMap", "loadPath", "changesetIncludeFlags",
    "clientSystemName", "tileTextureNamePrefix", "MaxDistance", "DamageCleanlinessDatas",
];

#[derive(Clone, Debug)]
pub struct PsinField {
    pub name_hash: u32,
    pub offset: u32,
    pub type_raw: u32,
    pub ref_hash: u32,
}

#[derive(Clone, Debug)]
pub struct PsinStruct {
    pub hash: u32,
    pub flags: u32,
    pub size: u32,
    pub fields: Vec<PsinField>,
}

#[derive(Clone, Debug)]
pub struct PsinEnum {
    pub hash: u32,
    pub values: Vec<(i64, u32)>,
}

#[derive(Clone, Debug)]
pub struct PsinInstance {
    pub tag: u32,
    pub size: u32,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PsinFixup {
    pub src_off: usize,
    pub dst_inst: usize,
    pub dst_off: usize,
}

#[derive(Clone, Debug)]
pub enum PsinValue {
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    F32(f32),
    U64(u64),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Hash(u32),
    Array {
        elem_type: u32,
        len: u32,
        target_inst: Option<usize>,
    },
    Enum {
        raw: i64,
        name_hash: Option<u32>,
        enum_hash: u32,
    },
    Struct(u32),
    Raw(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct PsinFieldValue {
    pub name_hash: u32,
    pub offset: u32,
    pub type_raw: u32,
    pub ref_hash: u32,
    pub value: PsinValue,
}

#[derive(Clone, Debug)]
pub struct PsinDecodedInstance {
    pub index: usize,
    pub tag: u32,
    pub size: u32,
    pub offset: usize,
    pub fields: Vec<PsinFieldValue>,
    pub raw: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PsinEntityPlacement {
    pub archetype_hash: u32,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub room_index: Option<i32>,
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PsinMloInfo {
    pub name_hash: u32,
    pub room_count: u32,
    pub portal_count: u32,
    pub entity_set_count: u32,
}

#[derive(Clone, Debug)]
pub struct PsinDocument {
    pub total_bytes: usize,
    pub sections: Vec<(String, usize)>,
    pub structs: Vec<PsinStruct>,
    pub enums: Vec<PsinEnum>,
    pub fixups: Vec<PsinFixup>,
    pub instances: Vec<PsinDecodedInstance>,
}

// puts the built in names into the shared name table
pub fn seed_names() {
    crate::names::add(KNOWN_NAMES.iter());
    crate::names::seed();
}

pub fn joaat(s: &str) -> u32 {
    let mut h = 0u32;
    for &b in s.as_bytes() {
        let c = b.to_ascii_lowercase() as u32;
        h = h.wrapping_add(c);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h.wrapping_add(h << 15)
}

fn known_joaat_map() -> &'static HashMap<u32, &'static str> {
    static MAP: OnceLock<HashMap<u32, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::with_capacity(KNOWN_NAMES.len());
        for &n in KNOWN_NAMES {
            m.insert(joaat(n), n);
        }
        m
    })
}

fn fmt_hash(h: u32) -> String {
    if let Some(&name) = known_joaat_map().get(&h) {
        format!("{name} (0x{h:08x})")
    } else if let Some(name) = crate::names::lookup(h) {
        format!("{name} (0x{h:08x})")
    } else {
        format!("0x{h:08x}")
    }
}

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn rd64(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3], d[o + 4], d[o + 5], d[o + 6], d[o + 7]])
}

pub(crate) fn parse_iff_chunks(data: &[u8]) -> Vec<(&[u8], &[u8])> {
    let mut chunks = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let tag = &data[pos..pos + 4];
        let sz = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if sz < 8 || pos + sz > data.len() {
            break;
        }
        chunks.push((tag, &data[pos + 8..pos + sz]));
        pos += sz;
    }
    chunks
}

pub(crate) fn parse_lsch(lsch: &[u8]) -> Vec<PsinStruct> {
    let mut off = 0usize;
    let mut hdrs = Vec::new();
    while off + 32 <= lsch.len() && lsch[off + 16..off + 24] == [0xAA; 8] {
        let hash = rd32(lsch, off);
        let flags = rd32(lsch, off + 8);
        let size = rd32(lsch, off + 24);
        let fcnt = (rd32(lsch, off + 28) >> 16) as usize;
        hdrs.push((hash, flags, size, fcnt));
        off += 32;
    }
    let mut structs = Vec::with_capacity(hdrs.len());
    for (hash, flags, size, fcnt) in hdrs {
        // field count is a file value never reserve more than the remaining descriptors
        let mut fields = Vec::with_capacity(fcnt.min(lsch.len().saturating_sub(off) / 16));
        for _ in 0..fcnt {
            if off + 16 > lsch.len() {
                break;
            }
            fields.push(PsinField {
                name_hash: rd32(lsch, off),
                offset: rd32(lsch, off + 4),
                type_raw: rd32(lsch, off + 8),
                ref_hash: rd32(lsch, off + 12),
            });
            off += 16;
        }
        structs.push(PsinStruct { hash, flags, size, fields });
    }
    structs
}

pub(crate) fn parse_lenm(lenm: &[u8]) -> Vec<PsinEnum> {
    let mut enums = Vec::new();
    let mut off = 0usize;
    while off + 24 <= lenm.len() && lenm[off + 8..off + 16] == [0xEE; 8] {
        let hash = rd32(lenm, off);
        let cnt = rd32(lenm, off + 16) as usize;
        off += 24;
        let mut values = Vec::with_capacity(cnt.min(4096));
        for _ in 0..cnt {
            if off + 16 > lenm.len() {
                break;
            }
            let val = rd64(lenm, off) as i64;
            let nh = rd32(lenm, off + 8);
            values.push((val, nh));
            off += 16;
        }
        enums.push(PsinEnum { hash, values });
    }
    enums
}

fn parse_lrsc(lrsc: &[u8], inst_count: usize, payload_len: usize) -> Vec<PsinFixup> {
    let mut fixups = Vec::new();
    // rage9 lrsc supports 16-byte (src_off, dst_inst, dst_off, pad) or 8-byte (src_off, dst_inst) records
    if lrsc.len() >= 16 && lrsc.len().is_multiple_of(16) {
        let mut off = 0usize;
        while off + 16 <= lrsc.len() {
            let src_off = rd32(lrsc, off) as usize;
            let dst_inst = rd32(lrsc, off + 4) as usize;
            let dst_off = rd32(lrsc, off + 8) as usize;
            if src_off < payload_len && dst_inst < inst_count {
                fixups.push(PsinFixup { src_off, dst_inst, dst_off });
            }
            off += 16;
        }
    }
    if fixups.is_empty() && lrsc.len() >= 8 && lrsc.len().is_multiple_of(8) {
        let mut off = 0usize;
        while off + 8 <= lrsc.len() {
            let src_off = rd32(lrsc, off) as usize;
            let dst_inst = rd32(lrsc, off + 4) as usize;
            if src_off < payload_len && dst_inst < inst_count {
                fixups.push(PsinFixup { src_off, dst_inst, dst_off: 0 });
            }
            off += 8;
        }
    }
    fixups
}

fn decode_field_val(
    block: &[u8],
    inst_offset: usize,
    f: &PsinField,
    enum_map: &HashMap<u32, &PsinEnum>,
    fixup_map: &HashMap<usize, usize>,
) -> PsinValue {
    let o = f.offset as usize;
    let kind = f.type_raw & 0xFF;
    match kind {
        0x01 if o < block.len() => PsinValue::Bool(block[o] != 0),
        0x02 | 0x10 if o < block.len() => PsinValue::U8(block[o]),
        0x11 | 0x12 if o + 2 <= block.len() => PsinValue::U16(u16::from_le_bytes([block[o], block[o + 1]])),
        0x13..=0x16 if o + 4 <= block.len() => {
            let v = rd32(block, o);
            let fv = f32::from_bits(v);
            if fv.is_finite() && fv.abs() > 1e-6 && fv.abs() < 1e7 && v >= 0x3800_0000 {
                PsinValue::F32(fv)
            } else {
                PsinValue::U32(v)
            }
        }
        0x17 | 0x18 if o + 8 <= block.len() => PsinValue::U64(rd64(block, o)),
        0x20..=0x22 if o + 12 <= block.len() => PsinValue::Vec3([
            f32::from_bits(rd32(block, o)),
            f32::from_bits(rd32(block, o + 4)),
            f32::from_bits(rd32(block, o + 8)),
        ]),
        0x23 | 0x24 if o + 16 <= block.len() => PsinValue::Vec4([
            f32::from_bits(rd32(block, o)),
            f32::from_bits(rd32(block, o + 4)),
            f32::from_bits(rd32(block, o + 8)),
            f32::from_bits(rd32(block, o + 12)),
        ]),
        0x44 | 0x48 | 0x4a if o + 4 <= block.len() => PsinValue::Hash(rd32(block, o)),
        0x52 | 0x5c if o + 12 <= block.len() => {
            let cnt = rd32(block, o + 8);
            let ptr_lo = rd32(block, o) as usize;
            let abs_off = inst_offset + o;
            let target_inst = fixup_map
                .get(&abs_off)
                .copied()
                .or_else(|| (ptr_lo != 0 && (ptr_lo >> 24) == 0).then_some(ptr_lo));
            PsinValue::Array {
                elem_type: f.ref_hash,
                len: cnt,
                target_inst,
            }
        }
        0x60 if o < block.len() => {
            let v = block[o] as i64;
            let name_hash = enum_map
                .get(&f.ref_hash)
                .and_then(|en| en.values.iter().find(|(ev, _)| *ev == v).map(|(_, nh)| *nh));
            PsinValue::Enum {
                raw: v,
                name_hash,
                enum_hash: f.ref_hash,
            }
        }
        0x05 => PsinValue::Struct(f.ref_hash),
        _ => {
            let end = (o + 8).min(block.len());
            PsinValue::Raw(block.get(o..end).unwrap_or(&[]).to_vec())
        }
    }
}

fn fmt_psin_value(v: &PsinValue, type_raw: u32) -> String {
    match v {
        PsinValue::Bool(b) => format!("bool({b})"),
        PsinValue::U8(x) => format!("u8({x})"),
        PsinValue::U16(x) => format!("u16({x})"),
        PsinValue::U32(x) => format!("{x} (0x{x:08x})"),
        PsinValue::F32(f) => {
            let bits = f.to_bits();
            format!("f32({f:.4}) / 0x{bits:08x}")
        }
        PsinValue::U64(x) => format!("u64(0x{x:016x})"),
        PsinValue::Vec3(a) => format!("vec3({:.3}, {:.3}, {:.3})", a[0], a[1], a[2]),
        PsinValue::Vec4(a) => format!("vec4({:.3}, {:.3}, {:.3}, {:.3})", a[0], a[1], a[2], a[3]),
        PsinValue::Hash(h) => format!("hash({})", fmt_hash(*h)),
        PsinValue::Array { len, target_inst, .. } => match target_inst {
            Some(ti) => format!("array(len={len} -> inst[{ti}])"),
            None => format!("array(len={len})"),
        },
        PsinValue::Enum { raw, name_hash: Some(nh), .. } => format!("enum({raw} = {})", fmt_hash(*nh)),
        PsinValue::Enum { raw, name_hash: None, enum_hash } => format!("enum({raw} @ {})", fmt_hash(*enum_hash)),
        PsinValue::Struct(h) => format!("struct({})", fmt_hash(*h)),
        PsinValue::Raw(bytes) => {
            let kind = type_raw & 0xFF;
            if bytes.is_empty() {
                format!("type=0x{kind:02x}")
            } else {
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                format!("type=0x{kind:02x} [{hex}]")
            }
        }
    }
}

pub fn parse_psin_doc(data: &[u8]) -> Result<PsinDocument, String> {
    if data.len() < 16 || &data[..4] != b"PSIN" {
        return Err("Not a valid PSIN container".into());
    }
    let chunks = parse_iff_chunks(data);
    let psin_body = chunks.iter().find(|(t, _)| *t == b"PSIN").map(|(_, b)| *b).unwrap_or(&[]);
    let lsch_body = chunks.iter().find(|(t, _)| *t == b"LSCH").map(|(_, b)| *b).unwrap_or(&[]);
    let lenm_body = chunks.iter().find(|(t, _)| *t == b"LENM").map(|(_, b)| *b).unwrap_or(&[]);
    let lrsc_body = chunks.iter().find(|(t, _)| *t == b"LRSC").map(|(_, b)| *b).unwrap_or(&[]);

    let mut i = 0usize;
    let mut raw_insts = Vec::new();
    while i + 8 <= psin_body.len() {
        if &psin_body[i..i + 8] == b"pppppppp" {
            i += 8;
            break;
        }
        if i + 16 > psin_body.len() {
            break;
        }
        raw_insts.push((rd32(psin_body, i), rd32(psin_body, i + 4)));
        i += 16;
    }
    let inst_payload = &psin_body[i..];
    let mut cursor = 0usize;
    let mut insts = Vec::with_capacity(raw_insts.len());
    for (tag, size) in raw_insts {
        insts.push(PsinInstance { tag, size, offset: cursor });
        cursor = cursor.saturating_add((size as usize + 15) & !15);
    }

    let structs = parse_lsch(lsch_body);
    let enums = parse_lenm(lenm_body);
    let fixups = parse_lrsc(lrsc_body, insts.len(), inst_payload.len());
    let fixup_map: HashMap<usize, usize> = fixups.iter().map(|fx| (fx.src_off, fx.dst_inst)).collect();
    let struct_map: HashMap<u32, &PsinStruct> = structs.iter().map(|s| (s.hash, s)).collect();
    let enum_map: HashMap<u32, &PsinEnum> = enums.iter().map(|e| (e.hash, e)).collect();

    let mut decoded = Vec::with_capacity(insts.len());
    for (idx, inst) in insts.iter().enumerate() {
        let end = (inst.offset + inst.size as usize).min(inst_payload.len());
        let block = inst_payload.get(inst.offset..end).unwrap_or(&[]);
        let mut fields = Vec::new();
        if let Some(st) = struct_map.get(&inst.tag) {
            for f in st.fields.iter().filter(|f| f.name_hash != 0x0000_0100) {
                fields.push(PsinFieldValue {
                    name_hash: f.name_hash,
                    offset: f.offset,
                    type_raw: f.type_raw,
                    ref_hash: f.ref_hash,
                    value: decode_field_val(block, inst.offset, f, &enum_map, &fixup_map),
                });
            }
        }
        decoded.push(PsinDecodedInstance {
            index: idx,
            tag: inst.tag,
            size: inst.size,
            offset: inst.offset,
            fields,
            raw: block.to_vec(),
        });
    }

    Ok(PsinDocument {
        total_bytes: data.len(),
        sections: chunks
            .iter()
            .map(|(t, b)| (String::from_utf8_lossy(t).into_owned(), b.len() + 8))
            .collect(),
        structs,
        enums,
        fixups,
        instances: decoded,
    })
}

impl PsinDocument {
    fn placement_from_instance(inst: &PsinDecodedInstance) -> Option<PsinEntityPlacement> {
        let mut arch = None;
        let mut pos = None;
        let mut rot = [0.0f32, 0.0, 0.0, 1.0];
        let mut scale = [1.0f32; 3];
        let mut room_index = None;
        let mut flags = 0u32;
        let h_arch = joaat("archetypeName");
        let h_arch2 = joaat("ArchetypeName");
        let h_pos = joaat("position");
        let h_pos2 = joaat("Position");
        let h_rot = joaat("rotation");
        let h_rot2 = joaat("Rotation");
        let h_sxy = joaat("scaleXY");
        let h_sz = joaat("scaleZ");
        let h_room = joaat("roomIndex");
        let h_flags = joaat("flags");
        let h_flags2 = joaat("Flags");
        for f in &inst.fields {
            if f.name_hash == h_arch || f.name_hash == h_arch2 {
                if let PsinValue::Hash(h) | PsinValue::U32(h) = f.value { arch = Some(h); }
            } else if f.name_hash == h_pos || f.name_hash == h_pos2 {
                if let PsinValue::Vec3(p) = f.value { pos = Some(p); }
            } else if f.name_hash == h_rot || f.name_hash == h_rot2 {
                if let PsinValue::Vec4(q) = f.value { rot = q; }
            } else if f.name_hash == h_sxy {
                if let PsinValue::F32(s) = f.value { scale[0] = s; scale[1] = s; }
            } else if f.name_hash == h_sz {
                if let PsinValue::F32(s) = f.value { scale[2] = s; }
            } else if f.name_hash == h_room {
                if let PsinValue::U32(r) = f.value { room_index = Some(r as i32); }
            } else if f.name_hash == h_flags || f.name_hash == h_flags2 {
                if let PsinValue::U32(fl) = f.value { flags = fl; }
            }
        }
        Some(PsinEntityPlacement {
            archetype_hash: arch?,
            position: pos.unwrap_or([0.0; 3]),
            rotation: rot,
            scale,
            room_index,
            flags,
        })
    }

    pub fn extract_placements(&self) -> Vec<PsinEntityPlacement> {
        let h_map = joaat("CMapData");
        let h_ent = joaat("CEntityDef");
        let h_ents_field = joaat("entities");
        let mut out = Vec::new();
        let mut seen = vec![false; self.instances.len()];
        let mut visited_insts = Vec::new();

        for inst in &self.instances {
            if inst.tag == h_map {
                for f in &inst.fields {
                    if f.name_hash == h_ents_field {
                        if let PsinValue::Array { len, target_inst: Some(ti), .. } = f.value {
                            visited_insts.push((ti, len as usize));
                        }
                    }
                }
            }
        }

        for (start_idx, len) in visited_insts {
            for idx in start_idx..(start_idx + len).min(self.instances.len()) {
                let inst = &self.instances[idx];
                if (inst.tag == h_ent || inst.tag == 0) && !seen[idx] {
                    if let Some(p) = Self::placement_from_instance(inst) {
                        seen[idx] = true;
                        out.push(p);
                    }
                }
            }
        }

        for (idx, inst) in self.instances.iter().enumerate() {
            if inst.tag == h_ent && !seen[idx] {
                if let Some(p) = Self::placement_from_instance(inst) {
                    seen[idx] = true;
                    out.push(p);
                }
            }
        }
        out
    }

    pub fn extract_mlos(&self) -> Vec<PsinMloInfo> {
        let h_mlo = joaat("CMloArchetypeDef");
        let h_name = joaat("Name");
        let h_rooms = joaat("rooms");
        let h_portals = joaat("portals");
        let h_sets = joaat("entitySets");
        let mut out = Vec::new();
        for inst in &self.instances {
            if inst.tag != h_mlo {
                continue;
            }
            let mut info = PsinMloInfo { name_hash: inst.tag, room_count: 0, portal_count: 0, entity_set_count: 0 };
            for f in &inst.fields {
                if f.name_hash == h_name {
                    if let PsinValue::Hash(h) | PsinValue::U32(h) = f.value { info.name_hash = h; }
                } else if f.name_hash == h_rooms {
                    if let PsinValue::Array { len, .. } = f.value { info.room_count = len; }
                } else if f.name_hash == h_portals {
                    if let PsinValue::Array { len, .. } = f.value { info.portal_count = len; }
                } else if f.name_hash == h_sets {
                    if let PsinValue::Array { len, .. } = f.value { info.entity_set_count = len; }
                }
            }
            out.push(info);
        }
        out
    }

    pub fn export_placements_csv(&self, models: Option<&HashMap<u32, u64>>, tsv: bool) -> String {
        let sep = if tsv { '\t' } else { ',' };
        let mut out = format!("archetype_hash{sep}drawable_offset{sep}pos_x{sep}pos_y{sep}pos_z{sep}rot_x{sep}rot_y{sep}rot_z{sep}rot_w{sep}scale_x{sep}scale_y{sep}scale_z{sep}room_index{sep}flags\n");
        for p in self.extract_placements() {
            let off_s = models
                .and_then(|m| m.get(&p.archetype_hash))
                .map(|o| format!("0x{o:x}"))
                .unwrap_or_default();
            let room_s = p.room_index.map(|r| r.to_string()).unwrap_or_default();
            let _ = writeln!(
                out,
                "0x{:08x}{sep}{off_s}{sep}{:.4}{sep}{:.4}{sep}{:.4}{sep}{:.5}{sep}{:.5}{sep}{:.5}{sep}{:.5}{sep}{:.3}{sep}{:.3}{sep}{:.3}{sep}{room_s}{sep}0x{:x}",
                p.archetype_hash, p.position[0], p.position[1], p.position[2],
                p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3],
                p.scale[0], p.scale[1], p.scale[2], p.flags
            );
        }
        out
    }
}

// decode rage 9 binary pso (psin + lsch + lenm + lsns + lrsc + chks)
pub fn parse_psin(data: &[u8]) -> String {
    let doc = match parse_psin_doc(data) {
        Ok(d) => d,
        Err(e) => return e,
    };

    let mut out = String::with_capacity(2048);
    let _ = writeln!(
        out,
        "=== RAGE 9 PSIN (Binary PSO Decoder) ===\nTotal: {} B | Chunks: {} | Structs: {} | Enums: {} | Instances: {} | Fixups: {}",
        doc.total_bytes,
        doc.sections.len(),
        doc.structs.len(),
        doc.enums.len(),
        doc.instances.len(),
        doc.fixups.len()
    );
    let _ = write!(out, "Sections:");
    for (tag, sz) in &doc.sections {
        let _ = write!(out, " {tag}({sz}B)");
    }
    out.push_str("\n\n");

    let placements = doc.extract_placements();
    let mlos = doc.extract_mlos();
    if !placements.is_empty() || !mlos.is_empty() {
        let _ = writeln!(out, "Scene Summary: {} entity placements, {} MLO interiors\n", placements.len(), mlos.len());
    }

    let _ = writeln!(out, "--- Decoded Instances ---");
    for inst in doc.instances.iter().take(64) {
        if !inst.fields.is_empty() {
            let _ = writeln!(out, "[{}] <{} size=0x{:x} fields={}>", inst.index, fmt_hash(inst.tag), inst.size, inst.fields.len());
            for f in inst.fields.iter().take(48) {
                let _ = writeln!(out, "  +0x{:03x} {} = {}", f.offset, fmt_hash(f.name_hash), fmt_psin_value(&f.value, f.type_raw));
            }
        } else {
            let ascii_preview: String = inst.raw.iter().take(64).map(|&b| if (32..127).contains(&b) { b as char } else { '.' }).collect();
            let _ = writeln!(out, "[{}] <heap/blob 0x{:08x} size=0x{:x}> \"{ascii_preview}\"", inst.index, inst.tag, inst.size);
        }
    }

    if !doc.enums.is_empty() {
        let _ = writeln!(out, "\n--- LENM Enums ({}) ---", doc.enums.len());
        for en in doc.enums.iter().take(24) {
            let _ = writeln!(out, "enum {} ({} values):", fmt_hash(en.hash), en.values.len());
            for (val, nh) in en.values.iter().take(16) {
                let _ = writeln!(out, "  {val} = {}", fmt_hash(*nh));
            }
        }
    }

    let _ = writeln!(out, "\n--- LSCH Schema ({} structs) ---", doc.structs.len());
    for st in doc.structs.iter().take(24) {
        let _ = writeln!(out, "struct {} size=0x{:x} flags=0x{:x} fields={}", fmt_hash(st.hash), st.size, st.flags, st.fields.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_psin_with_lsch_and_lenm() {
        let mut buf = Vec::new();
        // psin chunk: header + 1 instance (hash 0x11223344, size 16) + pppppppp + 16B payload
        let mut psin = Vec::new();
        psin.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        psin.extend_from_slice(&16u32.to_le_bytes());
        psin.extend_from_slice(&[0xAA; 8]);
        psin.extend_from_slice(b"pppppppp");
        psin.extend_from_slice(&42u32.to_le_bytes()); // offset 0: u32
        psin.push(1); // offset 4: bool
        psin.push(2); // offset 5: enum
        psin.extend_from_slice(&[0u8; 10]);

        buf.extend_from_slice(b"PSIN");
        buf.extend_from_slice(&((8 + psin.len()) as u32).to_be_bytes());
        buf.extend_from_slice(&psin);

        // lsch chunk: 1 struct with 3 fields
        let mut lsch = Vec::new();
        lsch.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&0x0300u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&[0xAA; 8]);
        lsch.extend_from_slice(&16u32.to_le_bytes());
        lsch.extend_from_slice(&(3u32 << 16).to_le_bytes());
        // field 0: Count @0x00 u32 (0x15)
        lsch.extend_from_slice(&joaat("Count").to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&0x15u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        // field 1: Enabled @0x04 bool (0x01)
        lsch.extend_from_slice(&joaat("Enabled").to_le_bytes());
        lsch.extend_from_slice(&4u32.to_le_bytes());
        lsch.extend_from_slice(&0x01u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        // field 2: State @0x05 enum (0x60) -> 0x55667788
        lsch.extend_from_slice(&joaat("State").to_le_bytes());
        lsch.extend_from_slice(&5u32.to_le_bytes());
        lsch.extend_from_slice(&0x60u32.to_le_bytes());
        lsch.extend_from_slice(&0x5566_7788u32.to_le_bytes());

        buf.extend_from_slice(b"LSCH");
        buf.extend_from_slice(&((8 + lsch.len()) as u32).to_be_bytes());
        buf.extend_from_slice(&lsch);

        // lenm chunk: enum 0x55667788 with 1 value (2 -> Active)
        let mut lenm = Vec::new();
        lenm.extend_from_slice(&0x5566_7788u32.to_le_bytes());
        lenm.extend_from_slice(&0u32.to_le_bytes());
        lenm.extend_from_slice(&[0xEE; 8]);
        lenm.extend_from_slice(&1u32.to_le_bytes());
        lenm.extend_from_slice(&0u32.to_le_bytes());
        lenm.extend_from_slice(&2i64.to_le_bytes());
        lenm.extend_from_slice(&joaat("Active").to_le_bytes());
        lenm.extend_from_slice(&0u32.to_le_bytes());

        buf.extend_from_slice(b"LENM");
        buf.extend_from_slice(&((8 + lenm.len()) as u32).to_be_bytes());
        buf.extend_from_slice(&lenm);

        let out = parse_psin(&buf);
        assert!(out.contains("Structs: 1 | Enums: 1 | Instances: 1"));
        assert!(out.contains("Count"));
        assert!(out.contains("42 (0x0000002a)"));
        assert!(out.contains("bool(true)"));
        assert!(out.contains("enum(2 = Active"));
    }

    #[test]
    fn resolves_lrsc_arrays_and_exports_placements_csv() {
        let mut buf = Vec::new();
        let mut psin = Vec::new();
        let h_map = joaat("CMapData");
        let h_ent = joaat("CEntityDef");
        // inst 0: CMapData (16B), inst 1: CEntityDef (32B)
        psin.extend_from_slice(&h_map.to_le_bytes());
        psin.extend_from_slice(&16u32.to_le_bytes());
        psin.extend_from_slice(&[0u8; 8]);
        psin.extend_from_slice(&h_ent.to_le_bytes());
        psin.extend_from_slice(&32u32.to_le_bytes());
        psin.extend_from_slice(&[0u8; 8]);
        psin.extend_from_slice(b"pppppppp");
        // payload 0 (16B): array(len=1) at +0
        psin.extend_from_slice(&0u64.to_le_bytes());
        psin.extend_from_slice(&1u32.to_le_bytes());
        psin.extend_from_slice(&0u32.to_le_bytes());
        // payload 1 (32B): archetypeName @0, position @4 (10,20,30), rotation @16 (0,0,0,1)
        psin.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        psin.extend_from_slice(&10.0f32.to_le_bytes());
        psin.extend_from_slice(&20.0f32.to_le_bytes());
        psin.extend_from_slice(&30.0f32.to_le_bytes());
        psin.extend_from_slice(&0.0f32.to_le_bytes());
        psin.extend_from_slice(&0.0f32.to_le_bytes());
        psin.extend_from_slice(&0.0f32.to_le_bytes());
        psin.extend_from_slice(&1.0f32.to_le_bytes());

        buf.extend_from_slice(b"PSIN");
        buf.extend_from_slice(&((8 + psin.len()) as u32).to_be_bytes());
        buf.extend_from_slice(&psin);

        // LSCH: CMapData (1 field: entities @0 type 0x52), CEntityDef (3 fields)
        let mut lsch = Vec::new();
        for (h, sz, fc) in [(h_map, 16u32, 1u32), (h_ent, 32u32, 3u32)] {
            lsch.extend_from_slice(&h.to_le_bytes());
            lsch.extend_from_slice(&[0u8; 12]);
            lsch.extend_from_slice(&[0xAA; 8]);
            lsch.extend_from_slice(&sz.to_le_bytes());
            lsch.extend_from_slice(&(fc << 16).to_le_bytes());
        }
        // CMapData.entities
        lsch.extend_from_slice(&joaat("entities").to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&0x52u32.to_le_bytes());
        lsch.extend_from_slice(&h_ent.to_le_bytes());
        // CEntityDef fields
        lsch.extend_from_slice(&joaat("archetypeName").to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&0x44u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&joaat("position").to_le_bytes());
        lsch.extend_from_slice(&4u32.to_le_bytes());
        lsch.extend_from_slice(&0x20u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());
        lsch.extend_from_slice(&joaat("rotation").to_le_bytes());
        lsch.extend_from_slice(&16u32.to_le_bytes());
        lsch.extend_from_slice(&0x23u32.to_le_bytes());
        lsch.extend_from_slice(&0u32.to_le_bytes());

        buf.extend_from_slice(b"LSCH");
        buf.extend_from_slice(&((8 + lsch.len()) as u32).to_be_bytes());
        buf.extend_from_slice(&lsch);

        // LRSC: fixup src_off=0 -> dst_inst=1
        let mut lrsc = Vec::new();
        lrsc.extend_from_slice(&0u32.to_le_bytes());
        lrsc.extend_from_slice(&1u32.to_le_bytes());
        lrsc.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(b"LRSC");
        buf.extend_from_slice(&((8 + lrsc.len()) as u32).to_be_bytes());
        buf.extend_from_slice(&lrsc);

        let doc = parse_psin_doc(&buf).unwrap();
        assert_eq!(doc.fixups.len(), 1);
        let placements = doc.extract_placements();
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].archetype_hash, 0xDEAD_BEEF);
        assert_eq!(placements[0].position, [10.0, 20.0, 30.0]);
        let csv = doc.export_placements_csv(None, false);
        assert!(csv.contains("0xdeadbeef,,10.0000,20.0000,30.0000"));
    }

    #[test]
    fn parses_carved_gta6_psin() {
        let p = std::path::Path::new(r"C:\pulpit\xx\carved_data\0x0000021000_binary.bin");
        if !p.exists() { return; }
        let data = std::fs::read(p).unwrap();
        let txt = parse_psin(&data);
        eprintln!("Carved PSIN dump (first 500 chars):\n{}", &txt[..txt.len().min(500)]);
    }
}

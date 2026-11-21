use super::*;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn descriptor_fixture(page: u32, logical_size: u64) -> Fixture {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "percystar-ttpf-address-{}-{nonce}.ttpf",
        std::process::id()
    ));
    // Only 512 physical bytes. The logical address space is independent of disk size.
    let mut bytes = vec![0u8; 512];
    bytes[..4].copy_from_slice(b"RDS5");
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    // Upper header bits are flags, not part of the 40-bit logical length.
    bytes[8..16].copy_from_slice(&(logical_size | (0xabu64 << 40)).to_le_bytes());
    bytes[16..20].copy_from_slice(&1u32.to_le_bytes());
    bytes[256..260].copy_from_slice(b"ETPF");
    bytes[260..264].copy_from_slice(&634u32.to_le_bytes());
    bytes[264..268].copy_from_slice(&1u32.to_le_bytes());
    bytes[268..272].copy_from_slice(&16u32.to_le_bytes());
    bytes[272..276].copy_from_slice(&(63u32 | (63 << 14)).to_le_bytes());
    bytes[276..280].copy_from_slice(&(71u32 << 3).to_le_bytes()); // 64x64 BC1
                                                                  // Higher descriptor flags must not leak into the 21-bit page origin.
    bytes[280..284].copy_from_slice(&(page | (0x5a0 << 21)).to_le_bytes());
    std::fs::write(&path, bytes).unwrap();
    Fixture(path)
}

#[test]
fn preserves_offsets_below_at_and_above_four_gib() {
    for page in [0x12345u32, 0x0ffffe, 0x100000, 0x100010] {
        let offset = u64::from(page) * 4096;
        let file = descriptor_fixture(page, offset + 8192);
        let (_, descriptors) = scan_ttpf_descriptors(&file.0).unwrap();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].byte_off, offset, "page {page:#x}");
        assert_eq!(
            descriptors[0].alloc_sz, 8192,
            "logical EOF must retain its high bits"
        );
        assert_eq!(
            (descriptors[0].w, descriptors[0].h, descriptors[0].dxgi),
            (64, 64, 71)
        );
    }
}

#[test]
fn original_pool_descriptors_above_four_gib() {
    let Ok(levels) = std::env::var("PERCYSTAR_LEVELS_RPF") else {
        return;
    };
    let levels = PathBuf::from(levels);
    let root = levels.parent().unwrap();
    for (relative, index, expected) in [
        ("data/ttdb/maps/ttdb_maps_2_7.ttpf", 15270, 4_295_573_504u64),
        (
            "data/ttdb/vehicles/ttdb_vehicles_loose_1_1.ttpf",
            3426,
            4_298_395_648u64,
        ),
    ] {
        let (_, descriptors) = scan_ttpf_descriptors(&root.join(relative)).unwrap();
        assert_eq!(descriptors[index].byte_off, expected);
        assert!(descriptors[index].alloc_sz >= 4096);
    }
}

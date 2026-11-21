//! RAGE9 attribute descriptors have a 7-bit format and a byte offset above it.
//! Slot 0 is position; slot 3 is UV0. Empty slots retain their semantic indices.
pub(super) struct VertexStream {
    pub positions: Vec<[f32; 3]>,
    pub uvs: Option<Vec<[f32; 2]>>,
}

fn word(bytes: &[u8], at: usize) -> Result<u32, &'static str> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .ok_or("Truncated vertex stream")?
            .try_into()
            .unwrap(),
    ))
}
fn vector(bytes: &[u8], at: usize) -> Result<[f32; 3], &'static str> {
    Ok([
        f32::from_bits(word(bytes, at)?),
        f32::from_bits(word(bytes, at + 4)?),
        f32::from_bits(word(bytes, at + 8)?),
    ])
}

pub(super) fn decode(bytes: &[u8]) -> Result<VertexStream, &'static str> {
    if bytes.len() < 64 {
        return Err("Missing vertex attribute descriptors");
    }
    let mut descriptors = [0u32; 16];
    for (i, descriptor) in descriptors.iter_mut().enumerate() {
        *descriptor = word(bytes, i * 4)?;
    }
    let format = descriptors[0] & 127;
    let start = (descriptors[0] >> 7) as usize;
    let stride = match format {
        66 if start >= 128 => 8,
        99 if start >= 64 => 12,
        _ => return Err("Unsupported position stream format or offset"),
    };
    let end = descriptors
        .iter()
        .filter(|&&d| d != 0)
        .map(|d| (d >> 7) as usize)
        .filter(|&offset| offset > start)
        .min()
        .unwrap_or(bytes.len())
        .min(bytes.len());
    let count = end
        .checked_sub(start)
        .ok_or("Position offset exceeds vertex buffer")?
        / stride;
    if !(3..=65536).contains(&count) {
        return Err("Invalid position stream count");
    }

    let mut positions = Vec::with_capacity(count);
    // Raw float3 has no quantization header; bytes at 96 can already be positions.
    let quantization = if format == 66 {
        let origin = vector(bytes, 96)?;
        let scale = vector(bytes, 112)?;
        let axis = word(bytes, 108)?;
        if !origin.iter().all(|v| v.is_finite())
            || !scale.iter().all(|v| v.is_finite())
        {
            return Err("Invalid packed position quantization");
        }
        Some((origin, scale, axis))
    } else {
        None
    };
    for i in 0..count {
        let at = start + i * stride;
        let position = if let Some((origin, scale, axis)) = quantization {
            let q = [
                u32::from_le_bytes([bytes[at + 1], bytes[at + 2], bytes[at + 3], 0]),
                u32::from_le_bytes([bytes[at], bytes[at + 4], bytes[at + 5], 0]),
                u16::from_le_bytes([bytes[at + 6], bytes[at + 7]]) as u32,
            ];
            let p = [
                origin[0] + q[0] as f32 * scale[0],
                origin[1] + q[1] as f32 * scale[1],
                origin[2] + q[2] as f32 * scale[2],
            ];
            match axis {
                0 => [p[2], p[1], p[0]],
                1 => [p[1], p[2], p[0]],
                _ => p,
            }
        } else {
            vector(bytes, at)?
        };
        if !position.iter().all(|v| v.is_finite()) {
            return Err("Non-finite vertex position");
        }
        positions.push(position);
    }

    let uv = descriptors[3];
    let uv_format = uv & 127;
    let uv_start = (uv >> 7) as usize;
    let uv_stride = match uv_format {
        40 => 4,
        66 => 8,
        _ => 0,
    };
    let uvs = if uv_stride > 0
        && uv_start >= 64
        && uv_start
            .checked_add(count * uv_stride)
            .is_some_and(|end| end <= bytes.len())
    {
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let at = uv_start + i * uv_stride;
            values.push(if uv_format == 40 {
                [
                    i16::from_le_bytes([bytes[at], bytes[at + 1]]) as f32 / 8192.0,
                    i16::from_le_bytes([bytes[at + 2], bytes[at + 3]]) as f32 / 8192.0,
                ]
            } else {
                [
                    f32::from_bits(word(bytes, at)?),
                    f32::from_bits(word(bytes, at + 4)?),
                ]
            });
        }
        values
            .iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some(values)
    } else {
        None
    };
    Ok(VertexStream { positions, uvs })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn set_word(bytes: &mut [u8], at: usize, value: u32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn float_fixture(start: usize) -> Vec<u8> {
        let mut bytes = vec![0; start + 36 + 24];
        set_word(&mut bytes, 0, (start as u32) << 7 | 99);
        set_word(&mut bytes, 12, ((start + 36) as u32) << 7 | 66);
        for (i, position) in [[-4.0f32, 0.0, 3.0], [4.0, 0.0, 3.0], [0.0, 5.0, -1.0]]
            .iter()
            .enumerate()
        {
            for (axis, value) in position.iter().enumerate() {
                set_word(&mut bytes, start + i * 12 + axis * 4, value.to_bits());
            }
        }
        for (i, value) in [0.0f32, 0.0, 1.0, 0.0, 0.5, 1.0].iter().enumerate() {
            set_word(&mut bytes, start + 36 + i * 4, value.to_bits());
        }
        bytes
    }
    #[test]
    fn float3_and_float_uvs_keep_empty_attribute_slots() {
        let stream = decode(&float_fixture(96)).unwrap();
        assert_eq!(
            stream.positions,
            [[-4.0, 0.0, 3.0], [4.0, 0.0, 3.0], [0.0, 5.0, -1.0]]
        );
        assert_eq!(stream.uvs.unwrap(), [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]);
    }
    #[test]
    fn float3_does_not_read_quantization_or_require_128_byte_header() {
        let mut compact = float_fixture(64);
        set_word(&mut compact, 12, 0);
        compact.truncate(100);
        assert_eq!(decode(&compact).unwrap().positions.len(), 3);
        let mut bytes = float_fixture(128);
        for at in (96..128).step_by(4) {
            set_word(&mut bytes, at, f32::NAN.to_bits());
        }
        assert_eq!(decode(&bytes).unwrap().positions[2], [0.0, 5.0, -1.0]);
    }
    #[test]
    fn later_attributes_bound_position_data_and_invalid_ranges_are_rejected() {
        let mut bytes = float_fixture(96);
        set_word(&mut bytes, 12, 0);
        set_word(&mut bytes, 12 * 4, (132 << 7) | 17);
        assert_eq!(decode(&bytes).unwrap().positions.len(), 3);
        set_word(&mut bytes, 0, (4096 << 7) | 99);
        assert!(decode(&bytes).is_err());
        let mut bytes = float_fixture(96);
        set_word(&mut bytes, 96 + 12, f32::NAN.to_bits());
        assert!(decode(&bytes).is_err());
        let mut bytes = float_fixture(96);
        set_word(&mut bytes, 0, (96 << 7) | 98);
        assert!(decode(&bytes).is_err());
        assert!(decode(&[0; 63]).is_err());
    }
    #[test]
    fn packed_positions_preserve_axis_order_and_signed_uvs() {
        let mut bytes = vec![0; 128 + 24 + 12];
        set_word(&mut bytes, 0, (128 << 7) | 66);
        set_word(&mut bytes, 12, (152 << 7) | 40);
        for i in 0..3 {
            set_word(&mut bytes, 112 + i * 4, 0.001f32.to_bits());
        }
        bytes[128..136].copy_from_slice(&[0x56, 0x12, 0x34, 0x05, 0x78, 0x09, 0xbc, 0x0a]);
        bytes[152..154].copy_from_slice(&(-4096i16).to_le_bytes());
        let p = [
            0x053412 as f32 * 0.001,
            0x097856 as f32 * 0.001,
            0x0abc as f32 * 0.001,
        ];
        for (axis, expected) in [(2, p), (1, [p[1], p[2], p[0]]), (0, [p[2], p[1], p[0]])] {
            set_word(&mut bytes, 108, axis);
            let stream = decode(&bytes).unwrap();
            assert_eq!(stream.positions[0], expected);
            assert_eq!(stream.uvs.unwrap()[0], [-0.5, 0.0]);
        }
    }

    #[test]
    fn packed_positions_keep_finite_zero_scale_axes() {
        let mut bytes = vec![0; 152];
        set_word(&mut bytes, 0, (128 << 7) | 66);
        set_word(&mut bytes, 108, 2);
        set_word(&mut bytes, 112, 1.0f32.to_bits());
        set_word(&mut bytes, 116, 1.0f32.to_bits());
        bytes[129] = 1;
        bytes[134] = 1;
        assert_eq!(decode(&bytes).unwrap().positions[0], [1.0, 0.0, 0.0]);
        set_word(&mut bytes, 116, f32::NAN.to_bits());
        assert!(decode(&bytes).is_err());
    }
    #[test]
    fn meshlet_append_uses_float_positions_and_rejects_invalid_input_atomically() {
        let vertex = float_fixture(96);
        let mut bytes = vertex.clone();
        let meshlet = bytes.len();
        bytes.resize(meshlet + 38, 0);
        set_word(&mut bytes, meshlet, 0x80000001);
        set_word(&mut bytes, meshlet + 4, 1);
        set_word(&mut bytes, meshlet + 8, 3);
        set_word(&mut bytes, meshlet + 28, (1 << 7) | (2 << 14));
        bytes[meshlet + 34..meshlet + 36].copy_from_slice(&1u16.to_le_bytes());
        bytes[meshlet + 36..meshlet + 38].copy_from_slice(&2u16.to_le_bytes());
        let mut build = super::super::MeshBuild::default();
        super::super::append_meshlet_geom(&bytes, (0, vertex.len(), meshlet, 38), 7, &mut build);
        assert_eq!(
            build.verts,
            [[-4.0, 0.0, 3.0], [4.0, 0.0, 3.0], [0.0, 5.0, -1.0]]
        );
        assert_eq!(build.tris, [[0, 1, 2]]);
        assert_eq!(build.tri_geom, [7]);
        set_word(&mut bytes, 108, f32::NAN.to_bits());
        let mut rejected = super::super::MeshBuild::default();
        super::super::append_meshlet_geom(&bytes, (0, vertex.len(), meshlet, 38), 0, &mut rejected);
        assert!(
            rejected.verts.is_empty() && rejected.tris.is_empty() && rejected.tri_geom.is_empty()
        );
    }
    #[test]
    fn original_float3_resources_keep_native_bounds() {
        let Ok(path) = std::env::var("PERCYSTAR_LEVELS_RPF") else {
            return;
        };
        let path = std::path::Path::new(&path);
        for (offset, triangles) in [
            (18_789_048_336, 16_380),
            (18_787_495_952, 80),
            (31_508_606_992, 10),
        ] {
            let mesh =
                super::super::decode_rage9_model_from_rpf(path, offset, "float3 fixture").unwrap();
            assert_eq!(mesh.tris.len(), triangles, "resource {offset}");
            let mut file = std::fs::File::open(path).unwrap();
            let (system, _, _) = super::super::read_virtual_from_file(&mut file, offset).unwrap();
            for position in &mesh.verts {
                for axis in 0..3 {
                    let min = super::super::rf32(&system, 48 + axis * 4);
                    let max = super::super::rf32(&system, 64 + axis * 4);
                    assert!(position[axis] >= min - 0.05 && position[axis] <= max + 0.05);
                }
            }
        }
    }
}

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use super::{GeomMaterial, MaterialPass, Mesh3D, Texture2D};

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;

const COMPONENT_UNSIGNED_INT: u32 = 5125;
const COMPONENT_FLOAT: u32 = 5126;

/// Writes `mesh` as a glTF 2.0 Binary (`.glb`) file to `path`.
pub fn write_glb(mesh: &Mesh3D, path: &Path) -> std::io::Result<()> {
    let bytes = build_glb_bytes(mesh)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Builds an in-memory glTF 2.0 Binary (`.glb`) byte buffer from `mesh`.
pub fn build_glb_bytes(mesh: &Mesh3D) -> Result<Vec<u8>, String> {
    if mesh.verts.is_empty() || mesh.tris.is_empty() {
        return Err("Mesh has no geometry to export".into());
    }

    let groups: Vec<(&str, std::ops::Range<usize>)> =
        match mesh.active_part.and_then(|i| mesh.parts.get(i)) {
            Some(p) => vec![(p.name.as_str(), mesh.visible_tris())],
            None if !mesh.parts.is_empty() => mesh
                .parts
                .iter()
                .map(|p| (p.name.as_str(), p.tris.clone()))
                .collect(),
            None => vec![(mesh.name.as_str(), 0..mesh.tris.len())],
        };

    let has_any_tris = groups
        .iter()
        .any(|(_, r)| r.start < r.end && r.start < mesh.tris.len());
    if !has_any_tris {
        return Err("Mesh has no visible triangles to export".into());
    }

    let shift = if mesh.active_part.is_some() {
        [mesh.center[0], mesh.center[1], 0.0]
    } else {
        [0.0; 3]
    };
    let has_uv = mesh.uvs.len() == mesh.verts.len() && !mesh.uvs.is_empty();
    let has_tri_geom = mesh.tri_geom.len() == mesh.tris.len();

    let mut bin_buf: Vec<u8> = Vec::new();
    let mut buffer_views: Vec<BufferViewDef> = Vec::new();
    let mut accessors: Vec<AccessorDef> = Vec::new();
    let mut images: Vec<ImageDef> = Vec::new();
    let mut textures: Vec<TextureDef> = Vec::new();
    let mut materials: Vec<MaterialDef> = Vec::new();
    let mut meshes_json: Vec<MeshDef> = Vec::new();
    let mut nodes: Vec<NodeDef> = Vec::new();
    let mut scene_nodes: Vec<usize> = Vec::new();

    // Texture deduplication by Arc pointer (for GeomMaterial) or single mesh.texture.
    let mut tex_ptr_map: HashMap<usize, usize> = HashMap::new();
    let mut override_mat_idx: Option<usize> = None;
    let mut geom_mat_map: HashMap<u16, usize> = HashMap::new();
    let mut default_mat_idx: Option<usize> = None;

    if let Some(ref tex) = mesh.texture {
        let tex_idx = append_texture_to_bin(tex, &mut bin_buf, &mut buffer_views, &mut images, &mut textures)?;
        let has_alpha = tex.pixels.iter().any(|p| p.a() < 255);
        materials.push(MaterialDef {
            name: "Material_Override".into(),
            base_color_texture: Some(tex_idx),
            base_color_factor: None,
            metallic: 0.0,
            roughness: 0.85,
            alpha_mode: if has_alpha { AlphaMode::Mask(0.5) } else { AlphaMode::Opaque },
            double_sided: has_alpha,
        });
        override_mat_idx = Some(0);
    }

    for (group_name, raw_range) in groups {
        let start = raw_range.start.min(mesh.tris.len());
        let end = raw_range.end.min(mesh.tris.len());
        if start >= end {
            continue;
        }

        // Remap vertices used by this group into a compact local vertex buffer.
        let mut vert_remap: HashMap<u32, u32> = HashMap::new();
        let mut local_positions: Vec<[f32; 3]> = Vec::new();
        let mut local_uvs: Vec<[f32; 2]> = Vec::new();
        let mut accum_normals: Vec<[f32; 3]> = Vec::new();

        // Bucket triangles by geom_id (or 0 if tri_geom is absent).
        let mut prim_buckets: BTreeMap<u16, Vec<u32>> = BTreeMap::new();

        for ti in start..end {
            let tri = mesh.tris[ti];
            let mut local_tri = [0u32; 3];
            let mut valid = true;
            for (k, &vi) in tri.iter().enumerate() {
                if (vi as usize) >= mesh.verts.len() {
                    valid = false;
                    break;
                }
                let mapped = match vert_remap.get(&vi) {
                    Some(&idx) => idx,
                    None => {
                        let idx = local_positions.len() as u32;
                        let v = mesh.verts[vi as usize];
                        local_positions.push([
                            sanitize_f32(v[0] - shift[0]),
                            sanitize_f32(v[1] - shift[1]),
                            sanitize_f32(v[2] - shift[2]),
                        ]);
                        if has_uv {
                            let uv = mesh.uvs[vi as usize];
                            local_uvs.push([sanitize_f32(uv[0]), sanitize_f32(uv[1])]);
                        }
                        accum_normals.push([0.0, 0.0, 0.0]);
                        vert_remap.insert(vi, idx);
                        idx
                    }
                };
                local_tri[k] = mapped;
            }
            if !valid {
                continue;
            }

            let face_n = if ti < mesh.normals.len() {
                mesh.normals[ti]
            } else {
                compute_face_normal(
                    local_positions[local_tri[0] as usize],
                    local_positions[local_tri[1] as usize],
                    local_positions[local_tri[2] as usize],
                )
            };
            for &lv in &local_tri {
                let n = &mut accum_normals[lv as usize];
                n[0] += sanitize_f32(face_n[0]);
                n[1] += sanitize_f32(face_n[1]);
                n[2] += sanitize_f32(face_n[2]);
            }

            let gid = if has_tri_geom { mesh.tri_geom[ti] } else { 0 };
            let bucket = prim_buckets.entry(gid).or_default();
            bucket.extend_from_slice(&local_tri);
        }

        if local_positions.is_empty() || prim_buckets.is_empty() {
            continue;
        }

        // Normalize accumulated vertex normals.
        let local_normals: Vec<[f32; 3]> = accum_normals
            .into_iter()
            .map(|n| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-12 {
                    [n[0] / len, n[1] / len, n[2] / len]
                } else {
                    [0.0, 0.0, 1.0]
                }
            })
            .collect();

        // Compute exact min/max for POSITION accessor.
        let (mut min_pos, mut max_pos) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &local_positions {
            for k in 0..3 {
                if p[k] < min_pos[k] {
                    min_pos[k] = p[k];
                }
                if p[k] > max_pos[k] {
                    max_pos[k] = p[k];
                }
            }
        }

        // Write POSITION, NORMAL, TEXCOORD_0 to BIN chunk.
        let pos_acc = append_vec3_accessor(
            &local_positions,
            Some((min_pos, max_pos)),
            &mut bin_buf,
            &mut buffer_views,
            &mut accessors,
        );
        let norm_acc = append_vec3_accessor(
            &local_normals,
            None,
            &mut bin_buf,
            &mut buffer_views,
            &mut accessors,
        );
        let uv_acc = if has_uv && local_uvs.len() == local_positions.len() {
            Some(append_vec2_accessor(
                &local_uvs,
                &mut bin_buf,
                &mut buffer_views,
                &mut accessors,
            ))
        } else {
            None
        };

        let mut primitives = Vec::with_capacity(prim_buckets.len());
        for (gid, indices) in prim_buckets {
            if indices.is_empty() {
                continue;
            }
            let idx_acc = append_index_accessor(
                &indices,
                &mut bin_buf,
                &mut buffer_views,
                &mut accessors,
            );

            let mat_idx = if let Some(ov) = override_mat_idx {
                Some(ov)
            } else if let Some(gmat) = mesh.geom_materials.get(gid as usize) {
                if let Some(&existing) = geom_mat_map.get(&gid) {
                    Some(existing)
                } else {
                    let m = build_geom_material(
                        gid,
                        gmat,
                        mesh.paint_color,
                        &mut tex_ptr_map,
                        &mut bin_buf,
                        &mut buffer_views,
                        &mut images,
                        &mut textures,
                        &mut materials,
                    )?;
                    geom_mat_map.insert(gid, m);
                    Some(m)
                }
            } else {
                let m = match default_mat_idx {
                    Some(existing) => existing,
                    None => {
                        let idx = materials.len();
                        materials.push(MaterialDef {
                            name: "Material_Default".into(),
                            base_color_texture: None,
                            base_color_factor: Some([0.75, 0.75, 0.78, 1.0]),
                            metallic: 0.0,
                            roughness: 0.85,
                            alpha_mode: AlphaMode::Opaque,
                            double_sided: false,
                        });
                        default_mat_idx = Some(idx);
                        idx
                    }
                };
                Some(m)
            };

            primitives.push(PrimitiveDef {
                pos_acc,
                norm_acc,
                uv_acc,
                idx_acc,
                material: mat_idx,
            });
        }

        if primitives.is_empty() {
            continue;
        }

        let mesh_idx = meshes_json.len();
        meshes_json.push(MeshDef {
            name: group_name.to_string(),
            primitives,
        });

        let node_idx = nodes.len();
        nodes.push(NodeDef {
            name: group_name.to_string(),
            mesh: Some(mesh_idx),
            children: Vec::new(),
        });
        scene_nodes.push(node_idx);
    }

    if meshes_json.is_empty() {
        return Err("No valid mesh primitives to export".into());
    }

    // Optional skeleton node hierarchy if crSkeletonData bone_count is present.
    if mesh.bone_count > 0 {
        let mut bone_children = Vec::with_capacity(mesh.bone_count);
        for b in 0..mesh.bone_count {
            let b_idx = nodes.len();
            nodes.push(NodeDef {
                name: format!("bone_{b}"),
                mesh: None,
                children: Vec::new(),
            });
            bone_children.push(b_idx);
        }
        let skel_root_idx = nodes.len();
        nodes.push(NodeDef {
            name: "Skeleton".into(),
            mesh: None,
            children: bone_children,
        });
        scene_nodes.push(skel_root_idx);
    }

    // Pad BIN chunk to 4-byte alignment with 0x00.
    while bin_buf.len() % 4 != 0 {
        bin_buf.push(0x00);
    }

    let json_str = build_gltf_json(
        bin_buf.len(),
        &buffer_views,
        &accessors,
        &images,
        &textures,
        &materials,
        &meshes_json,
        &nodes,
        &scene_nodes,
    );

    let mut json_bytes = json_str.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(0x20);
    }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_buf.len();
    let mut out = Vec::with_capacity(total_len);

    // 12-byte GLB header.
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());

    // JSON chunk.
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);

    // BIN chunk.
    out.extend_from_slice(&(bin_buf.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin_buf);

    Ok(out)
}

#[derive(Clone, Copy)]
enum AlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

struct BufferViewDef {
    byte_offset: usize,
    byte_length: usize,
    target: Option<u32>,
}

struct AccessorDef {
    buffer_view: usize,
    component_type: u32,
    count: usize,
    acc_type: &'static str,
    min_max: Option<([f32; 3], [f32; 3])>,
}

struct ImageDef {
    buffer_view: usize,
}

struct TextureDef {
    source: usize,
}

struct MaterialDef {
    name: String,
    base_color_texture: Option<usize>,
    base_color_factor: Option<[f32; 4]>,
    metallic: f32,
    roughness: f32,
    alpha_mode: AlphaMode,
    double_sided: bool,
}

struct PrimitiveDef {
    pos_acc: usize,
    norm_acc: usize,
    uv_acc: Option<usize>,
    idx_acc: usize,
    material: Option<usize>,
}

struct MeshDef {
    name: String,
    primitives: Vec<PrimitiveDef>,
}

struct NodeDef {
    name: String,
    mesh: Option<usize>,
    children: Vec<usize>,
}

#[inline]
fn sanitize_f32(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn compute_face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let nx = ab[1] * ac[2] - ab[2] * ac[1];
    let ny = ab[2] * ac[0] - ab[0] * ac[2];
    let nz = ab[0] * ac[1] - ab[1] * ac[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-12 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn pad_bin_4(bin_buf: &mut Vec<u8>) {
    while bin_buf.len() % 4 != 0 {
        bin_buf.push(0x00);
    }
}

fn append_vec3_accessor(
    data: &[[f32; 3]],
    min_max: Option<([f32; 3], [f32; 3])>,
    bin_buf: &mut Vec<u8>,
    buffer_views: &mut Vec<BufferViewDef>,
    accessors: &mut Vec<AccessorDef>,
) -> usize {
    pad_bin_4(bin_buf);
    let byte_offset = bin_buf.len();
    for v in data {
        bin_buf.extend_from_slice(&v[0].to_le_bytes());
        bin_buf.extend_from_slice(&v[1].to_le_bytes());
        bin_buf.extend_from_slice(&v[2].to_le_bytes());
    }
    let byte_length = bin_buf.len() - byte_offset;
    let bv_idx = buffer_views.len();
    buffer_views.push(BufferViewDef {
        byte_offset,
        byte_length,
        target: Some(TARGET_ARRAY_BUFFER),
    });
    let acc_idx = accessors.len();
    accessors.push(AccessorDef {
        buffer_view: bv_idx,
        component_type: COMPONENT_FLOAT,
        count: data.len(),
        acc_type: "VEC3",
        min_max,
    });
    acc_idx
}

fn append_vec2_accessor(
    data: &[[f32; 2]],
    bin_buf: &mut Vec<u8>,
    buffer_views: &mut Vec<BufferViewDef>,
    accessors: &mut Vec<AccessorDef>,
) -> usize {
    pad_bin_4(bin_buf);
    let byte_offset = bin_buf.len();
    for v in data {
        bin_buf.extend_from_slice(&v[0].to_le_bytes());
        bin_buf.extend_from_slice(&v[1].to_le_bytes());
    }
    let byte_length = bin_buf.len() - byte_offset;
    let bv_idx = buffer_views.len();
    buffer_views.push(BufferViewDef {
        byte_offset,
        byte_length,
        target: Some(TARGET_ARRAY_BUFFER),
    });
    let acc_idx = accessors.len();
    accessors.push(AccessorDef {
        buffer_view: bv_idx,
        component_type: COMPONENT_FLOAT,
        count: data.len(),
        acc_type: "VEC2",
        min_max: None,
    });
    acc_idx
}

fn append_index_accessor(
    indices: &[u32],
    bin_buf: &mut Vec<u8>,
    buffer_views: &mut Vec<BufferViewDef>,
    accessors: &mut Vec<AccessorDef>,
) -> usize {
    pad_bin_4(bin_buf);
    let byte_offset = bin_buf.len();
    for &i in indices {
        bin_buf.extend_from_slice(&i.to_le_bytes());
    }
    let byte_length = bin_buf.len() - byte_offset;
    let bv_idx = buffer_views.len();
    buffer_views.push(BufferViewDef {
        byte_offset,
        byte_length,
        target: Some(TARGET_ELEMENT_ARRAY_BUFFER),
    });
    let acc_idx = accessors.len();
    accessors.push(AccessorDef {
        buffer_view: bv_idx,
        component_type: COMPONENT_UNSIGNED_INT,
        count: indices.len(),
        acc_type: "SCALAR",
        min_max: None,
    });
    acc_idx
}

fn encode_texture_png(tex: &Texture2D) -> Result<Vec<u8>, String> {
    if tex.w == 0 || tex.h == 0 || tex.pixels.len() < tex.w * tex.h {
        return Err("Invalid texture dimensions".into());
    }
    let mut rgba = Vec::with_capacity(tex.w * tex.h * 4);
    for p in &tex.pixels[..tex.w * tex.h] {
        rgba.push(p.r());
        rgba.push(p.g());
        rgba.push(p.b());
        rgba.push(p.a());
    }
    let mut png_bytes = Vec::new();
    use image::ImageEncoder;
    image::codecs::png::PngEncoder::new(&mut png_bytes)
        .write_image(
            &rgba,
            tex.w as u32,
            tex.h as u32,
            image::ColorType::Rgba8.into(),
        )
        .map_err(|e| format!("PNG encode error: {e}"))?;
    Ok(png_bytes)
}

fn append_texture_to_bin(
    tex: &Texture2D,
    bin_buf: &mut Vec<u8>,
    buffer_views: &mut Vec<BufferViewDef>,
    images: &mut Vec<ImageDef>,
    textures: &mut Vec<TextureDef>,
) -> Result<usize, String> {
    let png_bytes = encode_texture_png(tex)?;
    pad_bin_4(bin_buf);
    let byte_offset = bin_buf.len();
    bin_buf.extend_from_slice(&png_bytes);
    let byte_length = png_bytes.len();
    pad_bin_4(bin_buf);

    let bv_idx = buffer_views.len();
    buffer_views.push(BufferViewDef {
        byte_offset,
        byte_length,
        target: None,
    });
    let img_idx = images.len();
    images.push(ImageDef { buffer_view: bv_idx });
    let tex_idx = textures.len();
    textures.push(TextureDef { source: img_idx });
    Ok(tex_idx)
}

#[allow(clippy::too_many_arguments)]
fn build_geom_material(
    gid: u16,
    gmat: &GeomMaterial,
    paint_color: [f32; 3],
    tex_ptr_map: &mut HashMap<usize, usize>,
    bin_buf: &mut Vec<u8>,
    buffer_views: &mut Vec<BufferViewDef>,
    images: &mut Vec<ImageDef>,
    textures: &mut Vec<TextureDef>,
    materials: &mut Vec<MaterialDef>,
) -> Result<usize, String> {
    let base_color_texture = if let Some(ref tex_arc) = gmat.texture {
        let ptr = Arc::as_ptr(tex_arc) as usize;
        if let Some(&existing) = tex_ptr_map.get(&ptr) {
            Some(existing)
        } else {
            let idx = append_texture_to_bin(tex_arc, bin_buf, buffer_views, images, textures)?;
            tex_ptr_map.insert(ptr, idx);
            Some(idx)
        }
    } else {
        None
    };

    let (alpha_mode, base_color_factor, metallic, roughness, double_sided) = match gmat.pass {
        MaterialPass::Opaque => {
            if gmat.is_vehicle_paint {
                (
                    AlphaMode::Opaque,
                    Some([
                        sanitize_f32(paint_color[0]),
                        sanitize_f32(paint_color[1]),
                        sanitize_f32(paint_color[2]),
                        1.0,
                    ]),
                    0.2,
                    0.25,
                    false,
                )
            } else {
                (AlphaMode::Opaque, None, 0.0, 0.85, false)
            }
        }
        MaterialPass::Cutout => (AlphaMode::Mask(0.5), None, 0.0, 0.85, true),
        MaterialPass::Alpha | MaterialPass::Decal => (AlphaMode::Blend, None, 0.0, 0.85, true),
        MaterialPass::Glass { texture_alpha } => {
            let factor = if texture_alpha {
                None
            } else {
                Some([0.85, 0.92, 0.98, 0.25])
            };
            (AlphaMode::Blend, factor, 0.05, 0.1, true)
        }
    };

    let mat_idx = materials.len();
    materials.push(MaterialDef {
        name: format!("Material_{gid}"),
        base_color_texture,
        base_color_factor,
        metallic,
        roughness,
        alpha_mode,
        double_sided,
    });
    Ok(mat_idx)
}

fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[allow(clippy::too_many_arguments)]
fn build_gltf_json(
    bin_len: usize,
    buffer_views: &[BufferViewDef],
    accessors: &[AccessorDef],
    images: &[ImageDef],
    textures: &[TextureDef],
    materials: &[MaterialDef],
    meshes: &[MeshDef],
    nodes: &[NodeDef],
    scene_nodes: &[usize],
) -> String {
    let mut j = String::with_capacity(2048);
    j.push_str("{\"asset\":{\"version\":\"2.0\",\"generator\":\"PercyStar glTF 2.0 Exporter\"},");

    // scene & scenes
    j.push_str("\"scene\":0,\"scenes\":[{\"nodes\":[");
    for (i, &n) in scene_nodes.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        let _ = write!(j, "{n}");
    }
    j.push_str("]}],");

    // nodes
    j.push_str("\"nodes\":[");
    for (i, node) in nodes.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        j.push_str("{\"name\":");
        push_json_str(&mut j, &node.name);
        if let Some(m) = node.mesh {
            let _ = write!(j, ",\"mesh\":{m}");
        }
        if !node.children.is_empty() {
            j.push_str(",\"children\":[");
            for (ci, &c) in node.children.iter().enumerate() {
                if ci > 0 {
                    j.push(',');
                }
                let _ = write!(j, "{c}");
            }
            j.push(']');
        }
        j.push('}');
    }
    j.push_str("],");

    // meshes
    j.push_str("\"meshes\":[");
    for (i, m) in meshes.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        j.push_str("{\"name\":");
        push_json_str(&mut j, &m.name);
        j.push_str(",\"primitives\":[");
        for (pi, p) in m.primitives.iter().enumerate() {
            if pi > 0 {
                j.push(',');
            }
            let _ = write!(
                j,
                "{{\"attributes\":{{\"POSITION\":{},\"NORMAL\":{}",
                p.pos_acc, p.norm_acc
            );
            if let Some(uv) = p.uv_acc {
                let _ = write!(j, ",\"TEXCOORD_0\":{uv}");
            }
            let _ = write!(j, "}},\"indices\":{},\"mode\":4", p.idx_acc);
            if let Some(mat) = p.material {
                let _ = write!(j, ",\"material\":{mat}");
            }
            j.push('}');
        }
        j.push_str("]}");
    }
    j.push_str("],");

    // materials
    if !materials.is_empty() {
        j.push_str("\"materials\":[");
        for (i, mat) in materials.iter().enumerate() {
            if i > 0 {
                j.push(',');
            }
            j.push_str("{\"name\":");
            push_json_str(&mut j, &mat.name);
            j.push_str(",\"pbrMetallicRoughness\":{");
            let mut has_pbr_field = false;
            if let Some(tex) = mat.base_color_texture {
                let _ = write!(j, "\"baseColorTexture\":{{\"index\":{tex}}}");
                has_pbr_field = true;
            }
            if let Some([r, g, b, a]) = mat.base_color_factor {
                if has_pbr_field {
                    j.push(',');
                }
                let _ = write!(
                    j,
                    "\"baseColorFactor\":[{:.6},{:.6},{:.6},{:.6}]",
                    r, g, b, a
                );
                has_pbr_field = true;
            }
            if has_pbr_field {
                j.push(',');
            }
            let _ = write!(
                j,
                "\"metallicFactor\":{:.4},\"roughnessFactor\":{:.4}}}",
                mat.metallic, mat.roughness
            );
            match mat.alpha_mode {
                AlphaMode::Opaque => {}
                AlphaMode::Mask(cutoff) => {
                    let _ = write!(j, ",\"alphaMode\":\"MASK\",\"alphaCutoff\":{:.3}", cutoff);
                }
                AlphaMode::Blend => {
                    j.push_str(",\"alphaMode\":\"BLEND\"");
                }
            }
            if mat.double_sided {
                j.push_str(",\"doubleSided\":true");
            }
            j.push('}');
        }
        j.push_str("],");
    }

    // textures, images, samplers
    if !textures.is_empty() {
        j.push_str("\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}],");
        j.push_str("\"images\":[");
        for (i, img) in images.iter().enumerate() {
            if i > 0 {
                j.push(',');
            }
            let _ = write!(
                j,
                "{{\"bufferView\":{},\"mimeType\":\"image/png\"}}",
                img.buffer_view
            );
        }
        j.push_str("],\"textures\":[");
        for (i, tex) in textures.iter().enumerate() {
            if i > 0 {
                j.push(',');
            }
            let _ = write!(j, "{{\"sampler\":0,\"source\":{}}}", tex.source);
        }
        j.push_str("],");
    }

    // accessors
    j.push_str("\"accessors\":[");
    for (i, acc) in accessors.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        let _ = write!(
            j,
            "{{\"bufferView\":{},\"componentType\":{},\"count\":{},\"type\":\"{}\"",
            acc.buffer_view, acc.component_type, acc.count, acc.acc_type
        );
        if let Some((min, max)) = acc.min_max {
            let _ = write!(
                j,
                ",\"min\":[{:.6},{:.6},{:.6}],\"max\":[{:.6},{:.6},{:.6}]",
                min[0], min[1], min[2], max[0], max[1], max[2]
            );
        }
        j.push('}');
    }
    j.push_str("],");

    // bufferViews
    j.push_str("\"bufferViews\":[");
    for (i, bv) in buffer_views.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        let _ = write!(
            j,
            "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}",
            bv.byte_offset, bv.byte_length
        );
        if let Some(target) = bv.target {
            let _ = write!(j, ",\"target\":{target}");
        }
        j.push('}');
    }
    j.push_str("],");

    // buffers
    let _ = write!(j, "\"buffers\":[{{\"byteLength\":{bin_len}}}]}}");
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::Color32;

    fn make_triangle_mesh() -> Mesh3D {
        let mut m = Mesh3D::from_parts(
            "test_tri".into(),
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![[0, 1, 2]],
            vec![0],
            vec![],
            None,
            "test".into(),
            1,
            1,
            "LOD0".into(),
        );
        m.bone_count = 3;
        m.geom_materials = vec![GeomMaterial {
            texture: Some(Arc::new(Texture2D {
                w: 2,
                h: 2,
                pixels: vec![
                    Color32::RED,
                    Color32::GREEN,
                    Color32::BLUE,
                    Color32::WHITE,
                ],
            })),
            is_vehicle_paint: false,
            pass: MaterialPass::Opaque,
            auxiliary_pass: false,
        }];
        m
    }

    #[test]
    fn glb_header_chunks_and_json_valid() {
        let mesh = make_triangle_mesh();
        let glb = build_glb_bytes(&mesh).expect("build_glb_bytes should succeed");

        assert!(glb.len() >= 20);
        let magic = u32::from_le_bytes(glb[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(glb[4..8].try_into().unwrap());
        let total_len = u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize;
        assert_eq!(magic, 0x4654_6C67);
        assert_eq!(version, 2);
        assert_eq!(total_len, glb.len());

        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json_type = u32::from_le_bytes(glb[16..20].try_into().unwrap());
        assert_eq!(json_type, 0x4E4F_534A);
        assert_eq!(json_len % 4, 0);

        let json_end = 20 + json_len;
        let json_str = std::str::from_utf8(&glb[20..json_end]).unwrap();
        assert!(json_str.starts_with('{'));
        assert!(json_str.trim_end_matches(' ').ends_with('}'));
        assert!(json_str.contains("\"version\":\"2.0\""));
        assert!(json_str.contains("\"POSITION\":0"));
        assert!(json_str.contains("\"NORMAL\":1"));
        assert!(json_str.contains("\"TEXCOORD_0\":2"));
        assert!(json_str.contains("\"indices\":3"));
        assert!(json_str.contains("\"mimeType\":\"image/png\""));
        assert!(json_str.contains("\"Skeleton\""));
        assert!(json_str.contains("\"bone_0\""));
        assert!(json_str.contains("\"bone_2\""));

        let bin_len = u32::from_le_bytes(glb[json_end..json_end + 4].try_into().unwrap()) as usize;
        let bin_type = u32::from_le_bytes(glb[json_end + 4..json_end + 8].try_into().unwrap());
        assert_eq!(bin_type, 0x004E_4942);
        assert_eq!(bin_len % 4, 0);
        assert_eq!(json_end + 8 + bin_len, total_len);
    }

    #[test]
    fn empty_mesh_returns_err() {
        let m = Mesh3D::from_parts(
            "empty".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            None,
            "empty".into(),
            0,
            0,
            "LOD0".into(),
        );
        assert!(build_glb_bytes(&m).is_err());
    }
}

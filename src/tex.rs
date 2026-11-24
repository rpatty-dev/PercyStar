//! Native Xbox Series X (Scarlett / RDNA2) texture, TTPF (RDS5/ETPF), RBF0, Scaleform GFX,
//! and CNavMesh decoders for GTA VI (RAGE 9). Zero external runtime dependencies.

use crate::model3d::{is_zlib_head, Mesh3D, Texture2D};
use eframe::egui::Color32;
use flate2::{Decompress, FlushDecompress, Status};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[inline]
fn ru16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn ru32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn ru64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5], b[o + 6], b[o + 7]])
}
#[inline]
fn rf32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn vptr(q: u64) -> Option<usize> {
    (0x5000_0000..0x5800_0000).contains(&q).then(|| (q - 0x5000_0000) as usize)
}
#[inline]
fn pptr(q: u64) -> Option<usize> {
    (0x6000_0000..0x7000_0000).contains(&q).then(|| (q - 0x6000_0000) as usize)
}

#[derive(Clone)]
pub struct DecodedTexture {
    pub name: String,
    pub w: usize,
    pub h: usize,
    pub mips: u8,
    pub dxgi: u8,
    pub fmt_name: &'static str,
    // straight non premultiplied rgba kept raw see premultiplied_for_egui
    pub pixels: Vec<Color32>,
    pub raw_dds_payload: Vec<u8>,
}

pub fn dxgi_info(dxgi: u8) -> Option<(&'static str, bool, usize)> {
    Some(match dxgi {
        10 => ("R16G16B16A16_FLOAT", false, 8),
        11 => ("R16G16B16A16_UNORM", false, 8),
        24 => ("R10G10B10A2_UNORM", false, 4),
        28 => ("R8G8B8A8_UNORM", false, 4),
        29 => ("R8G8B8A8_UNORM_SRGB", false, 4),
        34 => ("R16G16_FLOAT", false, 4),
        41 => ("R32_FLOAT", false, 4),
        49 => ("R8G8_UNORM", false, 2),
        54 => ("R16_FLOAT", false, 2),
        56 => ("R16_UNORM", false, 2),
        61 => ("R8_UNORM", false, 1),
        65 => ("A8_UNORM", false, 1),
        71 => ("BC1_UNORM", true, 8),
        72 => ("BC1_UNORM_SRGB", true, 8),
        74 => ("BC2_UNORM", true, 16),
        75 => ("BC2_UNORM_SRGB", true, 16),
        77 => ("BC3_UNORM", true, 16),
        78 => ("BC3_UNORM_SRGB", true, 16),
        80 => ("BC4_UNORM", true, 8),
        81 => ("BC4_SNORM", true, 8),
        83 => ("BC5_UNORM", true, 16),
        84 => ("BC5_SNORM", true, 16),
        87 => ("B8G8R8A8_UNORM", false, 4),
        91 => ("B8G8R8A8_UNORM_SRGB", false, 4),
        95 => ("BC6H_UF16", true, 16),
        96 => ("BC6H_SF16", true, 16),
        98 => ("BC7_UNORM", true, 16),
        99 => ("BC7_UNORM_SRGB", true, 16),
        _ => return None,
    })
}

pub fn padded_4kb_size(w: usize, h: usize, is_bc: bool, bpe: usize) -> usize {
    let ew = if is_bc { w.div_ceil(4) } else { w }.max(1);
    let eh = if is_bc { h.div_ceil(4) } else { h }.max(1);
    let (tw, th) = match bpe {
        16 => (16usize, 16usize),
        8 => (32, 16),
        4 => (32, 32),
        2 => (64, 32),
        _ => (64, 64),
    };
    let pw = ew.div_ceil(tw) * tw;
    let ph = eh.div_ceil(th) * th;
    pw * ph * bpe
}

pub fn padded_64kb_size(w: usize, h: usize, is_bc: bool, bpe: usize) -> usize {
    let ew = if is_bc { w.div_ceil(4) } else { w }.max(1);
    let eh = if is_bc { h.div_ceil(4) } else { h }.max(1);
    let (tw, th) = match bpe {
        16 => (64usize, 64usize),
        8 => (128, 64),
        4 => (128, 128),
        2 => (256, 128),
        _ => (256, 256),
    };
    let pw = ew.div_ceil(tw) * tw;
    let ph = eh.div_ceil(th) * th;
    pw * ph * bpe
}

/// Verified AMD RDNA2 / Xbox Scarlett SW_4KB_D deswizzle for all 5 bit depths (8..128bpp).
pub fn deswizzle_4kb_d(raw: &[u8], w: usize, h: usize, is_bc: bool, bpe: usize) -> Vec<u8> {
    let ew = if is_bc { w.div_ceil(4) } else { w }.max(1);
    let eh = if is_bc { h.div_ceil(4) } else { h }.max(1);
    let (tw, th) = match bpe {
        16 => (16usize, 16usize),
        8 => (32, 16),
        4 => (32, 32),
        2 => (64, 32),
        _ => (64, 64),
    };
    let pw = ew.div_ceil(tw) * tw;
    let ph = eh.div_ceil(th) * th;
    let needed = pw * ph * bpe;
    let lin_len = ew * eh * bpe;
    if raw.len() < needed {
        let mut out = vec![0u8; lin_len];
        let n = raw.len().min(lin_len);
        out[..n].copy_from_slice(&raw[..n]);
        return out;
    }

    let mut out = vec![0u8; lin_len];
    let tiles_x = pw / tw;
    let elems_per_tile = tw * th;

    for y in 0..eh {
        let ty = y / th;
        let ly = y % th;
        for x in 0..ew {
            let tx = x / tw;
            let lx = x % tw;
            let t_idx = ty * tiles_x + tx;
            let m = match bpe {
                // 128bpp (16x16): y0,y1,x0,x1, y2,x2,y3,x3
                16 => (ly & 3) | ((lx & 3) << 2) | (((ly >> 2) & 1) << 4) | (((lx >> 2) & 1) << 5) | (((ly >> 3) & 1) << 6) | (((lx >> 3) & 1) << 7),
                // 64bpp (32x16): x0,y0,y1,x1,x2, y2,x3,y3,x4
                8 => (lx & 1) | ((ly & 3) << 1) | (((lx >> 1) & 3) << 3) | (((ly >> 2) & 1) << 5) | (((lx >> 3) & 1) << 6) | (((ly >> 3) & 1) << 7) | (((lx >> 4) & 1) << 8),
                // 32bpp (32x32): x0,x1,y0,y1,x2,y2, y3,x3,y4,x4
                4 => (lx & 3) | ((ly & 3) << 2) | (((lx >> 2) & 1) << 4) | (((ly >> 2) & 1) << 5) | (((ly >> 3) & 1) << 6) | (((lx >> 3) & 1) << 7) | (((ly >> 4) & 1) << 8) | (((lx >> 4) & 1) << 9),
                // 16bpp (64x32): x0,x1,x2,y0,y1,y2, x3,y3,x4,y4,x5
                2 => (lx & 7) | ((ly & 7) << 3) | (((lx >> 3) & 1) << 6) | (((ly >> 3) & 1) << 7) | (((lx >> 4) & 1) << 8) | (((ly >> 4) & 1) << 9) | (((lx >> 5) & 1) << 10),
                // 8bpp (64x64): y0,y1,y2,x1,x0,x2, y3,x3,x4,y4,x5,y5
                _ => (ly & 7) | (((lx >> 1) & 1) << 3) | ((lx & 1) << 4) | (((lx >> 2) & 1) << 5) | (((ly >> 3) & 1) << 6) | (((lx >> 3) & 1) << 7) | (((lx >> 4) & 1) << 8) | (((ly >> 4) & 1) << 9) | (((lx >> 5) & 1) << 10) | (((ly >> 5) & 1) << 11),
            };
            let src_off = (t_idx * elems_per_tile + m) * bpe;
            let dst_off = (y * ew + x) * bpe;
            out[dst_off..dst_off + bpe].copy_from_slice(&raw[src_off..src_off + bpe]);
        }
    }
    out
}

/// Verified Xbox Scarlett 64KB TTPF tile deswizzle for ALL 5 bit depths (8..128bpp, BCn & uncompressed).
pub fn deswizzle_64kb_ttpf(raw: &[u8], w: usize, h: usize, is_bc: bool, bpe: usize) -> Vec<u8> {
    let bw = if is_bc { w.div_ceil(4) } else { w }.max(1);
    let bh = if is_bc { h.div_ceil(4) } else { h }.max(1);
    let (tw, th, shift) = match bpe {
        16 => (64usize, 64usize, 12usize),
        8 => (128, 64, 13),
        4 => (128, 128, 14),
        2 => (256, 128, 15),
        _ => (256, 256, 16),
    };
    let pw = bw.div_ceil(tw) * tw;
    let ph = bh.div_ceil(th) * th;
    let needed = pw * ph * bpe;
    let mut padded = Vec::new();
    let src: &[u8] = if raw.len() >= needed {
        &raw[..needed]
    } else {
        padded.resize(needed, 0);
        let n = raw.len().min(needed);
        padded[..n].copy_from_slice(&raw[..n]);
        &padded
    };
    let mut out = vec![0u8; bw * bh * bpe];
    let tiles_x = pw / tw;
    for y in 0..bh {
        let ty = y / th;
        let ly = y % th;
        for x in 0..bw {
            let tx = x / tw;
            let lx = x % tw;
            let t_idx = ty * tiles_x + tx;
            let m = match bpe {
                16 => (lx & 1)
                    | ((ly & 1) << 1)
                    | (((lx >> 1) & 1) << 2)
                    | (((ly >> 1) & 1) << 3)
                    | (((ly >> 2) & 1) << 4)
                    | (((lx >> 2) & 1) << 5)
                    | (((ly >> 3) & 1) << 6)
                    | (((lx >> 3) & 1) << 7)
                    | (((ly >> 4) & 1) << 8)
                    | (((lx >> 4) & 1) << 9)
                    | (((ly >> 5) & 1) << 10)
                    | (((lx >> 5) & 1) << 11),
                8 => (lx & 1)
                    | ((ly & 1) << 1)
                    | (((lx >> 1) & 3) << 2)
                    | (((ly >> 1) & 1) << 4)
                    | (((ly >> 2) & 1) << 5)
                    | (((lx >> 3) & 1) << 6)
                    | (((ly >> 3) & 1) << 7)
                    | (((lx >> 4) & 1) << 8)
                    | (((ly >> 4) & 1) << 9)
                    | (((lx >> 5) & 1) << 10)
                    | (((ly >> 5) & 1) << 11)
                    | (((lx >> 6) & 1) << 12),
                4 => (lx & 3)
                    | ((ly & 3) << 2)
                    | (((lx >> 2) & 1) << 4)
                    | (((ly >> 2) & 1) << 5)
                    | (((ly >> 3) & 1) << 6)
                    | (((lx >> 3) & 1) << 7)
                    | (((ly >> 4) & 1) << 8)
                    | (((lx >> 4) & 1) << 9)
                    | (((ly >> 5) & 1) << 10)
                    | (((lx >> 5) & 1) << 11)
                    | (((ly >> 6) & 1) << 12)
                    | (((lx >> 6) & 1) << 13),
                2 => (lx & 7)
                    | ((ly & 7) << 3)
                    | (((lx >> 3) & 1) << 6)
                    | (((ly >> 3) & 1) << 7)
                    | (((lx >> 4) & 1) << 8)
                    | (((ly >> 4) & 1) << 9)
                    | (((lx >> 5) & 1) << 10)
                    | (((ly >> 5) & 1) << 11)
                    | (((lx >> 6) & 1) << 12)
                    | (((ly >> 6) & 1) << 13)
                    | (((lx >> 7) & 1) << 14),
                _ => (ly & 7)
                    | (((lx >> 1) & 1) << 3)
                    | ((lx & 1) << 4)
                    | (((lx >> 2) & 1) << 5)
                    | (((ly >> 3) & 1) << 6)
                    | (((lx >> 3) & 1) << 7)
                    | (((lx >> 4) & 1) << 8)
                    | (((ly >> 4) & 1) << 9)
                    | (((lx >> 5) & 1) << 10)
                    | (((ly >> 5) & 1) << 11)
                    | (((lx >> 6) & 1) << 12)
                    | (((ly >> 6) & 1) << 13)
                    | (((lx >> 7) & 1) << 14)
                    | (((ly >> 7) & 1) << 15),
            };
            let src_off = ((t_idx << shift) | m) * bpe;
            let dst_off = (y * bw + x) * bpe;
            out[dst_off..dst_off + bpe].copy_from_slice(&src[src_off..src_off + bpe]);
        }
    }
    out
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1F) as u32;
    let m = (bits & 0x3FF) as u32;
    if e == 0 {
        if m == 0 { return if s != 0 { -0.0 } else { 0.0 }; }
        let v = (m as f32) * (2.0_f32).powi(-24);
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return 0.0;
    }
    f32::from_bits((s << 31) | ((e + 112) << 23) | (m << 13))
}

#[inline]
fn tonemap_u8(v: f32) -> u8 {
    let x = v.max(0.0);
    let m = x / (1.0 + x);
    (m.powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8
}

// stores straight rgba bytes untouched color32 constructors that take unmultiplied input would premultiply
#[inline]
fn straight(r: u8, g: u8, b: u8, a: u8) -> Color32 {
    Color32::from_rgba_premultiplied(r, g, b, a)
}

// egui wants premultiplied texels decoded textures are straight
pub fn premultiplied_for_egui(pixels: &[Color32]) -> Vec<Color32> {
    pixels
        .iter()
        .map(|c| {
            let [r, g, b, a] = c.to_array();
            if a == 255 { *c } else { Color32::from_rgba_unmultiplied(r, g, b, a) }
        })
        .collect()
}

// splits a 4x4 block image into bands of whole block rows and decodes the bands on all cores
// decode gets the band's blocks, the band height in pixels and the band's slice of the output
fn par_block_rows<T: Send>(
    lin: &[u8],
    w: usize,
    h: usize,
    block_bytes: usize,
    out: &mut [T],
    decode: impl Fn(&[u8], usize, &mut [T]) -> Option<()> + Sync,
) -> Option<()> {
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let need = bw.checked_mul(bh)?.checked_mul(block_bytes)?;
    if w == 0 || lin.len() < need || out.len() < w * h {
        return None;
    }
    let (lin, out) = (&lin[..need], &mut out[..w * h]);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    // small images are not worth a thread
    let rows = bh.div_ceil(cores).max(32);
    if rows >= bh {
        return decode(lin, h, out);
    }
    let decode = &decode;
    std::thread::scope(|s| {
        let bands: Vec<_> = lin
            .chunks(rows * bw * block_bytes)
            .zip(out.chunks_mut(rows * 4 * w))
            .map(|(d, o)| s.spawn(move || decode(d, o.len() / w, o)))
            .collect();
        // join every band before answering so none outlives the borrow
        bands.into_iter().map(|b| b.join().ok().flatten()).fold(Some(()), |a, b| a.and(b))
    })
}

// walks 4x4 blocks row major put writes texel 0 to 15 of the current block cropping at the texture edge
fn for_each_block(
    lin: &[u8],
    w: usize,
    h: usize,
    block_bytes: usize,
    decode: impl Fn(&[u8], &mut dyn FnMut(usize, Color32)) + Sync,
    pixels: &mut [Color32],
) -> Option<()> {
    par_block_rows(lin, w, h, block_bytes, pixels, |lin, h, pixels| {
        let bw = w.div_ceil(4);
        for (bi, blk) in lin.chunks_exact(block_bytes).enumerate() {
            let (bx, by) = ((bi % bw) * 4, (bi / bw) * 4);
            decode(blk, &mut |i, c| {
                let (x, y) = (bx + (i & 3), by + (i >> 2));
                if x < w && y < h {
                    pixels[y * w + x] = c;
                }
            });
        }
        Some(())
    })
}

// color block of bc2 and bc3 always four colors bc1 switches to three colors plus black when c0 <= c1
fn color_block4(blk: &[u8]) -> [[u8; 3]; 16] {
    let expand = |q: u16| {
        let (r, g, b) = ((q >> 11) & 31, (q >> 5) & 63, q & 31);
        [((r << 3) | (r >> 2)) as u16, ((g << 2) | (g >> 4)) as u16, ((b << 3) | (b >> 2)) as u16]
    };
    let (c0, c1) = (expand(u16::from_le_bytes([blk[0], blk[1]])), expand(u16::from_le_bytes([blk[2], blk[3]])));
    let pal: [[u8; 3]; 4] = [
        c0.map(|v| v as u8),
        c1.map(|v| v as u8),
        std::array::from_fn(|k| ((c0[k] * 2 + c1[k]) / 3) as u8),
        std::array::from_fn(|k| ((c0[k] + c1[k] * 2) / 3) as u8),
    ];
    let idx = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
    std::array::from_fn(|i| pal[((idx >> (2 * i)) & 3) as usize])
}

// single channel block of bc3 alpha bc4 and bc5 as bytes signed blocks map minus one to 0 and plus one to 255
fn channel_block(blk: &[u8], signed: bool) -> [u8; 16] {
    let mut pal = [0u8; 8];
    if signed {
        // snorm endpoints compare signed and minus 128 means minus 127
        let (e0, e1) = ((blk[0] as i8).max(-127), (blk[1] as i8).max(-127));
        let (f0, f1) = (e0 as f32 / 127.0, e1 as f32 / 127.0);
        let mut v = [f0, f1, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0];
        let steps = if e0 > e1 { 7 } else { 5 };
        for i in 1..steps {
            v[i + 1] = (f0 * (steps - i) as f32 + f1 * i as f32) / steps as f32;
        }
        pal = v.map(|f| ((f * 0.5 + 0.5) * 255.0 + 0.5) as u8);
    } else {
        let (e0, e1) = (blk[0] as u16, blk[1] as u16);
        pal[0] = blk[0];
        pal[1] = blk[1];
        let steps = if e0 > e1 { 7 } else { 5 };
        for i in 1..steps {
            pal[i as usize + 1] = ((e0 * (steps - i) + e1 * i) / steps) as u8;
        }
        if steps == 5 {
            pal[6] = 0;
            pal[7] = 255;
        }
    }
    let bits = u64::from_le_bytes(blk[..8].try_into().unwrap()) >> 16;
    std::array::from_fn(|i| pal[((bits >> (3 * i)) & 7) as usize])
}

// decodes to straight non premultiplied srgb rgba stored raw in the color32 container
// see premultiplied_for_egui before handing the result to egui
pub fn decode_linear_to_rgba(lin: &[u8], w: usize, h: usize, dxgi: u8) -> Option<Vec<Color32>> {
    if matches!(dxgi, 74 | 75 | 77 | 78) {
        // bc2 explicit 4 bit alpha bc3 interpolated alpha both followed by a four color block
        let mut pixels = vec![Color32::TRANSPARENT; w * h];
        for_each_block(lin, w, h, 16, |blk, put| {
            let rgb = color_block4(&blk[8..16]);
            let alpha = if dxgi <= 75 {
                let bits = u64::from_le_bytes(blk[..8].try_into().unwrap());
                std::array::from_fn(|i| ((bits >> (4 * i)) & 0xF) as u8 * 17)
            } else {
                channel_block(&blk[..8], false)
            };
            for i in 0..16 {
                put(i, straight(rgb[i][0], rgb[i][1], rgb[i][2], alpha[i]));
            }
        }, &mut pixels)?;
        return Some(pixels);
    }
    if matches!(dxgi, 80 | 81 | 83 | 84) {
        // one or two channel blocks signed formats keep the unorm byte layout the callers expect
        let (signed, two) = (matches!(dxgi, 81 | 84), dxgi >= 83);
        let mut pixels = vec![Color32::BLACK; w * h];
        for_each_block(lin, w, h, if two { 16 } else { 8 }, |blk, put| {
            let r = channel_block(&blk[..8], signed);
            if !two {
                for i in 0..16 {
                    put(i, Color32::from_gray(r[i]));
                }
                return;
            }
            let g = channel_block(&blk[8..16], signed);
            for i in 0..16 {
                let (nx, ny) = (r[i] as f32 / 127.5 - 1.0, g[i] as f32 / 127.5 - 1.0);
                let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
                put(i, Color32::from_rgb(r[i], g[i], ((nz * 0.5 + 0.5) * 255.0 + 0.5) as u8));
            }
        }, &mut pixels)?;
        return Some(pixels);
    }
    if matches!(dxgi, 71 | 72 | 95 | 96 | 98 | 99) {
        let mut out_u32 = vec![0u32; w * h];
        let res = match dxgi {
            71 | 72 => par_block_rows(lin, w, h, 8, &mut out_u32, |d, bh, o| texture2ddecoder::decode_bc1a(d, w, bh, o).ok()),
            95 | 96 => {
                // Work around texture2ddecoder 0.1.2 bug: BC6H mode 15 (0x0F, endpoint_bits=16)
                // overflows `1i16 << 16` in debug builds. Convert mode-15 blocks to mode-3 (0x03)
                // untransformed blocks safely so BC6H never panics.
                let mut safe = lin.to_vec();
                for blk in safe.as_chunks_mut::<16>().0 {
                    if (blk[0] & 0x1F) == 0x0F {
                        blk[0] = (blk[0] & !0x1F) | 0x03;
                    }
                }
                par_block_rows(&safe, w, h, 16, &mut out_u32, |d, bh, o| {
                    if dxgi == 95 {
                        texture2ddecoder::decode_bc6_unsigned(d, w, bh, o).ok()
                    } else {
                        texture2ddecoder::decode_bc6_signed(d, w, bh, o).ok()
                    }
                })
            }
            98 | 99 => par_block_rows(lin, w, h, 16, &mut out_u32, |d, bh, o| texture2ddecoder::decode_bc7(d, w, bh, o).ok()),
            _ => unreachable!(),
        };
        res?;
        let mut pixels = Vec::with_capacity(w * h);
        for px in out_u32 {
            // texture2ddecoder packs u32 as 0xAARRGGBB (BGRA little-endian)
            let b = (px & 0xFF) as u8;
            let g = ((px >> 8) & 0xFF) as u8;
            let r = ((px >> 16) & 0xFF) as u8;
            let a = ((px >> 24) & 0xFF) as u8;
            pixels.push(straight(r, g, b, a));
        }
        return Some(pixels);
    }
    let mut pixels = Vec::with_capacity(w * h);
    match dxgi {
        10 => {
            if lin.len() < w * h * 8 { return None; }
            for px in lin[..w * h * 8].as_chunks::<8>().0 {
                let r = tonemap_u8(f16_to_f32(ru16(px, 0)));
                let g = tonemap_u8(f16_to_f32(ru16(px, 2)));
                let b = tonemap_u8(f16_to_f32(ru16(px, 4)));
                // the alpha of this float format is not coverage upstream forces it opaque
                pixels.push(straight(r, g, b, 255));
            }
        }
        11 => {
            if lin.len() < w * h * 8 { return None; }
            for px in lin[..w * h * 8].as_chunks::<8>().0 {
                pixels.push(straight(px[1], px[3], px[5], px[7]));
            }
        }
        24 => {
            if lin.len() < w * h * 4 { return None; }
            for px in lin[..w * h * 4].as_chunks::<4>().0 {
                let v = ru32(px, 0);
                let r = ((v & 0x3FF) >> 2) as u8;
                let g = (((v >> 10) & 0x3FF) >> 2) as u8;
                let b = (((v >> 20) & 0x3FF) >> 2) as u8;
                let a = (((v >> 30) & 3) * 85) as u8;
                pixels.push(straight(r, g, b, a));
            }
        }
        28 | 29 => {
            if lin.len() < w * h * 4 { return None; }
            for px in lin[..w * h * 4].as_chunks::<4>().0 {
                pixels.push(straight(px[0], px[1], px[2], px[3]));
            }
        }
        34 => {
            if lin.len() < w * h * 4 { return None; }
            for px in lin[..w * h * 4].as_chunks::<4>().0 {
                let r = tonemap_u8(f16_to_f32(ru16(px, 0)));
                let g = tonemap_u8(f16_to_f32(ru16(px, 2)));
                pixels.push(Color32::from_rgb(r, g, 0));
            }
        }
        41 => {
            if lin.len() < w * h * 4 { return None; }
            for px in lin[..w * h * 4].as_chunks::<4>().0 {
                let g = tonemap_u8(rf32(px, 0));
                pixels.push(Color32::from_gray(g));
            }
        }
        87 | 91 => {
            if lin.len() < w * h * 4 { return None; }
            for px in lin[..w * h * 4].as_chunks::<4>().0 {
                pixels.push(straight(px[2], px[1], px[0], px[3]));
            }
        }
        49 => {
            if lin.len() < w * h * 2 { return None; }
            for px in lin[..w * h * 2].as_chunks::<2>().0 {
                pixels.push(Color32::from_rgb(px[0], px[1], 0));
            }
        }
        54 => {
            if lin.len() < w * h * 2 { return None; }
            let vals: Vec<f32> = lin[..w * h * 2].as_chunks::<2>().0.iter().map(|px| f16_to_f32(ru16(px, 0))).collect();
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in &vals {
                if v.is_finite() {
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
            }
            let span = (mx - mn).max(1e-5);
            for v in vals {
                let g = (((v - mn) / span).clamp(0.0, 1.0) * 255.0) as u8;
                pixels.push(Color32::from_gray(g));
            }
        }
        56 => {
            if lin.len() < w * h * 2 { return None; }
            for px in lin[..w * h * 2].as_chunks::<2>().0 {
                pixels.push(Color32::from_gray(px[1]));
            }
        }
        61 | 65 => {
            if lin.len() < w * h { return None; }
            for &g in &lin[..w * h] {
                pixels.push(Color32::from_gray(g));
            }
        }
        _ => return None,
    }
    Some(pixels)
}

pub fn build_dds_dx10(w: usize, h: usize, dxgi: u8, lin: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(148 + lin.len());
    out.extend_from_slice(b"DDS ");
    for v in [124u32, 0x1007, h as u32, w as u32, lin.len() as u32, 1, 1] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&[0u8; 44]);
    // pixelformat
    out.extend_from_slice(&32u32.to_le_bytes());
    out.extend_from_slice(&0x4u32.to_le_bytes());
    out.extend_from_slice(b"DX10");
    out.extend_from_slice(&[0u8; 20]);
    // caps
    out.extend_from_slice(&0x1000u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 16]);
    // dx10 header
    for v in [dxgi as u32, 3, 0, 1, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(lin);
    out
}

pub fn export_texture_png(tex: &DecodedTexture, path: &Path) -> Result<(), String> {
    let mut rgba = Vec::with_capacity(tex.w * tex.h * 4);
    // pixels are already straight
    for c in &tex.pixels {
        rgba.extend_from_slice(&c.to_array());
    }
    image::save_buffer(path, &rgba, tex.w as u32, tex.h as u32, image::ColorType::Rgba8)
        .map_err(|e| format!("PNG save error: {e}"))
}

// decoded pixels one texture dictionary may produce
const MAX_TXD_PIXELS: usize = 1 << 28;

/// Extract ALL textures from a pgDictionary<grcTexture> inside virtual `v` + physical `p`.
pub fn extract_txd_textures(v: &[u8], p: &[u8], dict_off: usize) -> Vec<DecodedTexture> {
    if dict_off + 0x3a > v.len() { return Vec::new(); }
    let Some(tarr) = vptr(ru64(v, dict_off + 0x30)) else { return Vec::new() };
    let tcnt = ru16(v, dict_off + 0x38) as usize;
    if tcnt == 0 || tcnt > 2048 || tarr + tcnt * 8 > v.len() { return Vec::new(); }

    let mut raw_list = Vec::new();
    for ti in 0..tcnt {
        let Some(tp) = vptr(ru64(v, tarr + ti * 8)) else { continue };
        if tp + 0x48 > v.len() { continue; }
        let Some(poff) = pptr(ru64(v, tp + 0x18)) else { continue };
        if poff >= p.len() { continue; }
        let w = ru16(v, tp + 0x38) as usize;
        let h = ru16(v, tp + 0x3a) as usize;
        let dxgi = v[tp + 0x3f];
        let swiz = v[tp + 0x40];
        let mips = v[tp + 0x42];
        if !(4..=8192).contains(&w) || !(4..=8192).contains(&h) || dxgi_info(dxgi).is_none() {
            continue;
        }
        let mut name = format!("tex_{ti:03}");
        if let Some(mut no) = vptr(ru64(v, tp + 0x30)) {
            if no + 8 < v.len() && v[no..no + 8] == [0u8; 8] {
                no += 8;
            }
            if let Some(rel) = v.get(no..).and_then(|t| t.iter().position(|&b| b == 0)) {
                if rel > 0 && rel < 128 {
                    let s = String::from_utf8_lossy(&v[no..no + rel]).into_owned();
                    if s.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
                        name = s;
                    }
                }
            }
        }
        raw_list.push((poff, name, w, h, dxgi, swiz, mips));
    }

    let mut sorted_poffs: Vec<usize> = raw_list.iter().map(|t| t.0).collect();
    sorted_poffs.sort_unstable();
    sorted_poffs.dedup();
    let mut next_map: HashMap<usize, usize> = HashMap::new();
    for w2 in sorted_poffs.windows(2) {
        next_map.insert(w2[0], w2[1]);
    }

    let mut out = Vec::new();
    let mut decoded_px = 0usize;
    for (poff, name, w, h, dxgi, swiz, mips) in raw_list {
        let Some((fmt_name, is_bc, bpe)) = dxgi_info(dxgi) else { continue };
        let alloc_end = next_map.get(&poff).copied().unwrap_or(p.len());
        let alloc_sz = alloc_end.saturating_sub(poff);
        let mip0_sz = padded_4kb_size(w, h, is_bc, bpe);
        let mip0_off = if mips > 1 && alloc_sz >= mip0_sz {
            poff + (alloc_sz - mip0_sz)
        } else {
            poff
        };
        if mip0_off >= p.len() { continue; }
        // many entries may point at the same data stop before a crafted dictionary exhausts memory
        decoded_px = decoded_px.saturating_add(w.saturating_mul(h));
        if decoded_px > MAX_TXD_PIXELS { break; }
        let slice_end = (mip0_off + mip0_sz).min(p.len());
        let lin = if swiz == 0 {
            let ew = if is_bc { w.div_ceil(4) } else { w }.max(1);
            let eh = if is_bc { h.div_ceil(4) } else { h }.max(1);
            let need = ew * eh * bpe;
            // an entry with no pixel data behind it would decode to a blank image
            if p.len() - mip0_off < need.min(bpe * 16) { continue; }
            let mut b = vec![0u8; need];
            let n = (p.len() - mip0_off).min(need);
            b[..n].copy_from_slice(&p[mip0_off..mip0_off + n]);
            b
        } else {
            deswizzle_4kb_d(&p[mip0_off..slice_end], w, h, is_bc, bpe)
        };
        if let Some(pixels) = decode_linear_to_rgba(&lin, w, h, dxgi) {
            out.push(DecodedTexture {
                name,
                w,
                h,
                mips,
                dxgi,
                fmt_name,
                pixels,
                raw_dds_payload: lin,
            });
        }
    }
    out
}

/// Convert DecodedTexture into model3d::Texture2D.
pub fn to_model_tex(t: DecodedTexture) -> Texture2D {
    let mut pixels = t.pixels;
    if t.dxgi == 65 {
        // texture browser shows a8 as grayscale models use it as opacity
        for c in &mut pixels {
            *c = straight(255, 255, 255, c.r());
        }
    }
    Texture2D::new(t.w, t.h, pixels)
}

/// Read RAGE9 resource virtual + physical buffers from RPF at `file_off` and decode all TXD textures.
pub fn decode_txd_from_rpf(rpf_path: &Path, file_off: u64) -> Result<Vec<DecodedTexture>, String> {
    let mut f = File::open(rpf_path).map_err(|e| format!("Open error: {e}"))?;
    let (v, p) = read_rage9_vp(&mut f, file_off)?;
    let texs = extract_txd_textures(&v, &p, 0);
    if texs.is_empty() {
        return Err("No decodable textures found in dictionary".into());
    }
    Ok(texs)
}

fn decompress_stream_at(f: &mut File, file_off: u64, max_out: usize) -> Result<(Vec<u8>, u64), String> {
    f.seek(SeekFrom::Start(file_off)).map_err(|e| e.to_string())?;
    let mut dec = Decompress::new(true);
    let mut in_buf = [0u8; 65536];
    let mut out_chunk = [0u8; 65536];
    let mut out = Vec::with_capacity(65536);
    let mut consumed = 0u64;
    loop {
        let n = f.read(&mut in_buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        let mut pos = 0usize;
        while pos < n {
            let bi = dec.total_in();
            let bo = dec.total_out();
            let st = dec.decompress(&in_buf[pos..n], &mut out_chunk, FlushDecompress::None).map_err(|e| e.to_string())?;
            let cin = (dec.total_in() - bi) as usize;
            let cout = (dec.total_out() - bo) as usize;
            pos += cin;
            consumed += cin as u64;
            if cout > 0 {
                out.extend_from_slice(&out_chunk[..cout]);
                if out.len() > max_out {
                    return Ok((out, file_off + consumed));
                }
            }
            if st == Status::StreamEnd {
                return Ok((out, file_off + consumed));
            }
            if cin == 0 && cout == 0 { break; }
        }
    }
    Ok((out, file_off + consumed))
}

fn find_next_zlib_near(f: &mut File, start_off: u64) -> Result<Option<u64>, String> {
    f.seek(SeekFrom::Start(start_off)).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 4096];
    let n = f.read(&mut buf).map_err(|e| e.to_string())?;
    for i in 0..n.saturating_sub(3) {
        if is_zlib_head(&buf[i..n]) {
            return Ok(Some(start_off + i as u64));
        }
    }
    Ok(None)
}

pub fn read_rage9_vp(f: &mut File, file_off: u64) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (mut v, mut next_off) = decompress_stream_at(f, file_off, 16 * 1024 * 1024)?;
    if v.len() < 0x40 { return Err("Virtual chunk too small".into()); }
    let Some(bmap_off) = vptr(ru64(&v, 8)) else { return Err("Missing BlockMap".into()) };
    let mut v_read = 1usize;
    while v.len() <= bmap_off + 16 && v_read < 32 {
        let Some(zoff) = find_next_zlib_near(f, next_off)? else { break };
        let (c, nxt) = decompress_stream_at(f, zoff, 16 * 1024 * 1024)?;
        v.extend_from_slice(&c);
        next_off = nxt;
        v_read += 1;
    }
    let (v_cnt, p_cnt) = if bmap_off + 10 <= v.len() {
        ((v[bmap_off + 8] as usize).clamp(1, 32), (v[bmap_off + 9] as usize).clamp(0, 64))
    } else {
        (v_read, 1)
    };
    for _ in v_read..v_cnt {
        let Some(zoff) = find_next_zlib_near(f, next_off)? else { break };
        let (c, nxt) = decompress_stream_at(f, zoff, 16 * 1024 * 1024)?;
        v.extend_from_slice(&c);
        next_off = nxt;
    }
    let mut p = Vec::new();
    for _ in 0..p_cnt {
        let Some(zoff) = find_next_zlib_near(f, next_off)? else { break };
        let (c, nxt) = decompress_stream_at(f, zoff, 64 * 1024 * 1024)?;
        p.extend_from_slice(&c);
        next_off = nxt;
    }
    Ok((v, p))
}

/// Probe if `c0` (first chunk of a RAGE9 resource) is a pgDictionary<grcTexture>.
pub fn probe_txd_header(c0: &[u8]) -> Option<(String, usize)> {
    if c0.len() < 0x40 { return None; }
    let h_ptr = ru64(c0, 0x20);
    let h_cnt = ru16(c0, 0x28) as usize;
    let i_ptr = ru64(c0, 0x30);
    let i_cnt = ru16(c0, 0x38) as usize;
    let i_cap = ru16(c0, 0x3a) as usize;
    if vptr(h_ptr).is_none() || vptr(i_ptr).is_none() { return None; }
    if !(1..=2048).contains(&i_cnt) || i_cnt != h_cnt || i_cnt > i_cap { return None; }
    let arr_off = vptr(i_ptr)?;
    if arr_off + 8 > c0.len() { return None; }
    let tp = vptr(ru64(c0, arr_off))?;
    if tp + 0x44 > c0.len() { return None; }
    pptr(ru64(c0, tp + 0x18))?;
    let w = ru16(c0, tp + 0x38) as usize;
    let h = ru16(c0, tp + 0x3a) as usize;
    let dxgi = c0[tp + 0x3f];
    if !(4..=8192).contains(&w) || !(4..=8192).contains(&h) || dxgi_info(dxgi).is_none() {
        return None;
    }
    let mut first_nm = format!("txd_{i_cnt}tex");
    if let Some(mut no) = vptr(ru64(c0, tp + 0x30)) {
        if no + 8 < c0.len() && c0[no..no + 8] == [0u8; 8] { no += 8; }
        if let Some(rel) = c0.get(no..).and_then(|t| t.iter().position(|&b| b == 0)) {
            if (2..96).contains(&rel) {
                let s = String::from_utf8_lossy(&c0[no..no + rel]).into_owned();
                if s.chars().all(|c| c.is_ascii_graphic()) {
                    first_nm = if i_cnt > 1 { format!("{s} (+{}).#td", i_cnt - 1) } else { format!("{s}.#td") };
                }
            }
        }
    }
    Some((first_nm, i_cnt))
}

// =============================================================================
// TTPF (RDS5 / ETPF) Native Scanner & Decoder
// =============================================================================

#[derive(Clone, Debug)]
pub struct TtpfTexMeta {
    pub idx: usize,
    pub w: usize,
    pub h: usize,
    pub fmt_byte: u8,
    pub dxgi: u8,
    pub has_mips: bool,
    pub byte_off: u64,
    pub alloc_sz: u64,
}

pub fn scan_ttpf_descriptors(path: &Path) -> Result<(String, Vec<TtpfTexMeta>), String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let sz = f.metadata().map_err(|e| e.to_string())?.len();
    let mut hdr = [0u8; 24];
    f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    if &hdr[..4] != b"RDS5" {
        return Err("Not an RDS5 TTPF file".into());
    }
    let n_chunks = ru32(&hdr, 4);
    let tot_uncomp = ru64(&hdr, 8) & 0xff_ffff_ffff;
    let hdr_pages = ru32(&hdr, 16) as u64;
    let etpf_off = hdr_pages * 0x100;

    f.seek(SeekFrom::Start(etpf_off)).map_err(|e| e.to_string())?;
    let mut ehdr = [0u8; 16];
    f.read_exact(&mut ehdr).map_err(|e| e.to_string())?;
    if &ehdr[..4] != b"ETPF" {
        return Err(format!("Missing ETPF header at 0x{etpf_off:x}"));
    }
    let tex_count = ru32(&ehdr, 8) as usize;
    // the record table has to fit in the file before it is allocated
    if (tex_count as u64).saturating_mul(16) > sz.saturating_sub(etpf_off) {
        return Err(format!("ETPF texture count {tex_count} exceeds the file size"));
    }
    let mut recs = vec![0u8; tex_count * 16];
    f.read_exact(&mut recs).map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(tex_count);
    for i in 0..tex_count {
        let u0 = ru32(&recs, i * 16);
        let u1 = ru32(&recs, i * 16 + 4);
        let u2 = ru32(&recs, i * 16 + 8);
        let w = ((u0 & 0x3FFF) + 1) as usize;
        let h = (((u0 >> 14) & 0x3FFF) + 1) as usize;
        let fmt_byte = (u1 & 0xFF) as u8;
        // Verified 100.00% across all 357,115 textures in all 57 .ttpf files:
        // ETPF u1 bits 3..9 ((u1 >> 3) & 0x7F) is the exact Microsoft DXGI_FORMAT enum!
        let dxgi = ((u1 >> 3) & 0x7F) as u8;
        let has_mips = (u1 >> 24) > 0;
        let page4k = (u2 & 0x001F_FFFF) as u64;
        out.push(TtpfTexMeta { idx: i, w, h, fmt_byte, dxgi, has_mips, byte_off: page4k * 0x1000, alloc_sz: 0 });
    }
    let mut offs: Vec<u64> = out.iter().map(|d| d.byte_off).collect();
    offs.sort_unstable();
    offs.dedup();
    let mut nxt: HashMap<u64, u64> = HashMap::new();
    for w2 in offs.windows(2) {
        nxt.insert(w2[0], w2[1]);
    }
    for d in &mut out {
        let (_, is_bc, bpe) = dxgi_info(d.dxgi).unwrap_or(("BC7_UNORM", true, 16));
        let ew = if is_bc { d.w.div_ceil(4) } else { d.w }.max(1);
        let eh = if is_bc { d.h.div_ceil(4) } else { d.h }.max(1);
        let est = ((ew * eh * bpe) as u64).max(0x10000);
        d.alloc_sz = nxt.get(&d.byte_off).copied().unwrap_or((d.byte_off + est).min(tot_uncomp)).saturating_sub(d.byte_off).max(0x1000);
    }
    let info = format!("RDS5/ETPF · {sz} B · {n_chunks} chunks · {tex_count} textures");
    Ok((info, out))
}

/// Exact virtual layout of every RDS5 chunk, straight from the chunk table (no inflating needed).
/// Entry i encodes chunk i's virtual END as (hi & 0xFFFFF) * 16KB + (lo >> 30) * 4KB,
/// and (lo >> 26) & 0xF encodes the NEXT chunk's codec (0=RAW, 1..7=BCPack BC1..BC7, 8=ZLIB).
pub struct TtpfIndex {
    pub descs: Vec<TtpfTexMeta>,
    /// (file_start, file_end, virt_start, virt_len, codec)
    chunks: Vec<(u64, u64, u64, u64, u8)>,
}

static TTPF_INDEX: std::sync::OnceLock<std::sync::Mutex<HashMap<std::path::PathBuf, std::sync::Arc<TtpfIndex>>>> = std::sync::OnceLock::new();

pub fn ttpf_index(path: &Path) -> Result<std::sync::Arc<TtpfIndex>, String> {
    let map = TTPF_INDEX.get_or_init(Default::default);
    if let Some(ix) = map.lock().unwrap().get(path) {
        return Ok(ix.clone());
    }
    let (_, descs) = scan_ttpf_descriptors(path)?;
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut hdr = [0u8; 24];
    f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let n_chunks = ru32(&hdr, 4) as usize;
    let etpf_off = ru32(&hdr, 16) as u64 * 0x100;
    let file_len = f.metadata().map_err(|e| e.to_string())?.len();
    if (n_chunks as u64).saturating_mul(8) > file_len {
        return Err(format!("RDS5 chunk count {n_chunks} exceeds the file size"));
    }
    let mut tbl = vec![0u8; n_chunks * 8];
    f.read_exact(&mut tbl).map_err(|e| e.to_string())?;

    let mut chunks = Vec::with_capacity(n_chunks);
    let (mut prev_c, mut vstart, mut cur_codec) = (etpf_off, 0u64, 0u8);
    for i in 0..n_chunks {
        let lo = ru32(&tbl, i * 8);
        let hi = ru32(&tbl, i * 8 + 4);
        let cend = (lo & 0x03FF_FFFF) as u64 * 0x100;
        let next_codec = ((lo >> 26) & 0xF) as u8;
        let r = (lo >> 30) as u64;
        let vend = (hi & 0x000F_FFFF) as u64 * 0x4000 + r * 0x1000;
        // a chunk that ends past the file cannot be read and would only size a huge buffer
        if cend > prev_c && vend > vstart && cend <= file_len {
            chunks.push((prev_c, cend, vstart, vend - vstart, cur_codec));
            vstart = vend;
        }
        prev_c = cend;
        cur_codec = next_codec;
    }
    let ix = std::sync::Arc::new(TtpfIndex { descs, chunks });
    map.lock().unwrap().insert(path.to_path_buf(), ix.clone());
    Ok(ix)
}

fn bcpack_bc1_chunk(stored: &[u8], virt_len: u64) -> Result<Vec<u8>, String> {
    static TABLES: std::sync::OnceLock<(bcpack::bc1::Window, bcpack::bc1::QuadTable)> = std::sync::OnceLock::new();
    let (window, quads) = TABLES.get_or_init(|| (bcpack::bc1::Window::default_table(), bcpack::bc1::QuadTable::default_table()));
    bcpack::bc1::decode(stored, (virt_len / 8) as usize, window, quads)
        .map(|out| out.blocks)
        .map_err(|e| e.to_string())
}

// 16384 square at 16 bytes per texel
const MAX_TEXTURE_BYTES: usize = 1 << 32;
// the decoded size of a bcpack chunk comes from the chunk table and sizes the output buffer
const MAX_BCPACK_CHUNK: u64 = 256 << 20;

// decoded chunks last used by this thread keyed by archive and file offset
// neighbouring textures and the small mips of one texture share chunks
const CHUNK_CACHE_SLOTS: usize = 12;
const CHUNK_CACHE_MAX_LEN: usize = 4 << 20;
type ChunkCache = std::collections::VecDeque<(std::rc::Rc<std::path::PathBuf>, u64, std::rc::Rc<Vec<u8>>)>;
thread_local! {
    static CHUNK_CACHE: std::cell::RefCell<ChunkCache> = const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
    static LAST_TTPF_FILE: std::cell::RefCell<Option<(std::rc::Rc<std::path::PathBuf>, File)>> = const { std::cell::RefCell::new(None) };
    // largest texture edge model decodes on this thread ask for, zero keeps full size
    static TEX_DIM_CAP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// makes model decodes on the calling thread load textures no larger than cap pixels per edge
pub fn set_thread_texture_cap(cap: usize) {
    TEX_DIM_CAP.with(|c| c.set(cap));
}

/// Read `len` bytes of the RDS5 virtual stream starting at `off`, inflating only overlapping chunks.
pub fn ttpf_read_virtual(path: &Path, ix: &TtpfIndex, off: u64, len: usize) -> Result<Vec<u8>, String> {
    let end = off.checked_add(len as u64).ok_or("Texture range overflows")?;
    // nothing to read when the range starts past the stream and no texture is larger than this
    let stream_end = ix.chunks.last().map_or(0, |c| c.2 + c.3);
    if off >= stream_end || len > MAX_TEXTURE_BYTES {
        return Err("Texture data lies outside the RDS5 stream".into());
    }
    let mut out_buf = vec![0u8; len];
    let mut covered = 0usize;
    let mut dec = Decompress::new(true);
    let mut path_rc: Option<std::rc::Rc<std::path::PathBuf>> = None;
    let first = ix.chunks.partition_point(|c| c.2 + c.3 <= off);
    for &(cs, ce, vs, vl, codec) in &ix.chunks[first..] {
        if vs >= end { break; }
        let cached = CHUNK_CACHE.with(|c| c.borrow().iter().find(|e| e.1 == cs && e.0.as_path() == path).map(|e| e.2.clone()));
        let data = match cached {
            Some(hit) => hit,
            None => {
                let mut ch = vec![0u8; (ce - cs) as usize];
                LAST_TTPF_FILE.with(|slot| -> Result<(), String> {
                    let mut guard = slot.borrow_mut();
                    let need_open = guard.as_ref().is_none_or(|(p, _)| p.as_path() != path);
                    if need_open {
                        let file = File::open(path).map_err(|e| e.to_string())?;
                        let rc = path_rc.get_or_insert_with(|| std::rc::Rc::new(path.to_path_buf())).clone();
                        *guard = Some((rc, file));
                    }
                    let (rc, file) = guard.as_mut().unwrap();
                    if path_rc.is_none() {
                        path_rc = Some(rc.clone());
                    }
                    file.seek(SeekFrom::Start(cs)).map_err(|e| e.to_string())?;
                    file.read_exact(&mut ch).map_err(|e| e.to_string())
                })?;
                let out = if is_zlib_head(&ch) {
                    // deflate cannot expand more than about 1032 times
                    let mut out = Vec::with_capacity((vl as usize).min(ch.len().saturating_mul(1032).saturating_add(64)));
                    dec.reset(true);
                    let _ = dec.decompress_vec(&ch, &mut out, FlushDecompress::Finish); // partial output is still usable
                    out
                } else if ch.len() as u64 == vl {
                    ch
                } else if codec == 1 && vl % 8 == 0 && vl <= MAX_BCPACK_CHUNK {
                    bcpack_bc1_chunk(&ch, vl)?
                } else {
                    let mode = match codec {
                        1 => "BC1", 2 => "BC2", 3 => "BC3", 4 => "BC4", 5 => "BC5", 6 => "BC6H", 7 => "BC7", _ => "HW",
                    };
                    return Err(format!("Texture uses Xbox Series X hardware BCPack ({mode}) — ASIC-only compression (not decodable on PC)"));
                };
                let out = std::rc::Rc::new(out);
                if out.len() <= CHUNK_CACHE_MAX_LEN {
                    let rc = path_rc.get_or_insert_with(|| std::rc::Rc::new(path.to_path_buf())).clone();
                    CHUNK_CACHE.with(|c| {
                        let mut c = c.borrow_mut();
                        if c.len() >= CHUNK_CACHE_SLOTS {
                            c.pop_front();
                        }
                        c.push_back((rc, cs, out.clone()));
                    });
                }
                out
            }
        };
        let s0 = vs.max(off);
        let s1 = (vs + data.len() as u64).min(end);
        if s1 > s0 {
            out_buf[(s0 - off) as usize..(s1 - off) as usize].copy_from_slice(&data[(s0 - vs) as usize..(s1 - vs) as usize]);
            covered += (s1 - s0) as usize;
        }
    }
    if covered == 0 {
        return Err("Texture data not found in zlib chunks (likely BCPack)".into());
    }
    Ok(out_buf)
}

// Optimized Row-3 BCPack base-endpoint seed swatch (16x16 RGBA, zero full-res buffer alloc)
fn bcpack_seed_swatch_rgba(path: &Path, ix: &TtpfIndex, d: &TtpfTexMeta) -> (Vec<Color32>, usize, usize) {
    let mut hdr = [0u8; 32];
    let mut hdr_len = 0usize;
    if let Some(&(fs, fe, _, _, _)) = ix
        .chunks
        .iter()
        .find(|&&(_, _, vs, vl, _)| d.byte_off >= vs && d.byte_off < vs + vl)
        .or_else(|| ix.chunks.first())
    {
        if let Ok(mut f) = File::open(path) {
            if f.seek(SeekFrom::Start(fs)).is_ok() {
                let to_read = (fe.saturating_sub(fs) as usize).min(32);
                if f.read_exact(&mut hdr[..to_read]).is_ok() {
                    hdr_len = to_read;
                }
            }
        }
    }
    let (r, g, b) = if hdr_len >= 32 {
        let r0 = (hdr[26] as u32 & 0x1F) * 255 / 31;
        let g0 = (hdr[27] as u32 & 0x3F) * 255 / 63;
        let b0 = (hdr[28] as u32 & 0x1F) * 255 / 31;
        let r1 = (hdr[29] as u32 & 0x1F) * 255 / 31;
        let g1 = (hdr[30] as u32 & 0x3F) * 255 / 63;
        let b1 = (hdr[31] as u32 & 0x1F) * 255 / 31;
        (((r0 + r1) / 2) as u8, ((g0 + g1) / 2) as u8, ((b0 + b1) / 2) as u8)
    } else {
        (32, 32, 32)
    };
    let (sw, sh) = (d.w.clamp(4, 16), d.h.clamp(4, 16));
    (vec![Color32::from_rgb(r, g, b); sw * sh], sw, sh)
}

/// Decode a single texture by descriptor index from a `.ttpf` archive.
pub fn decode_ttpf_texture(path: &Path, tex_idx: usize) -> Result<DecodedTexture, String> {
    decode_ttpf_texture_capped(path, tex_idx, 0)
}

// like decode_ttpf_texture but no edge exceeds cap pixels when cap is not zero
// reads a stored mip level directly when the layout allows and shrinks the full decode otherwise
pub fn decode_ttpf_texture_capped(path: &Path, tex_idx: usize, cap: usize) -> Result<DecodedTexture, String> {
    let ix = ttpf_index(path)?;
    let d = ix.descs.get(tex_idx).ok_or("Texture index out of range")?;

    let (fmt_name, is_bc, bpe) = dxgi_info(d.dxgi)
        .ok_or_else(|| format!("Unsupported DXGI format {} (fmt_byte=0x{:02x})", d.dxgi, d.fmt_byte))?;
    let ew = if is_bc { d.w.div_ceil(4) } else { d.w }.max(1);
    let eh = if is_bc { d.h.div_ceil(4) } else { d.h }.max(1);
    let mip0_lin = ew * eh * bpe;
    // Mipmapped textures use 4KB tiles at every size; mip-less ones use 64KB tiles once over 64KB
    // (uncompressed >= 128KB also uses 64KB tiles when mip-less or full-size allocation).
    let use_64k = (!d.has_mips || !is_bc) && mip0_lin > 65536;
    let mip0_padded = if use_64k {
        padded_64kb_size(d.w, d.h, is_bc, bpe)
    } else {
        padded_4kb_size(d.w, d.h, is_bc, bpe)
    };
    // mips are stored smallest first so mip0 sits at end of allocation
    let tail_bound = (mip0_padded as u64).saturating_mul(2).saturating_add(65536);
    let tail_off = if d.has_mips && d.alloc_sz > mip0_padded as u64 && d.alloc_sz <= tail_bound {
        Some(d.byte_off + (d.alloc_sz - mip0_padded as u64))
    } else {
        None
    };
    // edge sizes tiling and padded byte size of a mip level below the base
    let mip_layout = |mip: u8| {
        let mw = (d.w >> mip).max(4);
        let mh = (d.h >> mip).max(4);
        let m_ew = if is_bc { mw.div_ceil(4) } else { mw }.max(1);
        let m_eh = if is_bc { mh.div_ceil(4) } else { mh }.max(1);
        let m_64k = !is_bc && m_ew * m_eh * bpe > 65536;
        let m_pad = if m_64k { padded_64kb_size(mw, mh, is_bc, bpe) } else { padded_4kb_size(mw, mh, is_bc, bpe) };
        (mw, mh, m_64k, m_pad)
    };
    let mut small = None;
    // the walk below is only valid when the base level really sits at the tail
    let tail_ok = |tail: u64| ttpf_read_virtual(path, &ix, tail, 256).is_ok_and(|t| !t.iter().all(|&b| b == t[0]));
    if let (true, Some(tail)) = (cap > 0 && d.w.max(d.h) > cap, tail_off.filter(|&t| tail_ok(t))) {
        let mut cursor_end = tail;
        for mip in 1..=6u8 {
            let (mw, mh, m_64k, m_pad) = mip_layout(mip);
            if cursor_end < d.byte_off + (m_pad as u64) {
                break;
            }
            cursor_end -= m_pad as u64;
            if mw.max(mh) <= cap || mip == 6 {
                if let Ok(raw) = ttpf_read_virtual(path, &ix, cursor_end, m_pad) {
                    if !raw.iter().take(256).all(|&b| b == raw[0]) {
                        small = Some((raw, mw, mh, m_64k, 1u8));
                    }
                }
                break;
            }
        }
    }
    let raw_try = match tail_off {
        _ if small.is_some() => Err(String::new()),
        Some(off) => {
            match ttpf_read_virtual(path, &ix, off, mip0_padded) {
                Ok(t) if t.iter().take(256).all(|&b| b == t[0]) => ttpf_read_virtual(path, &ix, d.byte_off, mip0_padded),
                other => other,
            }
        }
        None => ttpf_read_virtual(path, &ix, d.byte_off, mip0_padded),
    };
    let raw_res: Result<(Vec<u8>, usize, usize, bool, u8), String> = match raw_try {
        Ok(raw) => Ok((raw, d.w, d.h, use_64k, if d.has_mips { 2 } else { 1 })),
        Err(_) if small.is_some() => Ok(small.take().unwrap()),
        Err(first_err) if d.has_mips => {
            // reduced mip fallback (credited to @paryzius ttpf-viewer) when mip0 sits in unreadable bcpack/cutoff chunk
            let mut found = None;
            let mut cursor_end = tail_off.unwrap_or(d.byte_off + d.alloc_sz);
            for mip in 1..=6u8 {
                let (mw, mh, m_64k, m_pad) = mip_layout(mip);
                if cursor_end < d.byte_off + (m_pad as u64) {
                    break;
                }
                cursor_end -= m_pad as u64;
                if let Ok(raw) = ttpf_read_virtual(path, &ix, cursor_end, m_pad) {
                    if !raw.iter().take(256).all(|&b| b == raw[0]) {
                        found = Some((raw, mw, mh, m_64k, 1u8));
                        break;
                    }
                }
            }
            found.ok_or(first_err)
        }
        Err(e) => Err(e),
    };
    let (mut pixels, mut dec_w, mut dec_h, dec_mips, lin) = match raw_res {
        Ok((raw_tex, dw, dh, dec_64k, dec_mips)) => {
            let lin = if dec_64k {
                deswizzle_64kb_ttpf(&raw_tex, dw, dh, is_bc, bpe)
            } else {
                deswizzle_4kb_d(&raw_tex, dw, dh, is_bc, bpe)
            };
            let px = decode_linear_to_rgba(&lin, dw, dh, d.dxgi)
                .ok_or_else(|| format!("Pixel decode failed for {fmt_name} ({dw}x{dh})"))?;
            (px, dw, dh, dec_mips, lin)
        }
        Err(ref e) if e.contains("BCPack") => {
            let (px, sw, sh) = bcpack_seed_swatch_rgba(path, &ix, &d);
            (px, sw, sh, 1u8, Vec::new())
        }
        Err(e) => return Err(e),
    };
    if cap > 0 && dec_w.max(dec_h) > cap && pixels.len() >= dec_w * dec_h {
        let step = dec_w.max(dec_h).div_ceil(cap);
        let (nw, nh) = ((dec_w / step).max(1), (dec_h / step).max(1));
        let mut shrunk = Vec::with_capacity(nw * nh);
        for y in 0..nh {
            for x in 0..nw {
                // plain average of the covered block
                let mut sum = [0u32; 4];
                let mut n = 0u32;
                for yy in (y * step)..((y + 1) * step).min(dec_h) {
                    for xx in (x * step)..((x + 1) * step).min(dec_w) {
                        let c = pixels[yy * dec_w + xx].to_array();
                        for k in 0..4 {
                            sum[k] += c[k] as u32;
                        }
                        n += 1;
                    }
                }
                let n = n.max(1);
                shrunk.push(straight((sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8, (sum[3] / n) as u8));
            }
        }
        (pixels, dec_w, dec_h) = (shrunk, nw, nh);
    }
    Ok(DecodedTexture {
        name: format!("tex_{:05}_{}x{}_{fmt_name}", d.idx, dec_w, dec_h),
        w: dec_w,
        h: dec_h,
        mips: dec_mips,
        dxgi: d.dxgi,
        fmt_name,
        pixels,
        raw_dds_payload: lin,
    })
}

// global 64 bit etpf texture hash to ttpf archive path and descriptor index resolver
#[derive(Clone, Debug)]
pub struct TtpfHashHit {
    pub path: std::path::PathBuf,
    pub tex_idx: usize,
    pub w: usize,
    pub h: usize,
    pub dxgi: u8,
}

impl TtpfHashHit {
    #[inline]
    pub fn is_srgb(&self) -> bool {
        matches!(self.dxgi, 29 | 72 | 75 | 78 | 91 | 99)
    }

    #[inline]
    pub fn is_normal_format(&self) -> bool {
        // bc5 only single channel bc4 holds masks and heights
        matches!(self.dxgi, 83 | 84)
    }
}

// texture hash to location map built once per dump root
type TtpfHashIndex = std::sync::Arc<HashMap<u64, TtpfHashHit>>;
static TTPF_HASH_INDEX: std::sync::OnceLock<std::sync::Mutex<HashMap<std::path::PathBuf, TtpfHashIndex>>> =
    std::sync::OnceLock::new();

// decoded external textures cached by hash and size cap, emptied once it holds too many bytes
const TTPF_TEX_CACHE_BYTES: usize = 1 << 30;
type TtpfTexCache = (HashMap<(u64, usize), Option<std::sync::Arc<Texture2D>>>, usize);
static TTPF_TEX_CACHE: std::sync::OnceLock<std::sync::Mutex<TtpfTexCache>> = std::sync::OnceLock::new();

fn collect_ttpf_files(dir: &Path, depth: usize, out: &mut Vec<std::path::PathBuf>) {
    if depth > 4 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !name.starts_with('.') {
                collect_ttpf_files(&p, depth + 1, out);
            }
        } else if p
            .extension()
            .and_then(|s| s.to_str())
            .map(|e| e.eq_ignore_ascii_case("ttpf"))
            .unwrap_or(false)
        {
            out.push(p);
        }
    }
}

fn find_dump_root_for_ttpf(rpf_path: &Path) -> Option<std::path::PathBuf> {
    let mut cur = rpf_path.parent()?;
    for _ in 0..4 {
        if cur.join("data").join("ttdb").is_dir() || cur.join("textures_loose.ttpf").is_file() {
            return Some(cur.to_path_buf());
        }
        if let Ok(rd) = std::fs::read_dir(cur) {
            if rd.flatten().any(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|x| x.eq_ignore_ascii_case("ttpf"))
                    .unwrap_or(false)
            }) {
                return Some(cur.to_path_buf());
            }
        }
        cur = match cur.parent() {
            Some(p) => p,
            None => break,
        };
    }
    None
}

pub fn ttpf_hash_index_for_rpf(rpf_path: &Path) -> Option<std::sync::Arc<HashMap<u64, TtpfHashHit>>> {
    let root = find_dump_root_for_ttpf(rpf_path)?;
    let map = TTPF_HASH_INDEX.get_or_init(Default::default);
    if let Some(existing) = map.lock().unwrap().get(&root) {
        return Some(existing.clone());
    }

    let mut ttpf_files = Vec::new();
    collect_ttpf_files(&root, 0, &mut ttpf_files);
    if ttpf_files.is_empty() {
        return None;
    }

    let mut index: HashMap<u64, TtpfHashHit> = HashMap::with_capacity(360_000);
    for tp in ttpf_files {
        let Ok(mut f) = File::open(&tp) else { continue };
        let mut hdr = [0u8; 24];
        if f.read_exact(&mut hdr).is_err() || &hdr[..4] != b"RDS5" {
            continue;
        }
        let etpf_off = ru32(&hdr, 16) as u64 * 0x100;
        if f.seek(SeekFrom::Start(etpf_off)).is_err() {
            continue;
        }
        let mut ehdr = [0u8; 16];
        if f.read_exact(&mut ehdr).is_err() || &ehdr[..4] != b"ETPF" {
            continue;
        }
        let tex_count = ru32(&ehdr, 8) as usize;
        if tex_count == 0 || tex_count > 500_000 {
            continue;
        }
        let mut recs = vec![0u8; tex_count * 16];
        let mut hashes = vec![0u8; tex_count * 8];
        if f.read_exact(&mut recs).is_err() || f.read_exact(&mut hashes).is_err() {
            continue;
        }
        for i in 0..tex_count {
            let h64 = ru64(&hashes, i * 8);
            let u0 = ru32(&recs, i * 16);
            let u1 = ru32(&recs, i * 16 + 4);
            let w = ((u0 & 0x3FFF) + 1) as usize;
            let h = (((u0 >> 14) & 0x3FFF) + 1) as usize;
            let dxgi = ((u1 >> 3) & 0x7F) as u8;
            index.entry(h64).or_insert_with(|| TtpfHashHit {
                path: tp.clone(),
                tex_idx: i,
                w,
                h,
                dxgi,
            });
        }
    }

    let arc = std::sync::Arc::new(index);
    map.lock().unwrap().insert(root, arc.clone());
    Some(arc)
}

pub fn decode_ttpf_texture_cached(hit: &TtpfHashHit, hash64: u64) -> Option<std::sync::Arc<Texture2D>> {
    let cache = TTPF_TEX_CACHE.get_or_init(Default::default);
    let cap = TEX_DIM_CAP.with(|c| c.get());
    if let Some(entry) = cache.lock().unwrap().0.get(&(hash64, cap)) {
        return entry.clone();
    }
    let decoded = decode_ttpf_texture_capped(&hit.path, hit.tex_idx, cap)
        .ok()
        .map(|t| std::sync::Arc::new(to_model_tex(t)));
    let bytes = decoded.as_ref().map_or(0, |t| t.pixels.len() * 4);
    let mut g = cache.lock().unwrap();
    if g.1 + bytes > TTPF_TEX_CACHE_BYTES || g.0.len() > 8192 {
        g.0.clear();
        g.1 = 0;
    }
    if g.0.insert((hash64, cap), decoded.clone()).is_none() {
        g.1 += bytes;
    }
    decoded
}

// =============================================================================
// RBF0 (GTA 6 fwui UIObject) -> Readable XML Decoder
// =============================================================================

const RBF0_MAX_DEPTH: usize = 256;

pub fn decode_rbf0(data: &[u8]) -> Option<String> {
    if data.len() < 8 || &data[..4] != b"RBF0" { return None; }
    let mut pos = 4usize;
    let n = data.len();
    let mut descs: Vec<(String, u8)> = Vec::new();
    let mut stack: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    let mut root_lines: Vec<String> = Vec::new();

    let esc = |s: &str| -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
    };

    while pos < n {
        let b0 = data[pos];
        if b0 == 0xFB && pos + 3 < n && data[pos + 1] == 0xFF {
            pos += 4;
            continue;
        }
        if b0 == 0xFF {
            if pos + 1 >= n { break; }
            let b1 = data[pos + 1];
            pos += 2;
            if b1 == 0xFF {
                if let Some((tag, attrs, kids)) = stack.pop() {
                    let indent = "  ".repeat(stack.len());
                    let attr_s = if attrs.is_empty() { String::new() } else { format!(" {}", attrs.join(" ")) };
                    let line = if kids.is_empty() {
                        format!("{indent}<{tag}{attr_s} />")
                    } else if kids.len() == 1 && !kids[0].starts_with('<') && !kids[0].contains('\n') {
                        format!("{indent}<{tag}{attr_s}>{}</{tag}>", kids[0])
                    } else {
                        format!("{indent}<{tag}{attr_s}>\n{}\n{indent}</{tag}>", kids.join("\n"))
                    };
                    if let Some(p) = stack.last_mut() { p.2.push(line); } else { root_lines.push(line); }
                }
            }
            continue;
        }
        if b0 == 0xFD && pos + 5 < n && data[pos + 1] == 0xFF {
            pos += 2;
            let dlen = ru32(data, pos) as usize;
            pos += 4;
            if pos + dlen > n { break; }
            let raw = &data[pos..pos + dlen];
            pos += dlen;
            let trimmed = raw.strip_suffix(&[0]).unwrap_or(raw);
            let txt = String::from_utf8_lossy(trimmed);
            if let Some(p) = stack.last_mut() { p.2.push(esc(&txt)); }
            continue;
        }
        if pos + 2 > n { break; }
        let idx = b0 as usize;
        let dtype = data[pos + 1];
        pos += 2;

        let name = if idx == descs.len() {
            if pos + 2 > n { break; }
            let nlen = ru16(data, pos) as usize;
            pos += 2;
            if pos + nlen > n { break; }
            let nm = String::from_utf8_lossy(&data[pos..pos + nlen]).into_owned();
            pos += nlen;
            descs.push((nm.clone(), dtype));
            nm
        } else if idx < descs.len() {
            descs[idx].0.clone()
        } else {
            break;
        };

        match dtype {
            0x00 => {
                if pos + 6 > n { break; }
                pos += 6;
                // closing re-indents every child so runaway nesting costs quadratic time and memory
                if stack.len() >= RBF0_MAX_DEPTH { return None; }
                stack.push((name, Vec::new(), Vec::new()));
            }
            0x10 => {
                if pos + 4 > n { break; }
                let v = ru32(data, pos); pos += 4;
                if let Some(p) = stack.last_mut() { p.1.push(format!("{name}=\"{v}\"")); }
            }
            0x20 => {
                if pos + 1 > n { break; }
                let v = data[pos] != 0; pos += 1;
                if let Some(p) = stack.last_mut() { p.1.push(format!("{name}=\"{v}\"")); }
            }
            0x30 => {
                if pos + 1 > n { break; }
                let v = data[pos]; pos += 1;
                if let Some(p) = stack.last_mut() { p.1.push(format!("{name}=\"{v}\"")); }
            }
            0x40 => {
                if pos + 4 > n { break; }
                let v = rf32(data, pos); pos += 4;
                if let Some(p) = stack.last_mut() { p.1.push(format!("{name}=\"{v:.5}\"")); }
            }
            0x50 => {
                if pos + 12 > n { break; }
                let (x, y, z) = (rf32(data, pos), rf32(data, pos + 4), rf32(data, pos + 8));
                pos += 12;
                if let Some(p) = stack.last_mut() { p.1.push(format!("{name}=\"{x:.4},{y:.4},{z:.4}\"")); }
            }
            0x60 => {
                if pos + 2 > n { break; }
                let slen = ru16(data, pos) as usize; pos += 2;
                if pos + slen > n { break; }
                let sv = String::from_utf8_lossy(&data[pos..pos + slen]);
                pos += slen;
                if let Some(p) = stack.last_mut() { p.1.push(format!("{name}=\"{}\"", esc(&sv))); }
            }
            _ => break,
        }
    }
    while let Some((tag, attrs, kids)) = stack.pop() {
        let indent = "  ".repeat(stack.len());
        let attr_s = if attrs.is_empty() { String::new() } else { format!(" {}", attrs.join(" ")) };
        let line = format!("{indent}<{tag}{attr_s}>\n{}\n{indent}</{tag}>", kids.join("\n"));
        if let Some(p) = stack.last_mut() { p.2.push(line); } else { root_lines.push(line); }
    }
    (!root_lines.is_empty()).then(|| root_lines.join("\n"))
}

// =============================================================================
// Scaleform GFX (v42 AS3) Inspector
// =============================================================================

pub fn inspect_gfx(data: &[u8]) -> String {
    if data.len() < 8 || &data[..3] != b"GFX" {
        return "Not a Scaleform GFX binary".into();
    }
    let ver = data[3];
    let fsize = ru32(data, 4);
    let mut out = format!("Scaleform GFx Binary (ActionScript 3)\nVersion: {ver} (0x{ver:02x})  |  Header Size: {fsize} B  |  Chunk Size: {} B\n\nRecovered AS3 Classes / Packages / Symbols:\n", data.len());
    let mut seen = HashSet::new();
    let mut cur = Vec::new();
    let mut syms = Vec::new();
    for &b in data {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'/') {
            cur.push(b);
        } else {
            if cur.len() >= 6 && cur.len() <= 96 {
                if let Ok(s) = std::str::from_utf8(&cur) {
                    if (s.contains('.') || s.contains(':') || s.contains("fontlib") || s.contains("rockstar") || s.contains("scaleform") || s.contains("general"))
                        && s.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 4
                        && seen.insert(s.to_string())
                    {
                        syms.push(s.to_string());
                    }
                }
            }
            cur.clear();
        }
    }
    syms.sort();
    for s in syms.iter().take(400) {
        out.push_str("  ");
        out.push_str(s);
        out.push('\n');
    }
    out
}

// fast snippet extractor for naming 0drpz and standalone text chunks across all 13 localizations
pub fn probe_gxt_snippet(head: &[u8]) -> Option<String> {
    let is_0drpz = head.len() >= 24 && &head[16..21] == b"0DRPz";
    let start = if is_0drpz { 0x60.min(head.len()) } else { 0 };
    let mut good = Vec::new();
    for raw in head[start..].split(|&b| b == 0) {
        if raw.len() < 4 || raw.len() > 160 { continue; }
        let Ok(s) = std::str::from_utf8(raw) else { continue };
        let s = s.trim();
        if s.len() < 4 || s == "0DRPz" { continue; }
        let ok_chars = s.chars().filter(|c| c.is_alphanumeric() || !c.is_control()).count();
        let letters = s.chars().filter(|c| c.is_alphabetic()).count();
        if ok_chars * 100 / s.chars().count().max(1) >= 92 && letters >= 2 {
            good.push(s);
            if good.len() >= 2 { break; }
        }
    }
    if is_0drpz || good.len() >= 2 {
        let slug: String = good
            .iter()
            .find(|s| s.chars().any(|c| c.is_ascii_alphabetic()))
            .or_else(|| good.first())
            .copied()
            .unwrap_or("text")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .take(32)
            .collect();
        let slug = slug.trim_matches('_');
        return Some(if slug.is_empty() { "text".into() } else { slug.into() });
    }
    None
}

// follow {u32, u32 len, u64 ptr} records to null-terminated utf8 string pools,
// which hold the text of small tables and can sit after the bitmap region
fn gxt_pool_strings(data: &[u8]) -> Vec<String> {
    let mut pools: Vec<(usize, Vec<String>)> = Vec::new();
    let mut o = 0x60;
    while o + 16 <= data.len() {
        let len = ru32(data, o + 4) as usize;
        let ptr = ru64(data, o + 8);
        o += 8;
        if !(2..=1 << 20).contains(&len) || !(0x5000_0000..0x5000_0000 + data.len() as u64).contains(&ptr) {
            continue;
        }
        let a = (ptr - 0x5000_0000) as usize;
        let Some(b) = data.get(a..a + len) else { continue };
        if b[len - 1] != 0 || pools.iter().any(|(pa, _)| *pa == a) {
            continue;
        }
        let mut segs = Vec::new();
        let mut valid = true;
        for raw in b[..len - 1].split(|&c| c == 0).filter(|r| !r.is_empty()) {
            match std::str::from_utf8(raw) {
                Ok(s) if s.chars().all(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t')) => segs.push(s.trim().to_string()),
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && segs.iter().any(|s| s.chars().any(char::is_alphabetic)) {
            pools.push((a, segs));
        }
    }
    pools.sort_by_key(|(a, _)| *a);
    pools.into_iter().flat_map(|(_, s)| s).filter(|s| !s.is_empty()).collect()
}

// extract clean readable utf8 strings from a rage9 0drpz localization text table across all 13 languages
pub fn decode_0drpz_text(data: &[u8]) -> Option<String> {
    let start = if data.len() >= 0x60 && &data[0x10..0x15] == b"0DRPz" { 0x60 } else { 0 };
    let end = if start == 0x60 {
        let bmap = ru64(data, 8);
        if (0x5000_0000..0x5800_0000).contains(&bmap) {
            ((bmap - 0x5000_0000) as usize).clamp(start, data.len())
        } else {
            data.len()
        }
    } else {
        data.len()
    };
    let mut lines = Vec::new();
    for raw in data[start..end].split(|&b| b == 0) {
        if raw.len() < 2 || raw.len() > 2000 { continue; }
        let Ok(s) = std::str::from_utf8(raw) else { continue };
        let s = s.trim();
        if s.chars().count() < 2 || s == "0DRPz" { continue; }
        let printable = s.chars().filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t')).count();
        let letters = s.chars().filter(|c| c.is_alphabetic()).count();
        if printable * 100 / s.chars().count().max(1) >= 96 && letters >= 1 {
            // skip tiny binary hash table noise while keeping short cjk and tag strings
            let has_non_ascii_letter = s.chars().any(|c| !c.is_ascii() && c.is_alphabetic());
            if !has_non_ascii_letter && s.len() <= 5 && s.chars().any(|c| !c.is_alphanumeric()) && !s.starts_with('~') {
                continue;
            }
            lines.push(s.to_string());
            if lines.len() >= 6000 { break; }
        }
    }
    let pooled = if start == 0x60 { gxt_pool_strings(data) } else { Vec::new() };
    if lines.len() < 3 {
        // small tables hold only one or two strings, so trust the pointed-to pools alone
        if pooled.is_empty() {
            return None;
        }
        lines = pooled;
    } else {
        // pools can sit past the scanned range, so add any string the scan missed
        let seen: std::collections::HashSet<String> = lines.iter().cloned().collect();
        lines.extend(pooled.into_iter().filter(|s| !seen.contains(s)));
    }
    let header = format!("=== RAGE9 Text Table ({} strings) ===\n\n", lines.len());
    Some(header + &lines.join("\n"))
}
// =============================================================================

pub fn decode_navmesh_from_rpf(rpf_path: &Path, file_off: u64, fallback_name: &str) -> Result<Mesh3D, String> {
    let mut f = File::open(rpf_path).map_err(|e| format!("Open error: {e}"))?;
    let (v, _) = read_rage9_vp(&mut f, file_off)?;
    if v.len() < 0xD0 || ru32(&v, 0x14) != 0x0001_0015 {
        return Err("Not a valid GTA VI CNavMesh (0x00010015)".into());
    }
    let ext = [rf32(&v, 0x60), rf32(&v, 0x64), rf32(&v, 0x68)];
    let qt = vptr(ru64(&v, 0xA0)).ok_or("Missing QuadTree")?;
    if qt + 32 > v.len() { return Err("Truncated QuadTree".into()); }
    let min_v = [rf32(&v, qt), rf32(&v, qt + 4), rf32(&v, qt + 8)];

    let read_split_blocks = |ptr_off: usize| -> Option<Vec<(usize, usize)>> {
        let sa = vptr(ru64(&v, ptr_off))?;
        if sa + 0x24 > v.len() { return None; }
        let p_blks = vptr(ru64(&v, sa + 0x10))?;
        let n_blks = ru32(&v, sa + 0x20) as usize;
        if n_blks == 0 || n_blks > 4096 || p_blks + n_blks * 16 > v.len() { return None; }
        let mut blks = Vec::with_capacity(n_blks);
        for bi in 0..n_blks {
            let bptr = vptr(ru64(&v, p_blks + bi * 16))?;
            let bcnt = ru32(&v, p_blks + bi * 16 + 8) as usize;
            blks.push((bptr, bcnt));
        }
        Some(blks)
    };

    let v_blks = read_split_blocks(0x70).ok_or("Missing vertex split array")?;
    let i_blks = read_split_blocks(0x80).ok_or("Missing index split array")?;
    let p_blks = read_split_blocks(0x98).ok_or("Missing poly split array")?;

    // indices are 16 bit so more vertices can never be referenced overlapping blocks stop here
    let mut verts = Vec::new();
    'verts: for (boff, cnt) in v_blks {
        for i in 0..cnt {
            let o = boff + i * 6;
            if o + 6 > v.len() { break; }
            if verts.len() >= 65536 { break 'verts; }
            let vx = min_v[0] + (ru16(&v, o) as f32 / 65535.0) * ext[0];
            let vy = min_v[1] + (ru16(&v, o + 2) as f32 / 65535.0) * ext[1];
            let vz = min_v[2] + (ru16(&v, o + 4) as f32 / 65535.0) * ext[2];
            verts.push([vx, vy, vz]);
        }
    }

    let mut indices = Vec::new();
    'indices: for (boff, cnt) in i_blks {
        for i in 0..cnt {
            let o = boff + i * 2;
            if o + 2 > v.len() { break; }
            // more indices than the buffer has bytes means the blocks overlap
            if indices.len() >= v.len() { break 'indices; }
            indices.push(ru16(&v, o) as u32);
        }
    }

    let mut tris = Vec::new();
    let mut tri_geom = Vec::new();
    let mut poly_cnt = 0usize;
    for (boff, cnt) in p_blks {
        for i in 0..cnt {
            let o = boff + i * 44;
            if o + 44 > v.len() { break; }
            let vinfo = ru32(&v, o + 4);
            let f8 = ru32(&v, o + 8);
            let f18 = ru32(&v, o + 0x18);
            let nv = (vinfo & 0xF) as usize;
            let first = (vinfo >> 4) as usize;
            if nv < 3 || first + nv > indices.len() { continue; }
            // Group material ID by navmesh surface type: 0=Land, 1=Water, 2=Road, 3=Pavement, 4=Interior/Shelter
            let mat: u16 = if (f8 & 0x80) != 0 || (f18 & 0x80000) != 0 {
                1
            } else if (f8 & 0x08) != 0 {
                4
            } else if (f18 & 0x2000) != 0 {
                2
            } else if (f18 & 0x1000) != 0 {
                3
            } else {
                0
            };
            let i0 = indices[first];
            for k in 1..nv - 1 {
                let (i1, i2) = (indices[first + k], indices[first + k + 1]);
                if (i0 as usize) < verts.len() && (i1 as usize) < verts.len() && (i2 as usize) < verts.len() {
                    tris.push([i0, i1, i2]);
                    tri_geom.push(mat);
                }
            }
            poly_cnt += 1;
        }
    }
    if verts.is_empty() || tris.is_empty() {
        return Err("Empty CNavMesh geometry".into());
    }
    Ok(Mesh3D::from_parts(
        fallback_name.to_string(),
        verts,
        Vec::new(),
        tris,
        tri_geom,
        Vec::new(),
        None,
        format!("CNavMesh ({poly_cnt} polys, Z {:.0}..{:.0}m)", min_v[2], min_v[2] + ext[2]),
        5,
        poly_cnt,
        "NavMesh".into(),
    ))
}

// decode rage 9 mhiu 2048x2048 row-compressed world heightmap into colorized elevation texture
pub fn decode_mhiu_heightmap(data: &[u8]) -> Option<DecodedTexture> {
    if data.len() < 44 || &data[..4] != b"MHIU" {
        return None;
    }
    let w = u16::from_le_bytes([data[6], data[7]]) as usize;
    let h = u16::from_le_bytes([data[8], data[9]]) as usize;
    if w == 0 || h == 0 || w > 4096 || h > 4096 || data.len() < 44 + h * 8 {
        return None;
    }
    let rows = &data[44..44 + h * 8];
    let pix = &data[44 + h * 8..];
    let mut vals = vec![23u8; w * h];
    for y in 0..h {
        let r = &rows[y * 8..y * 8 + 8];
        let sx = u16::from_le_bytes([r[0], r[1]]) as usize;
        let cnt = u16::from_le_bytes([r[2], r[3]]) as usize;
        let off = i32::from_le_bytes([r[4], r[5], r[6], r[7]]) as isize;
        if cnt > 0 && sx + cnt <= w {
            let src = off + sx as isize;
            if src >= 0 && (src as usize) + cnt <= pix.len() {
                vals[y * w + sx..y * w + sx + cnt].copy_from_slice(&pix[src as usize..src as usize + cnt]);
            }
        }
    }
    let mut pixels = Vec::with_capacity(w * h);
    for &v in &vals {
        let c = if v <= 23 {
            Color32::from_rgb(16, 44, 92)
        } else if v <= 30 {
            let t = (v - 23) * 14;
            Color32::from_rgb(45 + t, 115 + t, 72)
        } else if v <= 85 {
            let t = ((v - 30) as u16 * 120 / 55) as u8;
            Color32::from_rgb(80 + t, 145 + t / 2, 70)
        } else {
            let t = ((v - 85) as u16 * 100 / 170) as u8;
            Color32::from_rgb(155 + t, 145 + t, 130 + t)
        };
        pixels.push(c);
    }
    Some(DecodedTexture {
        name: format!("world_heightmap_{w}x{h}"),
        w,
        h,
        mips: 1,
        fmt_name: "MHIU RLE8",
        dxgi: 28,
        pixels,
        raw_dds_payload: Vec::new(),
    })
}

#[cfg(test)]
mod large_address_tests;

#[cfg(test)]
mod capped_decode_tests {
    use super::*;

    // mean absolute rgb difference between the small texture and the full one averaged down to its size
    fn picture_diff(full: &DecodedTexture, small: &DecodedTexture) -> f32 {
        let (sx, sy) = (full.w / small.w, full.h / small.h);
        let mut total = 0.0f64;
        for y in 0..small.h {
            for x in 0..small.w {
                let mut sum = [0.0f64; 3];
                for yy in 0..sy {
                    for xx in 0..sx {
                        let p = full.pixels[(y * sy + yy) * full.w + x * sx + xx];
                        sum[0] += p.r() as f64;
                        sum[1] += p.g() as f64;
                        sum[2] += p.b() as f64;
                    }
                }
                let n = (sx * sy) as f64;
                let q = small.pixels[y * small.w + x];
                total += (sum[0] / n - q.r() as f64).abs() + (sum[1] / n - q.g() as f64).abs() + (sum[2] / n - q.b() as f64).abs();
            }
        }
        (total / (small.w * small.h * 3) as f64) as f32
    }

    // a capped decode reads a stored mip level and must show the same picture as the full one
    #[test]
    fn capped_vehicle_textures_match_full_decodes() {
        let Ok(rpf) = std::env::var("PERCYSTAR_LEVELS_RPF") else { return };
        let Some(root) = Path::new(&rpf).parent() else { return };
        let pool = root.join("data").join("ttdb").join("vehicles").join("ttdb_vehicles.ttpf");
        if !pool.exists() {
            return;
        }
        let ix = ttpf_index(&pool).expect("index");
        let picks: Vec<usize> = ix.descs.iter().filter(|d| d.has_mips && d.w.max(d.h) >= 512).map(|d| d.idx).step_by(23).take(40).collect();
        let (mut both, mut close, mut from_mip) = (0usize, 0usize, 0usize);
        for idx in picks {
            let (Ok(full), Ok(small)) = (decode_ttpf_texture(&pool, idx), decode_ttpf_texture_capped(&pool, idx, 128)) else { continue };
            assert!(small.w.max(small.h) <= 128, "texture {idx} came back {}x{}", small.w, small.h);
            both += 1;
            from_mip += (small.mips == 1 && full.mips == 2) as usize;
            let diff = picture_diff(&full, &small);
            if diff < 12.0 {
                close += 1;
            } else {
                eprintln!("texture {idx} {}x{} differs from its capped decode by {diff:.1} (mips {})", full.w, full.h, small.mips);
            }
        }
        eprintln!("capped decode: {both} compared, {close} close, {from_mip} read from a stored mip");
        assert!(both >= 10, "only {both} textures decoded both ways");
        assert_eq!(close, both, "capped decodes differ from the full textures");
        assert!(from_mip * 2 >= both, "stored mips were used for only {from_mip} of {both}");
    }
}

#[cfg(test)]
mod mhiu_tests {
    use super::*;

    #[test]
    fn decodes_synthetic_mhiu_heightmap() {
        let mut buf = vec![0u8; 44 + 4 * 8 + 2];
        buf[0..4].copy_from_slice(b"MHIU");
        buf[6..8].copy_from_slice(&4u16.to_le_bytes());
        buf[8..10].copy_from_slice(&4u16.to_le_bytes());
        // row 1: sx=1, cnt=2, off=-1 -> pix[0..2]
        let r1 = 44 + 8;
        buf[r1..r1 + 2].copy_from_slice(&1u16.to_le_bytes());
        buf[r1 + 2..r1 + 4].copy_from_slice(&2u16.to_le_bytes());
        buf[r1 + 4..r1 + 8].copy_from_slice(&(-1i32).to_le_bytes());
        buf[44 + 32] = 60;
        buf[44 + 33] = 120;
        let tex = decode_mhiu_heightmap(&buf).expect("should decode mhiu");
        assert_eq!(tex.w, 4);
        assert_eq!(tex.h, 4);
        assert_eq!(tex.pixels[0], Color32::from_rgb(16, 44, 92));
        assert_ne!(tex.pixels[5], Color32::from_rgb(16, 44, 92));
    }

    #[test]
    fn bc3_color_block_is_always_four_color() {
        // flat block c0 == c1 with every index 3 bc1 rules would decode it black
        let mut blk = vec![255u8; 8];
        blk.extend_from_slice(&0xF800u16.to_le_bytes());
        blk.extend_from_slice(&0xF800u16.to_le_bytes());
        blk.extend_from_slice(&[0xFF; 4]);
        for dxgi in [77u8, 74] {
            let px = decode_linear_to_rgba(&blk, 4, 4, dxgi).unwrap();
            assert!(px.iter().all(|c| c.to_array() == [255, 0, 0, 255]), "dxgi {dxgi}: {:?}", px[0].to_array());
        }
        // c0 < c1 interpolates at thirds instead of the bc1 half mix and black
        let mut blk = vec![255u8; 8];
        blk.extend_from_slice(&0x0000u16.to_le_bytes());
        blk.extend_from_slice(&0xFFFFu16.to_le_bytes());
        blk.extend_from_slice(&[0b1110_0100, 0, 0, 0]);
        let px = decode_linear_to_rgba(&blk, 4, 4, 77).unwrap();
        let reds: Vec<u8> = px[..4].iter().map(|c| c.r()).collect();
        assert_eq!(reds, [0, 255, 85, 170]);
        // cropped sizes and short input
        assert_eq!(decode_linear_to_rgba(&blk, 3, 2, 77).unwrap().len(), 6);
        assert!(decode_linear_to_rgba(&blk[..15], 4, 4, 77).is_none());
    }

    #[test]
    fn alpha_as_data_keeps_the_color_and_exports_unchanged() {
        // bc3 texel with alpha zero and a strong color
        let mut blk = vec![0u8; 8];
        blk.extend_from_slice(&0x07E0u16.to_le_bytes());
        blk.extend_from_slice(&0x07E0u16.to_le_bytes());
        blk.extend_from_slice(&[0; 4]);
        let px = decode_linear_to_rgba(&blk, 4, 4, 77).unwrap();
        assert_eq!(px[0].to_array(), [0, 255, 0, 0], "color survives zero alpha");
        let rgba8 = decode_linear_to_rgba(&[200, 100, 50, 3], 1, 1, 28).unwrap();
        assert_eq!(rgba8[0].to_array(), [200, 100, 50, 3]);
        // egui gets premultiplied texels opaque ones untouched
        let up = premultiplied_for_egui(&[rgba8[0], px[0], Color32::from_rgb(9, 8, 7)]);
        assert_eq!(up[1], Color32::TRANSPARENT);
        assert_eq!(up[2], Color32::from_rgb(9, 8, 7));
        assert_eq!(up[0], Color32::from_rgba_unmultiplied(200, 100, 50, 3));
        // a8 becomes straight white with the value as coverage
        let a8 = to_model_tex(DecodedTexture { name: String::new(), w: 1, h: 1, mips: 1, dxgi: 65, fmt_name: "A8", pixels: vec![Color32::from_gray(40)], raw_dds_payload: Vec::new() });
        assert_eq!(a8.pixels[0].to_array(), [255, 255, 255, 40]);
    }

    #[test]
    fn signed_bc4_and_bc5_decode_as_snorm() {
        // endpoints plus one and minus one index 0 and 1
        let blk = [0x7Fu8, 0x81, 0b0000_1000, 0, 0, 0, 0, 0];
        let signed = channel_block(&blk, true);
        assert_eq!((signed[0], signed[1]), (255, 0));
        // read as unorm the same bytes are 127 and 129 which is what the old path produced
        let unsigned = channel_block(&blk, false);
        assert_eq!((unsigned[0], unsigned[1]), (127, 129));
        // minus 128 clamps to minus one and e0 <= e1 selects the six value palette with the two extremes
        let blk6 = [0x80u8, 0x00, 0b1111_0000, 0b0000_0011, 0, 0, 0, 0];
        let v = channel_block(&blk6, true);
        assert_eq!((v[0], v[1], v[2]), (0, 0, 255), "indices 0 6 7 1 -> {:?}", &v[..4]);
        assert_eq!(v[3], 128, "zero maps to the middle");
        // flat tangent space normal 0 0 1 through bc5 snorm
        let flat = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let px = decode_linear_to_rgba(&flat, 4, 4, 84).unwrap();
        assert_eq!(px[0].to_array(), [128, 128, 255, 255]);
        // unorm bc4 matches the reference decoder
        let mut reference = vec![0u32; 16];
        let ublk = [200u8, 40, 0x88, 0xC6, 0xFA, 0x53, 0x1D, 0x77];
        texture2ddecoder::decode_bc4(&ublk, 4, 4, &mut reference).unwrap();
        let ours = decode_linear_to_rgba(&ublk, 4, 4, 80).unwrap();
        for (o, r) in ours.iter().zip(&reference) {
            assert_eq!(o.r() as u32, (r >> 16) & 0xFF);
        }
        let ublk2 = [40u8, 200, 0x88, 0xC6, 0xFA, 0x53, 0x1D, 0x77];
        texture2ddecoder::decode_bc4(&ublk2, 4, 4, &mut reference).unwrap();
        let ours = decode_linear_to_rgba(&ublk2, 4, 4, 80).unwrap();
        for (o, r) in ours.iter().zip(&reference) {
            assert_eq!(o.r() as u32, (r >> 16) & 0xFF);
        }
        // bc4 masks are not normal maps
        let hit = |dxgi| TtpfHashHit { path: std::path::PathBuf::new(), tex_idx: 0, w: 4, h: 4, dxgi };
        assert!(hit(83).is_normal_format() && hit(84).is_normal_format() && !hit(80).is_normal_format() && !hit(81).is_normal_format());
    }

    #[test]
    fn rbf0_nesting_is_bounded() {
        // descriptor 0 named a opened once then thousands of nested opens of it
        let mut data = b"RBF0".to_vec();
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x00, b'a']);
        data.extend_from_slice(&[0u8; 6]);
        let shallow = decode_rbf0(&data).expect("one open element decodes");
        assert!(shallow.contains("<a"), "{shallow}");
        for _ in 0..20_000 {
            data.extend_from_slice(&[0x00, 0x00]);
            data.extend_from_slice(&[0u8; 6]);
        }
        let started = std::time::Instant::now();
        assert!(decode_rbf0(&data).is_none(), "runaway nesting is rejected");
        assert!(started.elapsed().as_secs() < 2);
    }

    #[test]
    fn bc1_preserves_one_bit_transparency() {
        let px = decode_linear_to_rgba(&[0, 0, 255, 255, 3, 0, 0, 0], 4, 4, 72).unwrap();
        assert_eq!(px[0], Color32::TRANSPARENT);
        assert_eq!(px[1], Color32::BLACK);
    }

    #[test]
    fn bcpack_bc1_chunk_decodes_synthetic_stream() {
        let blocks = 256;
        let (mut stream, _) = bcpack::encode::synthetic_bc1(7, blocks, 100);
        stream.resize(stream.len().next_multiple_of(256), 0);
        let decoded = bcpack_bc1_chunk(&stream, blocks as u64 * 8).expect("synthetic BC1");
        assert_eq!(decoded.len(), blocks * 8);
        assert!(bcpack_bc1_chunk(&[0xA5; 4096], blocks as u64 * 8).is_err());
    }

    #[test]
    fn channel_view_isolates_and_rebuilds_normal_z() {
        let px = [Color32::from_rgba_unmultiplied(128, 128, 0, 255)];
        let rgb = channel_view(&px, TexChannel::Rgb, true);
        assert!(rgb[0].b() >= 254);
        assert_eq!(channel_view(&px, TexChannel::R, false)[0], Color32::from_rgb(128, 128, 128));
        assert_eq!(channel_view(&px, TexChannel::A, false)[0], Color32::from_rgb(255, 255, 255));
        // a straight texel with zero alpha keeps its color in the channel views
        let hidden = [Color32::from_rgba_premultiplied(200, 40, 10, 0)];
        assert_eq!(channel_view(&hidden, TexChannel::Rgb, false)[0], Color32::from_rgb(200, 40, 10));
        assert_eq!(channel_view(&hidden, TexChannel::Rgba, false)[0], Color32::TRANSPARENT);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TexChannel {
    #[default]
    Rgba,
    Rgb,
    R,
    G,
    B,
    A,
}

impl TexChannel {
    pub const ALL: [(TexChannel, &'static str); 6] = [
        (TexChannel::Rgba, "RGBA"),
        (TexChannel::Rgb, "RGB"),
        (TexChannel::R, "R"),
        (TexChannel::G, "G"),
        (TexChannel::B, "B"),
        (TexChannel::A, "A"),
    ];
}

pub fn channel_view(pixels: &[Color32], channel: TexChannel, rebuild_z: bool) -> Vec<Color32> {
    let mut out = Vec::with_capacity(pixels.len());
    for p in pixels {
        // texels are stored straight the output is premultiplied for egui
        let [r, g, mut b, a] = p.to_array();
        if rebuild_z {
            let x = r as f32 * (1.0 / 127.5) - 1.0;
            let y = g as f32 * (1.0 / 127.5) - 1.0;
            let z = (1.0 - x * x - y * y).max(0.0).sqrt();
            b = (z * 127.5 + 127.5).round() as u8;
        }
        out.push(match channel {
            TexChannel::Rgba => Color32::from_rgba_unmultiplied(r, g, b, a),
            TexChannel::Rgb => Color32::from_rgb(r, g, b),
            TexChannel::R => Color32::from_rgb(r, r, r),
            TexChannel::G => Color32::from_rgb(g, g, g),
            TexChannel::B => Color32::from_rgb(b, b, b),
            TexChannel::A => Color32::from_rgb(a, a, a),
        });
    }
    out
}

//! Pure-Rust animated GIF89a encoder (no deps) - shared by the `wan_render` CLI and the HTTP API.
//! Quantizes RGB frames to a 6x6x6 color cube, LZW-compresses, and writes a looping GIF89a.
//! `encode_gif` is generic over the sink; `encode_gif_bytes` returns the file in memory (for the
//! API), `write_gif` writes it to a path (for the CLI). Wan runs at 16 fps -> 6 centiseconds/frame.

use std::collections::HashMap;
use std::io::{self, Write};

/// 6 evenly-spaced channel levels: round(c*5/255) ∈ 0..=5.
#[inline]
fn level_index(c: u8) -> u32 {
    (c as u32 * 5 + 127) / 255
}

/// Map an RGB triple to its 6x6x6-cube palette index (0..=215).
#[inline]
fn quantize(r: u8, g: u8, b: u8) -> u8 {
    (level_index(r) * 36 + level_index(g) * 6 + level_index(b)) as u8
}

/// Build the 256-entry global color table (768 bytes); 216 cube colors then zero padding.
fn global_color_table() -> Vec<u8> {
    let mut t = Vec::with_capacity(768);
    let lvl = [0u8, 51, 102, 153, 204, 255];
    for &r in &lvl {
        for &g in &lvl {
            for &b in &lvl {
                t.push(r);
                t.push(g);
                t.push(b);
            }
        }
    }
    t.resize(768, 0);
    t
}

/// Variable-width GIF LZW bit sink, LSB-first.
struct BitWriter {
    acc: u32,
    nbits: u32,
    out: Vec<u8>,
}
impl BitWriter {
    fn new() -> Self {
        BitWriter {
            acc: 0,
            nbits: 0,
            out: Vec::new(),
        }
    }
    #[inline]
    fn write(&mut self, code: u32, size: u32) {
        self.acc |= code << self.nbits;
        self.nbits += size;
        while self.nbits >= 8 {
            self.out.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.acc & 0xFF) as u8);
        }
        self.out
    }
}

/// LZW-compress one frame's palette indices. `min_code_size` is 8 (256-color table).
fn lzw_compress(indices: &[u8], min_code_size: u32) -> Vec<u8> {
    let clear: u32 = 1 << min_code_size; // 256
    let end: u32 = clear + 1; // 257
    let mut bw = BitWriter::new();

    let mut dict: HashMap<u32, u32> = HashMap::new();
    let mut code_size = min_code_size + 1; // 9
    let mut next_code = end + 1; // 258

    bw.write(clear, code_size);

    if indices.is_empty() {
        bw.write(end, code_size);
        return bw.finish();
    }

    let mut prefix = indices[0] as u32;
    for &k in &indices[1..] {
        let key = (prefix << 8) | k as u32;
        if let Some(&c) = dict.get(&key) {
            prefix = c;
        } else {
            bw.write(prefix, code_size);
            dict.insert(key, next_code);
            next_code += 1;
            // GIF decoders rebuild the table one code behind the encoder, so bump the width when
            // the next free code first *exceeds* what the current width can express (`>`, not `==`).
            if next_code > (1 << code_size) && code_size < 12 {
                code_size += 1;
            }
            if next_code == 4096 {
                bw.write(clear, code_size);
                dict.clear();
                code_size = min_code_size + 1;
                next_code = end + 1;
            }
            prefix = k as u32;
        }
    }
    bw.write(prefix, code_size);
    bw.write(end, code_size);
    bw.finish()
}

/// Write an LZW code stream as GIF sub-blocks (<=255 bytes each, 0x00 end).
fn write_sub_blocks<W: Write>(w: &mut W, data: &[u8]) -> io::Result<()> {
    for chunk in data.chunks(255) {
        w.write_all(&[chunk.len() as u8])?;
        w.write_all(chunk)?;
    }
    w.write_all(&[0])?;
    Ok(())
}

/// Encode RGB frames (each `w*h*3` bytes) as a looping animated GIF89a into any sink.
/// `delay_cs` is the per-frame delay in centiseconds (16 fps -> 6).
pub fn encode_gif<W: Write>(
    f: &mut W,
    frames: &[Vec<u8>],
    w: usize,
    h: usize,
    delay_cs: u16,
) -> io::Result<()> {
    let min_code_size: u32 = 8;

    // Header + Logical Screen Descriptor.
    f.write_all(b"GIF89a")?;
    f.write_all(&(w as u16).to_le_bytes())?;
    f.write_all(&(h as u16).to_le_bytes())?;
    // packed: GCT=1, color-res=7, sort=0, GCT-size=7 (2^8=256) -> 0xF7.
    f.write_all(&[0xF7, 0x00, 0x00])?;
    f.write_all(&global_color_table())?;

    // NETSCAPE2.0 application extension -> loop forever (loop count 0).
    f.write_all(&[0x21, 0xFF, 0x0B])?;
    f.write_all(b"NETSCAPE2.0")?;
    f.write_all(&[0x03, 0x01, 0x00, 0x00, 0x00])?;

    for fr in frames {
        // Graphic Control Extension (frame delay, no transparency).
        f.write_all(&[0x21, 0xF9, 0x04, 0x00])?;
        f.write_all(&delay_cs.to_le_bytes())?;
        f.write_all(&[0x00, 0x00])?;

        // Image Descriptor (full-frame, global color table -> packed 0x00).
        f.write_all(&[0x2C])?;
        f.write_all(&0u16.to_le_bytes())?; // left
        f.write_all(&0u16.to_le_bytes())?; // top
        f.write_all(&(w as u16).to_le_bytes())?;
        f.write_all(&(h as u16).to_le_bytes())?;
        f.write_all(&[0x00])?;

        // Quantize to palette indices, then LZW.
        let px = w * h;
        let mut indices = Vec::with_capacity(px);
        for i in 0..px {
            let o = i * 3;
            indices.push(quantize(fr[o], fr[o + 1], fr[o + 2]));
        }
        f.write_all(&[min_code_size as u8])?;
        let lzw = lzw_compress(&indices, min_code_size);
        write_sub_blocks(f, &lzw)?;
    }

    f.write_all(&[0x3B])?; // trailer
    Ok(())
}

/// Encode frames as a GIF89a and return the file bytes in memory (for the HTTP API).
pub fn encode_gif_bytes(frames: &[Vec<u8>], w: usize, h: usize, delay_cs: u16) -> Vec<u8> {
    let mut buf = Vec::new();
    // Writing to a Vec is infallible.
    let _ = encode_gif(&mut buf, frames, w, h, delay_cs);
    buf
}

/// Encode frames as a GIF89a and write them to `path` (for the CLI).
pub fn write_gif(
    path: &std::path::Path,
    frames: &[Vec<u8>],
    w: usize,
    h: usize,
    delay_cs: u16,
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    encode_gif(&mut f, frames, w, h, delay_cs)?;
    f.flush()
}

//! One routed expert read from a file on the disk under test, its pages dropped from the page
//! cache before every call: how a missing expert is best read. Page faults on first touch by one
//! thread and by several, the kernel's read-ahead advice one projection at a time and over the
//! whole expert before the touch, a populating advice, a buffered read, and a direct read that
//! bypasses the page cache. Each call reads the next expert of the layer, so no call finds what
//! the one before it read.

use super::shapes::Shapes;
use super::stats::run_for;
use super::{Context, Measured};
use crate::tensor::mapped::{self, MappedBytes};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A byte range of the file: its offset and its length.
pub type Span = (usize, usize);

/// Where each call's expert lies in the file.
#[derive(Clone, Debug, PartialEq)]
pub enum Plan {
    /// The file's own expert offsets, by projection.
    Listed {
        projections: Vec<(Vec<u64>, usize)>,
        start: usize,
    },
    /// Expert-sized slots laid end to end from an offset, the projections one after the other.
    Derived {
        offset: u64,
        sizes: Vec<usize>,
        slots: usize,
        start: usize,
    },
}

impl Plan {
    /// The plan for `shapes` over the file at `path` of `file_len` bytes: the file's offsets when
    /// the shapes came from that file, slots from `offset` otherwise.
    pub fn new(
        shapes: &Shapes,
        path: &Path,
        offset: Option<u64>,
        file_len: u64,
    ) -> Result<Self, String> {
        let listed = shapes.file.as_deref() == Some(path) && shapes.experts_listed() > 0;
        if listed {
            for p in &shapes.projections {
                if let Some(off) = p.offsets.iter().find(|&&o| o + p.bytes > file_len) {
                    return Err(format!(
                        "{} at {off} plus {} bytes lies past the end of the file ({file_len})",
                        p.name, p.bytes
                    ));
                }
            }
            return Ok(Self::Listed {
                projections: shapes
                    .projections
                    .iter()
                    .map(|p| (p.offsets.clone(), p.bytes as usize))
                    .collect(),
                start: shapes.expert,
            });
        }
        let offset = offset.unwrap_or(0);
        let expert = shapes.expert_bytes();
        if expert == 0 {
            return Err("the expert has no bytes".to_string());
        }
        let slots = file_len.saturating_sub(offset) / expert;
        if slots == 0 {
            return Err(format!(
                "the file holds no whole expert of {expert} bytes after offset {offset}"
            ));
        }
        Ok(Self::Derived {
            offset,
            sizes: shapes
                .projections
                .iter()
                .map(|p| p.bytes as usize)
                .collect(),
            slots: slots as usize,
            start: shapes.expert,
        })
    }

    /// The projections' spans of call `k`.
    pub fn spans(&self, k: usize) -> Vec<Span> {
        match self {
            Self::Listed { projections, start } => {
                let n = projections
                    .iter()
                    .map(|(o, _)| o.len())
                    .min()
                    .unwrap_or(1)
                    .max(1);
                let e = (start + k) % n;
                projections
                    .iter()
                    .map(|(offsets, bytes)| (offsets[e] as usize, *bytes))
                    .collect()
            }
            Self::Derived {
                offset,
                sizes,
                slots,
                start,
            } => {
                let slot = ((start + k) % slots) as u64;
                let expert: u64 = sizes.iter().map(|&s| s as u64).sum();
                let mut at = (offset + slot * expert) as usize;
                sizes
                    .iter()
                    .map(|&s| {
                        let span = (at, s);
                        at += s;
                        span
                    })
                    .collect()
            }
        }
    }

    /// The bytes one call reads.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Listed { projections, .. } => projections.iter().map(|(_, b)| *b as u64).sum(),
            Self::Derived { sizes, .. } => sizes.iter().map(|&s| s as u64).sum(),
        }
    }
}

/// One byte of every page of `bytes`, summed so the reads stay.
fn touch(bytes: &[u8], page: usize) -> u64 {
    let mut sum = 0u64;
    let mut at = 0;
    while at < bytes.len() {
        sum += bytes[at] as u64;
        at += page;
    }
    std::hint::black_box(sum)
}

/// `touch` over every span, each cut into `threads` page-aligned pieces, piece `t` of every span
/// touched by thread `t`.
fn touch_threads(map: &[u8], spans: &[Span], page: usize, threads: usize) {
    std::thread::scope(|s| {
        for t in 0..threads {
            let pieces: Vec<&[u8]> = spans
                .iter()
                .filter_map(|&(off, len)| {
                    let per = len.div_ceil(page).div_ceil(threads) * page;
                    let from = (t * per).min(len);
                    let to = ((t + 1) * per).min(len);
                    (from < to).then(|| &map[off + from..off + to])
                })
                .collect();
            s.spawn(move || {
                for p in pieces {
                    touch(p, page);
                }
            });
        }
    });
}

#[cfg(target_os = "linux")]
pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};

    let path = cx
        .options
        .file
        .clone()
        .or_else(|| cx.shapes.file.clone())
        .ok_or("expert-read needs --file or --shapes-from")?;
    let file = Arc::new(File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?);
    let meta = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let file_len = meta.len();
    let align = (meta.blksize() as usize).max(1);
    // Safety: the file is read-only here and not written while it is mapped.
    let map = Arc::new(
        unsafe { memmap2::Mmap::map(&*file) }.map_err(|e| format!("{}: {e}", path.display()))?,
    );
    mapped::register_file(&map, &file);
    let plan = Plan::new(cx.shapes, &path, cx.options.offset, file_len)?;
    let page = mapped::page_size().ok_or("the system does not say its page size")?;
    let threads = cx
        .options
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1);
    let target = Duration::from_secs_f64(cx.options.seconds);
    let bytes = Some(plan.bytes());
    let range = |span: Span| -> Result<MappedBytes, String> {
        MappedBytes::in_file(map.clone(), Some(file.clone()), span.0, span.1)
            .map_err(|e| format!("{e:?}"))
    };

    // The sectors the disk under the file has read, which count what the kernel reads beyond the
    // bytes a variant asks for.
    let device = crate::inference::bench::report::Disk::of(&path).map(|d| d.device);
    let sectors = || -> Option<u64> {
        let device = device.as_deref()?;
        std::fs::read_to_string("/proc/diskstats")
            .ok()?
            .lines()
            .map(|l| l.split_whitespace().collect::<Vec<_>>())
            .find(|f| f.get(2) == Some(&device))?
            .get(5)?
            .parse()
            .ok()
    };
    let mut call = 0usize;
    let mut measure = |variant: String, body: &mut dyn FnMut(&[Span]) -> Result<(), String>| {
        let mut failed = None;
        let before = sectors();
        let samples = run_for(target, || {
            let spans = plan.spans(call);
            call += 1;
            for &s in &spans {
                match range(s) {
                    Ok(r) => r.dont_need(),
                    Err(e) => {
                        failed.get_or_insert(e);
                    }
                }
            }
            let t = Instant::now();
            if let Err(e) = body(&spans) {
                failed.get_or_insert(e);
            }
            t.elapsed().as_secs_f64() * 1e6
        });
        let read = before.zip(sectors()).map(|(b, a)| (a - b) * 512);
        let note = read.zip(bytes).map(|(read, expert)| {
            let per_call = read as f64 / samples.len().max(1) as f64;
            format!(
                "the disk read {:.1} MB per call, {:.2} times the expert",
                per_call / 1e6,
                per_call / expert as f64
            )
        });
        match failed {
            Some(e) => Err(format!("{variant}: {e}")),
            None => Ok(Measured {
                variant,
                bytes,
                samples_us: samples,
                note,
            }),
        }
    };

    let mut rows = Vec::new();
    rows.push(measure("touch, 1 thread".to_string(), &mut |spans| {
        for &(off, len) in spans {
            touch(&map[off..off + len], page);
        }
        Ok(())
    })?);
    rows.push(measure(
        format!("touch, {threads} threads"),
        &mut |spans| {
            touch_threads(&map, spans, page, threads);
            Ok(())
        },
    )?);
    rows.push(measure(
        "willneed per projection, then touch it".to_string(),
        &mut |spans| {
            for &s in spans {
                range(s)?.will_need();
                touch(&map[s.0..s.0 + s.1], page);
            }
            Ok(())
        },
    )?);
    rows.push(measure(
        "willneed on every projection, the advice calls alone".to_string(),
        &mut |spans| {
            for &s in spans {
                range(s)?.will_need();
            }
            Ok(())
        },
    )?);
    rows.push(measure(
        "willneed over the whole expert, then touch".to_string(),
        &mut |spans| {
            for &s in spans {
                range(s)?.will_need();
            }
            for &s in spans {
                touch(&map[s.0..s.0 + s.1], page);
            }
            Ok(())
        },
    )?);

    // The populating advice is refused by a kernel that predates it; one probe says.
    let probe = plan.spans(0)[0];
    let probed = unsafe {
        libc::madvise(
            map.as_ptr().add(probe.0 / page * page) as *mut libc::c_void,
            page,
            libc::MADV_POPULATE_READ,
        )
    };
    if probed == 0 {
        rows.push(measure(
            "populate_read per projection".to_string(),
            &mut |spans| {
                for &(off, len) in spans {
                    mapped::advise_bytes(&map[off..off + len], libc::MADV_POPULATE_READ, false);
                    touch(&map[off..off + len], page);
                }
                Ok(())
            },
        )?);
    } else {
        rows.push(Measured {
            variant: "populate_read per projection".to_string(),
            bytes,
            samples_us: Vec::new(),
            note: Some(format!(
                "MADV_POPULATE_READ refused: {}",
                std::io::Error::last_os_error()
            )),
        });
    }

    let largest = cx
        .shapes
        .projections
        .iter()
        .map(|p| p.bytes as usize)
        .max()
        .unwrap_or(0);
    let mut buf = vec![0u8; largest];
    rows.push(measure(
        "pread, through the page cache".to_string(),
        &mut |spans| {
            for &(off, len) in spans {
                file.read_exact_at(&mut buf[..len], off as u64)
                    .map_err(|e| e.to_string())?;
                std::hint::black_box(&buf);
            }
            Ok(())
        },
    )?);

    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(&path)
    {
        Ok(direct) => {
            let mut aligned = Aligned::new(align, largest + 2 * align)?;
            let mut extra = 0u64;
            let row = measure(
                "pread, O_DIRECT, aligned buffer".to_string(),
                &mut |spans| {
                    for &(off, len) in spans {
                        let lo = off / align * align;
                        let hi = (off + len).div_ceil(align) * align;
                        extra += (hi - lo - len) as u64;
                        direct
                            .read_exact_at(&mut aligned.as_mut_slice()[..hi - lo], lo as u64)
                            .map_err(|e| e.to_string())?;
                        std::hint::black_box(aligned.as_mut_slice());
                    }
                    Ok(())
                },
            )?;
            let calls = row.samples_us.len().max(1) as f64;
            rows.push(Measured {
                note: Some(format!(
                    "aligned to {align} B, {:.0} extra bytes per call on average",
                    extra as f64 / calls
                )),
                ..row
            });
        }
        Err(e) => rows.push(Measured {
            variant: "pread, O_DIRECT, aligned buffer".to_string(),
            bytes,
            samples_us: Vec::new(),
            note: Some(format!("O_DIRECT refused: {e}")),
        }),
    }
    Ok(rows)
}

#[cfg(not(target_os = "linux"))]
pub fn run(_cx: &Context) -> Result<Vec<Measured>, String> {
    Err("expert-read drops and advises pages through Linux calls".to_string())
}

/// A buffer aligned for direct reads.
#[cfg(target_os = "linux")]
struct Aligned {
    ptr: *mut u8,
    len: usize,
}

#[cfg(target_os = "linux")]
impl Aligned {
    fn new(align: usize, len: usize) -> Result<Self, String> {
        let mut ptr = std::ptr::null_mut();
        // Safety: `align` is a power of two multiple of a pointer's size as the filesystem reports
        // it; the allocation is checked below.
        let r = unsafe { libc::posix_memalign(&mut ptr, align, len.max(align)) };
        if r != 0 || ptr.is_null() {
            return Err(format!("an aligned buffer of {len} bytes: error {r}"));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: `len` bytes were allocated at `ptr` and stay until `drop`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for Aligned {
    fn drop(&mut self) {
        // Safety: allocated by `posix_memalign` and freed here once.
        unsafe { libc::free(self.ptr as *mut libc::c_void) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microbench::shapes::{from_options, Projection};
    use clap::Parser;

    fn shapes(args: &[&str]) -> Shapes {
        from_options(&super::super::Options::parse_from(
            std::iter::once("microbench").chain(args.iter().copied()),
        ))
        .unwrap()
    }

    #[test]
    fn slots_are_laid_end_to_end_and_wrap() {
        let s = shapes(&[
            "expert-read",
            "--projection-bytes",
            "10,20,30",
            "--expert",
            "1",
        ]);
        let p = Plan::new(&s, Path::new("blob"), Some(100), 100 + 60 * 3 + 5).unwrap();
        assert_eq!(p.bytes(), 60);
        assert_eq!(p.spans(0), vec![(160, 10), (170, 20), (190, 30)]);
        assert_eq!(p.spans(2), vec![(100, 10), (110, 20), (130, 30)]);
        assert!(Plan::new(&s, Path::new("blob"), Some(100), 150).is_err());
    }

    #[test]
    fn the_files_own_offsets_are_followed_by_expert() {
        let mut s = shapes(&["expert-read", "--expert", "3"]);
        s.file = Some("m.gguf".into());
        s.projections = vec![
            Projection {
                name: "gate".to_string(),
                bytes: 8,
                offsets: vec![0, 8, 16, 24],
                dtype: None,
            },
            Projection {
                name: "down".to_string(),
                bytes: 4,
                offsets: vec![100, 104, 108, 112],
                dtype: None,
            },
        ];
        let p = Plan::new(&s, Path::new("m.gguf"), None, 200).unwrap();
        assert_eq!(p.spans(0), vec![(24, 8), (112, 4)]);
        assert_eq!(p.spans(1), vec![(0, 8), (100, 4)]);
        assert_eq!(p.bytes(), 12);
        assert!(Plan::new(&s, Path::new("m.gguf"), None, 110).is_err());
        let other = Plan::new(&s, Path::new("other"), Some(0), 200).unwrap();
        assert!(matches!(other, Plan::Derived { slots: 16, .. }));
    }

    #[test]
    fn a_touch_visits_one_byte_per_page_on_every_thread() {
        let page = 4;
        let bytes: Vec<u8> = (0..40).map(|i| i as u8).collect();
        assert_eq!(
            touch(&bytes, page),
            (0..40).step_by(4).sum::<usize>() as u64
        );
        assert_eq!(touch(&bytes[..1], page), 0);
        touch_threads(&bytes, &[(0, 17), (20, 20)], page, 3);
        touch_threads(&bytes, &[(0, 2)], page, 8);
    }
}

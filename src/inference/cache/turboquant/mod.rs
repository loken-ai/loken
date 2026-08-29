//! A 3-bit KV cache, **host reference only - there is no device path yet**.
//!
//! Nothing in this module runs on a card. There is no CUDA kernel, no `CudaSlice`, no
//! allocation, and no cache type that the engine can select: the loader cannot reach this
//! code and no request has ever gone through it. It is written down first, deliberately,
//! so that the thing that decides whether the kernel is worth writing exists before the
//! kernel does. Read [`measure`] before reading anything else - that is the part with an
//! answer in it.
//!
//! The scheme
//! ----------
//!
//! **A randomized Hadamard rotation** ([`hadamard`]) - a random sign flip followed by a
//! Walsh-Hadamard transform. Orthogonal, `n log n`, adds and subtracts only. It smears an
//! outlier across every coordinate, which is what lets a group of values share one scale
//! without the largest channel in the group setting that scale for everybody. A low-bit
//! scalar quantiser is not viable without it.
//!
//! **For K the rotation is free and exact.** Attention scores are `q . k`, and an
//! orthogonal `R` leaves a dot product alone, so `q . k == (Rq) . (Rk)`. K is rotated once
//! when it is written and Q once per step when it is read; there is no correction term to
//! carry and nothing to sketch. The rotation must be reproducible across those two calls,
//! which is why the sign flip is seeded from a fixed bit mixer written out in [`rng`]
//! rather than from a general-purpose generator whose stream is a property of a dependency
//! version.
//!
//! **A 3-bit Lloyd-Max codebook** ([`codebook`]). After the rotation a coordinate is close
//! to Gaussian, so the best eight levels are the ones that minimise mean squared error
//! against a normal. Those levels are solved here by iterating the boundary and centroid
//! conditions to a stated tolerance, not tabulated - eight constants lifted from a paper
//! are eight constants nobody in this repository can check, and the iteration is a
//! millisecond.
//!
//! **For V the rotation is not required.** V is averaged against softmax weights rather
//! than dotted with Q, so there is no orthogonality identity to exploit; outlier spreading
//! may still pay, which is a measurement rather than an argument, so [`measure`] reports V
//! both ways. V is quantised asymmetrically with a per-group zero point.
//!
//! **Storage** ([`pack`], [`quant`]). Three bits do not divide a byte, so eight values
//! pack into three bytes with nothing left over and no code straddling a group boundary.
//! A group carries an f16 scale, and for V an f16 zero point as well. That side
//! information is not free: at an eight-value group it costs 2 bits per value for K and 4
//! for V, which puts the scheme *above* the 4.5 bits per value of the Q4_0 cache already
//! in the tree. Every error figure this module reports is printed next to its bit rate for
//! that reason.
//!
//! What the measurement said
//! -------------------------
//!
//! Run against 28 layers of qwen3:0.6b, the scheme above did not survive contact with real
//! tensors intact. Several of the ideas it was built from are wrong as stated, and the
//! numbers behind each are in [`measure`]'s test output rather than only in this comment:
//!
//! * **The rotation works, on K.** Holding bits fixed it cut reconstruction error by more
//!   than half against the same scheme without it. That part is real.
//! * **The rotation does not pay on V.** A few per cent, which does not buy a transform on
//!   the read path.
//! * **The Gaussian-matched scale does not pay either.** A plain absmax beat it at every
//!   group size, so the codebook's levels are used and its calibration is not - see
//!   [`codebook`].
//! * **The rotation is the wrong lever.** What makes the incumbent Q4_0 cache hard to beat
//!   was never its bit width, it is its *grouping*: a block running along the token axis
//!   inside one channel never mixes a large channel with a small one, so the outliers are
//!   gone before quantisation starts and every level is spent on resolution. Grouping
//!   along `head_dim` and then rotating to make that grouping tolerable is an expensive
//!   route to a worse place.
//!
//! What phase 2 would have to add
//! ------------------------------
//!
//! A device implementation of exactly the layout [`pack`] documents, a rotation applied to
//! K on write and Q on read inside the attention kernels, and a cache type wired into the
//! loader alongside the existing quantised caches. None of that exists, and on the numbers
//! above none of it is justified. Until that changes, this module is a reference and a
//! measurement, and calling it a 3-bit KV cache would be calling a decision a feature.

pub mod codebook;
pub mod hadamard;
pub mod measure;
pub mod pack;
pub mod quant;
pub mod rng;

#[cfg(test)]
mod real_tensors;

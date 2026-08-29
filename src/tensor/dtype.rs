//! Element dtypes for the native tensor substrate.

use super::{Error, Result};

/// Declare the element dtypes, and derive from them everything that is asked of one.
///
/// A row is `Variant = "name", bytes, is_float`. The name is the one a checkpoint writes and
/// the one an error prints; the width is the element's, in bytes. Four lists over the same nine
/// variants used to say this - the enum, the width, the test for floatness and the name, each
/// able to forget a dtype the others knew.
macro_rules! dtypes {
    ($($(#[$note:meta])* $variant:ident = $name:literal, $bytes:expr, $float:literal;)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum DType {
            $($(#[$note])* $variant,)+
        }

        impl DType {
            /// What one element occupies.
            pub fn size_in_bytes(&self) -> usize {
                match self {
                    $(Self::$variant => $bytes,)+
                }
            }

            /// Whether arithmetic on it is floating point.
            pub fn is_float(&self) -> bool {
                match self {
                    $(Self::$variant => $float,)+
                }
            }

            /// The name a checkpoint writes.
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }
        }

        impl std::str::FromStr for DType {
            type Err = Error;
            /// The inverse of [`DType::as_str`].
            fn from_str(s: &str) -> Result<Self> {
                Ok(match s {
                    $($name => Self::$variant,)+
                    other => return Err(Error(format!("unknown dtype `{other}`"))),
                })
            }
        }
    };
}

dtypes! {
    //                    name   bytes  float
    U8   = "u8",   1, false;
    U32  = "u32",  4, false;
    /// 16-bit signed ints (audio PCM in the distributed protocol).
    I16  = "i16",  2, false;
    /// 32-bit signed ints (AWQ qweight/qzeros safetensors - fork extension).
    I32  = "i32",  4, false;
    I64  = "i64",  8, false;
    BF16 = "bf16", 2, true;
    F16  = "f16",  2, true;
    F32  = "f32",  4, true;
    F64  = "f64",  8, true;
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_canonical() {
        // GGML/safetensors-canonical element sizes.
        for (d, sz, fl) in [
            (DType::U8, 1, false),
            (DType::U32, 4, false),
            (DType::I16, 2, false),
            (DType::I32, 4, false),
            (DType::I64, 8, false),
            (DType::BF16, 2, true),
            (DType::F16, 2, true),
            (DType::F32, 4, true),
            (DType::F64, 8, true),
        ] {
            assert_eq!(d.size_in_bytes(), sz, "{d}");
            assert_eq!(d.is_float(), fl, "{d}");
        }
    }
}

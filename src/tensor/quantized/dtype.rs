//! The GGML block dtypes, and what each one costs.
//!
//! A dtype here is three numbers and a name: the id a GGUF file stores, how many values a
//! block holds, and how many bytes it occupies. Everything downstream - how a tensor is sized,
//! which kernel can serve it, how a row is padded - reads them from here.
//!
//! One row per format. The four things asked of a dtype used to be four `match` blocks over
//! the same sixteen variants, which is four places to forget a format in and no way to notice.

use super::*;

/// Declare the formats, and derive from them everything that is asked of a format.
///
/// A row is `Variant = id, values per block, bytes per block, the name GGUF spells it with`.
/// The unquantised carriers say one value per block, which is what makes a block count a length
/// in every unit. The spelling is a fifth thing asked of a format and it lives here for the
/// reason the other four do: a kernel named after a format, spelled out beside the kernel, is a
/// second table nobody updates when a format arrives.
macro_rules! ggml_dtypes {
    ($($(#[$note:meta])* $variant:ident = $id:expr, $block:expr, $bytes:expr, $spell:expr;)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum GgmlDType {
            $($(#[$note])* $variant,)+
        }

        impl GgmlDType {
            /// How GGUF spells this format, which is how every kernel named after it is spelled.
            pub const fn gguf_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $spell,)+
                }
            }

            /// The format a GGUF file's type id names.
            pub fn from_u32(v: u32) -> Result<Self> {
                Ok(match v {
                    $($id => Self::$variant,)+
                    other => return Err(Error(format!("unsupported ggml dtype id {other}"))),
                })
            }

            /// The id a GGUF file stores - the inverse of [`Self::from_u32`].
            pub fn to_u32(&self) -> u32 {
                match self {
                    $(Self::$variant => $id,)+
                }
            }

            /// Values per block.
            pub const fn block_size(&self) -> usize {
                match self {
                    $(Self::$variant => $block,)+
                }
            }

            /// Bytes per block.
            pub const fn type_size(&self) -> usize {
                match self {
                    $(Self::$variant => $bytes,)+
                }
            }
        }
    };
}

ggml_dtypes! {
    // In the file's own order, by the id it stores.
    //                    id  values  bytes
    F32                  = 0, 1, 4, "f32";
    F16                  = 1, 1, 2, "f16";
    Q4_0                 = 2, 32, 18, "q4_0";
    Q4_1                 = 3, 32, 20, "q4_1";
    Q5_0                 = 6, 32, 22, "q5_0";
    Q5_1                 = 7, 32, 24, "q5_1";
    Q8_0                 = 8, 32, 34, "q8_0";
    Q8_1                 = 9, 32, 36, "q8_1";
    Q2K                  = 10, 256, 84, "q2_K";
    Q3K                  = 11, 256, 110, "q3_K";
    Q4K                  = 12, 256, 144, "q4_K";
    Q5K                  = 13, 256, 176, "q5_K";
    Q6K                  = 14, 256, 210, "q6_K";
    Q8K                  = 15, 256, 292, "q8_K";
    BF16                 = 30, 1, 2, "bf16";
    /// OCP MXFP4: E2M1 codes against one E8M0 block scale - the format gpt-oss ships in.
    MxFp4                = 39, 32, 17, "mxfp4";
}

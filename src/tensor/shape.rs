//! Shapes, contiguous strides, and dimension indexing for the native tensor.

use super::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Explicit constructor from a dim slice (API-compat call sites).
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    /// Row-major (C-order) strides for a contiguous layout.
    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride = vec![1usize; self.0.len()];
        for i in (0..self.0.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * self.0[i + 1];
        }
        stride
    }

    pub fn dims1(&self) -> Result<usize> {
        match self.0[..] {
            [a] => Ok(a),
            _ => Err(Error(format!("expected rank-1 shape, got {:?}", self.0))),
        }
    }

    pub fn dims2(&self) -> Result<(usize, usize)> {
        match self.0[..] {
            [a, b] => Ok((a, b)),
            _ => Err(Error(format!("expected rank-2 shape, got {:?}", self.0))),
        }
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        match self.0[..] {
            [a, b, c] => Ok((a, b, c)),
            _ => Err(Error(format!("expected rank-3 shape, got {:?}", self.0))),
        }
    }

    pub fn dims5(&self) -> Result<(usize, usize, usize, usize, usize)> {
        match *self.dims() {
            [a, b, c, d, e] => Ok((a, b, c, d, e)),
            _ => Err(Error(format!("expected 5 dims, got {:?}", self.dims()))),
        }
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        match self.0[..] {
            [a, b, c, d] => Ok((a, b, c, d)),
            _ => Err(Error(format!("expected rank-4 shape, got {:?}", self.0))),
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Self(v.to_vec())
    }
}

impl From<usize> for Shape {
    fn from(a: usize) -> Self {
        Self(vec![a])
    }
}

impl From<(usize, usize)> for Shape {
    fn from((a, b): (usize, usize)) -> Self {
        Self(vec![a, b])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        Self(vec![a, b, c])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((a, b, c, d): (usize, usize, usize, usize)) -> Self {
        Self(vec![a, b, c, d])
    }
}

impl From<(usize,)> for Shape {
    fn from((a,): (usize,)) -> Self {
        Self(vec![a])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from((a, b, c, d, e): (usize, usize, usize, usize, usize)) -> Self {
        Self(vec![a, b, c, d, e])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from((a, b, c, d, e, f): (usize, usize, usize, usize, usize, usize)) -> Self {
        Self(vec![a, b, c, d, e, f])
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(v: &[usize; N]) -> Self {
        Self(v.to_vec())
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(v: &Vec<usize>) -> Self {
        Self(v.clone())
    }
}

impl From<&Shape> for Shape {
    fn from(s: &Shape) -> Self {
        s.clone()
    }
}

/// Relative dimension index (counted from the last axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D {
    Minus1,
    Minus2,
}

/// Anything that can resolve to an absolute axis of a shape: a `usize`
/// (absolute, bounds-checked) or `D::Minus*` (relative to the last axis).
pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        if *self >= shape.rank() {
            return Err(Error(format!(
                "{op}: dim {self} out of range for shape {:?}",
                shape.dims()
            )));
        }
        Ok(*self)
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        let back = match self {
            D::Minus1 => 1,
            D::Minus2 => 2,
        };
        if rank < back {
            return Err(Error(format!(
                "{op}: dim {self:?} out of range for shape {:?}",
                shape.dims()
            )));
        }
        Ok(rank - back)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_match_oracle() {
        for dims in [vec![7usize], vec![3, 5], vec![2, 3, 4], vec![2, 3, 4, 5]] {
            let native = Shape::from(dims.clone()).stride_contiguous();
            // Oracle: canonical row-major strides (suffix products).
            let mut oracle = vec![1usize; dims.len()];
            for i in (0..dims.len().saturating_sub(1)).rev() {
                oracle[i] = oracle[i + 1] * dims[i + 1];
            }
            assert_eq!(native, oracle, "{dims:?}");
        }
    }

    #[test]
    fn dim_resolution() {
        let s = Shape::from((2usize, 3usize, 4usize));
        assert_eq!(D::Minus1.to_index(&s, "t").unwrap(), 2);
        assert_eq!(D::Minus2.to_index(&s, "t").unwrap(), 1);
        assert_eq!(1usize.to_index(&s, "t").unwrap(), 1);
        assert!(5usize.to_index(&s, "t").is_err());
    }
}

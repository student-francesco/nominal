use std::cell::RefCell;
use std::fmt::{self, Display};
use std::iter::{FusedIterator, Iterator};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use ndarray::ArrayD;
use rand::RngExt;
use rand::seq::index;

const MAX_DIMS: usize = 5;

#[derive(Clone)]
pub struct Tensor {
    dims: [usize; MAX_DIMS],
    strides: [usize; MAX_DIMS],
    ndim: usize,
    storage: Rc<RefCell<TensorStorage>>,
}

pub struct TensorStorage {
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Self {
        let dims = padded_dims(shape);
        if dims.iter().product::<usize>() != data.len() {
            panic!("Dims do not match data length");
        }
        Self {
            dims,
            strides: strides_from_contiguous(dims),
            ndim: shape.len(),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    /// Create a tensor from an ndarray.
    pub fn from(array: ArrayD<f32>) -> Self {
        let shape = array.shape();
        if shape.len() > MAX_DIMS {
            panic!("Tensors with {} dimensions are not supported", shape.len());
        }
        let dims = padded_dims(shape);
        let strides = strides_from_contiguous(dims);

        let data = match array.as_slice() {
            Some(s) => s.to_vec(), // contiguous layout, use memcpy
            None => array.iter().copied().collect(),
        };

        Self {
            dims,
            strides,
            ndim: shape.len(),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    /// Create a zero-dimensional tensor with the given scalar value.
    pub fn scalar(value: f32) -> Self {
        Self::new(&[], vec![value])
    }

    /// Create a tensor of zeros with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let dims = padded_dims(shape);
        Self::new(shape, vec![0.0; dims.iter().product()])
    }

    /// Create a tensor of zeros with the same shape as this tensor.
    #[inline]
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.shape())
    }

    /// Create a tensor of ones with the given shape.
    #[inline]
    pub fn ones(shape: &[usize]) -> Self {
        Self::zeros(shape).add_f32(1.0)
    }

    /// Create a tensor of ones with the same shape as this tensor.
    #[inline]
    pub fn ones_like(&self) -> Self {
        Self::ones(self.shape())
    }

    /// Create a tensor of random values between -1 and 1 with the given shape.
    pub fn random(shape: &[usize]) -> Self {
        let mut rng = rand::rng();
        let data = (0..padded_dims(shape).iter().product())
            .map(|_| rng.random_range(-1.0..1.0))
            .collect::<Vec<_>>();
        Self::new(shape, data)
    }

    /// The logical shape of the tensor, without padding dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.dims[MAX_DIMS - self.ndim..]
    }

    /// The number of logical dimensions of the tensor.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Pretty-print the shape of the tensor.
    pub fn shape_pretty(&self) -> String {
        format!(
            "({})",
            self.shape()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Swap two logical dimensions. Negative dimensions count from the end.
    /// Tensors with fewer than two dims are treated as matrices,
    /// so a scalar becomes `(1, 1)` and a `(n,)` vector becomes `(n, 1)`.
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Self {
        let ndim = self.ndim.max(2);
        let start = MAX_DIMS - ndim;
        let d0 = start + normalize_dim(dim0, ndim);
        let d1 = start + normalize_dim(dim1, ndim);

        let mut dims = self.dims;
        let mut strides = self.strides;
        dims.swap(d0, d1);
        strides.swap(d0, d1);

        Self {
            dims,
            strides,
            ndim,
            storage: self.storage.clone(),
        }
    }

    /// Insert a dimension of size 1 at the given position
    pub fn unsqueeze(&self, dim: isize) -> Self {
        assert!(
            self.ndim < MAX_DIMS,
            "cannot unsqueeze a tensor that already has {MAX_DIMS} dims"
        );
        let start = MAX_DIMS - self.ndim;
        let slot = start + normalize_dim(dim, self.ndim + 1);

        // shift the dims before the insertion point one slot to the left
        let mut dims = self.dims;
        let mut strides = self.strides;
        dims.copy_within(start..slot, start - 1);
        strides.copy_within(start..slot, start - 1);
        dims[slot - 1] = 1;
        strides[slot - 1] = if slot < MAX_DIMS {
            strides[slot] * dims[slot]
        } else {
            1
        };

        Self {
            dims,
            strides,
            ndim: self.ndim + 1,
            storage: self.storage.clone(),
        }
    }

    /// Remove a dimension of size 1 at the given position.
    pub fn squeeze(&self, dim: isize) -> Self {
        let start = MAX_DIMS - self.ndim;
        let slot = start + normalize_dim(dim, self.ndim);
        assert_eq!(
            self.dims[slot],
            1,
            "cannot squeeze dimension {dim} of size {} in tensor of shape {}",
            self.dims[slot],
            self.shape_pretty()
        );

        // shift the dims before the removed one a slot to the right
        let mut dims = self.dims;
        let mut strides = self.strides;
        dims.copy_within(start..slot, start + 1);
        strides.copy_within(start..slot, start + 1);
        dims[start] = 1;
        strides[start] = if start + 1 < MAX_DIMS {
            strides[start + 1] * dims[start + 1]
        } else {
            1
        };

        Self {
            dims,
            strides,
            ndim: self.ndim - 1,
            storage: self.storage.clone(),
        }
    }

    /// Return whether the tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        let expected = strides_from_contiguous(self.dims);
        // compare strides and ignore size-1 dims
        (0..MAX_DIMS).all(|i| self.dims[i] == 1 || self.strides[i] == expected[i])
    }

    /// Make a contiguous copy of the tensor.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }

        let storage = self.storage.borrow();
        let data = self
            .offsets()
            .map(|off| storage.data[off])
            .collect::<Vec<_>>();
        Tensor::new(&self.shape(), data)
    }

    /// Calculate the offset at which to access data for a given index.
    #[inline]
    fn offset<const N: usize>(&self, idx: [usize; N]) -> usize {
        assert_eq!(
            N,
            self.ndim,
            "expected {} indices for tensor of shape {}, got {}",
            self.ndim,
            self.shape_pretty(),
            N
        );
        let start = MAX_DIMS - N;
        let mut off = 0;
        for i in 0..N {
            let axis = start + i;
            debug_assert!(
                idx[i] < self.dims[axis],
                "index {} out of bounds on axis {}",
                idx[i],
                i
            );
            off += idx[i] * self.strides[axis];
        }
        off
    }

    /// Iterate offsets of every element, in row-major order.
    pub fn offsets(&self) -> Offsets {
        Offsets {
            dims: self.dims,
            strides: self.strides,
            idx: [0; MAX_DIMS],
            off: 0,
            remaining: self.numel(),
        }
    }

    /// Unravel an offset into its given multi-dimensional index.
    #[inline]
    pub fn unravel(&self, mut flat: usize) -> [usize; MAX_DIMS] {
        let mut idx = [0usize; MAX_DIMS];
        for i in (0..MAX_DIMS).rev() {
            idx[i] = flat % self.dims[i];
            flat /= self.dims[i];
        }
        idx
    }

    /// Return the total number of elements in the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Extract the value of a single-element tensor.
    pub fn item(&self) -> f32 {
        debug_assert_eq!(
            self.storage.borrow().data.len(),
            1,
            "item() called on a non-scalar tensor"
        );
        self.storage.borrow().data[0]
    }

    /// Get a single point at any given index.
    #[inline]
    pub fn get<const N: usize>(&self, idx: [usize; N]) -> f32 {
        self.storage.borrow().data[self.offset(idx)]
    }

    /// Set the value of a single point at any given index.
    #[inline]
    pub fn set(&mut self, idx: [usize; 5], value: f32) {
        self.storage.borrow_mut().data[self.offset(idx)] = value;
    }

    // Operations

    pub fn pow(&self, exponent: f32) -> Self {
        let mut data = vec![0f32; self.storage.borrow().data.len()];
        for (a, out) in self.storage.borrow().data.iter().zip(data.iter_mut()) {
            *out = a.powf(exponent);
        }
        Self {
            dims: self.dims,
            strides: self.strides,
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    pub fn exp(&self) -> Self {
        let mut data = vec![0f32; self.storage.borrow().data.len()];
        for (a, out) in self.storage.borrow().data.iter().zip(data.iter_mut()) {
            *out = a.exp();
        }
        Self {
            dims: self.dims,
            strides: self.strides,
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    pub fn powf(&self, exponent: f32) -> Self {
        let mut data = vec![0f32; self.storage.borrow().data.len()];
        for (a, out) in self.storage.borrow().data.iter().zip(data.iter_mut()) {
            *out = a.powf(exponent);
        }
        Self {
            dims: self.dims,
            strides: self.strides,
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn add(&self, rhs: &Tensor) -> Self {
        assert_eq!(
            self.dims, rhs.dims,
            "dimensions must match in tensor addition: {:?} vs {:?}",
            self.dims, rhs.dims
        );
        let storage = self.storage.borrow();
        let other = rhs.storage.borrow();
        let data = self
            .offsets()
            .zip(rhs.offsets())
            .map(|(a, b)| storage.data[a] + other.data[b])
            .collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim.max(rhs.ndim),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn add_f32(&self, rhs: f32) -> Self {
        let storage = self.storage.borrow();
        let data = self.offsets().map(|off| storage.data[off] + rhs).collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn neg(&self) -> Self {
        let storage = self.storage.borrow();
        let data = self.offsets().map(|off| -storage.data[off]).collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn sub(&self, rhs: &Tensor) -> Self {
        assert_eq!(
            self.dims, rhs.dims,
            "dimensions must match in tensor subtraction: {:?} vs {:?}",
            self.dims, rhs.dims
        );
        let storage = self.storage.borrow();
        let other = rhs.storage.borrow();
        let data = self
            .offsets()
            .zip(rhs.offsets())
            .map(|(a, b)| storage.data[a] - other.data[b])
            .collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim.max(rhs.ndim),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn sub_f32(&self, rhs: f32) -> Self {
        let storage = self.storage.borrow();
        let data = self.offsets().map(|off| storage.data[off] - rhs).collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    /// Temporary implementation of matrix multiplication.
    /// Efficiency is not yet our first concern
    fn mul(&self, rhs: &Tensor) -> Self {
        assert!(
            self.dims[..MAX_DIMS - 2].iter().all(|&d| d == 1),
            "matmul only supports 2D tensors, got shape {}",
            self.shape_pretty()
        );
        assert!(
            rhs.dims[..MAX_DIMS - 2].iter().all(|&d| d == 1),
            "matmul only supports 2D tensors, got shape {}",
            rhs.shape_pretty()
        );

        let (m, k) = (self.dims[MAX_DIMS - 2], self.dims[MAX_DIMS - 1]);
        let (k2, n) = (rhs.dims[MAX_DIMS - 2], rhs.dims[MAX_DIMS - 1]);
        assert_eq!(
            k, k2,
            "dimensions must match in matrix multiplication: {}x{} * {}x{}",
            m, k, k2, n
        );
        let (sa_i, sa_p) = (self.strides[MAX_DIMS - 2], self.strides[MAX_DIMS - 1]);
        let (sb_p, sb_j) = (rhs.strides[MAX_DIMS - 2], rhs.strides[MAX_DIMS - 1]);
        // a[i][p] = a_data[i * sa_i + p * sa_p], b[p][j] = b_data[p * sb_p + j * sb_j]

        let storage = self.storage.borrow();
        let other = rhs.storage.borrow();

        let mut data = vec![0f32; m * n];
        for i in 0..m {
            let c_row = &mut data[i * n..i * n + n];
            for p in 0..k {
                let a_ip = storage.data[i * sa_i + p * sa_p];
                let b_start = p * sb_p;
                if sb_j == 1 {
                    // rows of b are contiguous, so use a slice
                    let b_row = &other.data[b_start..b_start + n];
                    for (c, b) in c_row.iter_mut().zip(b_row) {
                        *c += a_ip * b;
                    }
                } else {
                    for j in 0..n {
                        c_row[j] += a_ip * other.data[b_start + j * sb_j];
                    }
                }
            }
        }

        let mut dims = [1usize; MAX_DIMS];
        dims[MAX_DIMS - 2] = m;
        dims[MAX_DIMS - 1] = n;
        let strides = strides_from_contiguous(dims);

        Tensor {
            dims,
            strides,
            ndim: self.ndim.max(rhs.ndim),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn mul_f32(&self, rhs: f32) -> Self {
        let storage = self.storage.borrow();
        let data = self.offsets().map(|off| storage.data[off] * rhs).collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    pub fn mul_elementwise(&self, rhs: &Tensor) -> Self {
        assert_eq!(
            self.dims, rhs.dims,
            "dimensions must match in Hadamard product: {:?} vs {:?}",
            self.dims, rhs.dims
        );

        let storage = self.storage.borrow();
        let data = self
            .offsets()
            .zip(rhs.offsets())
            .map(|(a, b)| storage.data[a] * rhs.storage.borrow().data[b])
            .collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim.max(rhs.ndim),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn div(&self, rhs: &Tensor) -> Self {
        assert_eq!(
            self.dims, rhs.dims,
            "dimensions must match in tensor division: {:?} vs {:?}",
            self.dims, rhs.dims
        );
        let storage = self.storage.borrow();
        let data = self
            .offsets()
            .zip(rhs.offsets())
            .map(|(a, b)| storage.data[a] / rhs.storage.borrow().data[b])
            .collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim.max(rhs.ndim),
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }

    fn div_f32(&self, rhs: f32) -> Self {
        let storage = self.storage.borrow();
        let data = self.offsets().map(|off| storage.data[off] / rhs).collect();
        Self {
            dims: self.dims,
            strides: strides_from_contiguous(self.dims),
            ndim: self.ndim,
            storage: Rc::new(RefCell::new(TensorStorage { data })),
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape{:?}", self.shape_pretty())
    }
}

macro_rules! forward_binop {
    ($trait:ident, $method:ident, $method_f32:ident) => {
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$method(&self, &rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$method(&self, rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$method(self, &rhs)
            }
        }

        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$method(self, rhs)
            }
        }

        impl $trait<f32> for Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: f32) -> Tensor {
                Tensor::$method_f32(&self, rhs)
            }
        }

        impl $trait<f32> for &Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: f32) -> Tensor {
                Tensor::$method_f32(&self, rhs)
            }
        }
    };
}

impl Neg for Tensor {
    type Output = Tensor;
    #[inline]
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}

/// Storage offsets of a tensor's elements, row-major
pub struct Offsets {
    dims: [usize; MAX_DIMS],
    strides: [usize; MAX_DIMS],
    idx: [usize; MAX_DIMS],
    off: usize,
    remaining: usize,
}
impl Iterator for Offsets {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let current = self.off;

        for axis in (0..MAX_DIMS).rev() {
            self.idx[axis] += 1;
            self.off += self.strides[axis];
            if self.idx[axis] < self.dims[axis] {
                break;
            }
            // Reset to 0 for next axis
            self.idx[axis] = 0;
            self.off -= self.strides[axis] * self.dims[axis];
        }

        Some(current)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Offsets {}
impl FusedIterator for Offsets {}

forward_binop!(Add, add, add_f32);
forward_binop!(Sub, sub, sub_f32);
forward_binop!(Mul, mul, mul_f32);
forward_binop!(Div, div, div_f32);

// Helper functions

#[inline]
fn normalize_dim(dim: isize, ndim: usize) -> usize {
    let n = ndim as isize;
    let d = if dim < 0 { dim + n } else { dim };
    assert!(
        (0..n).contains(&d),
        "dimension {dim} out of range for tensor with {ndim} dims"
    );
    d as usize
}

/// Right-align a logical shape into the padded dims array.
#[inline]
fn padded_dims(shape: &[usize]) -> [usize; MAX_DIMS] {
    assert!(
        shape.len() <= MAX_DIMS,
        "Tensors with {} dimensions are not supported",
        shape.len()
    );
    let mut dims = [1usize; MAX_DIMS];
    dims[MAX_DIMS - shape.len()..].copy_from_slice(shape);
    dims
}

/// Calculate the strides from a tensor with a contiguous layout.
#[inline]
fn strides_from_contiguous(dims: [usize; MAX_DIMS]) -> [usize; MAX_DIMS] {
    let mut strides = [1usize; MAX_DIMS];
    for i in (0..MAX_DIMS - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

// Implementation tests

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;

    #[test]
    fn test_contiguous_strides() {
        let strides = strides_from_contiguous([2, 3, 4, 1, 1]);
        assert_eq!(strides, [12, 4, 1, 1, 1]);
    }

    #[test]
    fn test_access_at_index_returns_correct_data() {
        let matrix: Array2<f32> = array![
            [0., 1.2, 4.1, 5.1],
            [1.1, 2.1, 9.1, 2.8],
            [4.3, 8.5, 2.2, -1.2]
        ];
        let tensor = Tensor::from(matrix.into_dyn());
        assert_eq!(tensor.shape(), [3, 4]);
        assert_eq!(tensor.get([0, 0]), 0.);
        assert_eq!(tensor.get([1, 2]), 9.1);
        assert_eq!(tensor.get([1, 1]), 2.1);
        assert_eq!(tensor.get([2, 3]), -1.2);
    }

    #[test]
    fn test_matmul() {
        let a_arr: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.]];
        let b_arr: Array2<f32> = array![[7., 8.], [9., 10.], [11., 12.]];
        let a = Tensor::from(a_arr.into_dyn());
        let b = Tensor::from(b_arr.into_dyn());

        let result = a * b;

        assert_eq!(result.shape(), [2, 2]);
        assert_eq!(result.get([0, 0]), 58.);
        assert_eq!(result.get([0, 1]), 64.);
        assert_eq!(result.get([1, 0]), 139.);
        assert_eq!(result.get([1, 1]), 154.);
    }

    #[test]
    fn test_scalar_has_no_dims() {
        let s = Tensor::scalar(3.0);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.shape(), [] as [usize; 0]);
        assert_eq!(s.shape_pretty(), "()");
    }

    #[test]
    fn test_transpose_swaps_matrix_dims() {
        let t = Tensor::zeros(&[8, 1]).transpose(-2, -1);
        assert_eq!(t.shape(), [1, 8]);

        let t = Tensor::zeros(&[2, 3, 4]).transpose(-2, -1);
        assert_eq!(t.shape(), [2, 4, 3]);

        let t = Tensor::zeros(&[8]).transpose(-2, -1);
        assert_eq!(t.shape(), [8, 1]);

        let t = Tensor::scalar(1.0).transpose(-2, -1);
        assert_eq!(t.shape(), [1, 1]);
    }

    #[test]
    fn test_unsqueeze() {
        let v = Tensor::from(array![1., 2., 3.].into_dyn());
        assert_eq!(v.shape(), [3]);

        let col = v.unsqueeze(-1);
        assert_eq!(col.shape(), [3, 1]);
        assert_eq!(col.get([2, 0]), 3.);

        let row = v.unsqueeze(0);
        assert_eq!(row.shape(), [1, 3]);
        assert_eq!(row.get([0, 2]), 3.);

        let t = Tensor::zeros(&[2, 3]).unsqueeze(1);
        assert_eq!(t.shape(), [2, 1, 3]);
    }

    #[test]
    fn test_squeeze() {
        let t = Tensor::zeros(&[2, 1, 3]).squeeze(1);
        assert_eq!(t.shape(), [2, 3]);

        let t = Tensor::zeros(&[3, 1]).squeeze(-1);
        assert_eq!(t.shape(), [3]);

        let t = Tensor::zeros(&[1]).squeeze(0);
        assert_eq!(t.ndim(), 0);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip_keeps_data() {
        let m = Tensor::from(array![[1., 2.], [3., 4.]].into_dyn());
        let t = m.unsqueeze(1).squeeze(1);
        assert_eq!(t.shape(), [2, 2]);
        assert_eq!(t.get([1, 0]), 3.);
    }

    #[test]
    #[should_panic(expected = "cannot squeeze")]
    fn test_squeeze_non_unit_dim_panics() {
        Tensor::zeros(&[2, 3]).squeeze(0);
    }

    #[test]
    fn test_matmul_matrix_by_column_vector() {
        let w = Tensor::from(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let x = Tensor::from(array![1., 0., -1.].into_dyn()).unsqueeze(-1);
        let y = w * x;
        assert_eq!(y.shape(), [2, 1]);
        assert_eq!(y.get([0, 0]), -2.);
        assert_eq!(y.get([1, 0]), -2.);
    }

    /// Read every element of a 2D tensor in row-major order through `get`.
    fn to_vec2(t: &Tensor) -> Vec<f32> {
        let [rows, cols] = t.shape() else {
            panic!("expected a 2D tensor, got shape {}", t.shape_pretty());
        };
        (0..*rows)
            .flat_map(|i| (0..*cols).map(move |j| t.get([i, j])))
            .collect()
    }

    #[test]
    fn test_matmul_transposed_lhs() {
        // aᵀ = [[1, 4], [2, 5], [3, 6]]
        let a = Tensor::from(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let b = Tensor::from(array![[1., 0.], [0., 1.]].into_dyn());

        let result = a.transpose(0, 1) * b;

        assert_eq!(result.shape(), [3, 2]);
        assert_eq!(to_vec2(&result), [1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_matmul_transposed_rhs() {
        // bᵀ = [[7, 9, 11], [8, 10, 12]]
        let a = Tensor::from(array![[1., 2.], [3., 4.]].into_dyn());
        let b = Tensor::from(array![[7., 8.], [9., 10.], [11., 12.]].into_dyn());

        let result = a * b.transpose(0, 1);

        assert_eq!(result.shape(), [2, 3]);
        assert_eq!(to_vec2(&result), [23., 29., 35., 53., 67., 81.]);
    }

    #[test]
    fn test_matmul_both_transposed() {
        // (bᵀ aᵀ) = (a b)ᵀ
        let a = Tensor::from(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let b = Tensor::from(array![[7., 8.], [9., 10.], [11., 12.]].into_dyn());

        let result = b.transpose(0, 1) * a.transpose(0, 1);

        assert_eq!(result.shape(), [2, 2]);
        assert_eq!(to_vec2(&result), [58., 139., 64., 154.]);
    }

    #[test]
    fn test_matmul_views_match_contiguous_copies() {
        let a = Tensor::random(&[4, 3]).transpose(0, 1);
        let b = Tensor::random(&[5, 4]).transpose(0, 1);
        assert!(!a.is_contiguous() && !b.is_contiguous());

        let from_views = &a * &b;
        let from_copies = a.contiguous() * b.contiguous();

        assert_eq!(from_views.shape(), [3, 5]);
        assert_eq!(to_vec2(&from_views), to_vec2(&from_copies));
    }

    #[test]
    fn test_matmul_output_is_contiguous() {
        let a = Tensor::random(&[3, 2]).transpose(0, 1);
        let b = Tensor::random(&[4, 3]).transpose(0, 1);
        assert!((a * b).is_contiguous());
    }

    #[test]
    fn test_matmul_with_scalar() {
        // Value::neg multiplies by a 0-D scalar, which matmul treats as 1x1
        let col = Tensor::from(array![1., -2., 3.].into_dyn()).unsqueeze(-1);

        let result = col * Tensor::scalar(-1.0);

        assert_eq!(result.shape(), [3, 1]);
        assert_eq!(to_vec2(&result), [-1., 2., -3.]);
    }

    #[test]
    fn test_matmul_backward_shapes() {
        // out = a @ b; da = grad @ bᵀ, db = aᵀ @ grad
        let a = Tensor::random(&[1, 8]);
        let b = Tensor::random(&[8, 1]);
        let grad = Tensor::ones(&[1, 1]);

        let grad_a = &grad * b.transpose(-2, -1);
        let grad_b = a.transpose(-2, -1) * &grad;

        assert_eq!(grad_a.shape(), a.shape());
        assert_eq!(grad_b.shape(), b.shape());
    }

    #[test]
    #[should_panic(expected = "dimensions must match in matrix multiplication")]
    fn test_matmul_inner_dim_mismatch_panics() {
        let _ = Tensor::zeros(&[2, 3]) * Tensor::zeros(&[2, 3]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from(array![[1., 2.], [3., 4.]].into_dyn());
        let b = Tensor::from(array![[10., 20.], [30., 40.]].into_dyn());

        let result = a + b;

        assert_eq!(result.shape(), [2, 2]);
        assert_eq!(to_vec2(&result), [11., 22., 33., 44.]);
    }

    #[test]
    fn test_add_transposed_lhs() {
        let a = Tensor::from(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let z = Tensor::zeros(&[3, 2]);

        let result = a.transpose(0, 1) + z;

        assert_eq!(result.shape(), [3, 2]);
        assert_eq!(to_vec2(&result), [1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_add_transposed_rhs() {
        let a = Tensor::from(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let z = Tensor::zeros(&[3, 2]);

        let result = z + a.transpose(0, 1);

        assert_eq!(to_vec2(&result), [1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn test_add_both_transposed() {
        let a = Tensor::from(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let b = Tensor::from(array![[10., 20., 30.], [40., 50., 60.]].into_dyn());

        let result = a.transpose(0, 1) + b.transpose(0, 1);

        assert_eq!(to_vec2(&result), [11., 44., 22., 55., 33., 66.]);
    }

    #[test]
    fn test_add_output_is_contiguous() {
        let a = Tensor::random(&[2, 3]).transpose(0, 1);
        let b = Tensor::random(&[3, 2]);
        assert!((&a + &b).is_contiguous());
        assert!((&b + &a).is_contiguous());
    }

    #[test]
    fn test_add_scalar_to_1x1_keeps_larger_ndim() {
        let result = Tensor::ones(&[1, 1]) + Tensor::scalar(2.0);
        assert_eq!(result.shape(), [1, 1]);
        assert_eq!(result.get([0, 0]), 3.);
    }

    #[test]
    #[should_panic(expected = "dimensions must match in tensor addition")]
    fn test_add_shape_mismatch_panics() {
        let _ = Tensor::zeros(&[2, 3]) + Tensor::zeros(&[3, 2]);
    }
}

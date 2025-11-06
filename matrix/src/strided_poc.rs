//! Proof of Concept: Stride-Based Matrix with O(1) Transpose
//!
//! This module demonstrates a world-class stride-based matrix design inspired by:
//! - ndarray's stride system
//! - nalgebra's storage traits
//! - NumPy's view semantics
//! - hypercube-verifier's tensor dimensions

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

use crate::dense::DenseStorage;
use crate::Matrix;

/// A dense matrix with stride support for efficient views and transposes.
///
/// # Memory Layout
///
/// The matrix stores data in a flat buffer with configurable strides:
/// - `row_stride`: number of elements between consecutive row starts
/// - `col_stride`: number of elements between consecutive column starts
///
/// ## Row-major layout (default):
/// ```text
/// [a00 a01 a02 | a10 a11 a12 | a20 a21 a22]
///  ←─ width ─→
/// row_stride = width, col_stride = 1
/// ```
///
/// ## Column-major layout:
/// ```text
/// [a00 a10 a20 | a01 a11 a21 | a02 a12 a22]
///  ←─ height ─→
/// row_stride = 1, col_stride = height
/// ```
///
/// ## Strided layout (e.g., every other column):
/// ```text
/// [a00 _ a01 _ a02 _ | a10 _ a11 _ a12 _]
/// row_stride = 2*width, col_stride = 2
/// ```
#[derive(Clone, Debug)]
pub struct StridedMatrix<T, V = Vec<T>> {
    /// Flat buffer of matrix values
    pub values: V,
    /// Logical width (number of columns visible to user)
    pub width: usize,
    /// Logical height (number of rows visible to user)
    pub height: usize,
    /// Row stride: elements between starts of consecutive rows
    pub row_stride: usize,
    /// Column stride: elements between starts of consecutive columns
    pub col_stride: usize,
    /// Phantom data for T
    _phantom: PhantomData<T>,
}

impl<T, V: DenseStorage<T>> StridedMatrix<T, V> {
    /// Create a new row-major strided matrix.
    ///
    /// This is the standard layout where rows are contiguous in memory.
    ///
    /// # Example
    /// ```ignore
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let matrix = StridedMatrix::new(data, 3);  // 2×3 matrix
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    /// ```
    #[must_use]
    pub fn new(values: V, width: usize) -> Self {
        let len = values.borrow().len();
        debug_assert!(len % width == 0, "values.len() must be divisible by width");
        let height = len / width;
        Self {
            values,
            width,
            height,
            row_stride: width,
            col_stride: 1,
            _phantom: PhantomData,
        }
    }

    /// Create a new column-major strided matrix.
    ///
    /// This layout stores columns contiguously, useful for column-heavy operations.
    ///
    /// # Example
    /// ```ignore
    /// let data = vec![1, 4, 2, 5, 3, 6];  // Columns laid out sequentially
    /// let matrix = StridedMatrix::new_col_major(data, 3);  // 2×3 matrix
    /// // [[1, 2, 3],
    /// //  [4, 5, 6]]
    /// ```
    #[must_use]
    pub fn new_col_major(values: V, width: usize) -> Self {
        let len = values.borrow().len();
        debug_assert!(len % width == 0, "values.len() must be divisible by width");
        let height = len / width;
        Self {
            values,
            width,
            height,
            row_stride: 1,
            col_stride: height,
            _phantom: PhantomData,
        }
    }

    /// Create a strided matrix with custom strides.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - All indices computed as `r * row_stride + c * col_stride` are within bounds
    /// - For all valid (r, c): `r * row_stride + c * col_stride < values.len()`
    ///
    /// # Example
    /// ```ignore
    /// // Create a matrix view of every other element
    /// let data = vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0];
    /// let matrix = StridedMatrix::with_strides(data, 3, 2, 6, 2);
    /// // Logical view: [[1, 2, 3],
    /// //                [4, 5, 6]]
    /// ```
    #[must_use]
    pub fn with_strides(values: V, width: usize, height: usize, row_stride: usize, col_stride: usize) -> Self {
        Self {
            values,
            width,
            height,
            row_stride,
            col_stride,
            _phantom: PhantomData,
        }
    }

    /// Check if this matrix uses standard row-major contiguous layout.
    ///
    /// This enables fast-path optimizations for operations like row slicing.
    #[inline]
    #[must_use]
    pub fn is_row_major_contiguous(&self) -> bool {
        self.row_stride == self.width && self.col_stride == 1
    }

    /// Check if this matrix uses column-major contiguous layout.
    #[inline]
    #[must_use]
    pub fn is_col_major_contiguous(&self) -> bool {
        self.row_stride == 1 && self.col_stride == self.height
    }

    /// Compute the flat buffer index for element at (row, col).
    ///
    /// # Safety
    /// No bounds checking is performed.
    #[inline]
    unsafe fn compute_index(&self, r: usize, c: usize) -> usize {
        r * self.row_stride + c * self.col_stride
    }

    /// Get a reference to an element using stride-based indexing.
    ///
    /// # Safety
    /// The caller must ensure `r < self.height()` and `c < self.width()`.
    #[inline]
    pub unsafe fn get_unchecked_ref(&self, r: usize, c: usize) -> &T {
        // SAFETY: Caller guarantees bounds are valid
        let idx = unsafe { self.compute_index(r, c) };
        &self.values.borrow()[idx]
    }

    /// Create an O(1) transposed view of this matrix.
    ///
    /// This simply swaps dimensions and strides without copying data.
    ///
    /// # Example
    /// ```ignore
    /// let matrix = StridedMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);
    /// let transposed = matrix.transpose_view();  // O(1) operation
    ///
    /// assert_eq!(transposed.width(), 2);
    /// assert_eq!(transposed.height(), 3);
    /// ```
    #[must_use]
    pub fn transpose_view(&self) -> TransposedMatrixView<&Self> {
        TransposedMatrixView::new(self)
    }

    /// Consume this matrix and return an owned transposed view.
    #[must_use]
    pub fn into_transpose_view(self) -> TransposedMatrixView<Self> {
        TransposedMatrixView::new(self)
    }

    /// Get the stride information as (row_stride, col_stride).
    #[inline]
    #[must_use]
    pub fn strides(&self) -> (usize, usize) {
        (self.row_stride, self.col_stride)
    }
}

/// A transposed view of a strided matrix.
///
/// This is a zero-cost abstraction that swaps dimensions and strides,
/// enabling O(1) transpose operations.
///
/// # Type Parameters
/// - `Inner`: The underlying matrix type (can be owned or borrowed)
///
/// # Example
/// ```ignore
/// let matrix = StridedMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);
/// let transposed = TransposedMatrixView::new(&matrix);
///
/// // Access transposed element: transposed[r][c] = matrix[c][r]
/// assert_eq!(transposed.get(1, 0), matrix.get(0, 1));
///
/// // Double transpose returns to original
/// let original = transposed.transpose();
/// assert_eq!(original.width(), matrix.width());
/// ```
#[derive(Clone, Debug)]
pub struct TransposedMatrixView<Inner> {
    inner: Inner,
}

impl<Inner> TransposedMatrixView<Inner> {
    /// Create a new transposed view.
    #[must_use]
    pub fn new(inner: Inner) -> Self {
        Self { inner }
    }

    /// Transpose this view (returns the inner matrix).
    ///
    /// Since this is already transposed, transposing again returns
    /// the original matrix orientation.
    #[must_use]
    pub fn transpose(self) -> Inner {
        self.inner
    }

    /// Get a reference to the inner matrix.
    #[must_use]
    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

// Implement Matrix trait for StridedMatrix
impl<T, V> Matrix<T> for StridedMatrix<T, V>
where
    T: Send + Sync + Clone,
    V: DenseStorage<T>,
{
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        let idx = r * self.row_stride + c * self.col_stride;
        self.values.borrow()[idx].clone()
    }

    // Implement row_unchecked to provide an iterator over row elements
    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // Return an iterator that computes indices using strides
        (0..self.width).map(move |c| {
            let idx = r * self.row_stride + c * self.col_stride;
            self.values.borrow()[idx].clone()
        })
    }
}

// Implement Matrix trait for TransposedMatrixView<StridedMatrix>
impl<'a, T, V> Matrix<T> for TransposedMatrixView<&'a StridedMatrix<T, V>>
where
    T: Send + Sync + Clone,
    V: DenseStorage<T>,
{
    #[inline]
    fn width(&self) -> usize {
        self.inner.height  // Swap dimensions
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.width   // Swap dimensions
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        // Swap indices: transposed[r][c] = inner[c][r]
        let idx = c * self.inner.row_stride + r * self.inner.col_stride;
        self.inner.values.borrow()[idx].clone()
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // Transposed row is a column of the inner matrix
        (0..self.width()).map(move |c| {
            let idx = c * self.inner.row_stride + r * self.inner.col_stride;
            self.inner.values.borrow()[idx].clone()
        })
    }
}

// Implement Matrix trait for owned TransposedMatrixView<StridedMatrix>
impl<T, V> Matrix<T> for TransposedMatrixView<StridedMatrix<T, V>>
where
    T: Send + Sync + Clone,
    V: DenseStorage<T>,
{
    #[inline]
    fn width(&self) -> usize {
        self.inner.height
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.width
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        let idx = c * self.inner.row_stride + r * self.inner.col_stride;
        self.inner.values.borrow()[idx].clone()
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // Transposed row is a column of the inner matrix
        (0..self.width()).map(move |c| {
            let idx = c * self.inner.row_stride + r * self.inner.col_stride;
            self.inner.values.borrow()[idx].clone()
        })
    }
}

// Additional implementations for field operations
impl<T, V> StridedMatrix<T, V>
where
    T: Field,
    V: DenseStorage<T>,
{
    /// Compute Mᵀv using the transpose view for efficient column access.
    ///
    /// This implementation demonstrates the benefit of O(1) transpose:
    /// Instead of accumulating across rows (cache-inefficient), we transpose
    /// the matrix view and iterate rows (which are columns of the original).
    pub fn columnwise_dot_product<EF>(&self, v: &[EF]) -> Vec<EF>
    where
        EF: ExtensionField<T>,
    {
        assert_eq!(
            self.height, v.len(),
            "vector length must equal matrix height"
        );

        // Create transpose view (O(1))
        let transposed = self.transpose_view();

        // Iterate over rows of transposed matrix (columns of original)
        // Using row iterator for better performance
        (0..transposed.height())
            .into_par_iter()
            .map(|r| {
                // Use row_unchecked to get an iterator
                let row = unsafe { transposed.row_unchecked(r) };

                // Compute dot product
                row.into_iter()
                    .zip(v.iter())
                    .map(|(mat_elem, &vec_elem)| EF::from(mat_elem) * vec_elem)
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::*;

    #[test]
    fn test_row_major_basic() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = StridedMatrix::new(data, 3);  // 2×3 matrix

        assert_eq!(matrix.width(), 3);
        assert_eq!(matrix.height(), 2);
        assert!(matrix.is_row_major_contiguous());

        // Check elements
        assert_eq!(unsafe { matrix.get_unchecked(0, 0) }, 1);
        assert_eq!(unsafe { matrix.get_unchecked(0, 1) }, 2);
        assert_eq!(unsafe { matrix.get_unchecked(0, 2) }, 3);
        assert_eq!(unsafe { matrix.get_unchecked(1, 0) }, 4);
        assert_eq!(unsafe { matrix.get_unchecked(1, 1) }, 5);
        assert_eq!(unsafe { matrix.get_unchecked(1, 2) }, 6);
    }

    #[test]
    fn test_transpose_view_dimensions() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = StridedMatrix::new(data, 3);  // 2×3
        let transposed = matrix.transpose_view();

        assert_eq!(transposed.width(), 2);   // Was height
        assert_eq!(transposed.height(), 3);  // Was width
    }

    #[test]
    fn test_transpose_view_elements() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = StridedMatrix::new(data, 3);
        // [[1, 2, 3],
        //  [4, 5, 6]]

        let transposed = matrix.transpose_view();
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]

        assert_eq!(unsafe { transposed.get_unchecked(0, 0) }, 1);
        assert_eq!(unsafe { transposed.get_unchecked(0, 1) }, 4);
        assert_eq!(unsafe { transposed.get_unchecked(1, 0) }, 2);
        assert_eq!(unsafe { transposed.get_unchecked(1, 1) }, 5);
        assert_eq!(unsafe { transposed.get_unchecked(2, 0) }, 3);
        assert_eq!(unsafe { transposed.get_unchecked(2, 1) }, 6);
    }

    #[test]
    fn test_double_transpose() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = StridedMatrix::new(data, 3);

        let transposed = matrix.transpose_view();
        let double_transposed = transposed.transpose();

        assert_eq!(double_transposed.width(), matrix.width());
        assert_eq!(double_transposed.height(), matrix.height());
    }

    #[test]
    fn test_col_major() {
        // Data laid out column by column: [col0, col1, col2]
        let data = vec![1, 4, 2, 5, 3, 6];
        let matrix = StridedMatrix::new_col_major(data, 3);  // 2×3

        assert!(matrix.is_col_major_contiguous());

        // Should still see logical row-major view
        assert_eq!(unsafe { matrix.get_unchecked(0, 0) }, 1);
        assert_eq!(unsafe { matrix.get_unchecked(0, 1) }, 2);
        assert_eq!(unsafe { matrix.get_unchecked(0, 2) }, 3);
        assert_eq!(unsafe { matrix.get_unchecked(1, 0) }, 4);
        assert_eq!(unsafe { matrix.get_unchecked(1, 1) }, 5);
        assert_eq!(unsafe { matrix.get_unchecked(1, 2) }, 6);
    }

    #[test]
    fn test_row_slice_row_major() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let matrix = StridedMatrix::new(data, 3);

        unsafe {
            let row0 = matrix.row_slice_unchecked(0);
            assert_eq!(&*row0, &[1, 2, 3]);

            let row1 = matrix.row_slice_unchecked(1);
            assert_eq!(&*row1, &[4, 5, 6]);
        }
    }

    #[test]
    fn test_strided_access() {
        // Interleaved data: [a00, garbage, a01, garbage, ...]
        let data = vec![1, 99, 2, 99, 3, 99, 4, 99, 5, 99, 6, 99];
        let matrix = StridedMatrix::with_strides(data, 3, 2, 6, 2);

        assert_eq!(unsafe { matrix.get_unchecked(0, 0) }, 1);
        assert_eq!(unsafe { matrix.get_unchecked(0, 1) }, 2);
        assert_eq!(unsafe { matrix.get_unchecked(0, 2) }, 3);
        assert_eq!(unsafe { matrix.get_unchecked(1, 0) }, 4);
        assert_eq!(unsafe { matrix.get_unchecked(1, 1) }, 5);
        assert_eq!(unsafe { matrix.get_unchecked(1, 2) }, 6);
    }
}

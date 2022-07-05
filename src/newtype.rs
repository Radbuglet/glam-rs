use core::{
    fmt, hash,
    iter::{Product, Sum},
    marker::PhantomData,
    mem::transmute,
    ops::{
        Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Index, IndexMut, Mul, MulAssign,
        Neg, Not, Rem, RemAssign, Shl, Shr, Sub, SubAssign,
    },
};

// === `BackingVec` === //

pub(crate) mod backing_vec {
    pub trait Sealed {}
}

pub trait BackingVec:
    // This bound ensures that users cannot extend this trait.
    backing_vec::Sealed +
    // These bounds encode some of the properties common to all backing vectors. The remaining
    // "properties" are derived by snippets of code generated for every vector type.
    fmt::Debug + fmt::Display + Copy + PartialEq + Default +
    Index<usize, Output = Self> + IndexMut<usize>
{
}

// === `VecFlavor` and `FlavorConvertFrom` === //

pub trait VecFlavor {
    type Backing: BackingVec;
}

pub trait FlavorConvertFrom<O>: VecFlavor {
    fn vec_backing_from(other: O) -> Self::Backing;
}

impl<F: VecFlavor> FlavorConvertFrom<NewTypeVector<F>> for F {
    fn vec_backing_from(other: NewTypeVector<F>) -> Self::Backing {
        other.into_raw()
    }
}

// The following `impl`s must be implemented manually with every derivation of `BackingVec`:
// impl<F: VecFlavor> FlavorConvertFrom<F::Backing> for F {
//     fn convert_from(other: F::Backing) -> Self::Backing {
//         other
//     }
// }

// === `NewTypeVector` === //

pub type NewTypeVector<F> = NewTypeVectorImpl<<F as VecFlavor>::Backing, F>;

// We keep `B` as its own parameter—despite it being trivially re-derivable—so Rust can figure out
// the difference between an `impl` on all flavors that have a `BackingVec = IVec3` and an `impl` on
// all flavors that have a `BackingVec = UVec3`. For most intents and purposes, we can just use
// `NewTypeVector` directly. It is, after all, the only valid choice for generic parameter pairs.
#[repr(transparent)]
pub struct NewTypeVectorImpl<B: BackingVec, F: VecFlavor<Backing = B>> {
    _flavor: PhantomData<fn(F) -> F>,
    vec: B,
}

// Constructors

impl<F: VecFlavor> NewTypeVector<F> {
    pub const fn from_raw(vec: F::Backing) -> Self {
        Self {
            _flavor: PhantomData,
            vec,
        }
    }

    pub fn from<O>(other: O) -> Self
    where
        F: FlavorConvertFrom<O>,
    {
        Self::from_raw(F::vec_backing_from(other))
    }

    pub fn into<T: FlavorConvertFrom<Self>>(self) -> NewTypeVector<T> {
        NewTypeVector::<T>::from(self)
    }

    pub const fn from_raw_ref(vec: &F::Backing) -> &Self {
        unsafe {
            // Safety: `NewTypeVectorImpl` is `repr(transparent)` w.r.t `F::Backing` and thus so is
            // its reference.
            transmute(vec)
        }
    }

    pub fn from_raw_mut(vec: &mut F::Backing) -> &mut Self {
        unsafe {
            // Safety: `NewTypeVectorImpl` is `repr(transparent)` w.r.t `F::Backing` and thus so is
            // its reference.
            transmute(vec)
        }
    }

    pub const fn into_raw(self) -> F::Backing {
        self.vec
    }

    pub const fn raw(&self) -> &F::Backing {
        &self.vec
    }

    pub fn raw_mut(&mut self) -> &mut F::Backing {
        &mut self.vec
    }

    pub(crate) fn map_raw<C>(self, f: C) -> Self
    where
        C: FnOnce(F::Backing) -> F::Backing,
    {
        Self::from_raw(f(self.into_raw()))
    }
}

// Basic `impl`s

impl<F: VecFlavor> fmt::Debug for NewTypeVector<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.raw(), f)
    }
}

impl<F: VecFlavor> fmt::Display for NewTypeVector<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.raw(), f)
    }
}

impl<F: VecFlavor> Copy for NewTypeVector<F> {}

impl<F: VecFlavor> Clone for NewTypeVector<F> {
    fn clone(&self) -> Self {
        Self {
            _flavor: self._flavor,
            vec: self.vec,
        }
    }
}

impl<F: VecFlavor> PartialEq for NewTypeVector<F> {
    fn eq(&self, other: &Self) -> bool {
        self.vec == other.vec
    }
}

impl<F: VecFlavor> Eq for NewTypeVector<F> where F::Backing: Eq {}

impl<F: VecFlavor> hash::Hash for NewTypeVector<F>
where
    F::Backing: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vec.hash(state);
    }
}

impl<F: VecFlavor> Default for NewTypeVector<F> {
    fn default() -> Self {
        Self::from_raw(Default::default())
    }
}

impl<F: VecFlavor> Index<usize> for NewTypeVector<F> {
    type Output = Self;

    fn index(&self, index: usize) -> &Self::Output {
        Self::from_raw_ref(&self.raw()[index])
    }
}

impl<F: VecFlavor> IndexMut<usize> for NewTypeVector<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        Self::from_raw_mut(&mut self.raw_mut()[index])
    }
}

// Misc impls

impl<'a, F: VecFlavor> Sum<&'a NewTypeVector<F>> for NewTypeVector<F>
where
    F: VecFlavor,
    F::Backing: 'a + Sum<&'a F::Backing>,
{
    fn sum<I: Iterator<Item = &'a NewTypeVector<F>>>(iter: I) -> Self {
        Self::from_raw(iter.map(|v| v.raw()).sum())
    }
}

impl<'a, F: VecFlavor> Product<&'a NewTypeVector<F>> for NewTypeVector<F>
where
    F: VecFlavor,
    F::Backing: 'a + Product<&'a F::Backing>,
{
    fn product<I: Iterator<Item = &'a NewTypeVector<F>>>(iter: I) -> Self {
        Self::from_raw(iter.map(|v| v.raw()).product())
    }
}

// `core::ops` impls

impl<F, R> Add<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: Add<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn add(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs + F::vec_backing_from(rhs))
    }
}

impl<F, R> AddAssign<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: AddAssign<F::Backing>,
{
    fn add_assign(&mut self, rhs: R) {
        *self.raw_mut() += F::vec_backing_from(rhs);
    }
}

impl<F, R> Sub<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: Sub<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn sub(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs - F::vec_backing_from(rhs))
    }
}

impl<F, R> SubAssign<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: SubAssign<F::Backing>,
{
    fn sub_assign(&mut self, rhs: R) {
        *self.raw_mut() -= F::vec_backing_from(rhs);
    }
}

impl<F, R> Mul<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: Mul<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn mul(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs * F::vec_backing_from(rhs))
    }
}

impl<F, R> MulAssign<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: MulAssign<F::Backing>,
{
    fn mul_assign(&mut self, rhs: R) {
        *self.raw_mut() *= F::vec_backing_from(rhs);
    }
}

impl<F, R> Div<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: Div<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn div(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs / F::vec_backing_from(rhs))
    }
}

impl<F, R> DivAssign<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: DivAssign<F::Backing>,
{
    fn div_assign(&mut self, rhs: R) {
        *self.raw_mut() /= F::vec_backing_from(rhs);
    }
}

impl<F, R> Rem<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: Rem<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn rem(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs % F::vec_backing_from(rhs))
    }
}

impl<F, R> RemAssign<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: RemAssign<F::Backing>,
{
    fn rem_assign(&mut self, rhs: R) {
        *self.raw_mut() %= F::vec_backing_from(rhs);
    }
}

impl<F> Not for NewTypeVector<F>
where
    F: VecFlavor,
    F::Backing: Not<Output = F::Backing>,
{
    type Output = Self;

    fn not(self) -> Self::Output {
        self.map_raw(|lhs| !lhs)
    }
}

impl<F> Neg for NewTypeVector<F>
where
    F: VecFlavor,
    F::Backing: Neg<Output = F::Backing>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map_raw(|lhs| -lhs)
    }
}

// TODO: Explain why this choice is reasonable.
impl<F, R> Shl<R> for NewTypeVector<F>
where
    F: VecFlavor,
    F::Backing: Shl<R, Output = F::Backing>,
{
    type Output = Self;

    fn shl(self, rhs: R) -> Self::Output {
        self.map_raw(|v| v << rhs)
    }
}

impl<F, R> Shr<R> for NewTypeVector<F>
where
    F: VecFlavor,
    F::Backing: Shr<R, Output = F::Backing>,
{
    type Output = Self;

    fn shr(self, rhs: R) -> Self::Output {
        self.map_raw(|v| v >> rhs)
    }
}

impl<F, R> BitAnd<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: BitAnd<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn bitand(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs & F::vec_backing_from(rhs))
    }
}

impl<F, R> BitOr<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: BitOr<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn bitor(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs | F::vec_backing_from(rhs))
    }
}

impl<F, R> BitXor<R> for NewTypeVector<F>
where
    F: FlavorConvertFrom<R>,
    F::Backing: BitXor<F::Backing, Output = F::Backing>,
{
    type Output = Self;

    fn bitxor(self, rhs: R) -> Self::Output {
        self.map_raw(|lhs| lhs ^ F::vec_backing_from(rhs))
    }
}

// `ShlAssign`, `ShrAssign`, `BitAndAssign`, `BitOrAssign`, and `BitXorAssign` are never implemented
// on the underlying types.

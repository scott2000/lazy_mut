//! This crate provides a structure that can be used to lazily initialized values as well as a
//! macro for creating lazy variables.
//!
//! `LazyMut<T>` dereferences to `T`, however it will throw an error if it is dereferenced immutably
//! before it is initialized. Mutable dereferences are allowed and will automatically initialize the
//! variable, to allow assignments like `*VEC = vec![1, 2, 3]` on an uninitialized `VEC`. You
//! should still include the explicit initialization for clarity, even if it is unnecessary.
//!
//! # Local Variables
//!
//! Using the `lazy_mut` macro makes declaring lazy variables easier:
//!
//! ```
//! #[macro_use]
//! extern crate lazy_mut;
//!
//! # fn main() {
//! lazy_mut! {
//!     let mut num: u32 = 2 + 3;
//! }
//!
//! num.init();
//! assert_eq!(*num, 5);
//! # }
//! ```
//!
//! Another, more lengthy way to write this would be:
//!
//! ```
//! use lazy_mut::LazyMut;
//!
//! let mut num = {
//!     fn init() -> u32 { 2 + 3 }
//!     LazyMut::Init(init)
//! };
//!
//! num.init();
//! assert_eq!(*num, 5);
//! ```
//!
//! The function definition makes this code harder to read and understand than the example with the
//! macro. However, this code does emphasize the fact that the variable is lazy and must be
//! initialized before use. You should consider this trade-off when choosing how to use `LazyMut`.
//!
//! # Static Variables
//!
//! The `lazy_mut` macro also works for static variables:
//!
//! ```
//! #[macro_use]
//! extern crate lazy_mut;
//!
//! lazy_mut! {
//!     static mut VEC: Vec<u64> = Vec::new();
//! }
//!
//! # fn main() {
//! # unsafe {
//! VEC.init(); // Although technically unnecessary, it is more clear to explicitly initialize it
//! VEC.push(17);
//! VEC.push(64);
//!
//! assert_eq!(*VEC, vec![17, 64]);
//! # }
//! # }
//! ```
//!
//! Another way to write this would be:
//!
//! ```
//! use lazy_mut::LazyMut;
//!
//! static mut VEC: LazyMut<Vec<u64>> = LazyMut::Init(Vec::new);
//!
//! # fn main() {
//! # unsafe {
//! VEC.push(17);
//! VEC.push(64);
//!
//! assert_eq!(*VEC, vec![17, 64]);
//! # }
//! # }
//! ```
//!
//! Note that with the direct definition the function `Vec::new` can be passed directly, making it
//! simpler to write. `LazyMut` can be used to make simple initializers for types that require heap
//! allocations at runtime, such as collections, strings, or boxed types.
#![deny(missing_docs)]

use std::ops::{Deref, DerefMut};
use std::fmt::{self, Display};

/// A macro that creates lazy variables
///
/// # Usage
///
/// ```ignore
/// lazy_mut! {
///     // Local variables
///     let mut NAME: TY = EXPR;
///
///     // Static variables
///     [pub [(VIS)]] static mut NAME: TY = EXPR;
/// }
/// ```
#[macro_export]
macro_rules! lazy_mut {
    (/* empty */) => {};
    ($(#[$attr:meta])* let mut $N:ident: $T:ty = $e:expr; $($t:tt)*) => {
        $(#[$attr])*
        let mut $N: $crate::LazyMut<$T> = {
            fn init() -> $T { $e }
            $crate::LazyMut::Init(init)
        };
        lazy_mut!($($t)*);
    };
    ($(#[$attr:meta])* static mut $N:ident: $T:ty = $e:expr; $($t:tt)*) => {
        $(#[$attr])*
        static mut $N: $crate::LazyMut<$T> = {
            fn init() -> $T { $e }
            $crate::LazyMut::Init(init)
        };
        lazy_mut!($($t)*);
    };
    ($(#[$attr:meta])* pub static mut $N:ident: $T:ty = $e:expr; $($t:tt)*) => {
        $(#[$attr])*
        pub static mut $N: $crate::LazyMut<$T> = {
            fn init() -> $T { $e }
            $crate::LazyMut::Init(init)
        };
        lazy_mut!($($t)*);
    };
    ($(#[$attr:meta])* pub ($($vis:tt)+) static mut $N:ident: $T:ty = $e:expr;
        $($t:tt)*) => {
        $(#[$attr])*
        pub ($($vis)+) static mut $N: $crate::LazyMut<$T> = {
            fn init() -> $T { $e }
            $crate::LazyMut::Init(init)
        };
        lazy_mut!($($t)*);
    };
}

/// A mutable lazy value with either an initializer or a value
///
/// See the module-level documentation for more information on usage.
#[derive(Clone, Debug)]
pub enum LazyMut<T> {
    /// An initializer that will be run to obtain the first value
    Init(fn() -> T),
    /// The value from the initializer
    Value(T),
}

impl<T> LazyMut<T> {
    /// Returns the wrapped value, initializing if needed
    pub fn unwrap(self) -> T {
        use LazyMut::*;
        match self {
            Init(init) => init(),
            Value(val) => val,
        }
    }

    /// Initializes the wrapped value if it is uninitialized
    pub fn init(&mut self) -> &mut LazyMut<T> {
        use LazyMut::*;
        let new = match self {
            &mut Init(init) => Value(init()),
            other => return other,
        };
        *self = new;
        self
    }

    /// Initializes the wrapped value, panicking if it was already initialized
    pub fn init_once(&mut self) -> &mut LazyMut<T> {
        use LazyMut::*;
        let new = match self {
            &mut Init(init) => Value(init()),
            _ => panic!("call to `init_once` on already initialized value"),
        };
        *self = new;
        self
    }

    /// Tries to get a reference to the value, returns `None` if the value is uninitialized
    ///
    /// Uses associated function syntax (`LazyMut::get(&VAL)`)
    pub fn get(this: &LazyMut<T>) -> Option<&T> {
        use LazyMut::*;
        match this {
            &Init(_) => None,
            &Value(ref val) => Some(val),
        }
    }

    /// Tries to get a mutable reference the value, returns `None` if the value is uninitialized
    ///
    /// Uses associated function syntax (`LazyMut::get_mut(&mut VAL)`)
    pub fn get_mut(this: &mut LazyMut<T>) -> Option<&mut T> {
        use LazyMut::*;
        match this {
            &mut Init(_) => None,
            &mut Value(ref mut val) => Some(val),
        }
    }

    /// Returns `true` if the wrapped value has been initialized
    pub fn is_initialized(&self) -> bool {
        use LazyMut::*;
        match self {
            &Init(_) => false,
            &Value(_) => true,
        }
    }
}

impl<T> Deref for LazyMut<T> {
    type Target = T;
    fn deref(&self) -> &T {
        use LazyMut::*;
        match self {
            &Init(_) => panic!("cannot dereference uninitialized value"),
            &Value(ref val) => val,
        }
    }
}

impl<T> DerefMut for LazyMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.init();
        use LazyMut::*;
        match self {
            &mut Init(_) => unreachable!(),
            &mut Value(ref mut val) => val,
        }
    }
}

impl<T> Display for LazyMut<T> where T: Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use LazyMut::*;
        match self {
            &Init(_) => write!(f, "{{uninitialized}}"),
            &Value(ref val) => val.fmt(f),
        }
    }
}
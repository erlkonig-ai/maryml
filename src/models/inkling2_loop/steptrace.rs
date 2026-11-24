//! Host timestamps inside one forward, off unless a bench turns them on
//! ([`enable`]). A mark is one `Instant::now()` pushed onto a thread-local
//! vector: no device sync, no formatting, no lock.
//!
//! Marks cover the thread that builds the forward. What the device thread
//! submits, and when the device ran it, is a device trace's to say.
//!
//! [`route_counts`] is separate and does sync: it reads each visit's picks, for
//! counting distinct experts in an untimed run.

use std::cell::{Cell, RefCell};
use std::time::Instant;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<bool> = const { Cell::new(false) };
    static MARKS: RefCell<Vec<(&'static str, usize, usize, Instant)>> = const { RefCell::new(Vec::new()) };
    static DISTINCT: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

/// Turn marks on or off for this thread.
pub fn enable(on: bool) {
    ON.with(|o| o.set(on));
}

pub fn mark(name: &'static str, layer: usize, pass: usize) {
    if ON.with(|o| o.get()) {
        MARKS.with(|m| m.borrow_mut().push((name, layer, pass, Instant::now())));
    }
}

/// The marks since the last call, in order.
pub fn take() -> Vec<(&'static str, usize, usize, Instant)> {
    MARKS.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// Record `(rows, distinct experts)` per visit. Reads the device.
pub fn route_counts(on: bool) {
    COUNT.with(|c| c.set(on));
}

pub fn counting() -> bool {
    COUNT.with(|c| c.get())
}

pub fn add_distinct(rows: usize, distinct: usize) {
    DISTINCT.with(|d| d.borrow_mut().push((rows, distinct)));
}

pub fn take_distinct() -> Vec<(usize, usize)> {
    DISTINCT.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

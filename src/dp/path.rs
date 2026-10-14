//! Path reconstruction from argbest predecessors.
//!
//! A forward DP with predecessor tracking stores, for every node, the smallest predecessor
//! whose term realises the node's value (or [`NONE`]); following these links from a node walks
//! an optimal path back to the node where it starts (a window source, or a node whose initial
//! term won). A backward DP stores argbest successors, and the same walk goes forward to a
//! sink. This is the witness construction of selective semirings
//! ([\[Hua08\]](crate::references#hua08), [\[Moh02\]](crate::references#moh02)); for MaxPlus it
//! reproduces the critical path of [\[KW59\]](crate::references#kw59).
//!
//! Complexity: `O(path length)`.

use crate::io::NONE;
use crate::window::Window;

/// Follows `pred` (indexed by `v − window.lo`, absolute ids) from `end` until a [`NONE`]
/// link, and returns the visited nodes **in walk order**, starting with `end`.
///
/// For a forward DP the result is the path reversed (`end` first, its start last); for a
/// backward DP it is the path in order (`end` first, the sink last).
///
/// # Panics
/// If `end` is not in the window, if `pred.len() != window.len()`, or if a link leaves the
/// window or the walk does not terminate within `window.len()` steps (a corrupted array).
///
/// ```
/// use dag::{NONE, Window};
/// use dag::dp::path::path_to;
/// let pred = [NONE, 0, 1, 1];
/// assert_eq!(path_to(&pred, Window { lo: 0, hi: 4 }, 3), [3, 1, 0]);
/// ```
pub fn path_to(pred: &[u32], window: Window, end: usize) -> Vec<u32> {
    assert_eq!(
        pred.len(),
        window.len(),
        "pred has {} entries, the window {window} has {} nodes",
        pred.len(),
        window.len()
    );
    assert!(
        window.contains(end),
        "node {end} is not in the window {window}"
    );
    let mut path = vec![end as u32];
    let mut v = end;
    loop {
        let p = pred[v - window.lo];
        if p == NONE {
            return path;
        }
        let p = p as usize;
        assert!(
            window.contains(p) && p != v,
            "pred[{v}] = {p} is not a valid link in the window {window}"
        );
        assert!(
            path.len() <= window.len(),
            "the predecessor links starting at {end} form a cycle"
        );
        path.push(p as u32);
        v = p;
    }
}

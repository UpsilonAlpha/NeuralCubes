use super::ParseError;
use std::fmt;

/// A set of neighbor counts (0..=26), stored as a bitmask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CountSet(u32);

impl CountSet {
    pub fn new(counts: impl IntoIterator<Item = u32>) -> Self {
        Self(counts.into_iter().fold(0, |m, c| m | 1 << c))
    }

    #[inline]
    pub fn contains(self, n: u32) -> bool {
        self.0 >> n & 1 == 1
    }

    /// Parses `5-7,12` style lists. Counts above `max` are rejected.
    pub fn parse(s: &str, max: u32) -> Result<Self, ParseError> {
        let mut mask = 0u32;
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let num = |t: &str| {
                t.trim()
                    .parse::<u32>()
                    .ok()
                    .filter(|&n| n <= max)
                    .ok_or_else(|| ParseError(format!("`{t}` is not a neighbor count in 0..={max}")))
            };
            let (lo, hi) = match part.split_once('-') {
                Some((a, b)) => (num(a)?, num(b)?),
                None => (num(part)?, num(part)?),
            };
            if lo > hi {
                return Err(ParseError(format!("range `{part}` runs backwards")));
            }
            for n in lo..=hi {
                mask |= 1 << n;
            }
        }
        Ok(Self(mask))
    }
}

impl fmt::Display for CountSet {
    /// Writes the set compactly, collapsing runs of three or more into ranges.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        let mut n = 0;
        while n < 32 {
            if !self.contains(n) {
                n += 1;
                continue;
            }
            let start = n;
            while n + 1 < 32 && self.contains(n + 1) {
                n += 1;
            }
            match n - start {
                0 => parts.push(format!("{start}")),
                1 => parts.push(format!("{start},{n}")),
                _ => parts.push(format!("{start}-{n}")),
            }
            n += 1;
        }
        f.write_str(&parts.join(","))
    }
}

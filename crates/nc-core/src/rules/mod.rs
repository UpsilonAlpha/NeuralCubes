//! Built-in rule families.
//!
//! | Family        | Cell type | Notation       | Example     |
//! |---------------|-----------|----------------|-------------|
//! | `life-like`   | `bool`    | `S/B/2/N`      | `4/4/2/M`   |
//! | `generations` | `u8`      | `S/B/C/N`, C>2 | `4/4/5/M`   |
//!
//! `S` and `B` list the neighbor counts for survival and birth (`5-7,12`), `C` is the
//! number of states, and `N` is the neighborhood: `M` (Moore, 26) or `N` (von Neumann, 6).

mod counts;
mod generations;
mod life_like;

pub use counts::CountSet;
pub use generations::Generations;
pub use life_like::LifeLike;

use crate::neighborhood::Neighborhood;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// The fields of `S/B/C/N` notation, shared by the totalistic families.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Totalistic {
    pub survive: CountSet,
    pub birth: CountSet,
    pub states: u8,
    pub neighborhood: Neighborhood,
}

impl Totalistic {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let parts: Vec<&str> = s.trim().split('/').map(str::trim).collect();
        let [survive, birth, states, hood] = parts[..] else {
            return Err(ParseError(format!("expected S/B/C/N (e.g. 4/4/5/M), got `{s}`")));
        };
        let neighborhood = match hood.to_ascii_uppercase().as_str() {
            "M" => Neighborhood::Moore,
            "N" | "VN" => Neighborhood::VonNeumann,
            other => return Err(ParseError(format!("unknown neighborhood `{other}`; use M or N"))),
        };
        let states: u8 = states
            .parse()
            .ok()
            .filter(|&c| c >= 2)
            .ok_or_else(|| ParseError(format!("state count must be an integer from 2 to 255, got `{states}`")))?;
        let max = neighborhood.size() as u32;
        Ok(Self {
            survive: CountSet::parse(survive, max)?,
            birth: CountSet::parse(birth, max)?,
            states,
            neighborhood,
        })
    }

    pub fn notation(&self) -> String {
        format!("{}/{}/{}/{}", self.survive, self.birth, self.states, self.neighborhood.letter())
    }
}

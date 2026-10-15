use super::{CountSet, ParseError, Totalistic};
use crate::neighborhood::{Neighborhood, Neighbors};
use crate::rng::Rng;
use crate::rule::Rule;

/// Binary outer-totalistic rules: a cell is alive or dead, and only the number of live
/// neighbors matters. Conway's Life is the 2D member of this family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LifeLike {
    pub survive: CountSet,
    pub birth: CountSet,
    pub neighborhood: Neighborhood,
}

impl LifeLike {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let t = Totalistic::parse(s)?;
        if t.states != 2 {
            return Err(ParseError(format!(
                "life-like rules have 2 states, `{s}` has {}",
                t.states
            )));
        }
        Ok(Self {
            survive: t.survive,
            birth: t.birth,
            neighborhood: t.neighborhood,
        })
    }
}

impl Rule for LifeLike {
    type Cell = bool;

    fn notation(&self) -> String {
        Totalistic {
            survive: self.survive,
            birth: self.birth,
            states: 2,
            neighborhood: self.neighborhood,
        }
        .notation()
    }

    fn family(&self) -> &'static str {
        "life-like"
    }

    fn neighborhood(&self) -> Neighborhood {
        self.neighborhood
    }

    #[inline]
    fn next(&self, alive: bool, n: &Neighbors<'_, bool>) -> bool {
        let live = n.count(|c| c);
        if alive {
            self.survive.contains(live)
        } else {
            self.birth.contains(live)
        }
    }

    fn seed_cell(&self, _rng: &mut Rng) -> bool {
        true
    }

    fn shade(&self, alive: bool) -> Option<f32> {
        alive.then_some(1.0)
    }
}

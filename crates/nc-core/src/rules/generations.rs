use super::{CountSet, ParseError, Totalistic};
use crate::neighborhood::{Neighborhood, Neighbors};
use crate::rng::Rng;
use crate::rule::Rule;

/// Totalistic rules with decay. State 0 is empty, 1 is alive, and `2..states` are
/// dying: a live cell that fails to survive starts counting up through the dying states
/// and then becomes empty. Only fully alive cells count as neighbors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generations {
    pub survive: CountSet,
    pub birth: CountSet,
    pub states: u8,
    pub neighborhood: Neighborhood,
}

impl Generations {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let t = Totalistic::parse(s)?;
        Ok(Self {
            survive: t.survive,
            birth: t.birth,
            states: t.states,
            neighborhood: t.neighborhood,
        })
    }
}

impl Rule for Generations {
    type Cell = u8;

    fn notation(&self) -> String {
        Totalistic {
            survive: self.survive,
            birth: self.birth,
            states: self.states,
            neighborhood: self.neighborhood,
        }
        .notation()
    }

    fn family(&self) -> &'static str {
        "generations"
    }

    fn neighborhood(&self) -> Neighborhood {
        self.neighborhood
    }

    #[inline]
    fn next(&self, state: u8, n: &Neighbors<'_, u8>) -> u8 {
        match state {
            0 => u8::from(self.birth.contains(n.count(|c| c == 1))),
            1 if self.survive.contains(n.count(|c| c == 1)) => 1,
            s if s + 1 >= self.states => 0,
            s => s + 1,
        }
    }

    fn seed_cell(&self, _rng: &mut Rng) -> u8 {
        1
    }

    fn shade(&self, state: u8) -> Option<f32> {
        match state {
            0 => None,
            s => Some(1.0 - (s - 1) as f32 / (self.states - 1) as f32),
        }
    }
}

/*
 * Idea from: https://github.com/rustomax/rust-collatz
 *
 */

use std::env;
use std::process;
use std::fs::File;
use std::io::Write;

pub struct Collatz {
    curr: u64,
    done: bool,
}

impl Collatz {
    pub fn new(n: u64) -> Collatz {
        Collatz {curr: n, done: false}
    }
}

impl Iterator for Collatz {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.done {
            return None;
        }

        let result = Some(self.curr);

        self.curr = match self.curr {
            n if n <= 1 => {self.done = true; 1},
            n if n & 1 == 0 => {n / 2},
            n => {n * 3 + 1}
        };

        return result;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("usage: ./collatz <N>");
        process::exit(1);
    }

    let limit: u64 = args[1].parse::<u64>().unwrap();

    let mut f = File::create("out.txt").expect("Unable to create file");

    for n in 1..=limit {
        let xs: Vec<u64> = Collatz::new(n).collect();
        let sz: usize = xs.len() - 1;
        let _r = write!(f, "{}\n", sz);
    }
}

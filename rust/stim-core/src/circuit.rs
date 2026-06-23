//! Minimal circuit IR and a subset parser for Stim's `.stim` text format.
//!
//! Analogue of `src/stim/circuit/circuit.h` + `circuit_instruction.h`, scoped to
//! the gates the PoC frame simulator supports. The parser understands:
//!   * `NAME(arg, arg, ...) target target ...`
//!   * integer qubit targets
//!   * `#` line comments
//!   * `REPEAT n { ... }` blocks (flattened on parse)
//!   * annotation instructions that don't affect measurement sampling
//!     (`TICK`, `QUBIT_COORDS`, `SHIFT_COORDS`, `DETECTOR`, `OBSERVABLE_INCLUDE`)
//!     which are ignored.

/// Supported gates. Pauli gates (X/Y/Z) are intentionally frame no-ops: in the
/// Pauli-frame / measurement-flip picture a deterministic Pauli only shifts the
/// reference sample, not the error frame (matching Stim's frame simulator).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    I,
    X,
    Y,
    Z,
    H,
    S,
    Cx,
    Cz,
    Swap,
    R,
    M,
    Mr,
    XError,
    YError,
    ZError,
    Depolarize1,
    Depolarize2,
}

impl Gate {
    /// Number of measurement results this gate produces per target.
    pub fn measurements_per_target(self) -> usize {
        matches!(self, Gate::M | Gate::Mr) as usize
    }

    pub fn is_two_qubit(self) -> bool {
        matches!(self, Gate::Cx | Gate::Cz | Gate::Swap | Gate::Depolarize2)
    }
}

#[derive(Clone, Debug)]
pub struct Instruction {
    pub gate: Gate,
    pub args: Vec<f64>,
    pub targets: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Circuit {
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    pub fn new() -> Self {
        Circuit::default()
    }

    /// Highest qubit index referenced + 1.
    pub fn num_qubits(&self) -> usize {
        self.instructions
            .iter()
            .filter(|i| !matches!(i.gate, Gate::I)) // I may carry no targets
            .flat_map(|i| i.targets.iter())
            .map(|&q| q as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Total number of measurement results the circuit produces.
    pub fn num_measurements(&self) -> usize {
        self.instructions
            .iter()
            .map(|i| i.gate.measurements_per_target() * i.targets.len())
            .sum()
    }

    /// Parses a circuit from Stim text format (supported subset).
    pub fn from_text(text: &str) -> Result<Circuit, String> {
        let mut parser = Parser {
            tokens: tokenize(text),
            pos: 0,
        };
        let mut circuit = Circuit::new();
        parser.parse_block(&mut circuit.instructions, false)?;
        Ok(circuit)
    }
}

fn gate_from_name(name: &str) -> Option<Gate> {
    Some(match name {
        "I" => Gate::I,
        "X" => Gate::X,
        "Y" => Gate::Y,
        "Z" => Gate::Z,
        "H" | "H_XZ" => Gate::H,
        "S" | "SQRT_Z" => Gate::S,
        "CX" | "ZCX" | "CNOT" => Gate::Cx,
        "CZ" | "ZCZ" => Gate::Cz,
        "SWAP" => Gate::Swap,
        "R" | "RZ" => Gate::R,
        "M" | "MZ" => Gate::M,
        "MR" | "MRZ" => Gate::Mr,
        "X_ERROR" => Gate::XError,
        "Y_ERROR" => Gate::YError,
        "Z_ERROR" => Gate::ZError,
        "DEPOLARIZE1" => Gate::Depolarize1,
        "DEPOLARIZE2" => Gate::Depolarize2,
        _ => return None,
    })
}

/// Instructions that are accepted but ignored for measurement sampling.
fn is_ignored_annotation(name: &str) -> bool {
    matches!(
        name,
        "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" | "DETECTOR" | "OBSERVABLE_INCLUDE"
    )
}

#[derive(Debug, PartialEq)]
enum Token {
    Word(String),
    Number(f64),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Newline,
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    for line in text.lines() {
        let line = match line.split('#').next() {
            Some(l) => l,
            None => line,
        };
        let mut chars = line.char_indices().peekable();
        while let Some(&(start, c)) = chars.peek() {
            match c {
                ' ' | '\t' | '\r' => {
                    chars.next();
                }
                '(' => {
                    chars.next();
                    tokens.push(Token::LParen);
                }
                ')' => {
                    chars.next();
                    tokens.push(Token::RParen);
                }
                '{' => {
                    chars.next();
                    tokens.push(Token::LBrace);
                }
                '}' => {
                    chars.next();
                    tokens.push(Token::RBrace);
                }
                ',' => {
                    chars.next();
                    tokens.push(Token::Comma);
                }
                '*' => {
                    // Combined Pauli target separator; unsupported but skip char.
                    chars.next();
                }
                _ => {
                    // Consume a run of non-delimiter characters.
                    let mut end = start;
                    while let Some(&(i, ch)) = chars.peek() {
                        if ch.is_whitespace() || "(){},#".contains(ch) {
                            break;
                        }
                        end = i + ch.len_utf8();
                        chars.next();
                    }
                    let word = &line[start..end];
                    if let Ok(n) = word.parse::<f64>() {
                        tokens.push(Token::Number(n));
                    } else {
                        tokens.push(Token::Word(word.to_string()));
                    }
                }
            }
        }
        tokens.push(Token::Newline);
    }
    tokens
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Token::Newline)) {
            self.pos += 1;
        }
    }

    fn parse_block(&mut self, out: &mut Vec<Instruction>, in_repeat: bool) -> Result<(), String> {
        loop {
            self.skip_newlines();
            match self.peek() {
                None => {
                    if in_repeat {
                        return Err("unterminated REPEAT block".into());
                    }
                    return Ok(());
                }
                Some(Token::RBrace) => {
                    if !in_repeat {
                        return Err("unexpected '}'".into());
                    }
                    self.next();
                    return Ok(());
                }
                Some(Token::Word(w)) if w == "REPEAT" => {
                    self.next();
                    let count = match self.next() {
                        Some(Token::Number(n)) => *n as usize,
                        other => return Err(format!("expected REPEAT count, got {:?}", other)),
                    };
                    // Expect '{' (possibly after a newline).
                    self.skip_newlines();
                    if !matches!(self.next(), Some(Token::LBrace)) {
                        return Err("expected '{' after REPEAT count".into());
                    }
                    let mut body = Vec::new();
                    self.parse_block(&mut body, true)?;
                    for _ in 0..count {
                        out.extend(body.iter().cloned());
                    }
                }
                Some(Token::Word(_)) => {
                    if let Some(inst) = self.parse_instruction()? {
                        out.push(inst);
                    }
                }
                other => return Err(format!("unexpected token {:?}", other)),
            }
        }
    }

    fn parse_instruction(&mut self) -> Result<Option<Instruction>, String> {
        let name = match self.next() {
            Some(Token::Word(w)) => w.clone(),
            other => return Err(format!("expected instruction name, got {:?}", other)),
        };

        // Optional parenthesized args.
        let mut args = Vec::new();
        if matches!(self.peek(), Some(Token::LParen)) {
            self.next();
            loop {
                match self.next() {
                    Some(Token::Number(n)) => args.push(*n),
                    Some(Token::Comma) => {}
                    Some(Token::RParen) => break,
                    other => return Err(format!("bad argument list near {:?}", other)),
                }
            }
        }

        // Targets until newline.
        let mut targets = Vec::new();
        while let Some(tok) = self.peek() {
            match tok {
                Token::Newline => {
                    self.next();
                    break;
                }
                Token::Number(n) => {
                    let v = *n;
                    if v < 0.0 || v.fract() != 0.0 {
                        return Err(format!("unsupported target '{}' in '{}'", v, name));
                    }
                    targets.push(v as u32);
                    self.next();
                }
                Token::Word(w) => {
                    return Err(format!("unsupported target '{}' in '{}'", w, name));
                }
                _ => {
                    self.next();
                }
            }
        }

        if is_ignored_annotation(&name) {
            return Ok(None);
        }
        let gate = gate_from_name(&name)
            .ok_or_else(|| format!("unsupported gate '{}'", name))?;
        if gate.is_two_qubit() && targets.len() % 2 != 0 {
            return Err(format!("'{}' needs an even number of targets", name));
        }
        Ok(Some(Instruction { gate, args, targets }))
    }
}

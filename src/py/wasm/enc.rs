//! A WebAssembly binary encoder: GC types (one recursion group), functions,
//! imports, globals, memory, exception tags (legacy `try` / `catch`) and data.

// a general encoder: not every type and instruction is used by the backend
#![allow(dead_code)]

/// A heap type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Heap {
    Eq,
    Any,
    I31,
    Func,
    Extern,
    None,
    /// a defined type
    Idx(u32),
}

/// A value type; references are nullable unless `NonNull`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Val {
    I32,
    I64,
    F64,
    Ref(Heap),
    NonNull(Heap),
}

/// A field or array element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    Val(Val),
    I8,
}

/// A composite type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Comp {
    Func(Vec<Val>, Vec<Val>),
    /// fields: (type, mutable)
    Struct(Vec<(Storage, bool)>),
    Array(Storage, bool),
}

/// A type of the recursion group: open to subtypes (`open`) and / or a
/// subtype of `sup`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubType {
    pub comp: Comp,
    pub sup: Option<u32>,
    pub open: bool,
}

pub fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

pub fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        out.push(if done { b } else { b | 0x80 });
        if done {
            return;
        }
    }
}

fn name(out: &mut Vec<u8>, s: &str) {
    uleb(out, s.len() as u64);
    out.extend(s.as_bytes());
}

pub fn heap(out: &mut Vec<u8>, h: Heap) {
    match h {
        Heap::Eq => out.push(0x6d),
        Heap::Any => out.push(0x6e),
        Heap::I31 => out.push(0x6c),
        Heap::Func => out.push(0x70),
        Heap::Extern => out.push(0x6f),
        Heap::None => out.push(0x71),
        Heap::Idx(i) => sleb(out, i as i64),
    }
}

pub fn val(out: &mut Vec<u8>, v: Val) {
    match v {
        Val::I32 => out.push(0x7f),
        Val::I64 => out.push(0x7e),
        Val::F64 => out.push(0x7c),
        Val::Ref(h) => {
            out.push(0x63);
            heap(out, h)
        }
        Val::NonNull(h) => {
            out.push(0x64);
            heap(out, h)
        }
    }
}

fn storage(out: &mut Vec<u8>, s: Storage) {
    match s {
        Storage::Val(v) => val(out, v),
        Storage::I8 => out.push(0x78),
    }
}

fn comp(out: &mut Vec<u8>, c: &Comp) {
    match c {
        Comp::Func(ps, rs) => {
            out.push(0x60);
            uleb(out, ps.len() as u64);
            ps.iter().for_each(|p| val(out, *p));
            uleb(out, rs.len() as u64);
            rs.iter().for_each(|r| val(out, *r));
        }
        Comp::Struct(fs) => {
            out.push(0x5f);
            uleb(out, fs.len() as u64);
            for (s, m) in fs {
                storage(out, *s);
                out.push(*m as u8);
            }
        }
        Comp::Array(s, m) => {
            out.push(0x5e);
            storage(out, *s);
            out.push(*m as u8);
        }
    }
}

/// A block type.
#[derive(Clone, Copy, Debug)]
pub enum Block {
    Empty,
    Val(Val),
    /// a function type index (parameters and results)
    Type(u32),
}

/// Instructions of a function body.
#[derive(Clone, Debug, Default)]
pub struct Code(pub Vec<u8>);

macro_rules! simple {
    ($($name:ident = $op:expr;)*) => {
        $(pub fn $name(&mut self) -> &mut Self { self.0.push($op); self })*
    };
}
macro_rules! gc {
    ($($name:ident = $op:expr;)*) => {
        $(pub fn $name(&mut self, t: u32) -> &mut Self { self.0.push(0xfb); uleb(&mut self.0, $op); uleb(&mut self.0, t as u64); self })*
    };
}
macro_rules! mem {
    ($($name:ident = $op:expr, $align:expr;)*) => {
        $(pub fn $name(&mut self, offset: u32) -> &mut Self { self.0.push($op); uleb(&mut self.0, $align); uleb(&mut self.0, offset as u64); self })*
    };
}

impl Code {
    fn op_idx(&mut self, op: u8, i: u32) -> &mut Self {
        self.0.push(op);
        uleb(&mut self.0, i as u64);
        self
    }
    fn block_type(&mut self, b: Block) {
        match b {
            Block::Empty => self.0.push(0x40),
            Block::Val(v) => val(&mut self.0, v),
            Block::Type(t) => sleb(&mut self.0, t as i64),
        }
    }
    pub fn block(&mut self, b: Block) -> &mut Self {
        self.0.push(0x02);
        self.block_type(b);
        self
    }
    pub fn loop_(&mut self, b: Block) -> &mut Self {
        self.0.push(0x03);
        self.block_type(b);
        self
    }
    pub fn if_(&mut self, b: Block) -> &mut Self {
        self.0.push(0x04);
        self.block_type(b);
        self
    }
    /// legacy exception handling: `try` ... `catch` / `catch_all` ... `end`
    pub fn try_(&mut self, b: Block) -> &mut Self {
        self.0.push(0x06);
        self.block_type(b);
        self
    }
    pub fn catch(&mut self, tag: u32) -> &mut Self {
        self.op_idx(0x07, tag)
    }
    pub fn throw(&mut self, tag: u32) -> &mut Self {
        self.op_idx(0x08, tag)
    }
    pub fn rethrow(&mut self, depth: u32) -> &mut Self {
        self.op_idx(0x09, depth)
    }
    pub fn br(&mut self, depth: u32) -> &mut Self {
        self.op_idx(0x0c, depth)
    }
    pub fn br_if(&mut self, depth: u32) -> &mut Self {
        self.op_idx(0x0d, depth)
    }
    pub fn br_table(&mut self, targets: &[u32], default: u32) -> &mut Self {
        self.0.push(0x0e);
        uleb(&mut self.0, targets.len() as u64);
        for t in targets {
            uleb(&mut self.0, *t as u64);
        }
        uleb(&mut self.0, default as u64);
        self
    }
    pub fn call(&mut self, f: u32) -> &mut Self {
        self.op_idx(0x10, f)
    }
    pub fn call_ref(&mut self, t: u32) -> &mut Self {
        self.op_idx(0x14, t)
    }
    pub fn local_get(&mut self, i: u32) -> &mut Self {
        self.op_idx(0x20, i)
    }
    pub fn local_set(&mut self, i: u32) -> &mut Self {
        self.op_idx(0x21, i)
    }
    pub fn local_tee(&mut self, i: u32) -> &mut Self {
        self.op_idx(0x22, i)
    }
    pub fn global_get(&mut self, i: u32) -> &mut Self {
        self.op_idx(0x23, i)
    }
    pub fn global_set(&mut self, i: u32) -> &mut Self {
        self.op_idx(0x24, i)
    }
    pub fn i32_const(&mut self, v: i32) -> &mut Self {
        self.0.push(0x41);
        sleb(&mut self.0, v as i64);
        self
    }
    pub fn i64_const(&mut self, v: i64) -> &mut Self {
        self.0.push(0x42);
        sleb(&mut self.0, v);
        self
    }
    pub fn f64_const(&mut self, v: f64) -> &mut Self {
        self.0.push(0x44);
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn ref_null(&mut self, h: Heap) -> &mut Self {
        self.0.push(0xd0);
        heap(&mut self.0, h);
        self
    }
    pub fn ref_func(&mut self, f: u32) -> &mut Self {
        self.op_idx(0xd2, f)
    }
    pub fn ref_test(&mut self, h: Heap) -> &mut Self {
        self.0.extend([0xfb, 21]);
        heap(&mut self.0, h);
        self
    }
    /// `ref.cast` to a nullable reference
    pub fn ref_cast(&mut self, h: Heap) -> &mut Self {
        self.0.extend([0xfb, 23]);
        heap(&mut self.0, h);
        self
    }
    pub fn array_new_data(&mut self, t: u32, data: u32) -> &mut Self {
        self.0.push(0xfb);
        uleb(&mut self.0, 9);
        uleb(&mut self.0, t as u64);
        uleb(&mut self.0, data as u64);
        self
    }
    pub fn array_new_fixed(&mut self, t: u32, n: u32) -> &mut Self {
        self.0.push(0xfb);
        uleb(&mut self.0, 8);
        uleb(&mut self.0, t as u64);
        uleb(&mut self.0, n as u64);
        self
    }
    pub fn struct_get(&mut self, t: u32, f: u32) -> &mut Self {
        self.0.push(0xfb);
        uleb(&mut self.0, 2);
        uleb(&mut self.0, t as u64);
        uleb(&mut self.0, f as u64);
        self
    }
    pub fn struct_set(&mut self, t: u32, f: u32) -> &mut Self {
        self.0.push(0xfb);
        uleb(&mut self.0, 5);
        uleb(&mut self.0, t as u64);
        uleb(&mut self.0, f as u64);
        self
    }
    pub fn array_copy(&mut self, dst: u32, src: u32) -> &mut Self {
        self.0.push(0xfb);
        uleb(&mut self.0, 17);
        uleb(&mut self.0, dst as u64);
        uleb(&mut self.0, src as u64);
        self
    }
    pub fn array_len(&mut self) -> &mut Self {
        self.0.extend([0xfb, 15]);
        self
    }
    pub fn ref_i31(&mut self) -> &mut Self {
        self.0.extend([0xfb, 28]);
        self
    }
    pub fn i31_get_s(&mut self) -> &mut Self {
        self.0.extend([0xfb, 29]);
        self
    }
    pub fn memory_size(&mut self) -> &mut Self {
        self.0.extend([0x3f, 0x00]);
        self
    }
    pub fn memory_grow(&mut self) -> &mut Self {
        self.0.extend([0x40, 0x00]);
        self
    }
    gc! {
        struct_new = 0; struct_new_default = 1;
        array_new = 6; array_new_default = 7; array_get = 11; array_get_s = 12; array_get_u = 13; array_set = 14; array_fill = 16;
    }
    mem! {
        i32_load8_u = 0x2d, 0; i32_store8 = 0x3a, 0; f64_store = 0x39, 3; f64_load = 0x2b, 3; i64_store = 0x37, 3;
    }
    simple! {
        unreachable = 0x00; nop = 0x01; else_ = 0x05; end = 0x0b; return_ = 0x0f; drop = 0x1a; select = 0x1b;
        ref_is_null = 0xd1; ref_eq = 0xd3; ref_as_non_null = 0xd4;
        i32_eqz = 0x45; i32_eq = 0x46; i32_ne = 0x47; i32_lt_s = 0x48; i32_lt_u = 0x49; i32_gt_s = 0x4a; i32_gt_u = 0x4b; i32_le_s = 0x4c; i32_ge_s = 0x4e; i32_ge_u = 0x4f;
        i64_eqz = 0x50; i64_eq = 0x51; i64_ne = 0x52; i64_lt_s = 0x53; i64_lt_u = 0x54; i64_gt_s = 0x55; i64_le_s = 0x57; i64_ge_s = 0x59; i64_ge_u = 0x5a;
        f64_eq = 0x61; f64_ne = 0x62; f64_lt = 0x63; f64_gt = 0x64; f64_le = 0x65; f64_ge = 0x66;
        i32_clz = 0x67; i32_add = 0x6a; i32_sub = 0x6b; i32_mul = 0x6c; i32_div_s = 0x6d; i32_div_u = 0x6e; i32_rem_s = 0x6f; i32_rem_u = 0x70; i32_and = 0x71; i32_or = 0x72; i32_xor = 0x73; i32_shl = 0x74; i32_shr_s = 0x75; i32_shr_u = 0x76;
        i64_clz = 0x79; i64_ctz = 0x7a; i64_popcnt = 0x7b; i64_add = 0x7c; i64_sub = 0x7d; i64_mul = 0x7e; i64_div_s = 0x7f; i64_div_u = 0x80; i64_rem_s = 0x81; i64_rem_u = 0x82;
        i64_and = 0x83; i64_or = 0x84; i64_xor = 0x85; i64_shl = 0x86; i64_shr_s = 0x87; i64_shr_u = 0x88; i64_rotl = 0x89;
        f64_abs = 0x99; f64_neg = 0x9a; f64_ceil = 0x9b; f64_floor = 0x9c; f64_trunc = 0x9d; f64_nearest = 0x9e; f64_sqrt = 0x9f;
        f64_add = 0xa0; f64_sub = 0xa1; f64_mul = 0xa2; f64_div = 0xa3; f64_min = 0xa4; f64_max = 0xa5; f64_copysign = 0xa6;
        i32_wrap_i64 = 0xa7; i64_extend_i32_s = 0xac; i64_extend_i32_u = 0xad; i64_trunc_f64_s = 0xb0; f64_convert_i64_s = 0xb9; f64_convert_i32_s = 0xb7;
        i64_reinterpret_f64 = 0xbd; f64_reinterpret_i64 = 0xbf;
    }
}

/// A function: its type, extra locals and body (without the final `end`).
#[derive(Clone, Debug)]
pub struct Func {
    pub ty: u32,
    pub locals: Vec<Val>,
    pub code: Code,
}

/// A module under construction.
#[derive(Default)]
pub struct Module {
    pub types: Vec<SubType>,
    /// (module, name, type)
    pub imports: Vec<(String, String, u32)>,
    pub funcs: Vec<Func>,
    /// (type, mutable, init expression)
    pub globals: Vec<(Val, bool, Code)>,
    /// (name, function)
    pub exports: Vec<(String, u32)>,
    /// tag types (function types without results)
    pub tags: Vec<u32>,
    /// passive data segments
    pub data: Vec<Vec<u8>>,
    /// functions referenced by `ref.func`
    pub declared: Vec<u32>,
    /// pages of the exported memory "mem"
    pub memory: u32,
}

impl Module {
    /// The index of a type (equal types share one).
    pub fn ty(&mut self, t: SubType) -> u32 {
        if let Some(i) = self.types.iter().position(|x| *x == t) {
            return i as u32;
        }
        self.types.push(t);
        (self.types.len() - 1) as u32
    }
    /// A function type.
    pub fn func_ty(&mut self, params: Vec<Val>, results: Vec<Val>) -> u32 {
        self.ty(SubType { comp: Comp::Func(params, results), sup: None, open: false })
    }
    pub fn data(&mut self, bytes: &[u8]) -> u32 {
        if let Some(i) = self.data.iter().position(|d| d == bytes) {
            return i as u32;
        }
        self.data.push(bytes.to_vec());
        (self.data.len() - 1) as u32
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut out = b"\0asm\x01\0\0\0".to_vec();
        let mut section = |id: u8, body: Vec<u8>| {
            out.push(id);
            uleb(&mut out, body.len() as u64);
            out.extend(body);
        };
        // types: one recursion group
        let mut b = vec![1, 0x4e];
        uleb(&mut b, self.types.len() as u64);
        for t in &self.types {
            if t.sup.is_some() || t.open {
                b.push(if t.open { 0x50 } else { 0x4f });
                match t.sup {
                    Some(s) => {
                        b.push(1);
                        uleb(&mut b, s as u64);
                    }
                    None => b.push(0),
                }
            }
            comp(&mut b, &t.comp);
        }
        section(1, b);
        let mut b = vec![];
        uleb(&mut b, self.imports.len() as u64);
        for (m, n, t) in &self.imports {
            name(&mut b, m);
            name(&mut b, n);
            b.push(0);
            uleb(&mut b, *t as u64);
        }
        section(2, b);
        let mut b = vec![];
        uleb(&mut b, self.funcs.len() as u64);
        for f in &self.funcs {
            uleb(&mut b, f.ty as u64);
        }
        section(3, b);
        section(5, vec![1, 0, self.memory.max(1) as u8]);
        let mut b = vec![];
        uleb(&mut b, self.tags.len() as u64);
        for t in &self.tags {
            b.push(0);
            uleb(&mut b, *t as u64);
        }
        section(13, b);
        let mut b = vec![];
        uleb(&mut b, self.globals.len() as u64);
        for (v, m, init) in &self.globals {
            val(&mut b, *v);
            b.push(*m as u8);
            b.extend(&init.0);
            b.push(0x0b);
        }
        section(6, b);
        let mut b = vec![];
        uleb(&mut b, self.exports.len() as u64 + 1);
        for (n, f) in &self.exports {
            name(&mut b, n);
            b.push(0);
            uleb(&mut b, *f as u64);
        }
        name(&mut b, "mem");
        b.extend([2, 0]);
        section(7, b);
        if !self.declared.is_empty() {
            // a declarative element segment: functions that `ref.func` names
            let mut b = vec![1, 3, 0];
            uleb(&mut b, self.declared.len() as u64);
            for f in &self.declared {
                uleb(&mut b, *f as u64);
            }
            section(9, b);
        }
        let mut b = vec![];
        uleb(&mut b, self.data.len() as u64);
        section(12, b);
        let mut b = vec![];
        uleb(&mut b, self.funcs.len() as u64);
        for f in &self.funcs {
            let mut body = vec![];
            // locals, run-length encoded
            let mut runs: Vec<(u32, Val)> = vec![];
            for l in &f.locals {
                match runs.last_mut() {
                    Some((n, v)) if v == l => *n += 1,
                    _ => runs.push((1, *l)),
                }
            }
            uleb(&mut body, runs.len() as u64);
            for (n, v) in runs {
                uleb(&mut body, n as u64);
                val(&mut body, v);
            }
            body.extend(&f.code.0);
            body.push(0x0b);
            uleb(&mut b, body.len() as u64);
            b.extend(body);
        }
        section(10, b);
        let mut b = vec![];
        uleb(&mut b, self.data.len() as u64);
        for d in &self.data {
            b.push(1);
            uleb(&mut b, d.len() as u64);
            b.extend(d);
        }
        section(11, b);
        out
    }
}

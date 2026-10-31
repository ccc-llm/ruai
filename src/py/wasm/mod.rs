//! py0 → WebAssembly, without Rust: the typed IR compiled straight to a wasm
//! module that uses WasmGC (objects, lists, dicts and strings are GC structs and
//! arrays) and exception handling (`try` / `catch`). The module imports a few
//! functions from a JavaScript host ([`HOST_JS`]: output, float formatting and
//! parsing, math); it runs in browsers and in Node.js
//! (`node ruai_wasm.mjs prog.wasm`).
//!
//! The translator itself builds for `wasm32-unknown-unknown`, so py0 can be
//! compiled to wasm inside a browser ([`compile_source`]).
//!
//! Not every py0 program compiles yet: generators, closures over reassigned
//! variables, files, `re` and the Rust-backed packages report "the wasm
//! backend does not support ...".

mod builtins;
mod enc;
mod expr;
mod host;
mod rt;
mod show;
mod stmt;

use crate::py::ir::*;
use enc::{Block, Code, Comp, Heap, Module, Storage, SubType, Val};
use std::collections::HashMap;

/// The JavaScript host of compiled programs (an ES module): `run(bytes, io)`
/// in a browser, `node ruai_wasm.mjs prog.wasm [input]` on the command line.
pub const HOST_JS: &str = include_str!("host.mjs");

pub(crate) type Res<T> = Result<T, String>;

fn unsupported<T>(what: impl std::fmt::Display) -> Res<T> {
    Err(format!("the wasm backend does not support {} yet", what))
}

/// Compile py0 source to a wasm module (`Err`: the translator's message, or
/// what the wasm backend does not support).
pub fn compile_source(src: &str) -> Result<Vec<u8>, String> {
    let prog = crate::py::front::lower(src).map_err(|e| format!("line {}: {}", e.line, e.msg))?;
    crate::py::own::analyze(&prog).map_err(|e| format!("line {}: {}", e.line, e.msg))?;
    compile(&prog)
}

/// Compile a checked program.
pub fn compile(p: &Program) -> Res<Vec<u8>> {
    if !p.crates.is_empty() || !p.python_modules.is_empty() {
        return unsupported("Rust crates, packages implemented in Rust or Python modules");
    }
    let mut w = W::new(p);
    w.program()?;
    Ok(w.m.finish())
}

/// Element kinds of lists (and stored values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum K {
    I64,
    F64,
    I32,
    Ref,
}

/// Fixed types.
pub(crate) struct Types {
    pub bytes: u32,
    pub str: u32,
    pub arr: [u32; 4],
    pub list: [u32; 4],
    pub box_i64: u32,
    pub box_f64: u32,
    pub arr_i32: u32,
    pub exc: u32,
    pub dict: u32,
    pub hashfn: u32,
    pub eqfn: u32,
    pub sb: u32,
}

/// Host imports, in order (see host.mjs): a small call channel. Arguments
/// are pushed one by one, `call` / `fcall` run a named operation (its name is
/// the last string argument); strings come back through `pending` + `take`.
#[derive(Clone, Copy)]
pub(crate) enum H {
    /// write the argument string to fd
    Write,
    /// push a string argument (ptr, len)
    Arg,
    IntArg,
    FloatArg,
    /// run an operation → i64
    Call,
    /// run an operation → f64
    FCall,
    /// the length of the pending string
    Pending,
    /// copy the pending string to ptr
    Take,
    /// make item i of the last list result pending → its length
    Item,
    /// the error kind of the last operation (0: none; its message is pending)
    Failed,
}

/// A label of the block structure (for `br` depths).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Lab {
    Plain,
    Break(u32),
    Continue(u32),
}

/// The function being compiled.
#[derive(Default)]
pub(crate) struct Fx {
    pub code: Code,
    pub params: Vec<Val>,
    pub locals: Vec<Val>,
    pub vars: HashMap<VarId, u32>,
    pub labels: Vec<Lab>,
    /// loops: (id, number of `finally` blocks open when it started)
    pub loops: Vec<(u32, usize)>,
    /// `finally` bodies of the enclosing `try`s
    pub finals: Vec<Vec<Stmt>>,
    pub fid: Option<FuncId>,
    pub ret: Option<Val>,
    pub next_label: u32,
}

impl Fx {
    pub fn local(&mut self, v: Val) -> u32 {
        self.locals.push(v);
        (self.params.len() + self.locals.len() - 1) as u32
    }
    /// The `br` depth of a label.
    pub fn depth(&self, l: Lab) -> u32 {
        let i = self.labels.iter().rposition(|x| *x == l).expect("label");
        (self.labels.len() - 1 - i) as u32
    }
}

pub(crate) struct W<'a> {
    pub p: &'a Program,
    pub m: Module,
    pub t: Types,
    pub fx: Fx,
    /// function index of each FuncId
    pub user: Vec<u32>,
    /// struct type of each class
    pub class_ty: Vec<u32>,
    /// runtime functions by key
    pub rts: HashMap<String, u32>,
    /// globals holding string constants
    pub strs: HashMap<String, u32>,
    /// globals of module constants
    pub consts: Vec<Option<u32>>,
    /// exception class ids by name
    pub exc_ids: HashMap<String, i32>,
    pub tag: u32,
    pub exit_tag: u32,
    nimports: u32,
}

const IMPORTS: &[(&str, &[Val], &[Val])] = &[
    ("write", &[Val::I32], &[]),
    ("arg", &[Val::I32, Val::I32], &[]),
    ("int_arg", &[Val::I64], &[]),
    ("float_arg", &[Val::F64], &[]),
    ("call", &[], &[Val::I64]),
    ("fcall", &[], &[Val::F64]),
    ("pending", &[], &[Val::I32]),
    ("take", &[Val::I32], &[]),
    ("item", &[Val::I32], &[Val::I32]),
    ("failed", &[], &[Val::I32]),
];

/// Error kinds of host operations (host.mjs `KINDS`).
pub(crate) const HOST_ERRORS: &[&str] = &["", "ValueError", "TypeError", "OverflowError", "IndexError", "EOFError", "ZeroDivisionError", "KeyError"];

impl<'a> W<'a> {
    fn new(p: &'a Program) -> W<'a> {
        let mut m = Module::default();
        let st = |m: &mut Module, comp: Comp| m.ty(SubType { comp, sup: None, open: false });
        let bytes = st(&mut m, Comp::Array(Storage::I8, true));
        let str = st(&mut m, Comp::Struct(vec![(Storage::Val(Val::NonNull(Heap::Idx(bytes))), false), (Storage::Val(Val::I32), false)]));
        let arr = [Val::I64, Val::F64, Val::I32, Val::Ref(Heap::Eq)].map(|v| st(&mut m, Comp::Array(Storage::Val(v), true)));
        let list = arr.map(|a| st(&mut m, Comp::Struct(vec![(Storage::Val(Val::NonNull(Heap::Idx(a))), true), (Storage::Val(Val::I32), true)])));
        let box_i64 = st(&mut m, Comp::Struct(vec![(Storage::Val(Val::I64), false)]));
        let box_f64 = st(&mut m, Comp::Struct(vec![(Storage::Val(Val::F64), false)]));
        let arr_i32 = arr[2];
        let sref = Val::Ref(Heap::Idx(str));
        // exception: class name, chain of class ids (most specific first), message, object
        let exc = st(&mut m, Comp::Struct(vec![(Storage::Val(sref), false), (Storage::Val(Val::Ref(Heap::Idx(arr_i32))), false), (Storage::Val(sref), false), (Storage::Val(Val::Ref(Heap::Eq)), false)]));
        let hashfn = m.func_ty(vec![Val::Ref(Heap::Eq)], vec![Val::I64]);
        let eqfn = m.func_ty(vec![Val::Ref(Heap::Eq), Val::Ref(Heap::Eq)], vec![Val::I32]);
        let aeq = Val::NonNull(Heap::Idx(arr[3]));
        let dict = st(
            &mut m,
            Comp::Struct(vec![
                (Storage::Val(aeq), true),                               // keys
                (Storage::Val(aeq), true),                               // values
                (Storage::Val(Val::NonNull(Heap::Idx(arr[0]))), true), // hashes
                (Storage::Val(Val::NonNull(Heap::Idx(arr[2]))), true), // index: slot → entry + 1 (0: free)
                (Storage::Val(Val::I32), true),                          // entries used (with deleted ones)
                (Storage::Val(Val::I32), true),                          // live entries
                (Storage::Val(Val::NonNull(Heap::Idx(hashfn))), false),
                (Storage::Val(Val::NonNull(Heap::Idx(eqfn))), false),
            ]),
        );
        let sb = st(&mut m, Comp::Struct(vec![(Storage::Val(Val::NonNull(Heap::Idx(bytes))), true), (Storage::Val(Val::I32), true), (Storage::Val(Val::I32), true)]));
        for (n, ps, rs) in IMPORTS {
            let t = m.func_ty(ps.to_vec(), rs.to_vec());
            m.imports.push(("host".into(), n.to_string(), t));
        }
        let tag_ty = m.func_ty(vec![Val::Ref(Heap::Idx(exc))], vec![]);
        let exit_ty = m.func_ty(vec![Val::I32], vec![]);
        m.tags = vec![tag_ty, exit_ty];
        let nimports = m.imports.len() as u32;
        W {
            p,
            m,
            t: Types { bytes, str, arr, list, box_i64, box_f64, arr_i32, exc, dict, hashfn, eqfn, sb },
            fx: Fx::default(),
            user: vec![],
            class_ty: vec![],
            rts: HashMap::new(),
            strs: HashMap::new(),
            consts: vec![],
            exc_ids: HashMap::new(),
            tag: 0,
            exit_tag: 1,
            nimports,
        }
    }

    pub fn c(&mut self) -> &mut Code {
        &mut self.fx.code
    }

    pub fn host(&self, h: H) -> u32 {
        h as u32
    }

    // ------------------------------------------------------------ types

    /// The wasm type of a value of type `t` (`None`: no value, the type None).
    pub fn val(&mut self, t: &Ty) -> Res<Option<Val>> {
        Ok(Some(match t {
            Ty::Int => Val::I64,
            Ty::Float => Val::F64,
            Ty::Bool => Val::I32,
            Ty::None => return Ok(None),
            Ty::Str => Val::Ref(Heap::Idx(self.t.str)),
            Ty::List(e) => {
                let k = self.kind(e)?;
                Val::Ref(Heap::Idx(self.t.list[k as usize]))
            }
            Ty::Dict(..) | Ty::Set(_) => Val::Ref(Heap::Idx(self.t.dict)),
            Ty::Tuple(ts) => Val::Ref(Heap::Idx(self.tuple_ty(ts)?)),
            Ty::Class(c) | Ty::Obj(c) => {
                if self.p.classes[*c].ext.is_some() {
                    return unsupported(format!("the type `{}` of a package", self.p.classes[*c].name));
                }
                Val::Ref(Heap::Idx(self.class_ty[*c]))
            }
            Ty::Optional(i) => match self.val(i)? {
                Some(Val::I64) => Val::Ref(Heap::Idx(self.t.box_i64)),
                Some(Val::F64) => Val::Ref(Heap::Idx(self.t.box_f64)),
                Some(Val::I32) => Val::Ref(Heap::I31),
                Some(v) => v,
                None => Val::I32,
            },
            Ty::Exc => Val::Ref(Heap::Idx(self.t.exc)),
            Ty::Func(..) => return unsupported("functions as values"),
            Ty::Gen(_) => return unsupported("generators"),
            Ty::Deque(_) => return unsupported("collections.deque"),
            Ty::File => return unsupported("files"),
            Ty::Pattern(_) | Ty::Match(_) => return unsupported("the re module"),
            Ty::Var(_) => return unsupported("an uninferred type"),
        }))
    }

    /// The element kind of a list of `t`.
    pub fn kind(&mut self, t: &Ty) -> Res<K> {
        Ok(match self.val(t)? {
            Some(Val::I64) => K::I64,
            Some(Val::F64) => K::F64,
            Some(Val::I32) | None => K::I32,
            Some(_) => K::Ref,
        })
    }

    pub fn tuple_ty(&mut self, ts: &[Ty]) -> Res<u32> {
        let mut fs = vec![];
        for t in ts {
            let v = self.val(t)?.unwrap_or(Val::I32);
            fs.push((Storage::Val(v), false));
        }
        Ok(self.m.ty(SubType { comp: Comp::Struct(fs), sup: None, open: false }))
    }

    fn class_types(&mut self) -> Res<()> {
        // reserve indices first: fields may refer to any class
        for _ in &self.p.classes {
            self.m.types.push(SubType { comp: Comp::Struct(vec![(Storage::Val(Val::I64), false); 3]), sup: None, open: true });
            self.class_ty.push((self.m.types.len() - 1) as u32);
        }
        for (c, cl) in self.p.classes.iter().enumerate() {
            if cl.ext.is_some() {
                continue;
            }
            let mut fs = vec![(Storage::Val(Val::I32), false)];
            for f in &cl.fields {
                let v = self.val(&f.ty)?.unwrap_or(Val::I32);
                fs.push((Storage::Val(v), true));
            }
            let sup = cl.base.map(|b| self.class_ty[b]);
            self.m.types[self.class_ty[c] as usize] = SubType { comp: Comp::Struct(fs), sup, open: true };
        }
        Ok(())
    }

    /// The field index of `name` in class `c`'s struct.
    pub fn field_index(&self, c: ClassId, name: &str) -> Res<u32> {
        match self.p.classes[c].fields.iter().position(|f| f.name == name) {
            Some(i) => Ok(i as u32 + 1),
            None => unsupported(format!("the attribute `{}`", name)),
        }
    }

    /// The id of an exception class.
    pub fn exc_id(&mut self, name: &str) -> i32 {
        let n = self.exc_ids.len() as i32;
        *self.exc_ids.entry(name.to_string()).or_insert(n)
    }

    // ------------------------------------------------------------ functions

    /// Reserve a function index for a function to be defined.
    pub fn reserve(&mut self) -> u32 {
        let t = self.m.func_ty(vec![], vec![]);
        self.m.funcs.push(enc::Func { ty: t, locals: vec![], code: Code::default() });
        self.nimports + self.m.funcs.len() as u32 - 1
    }

    /// Define a reserved function from the current `fx`.
    pub fn define(&mut self, idx: u32, results: Vec<Val>) {
        let fx = std::mem::take(&mut self.fx);
        let ty = self.m.func_ty(fx.params.clone(), results);
        self.m.funcs[(idx - self.nimports) as usize] = enc::Func { ty, locals: fx.locals, code: fx.code };
    }

    /// A runtime function, generated on first use by `gen` (which fills `fx`
    /// with the parameters and code, and returns the results).
    pub fn rt_fn(&mut self, key: &str, gen: impl FnOnce(&mut W<'a>) -> Res<Vec<Val>>) -> Res<u32> {
        if let Some(&f) = self.rts.get(key) {
            return Ok(f);
        }
        let idx = self.reserve();
        self.rts.insert(key.to_string(), idx);
        let saved = std::mem::take(&mut self.fx);
        let r = gen(self);
        let results = match r {
            Ok(r) => r,
            Err(e) => {
                self.fx = saved;
                return Err(e);
            }
        };
        self.define(idx, results);
        self.fx = saved;
        Ok(idx)
    }

    fn program(&mut self) -> Res<()> {
        self.class_types()?;
        self.consts = vec![None; self.p.consts.len()];
        self.const_globals()?;
        let init = self.reserve();
        for _ in &self.p.funcs {
            let i = self.reserve();
            self.user.push(i);
        }
        for (fid, f) in self.p.funcs.iter().enumerate() {
            if f.ext.is_some() {
                return unsupported(format!("`{}` (implemented in Rust)", f.name));
            }
            self.func(fid)?;
        }
        let consts = self.const_code()?;
        // run: constants, then the module body; uncaught exceptions → traceback, exit code 1
        let main = self.user[self.p.main];
        let report = self.rt_report()?;
        self.fx = Fx::default();
        let e = self.fx.local(Val::Ref(Heap::Idx(self.t.exc)));
        let (tag, exit_tag) = (self.tag, self.exit_tag);
        self.c().try_(Block::Val(Val::I32)).call(init).call(main).i32_const(0);
        self.c().catch(tag).local_set(e).local_get(e).call(report).i32_const(1);
        self.c().catch(exit_tag).end();
        let run = self.reserve();
        self.define(run, vec![Val::I32]);
        self.m.exports.push(("run".into(), run));
        // last: string constants (all are known now), then module constants
        self.fx = Fx::default();
        let mut keys: Vec<(String, u32)> = self.strs.iter().map(|(k, v)| (k.clone(), *v)).collect();
        keys.sort_by_key(|k| k.1);
        for (s, g) in keys {
            self.str_new(&s);
            self.c().global_set(g);
        }
        self.fx.code.0.extend(consts.code.0);
        self.fx.locals = consts.locals;
        self.define(init, vec![]);
        Ok(())
    }

    fn func(&mut self, fid: FuncId) -> Res<()> {
        let f = &self.p.funcs[fid];
        if f.is_gen {
            return unsupported("generator functions");
        }
        if f.cache {
            return unsupported("@functools.cache");
        }
        if !f.dispatch.is_empty() {
            return self.dispatcher(fid);
        }
        if !f.captures.is_empty() && assigns_capture(f) {
            return unsupported("a nested function that rebinds a variable of the enclosing function (nonlocal)");
        }
        self.fx = Fx { fid: Some(fid), ..Default::default() };
        let mut params = vec![];
        if let Some(sv) = f.self_var {
            let v = self.val(&f.vars[sv].ty)?.unwrap();
            params.push(v);
            self.fx.vars.insert(sv, 0);
        }
        for q in &f.params {
            if Some(q.var) == f.self_var {
                continue;
            }
            if let Some(v) = self.val(&f.vars[q.var].ty)? {
                self.fx.vars.insert(q.var, params.len() as u32);
                params.push(v);
            }
        }
        self.fx.params = params;
        for (i, v) in f.vars.iter().enumerate() {
            if self.fx.vars.contains_key(&i) {
                continue;
            }
            if let Some(t) = self.val(&v.ty)? {
                let l = self.fx.local(t);
                self.fx.vars.insert(i, l);
            }
        }
        let ret = if f.is_init { None } else { self.val(&f.ret)? };
        self.fx.ret = ret;
        let body = f.body.clone();
        self.block(&body)?;
        // falling off the end returns None (a non-None function always returns before)
        if let Some(r) = ret {
            self.zero(r);
        }
        let idx = self.user[fid];
        self.define(idx, ret.into_iter().collect());
        Ok(())
    }

    /// A method of a class with subclasses: calls the implementation of the
    /// object's class.
    fn dispatcher(&mut self, fid: FuncId) -> Res<()> {
        let f = &self.p.funcs[fid];
        let disp = f.dispatch.clone();
        let base = f.class.unwrap();
        self.fx = Fx::default();
        let mut params = vec![Val::Ref(Heap::Idx(self.class_ty[base]))];
        for q in &f.params {
            if Some(q.var) == f.self_var {
                continue;
            }
            if let Some(v) = self.val(&f.vars[q.var].ty)? {
                params.push(v);
            }
        }
        let n = params.len() as u32;
        self.fx.params = params;
        let ret = self.val(&f.ret)?;
        let cty = self.class_ty[base];
        for (c, imp) in disp {
            // the implementation takes its own class
            let own = self.class_ty[self.p.funcs[imp].class.unwrap_or(c)];
            self.c().local_get(0).struct_get(cty, 0).i32_const(c as i32).i32_eq().if_(Block::Empty);
            self.c().local_get(0).ref_cast(Heap::Idx(own));
            for i in 1..n {
                self.c().local_get(i);
            }
            let fi = self.user[imp];
            self.c().call(fi).return_().end();
        }
        self.c().unreachable();
        let idx = self.user[fid];
        self.define(idx, ret.into_iter().collect());
        Ok(())
    }

    /// Push the zero value of a type.
    pub fn zero(&mut self, v: Val) {
        match v {
            Val::I32 => self.c().i32_const(0),
            Val::I64 => self.c().i64_const(0),
            Val::F64 => self.c().f64_const(0.0),
            Val::Ref(h) | Val::NonNull(h) => self.c().ref_null(h),
        };
    }

    /// Globals for the module constants.
    fn const_globals(&mut self) -> Res<()> {
        for i in 0..self.p.consts.len() {
            let t = self.p.consts[i].value.ty.clone();
            if let Some(v) = self.val(&t)? {
                let mut init = Code::default();
                match v {
                    Val::I32 => init.i32_const(0),
                    Val::I64 => init.i64_const(0),
                    Val::F64 => init.f64_const(0.0),
                    Val::Ref(h) | Val::NonNull(h) => init.ref_null(h),
                };
                self.m.globals.push((v, true, init));
                self.consts[i] = Some((self.m.globals.len() - 1) as u32);
            }
        }
        Ok(())
    }

    /// The code computing the module constants.
    fn const_code(&mut self) -> Res<Fx> {
        self.fx = Fx::default();
        for (i, c) in self.p.consts.iter().enumerate() {
            let e = c.value.clone();
            self.expr(&e)?;
            if let Some(g) = self.consts[i] {
                self.c().global_set(g);
            }
        }
        Ok(std::mem::take(&mut self.fx))
    }

    /// A string constant (a global set when the program starts).
    pub fn str_const(&mut self, s: &str) {
        let g = match self.strs.get(s) {
            Some(g) => *g,
            None => {
                let init = {
                    let mut c = Code::default();
                    c.ref_null(Heap::Idx(self.t.str));
                    c
                };
                self.m.globals.push((Val::Ref(Heap::Idx(self.t.str)), true, init));
                let g = (self.m.globals.len() - 1) as u32;
                self.strs.insert(s.to_string(), g);
                g
            }
        };
        self.c().global_get(g);
    }

    /// Make a new string from a literal.
    fn str_new(&mut self, s: &str) {
        let d = self.m.data(s.as_bytes());
        let (bytes, st) = (self.t.bytes, self.t.str);
        self.c().i32_const(0).i32_const(s.len() as i32).array_new_data(bytes, d);
        self.c().i32_const(s.chars().count() as i32).struct_new(st);
    }
}

/// Does a nested function assign one of its captured variables (its last parameters)?
fn assigns_capture(f: &Func) -> bool {
    let n = f.params.len();
    let caps: Vec<VarId> = f.params[n.saturating_sub(f.captures.len())..].iter().map(|q| q.var).collect();
    fn walk(b: &[Stmt], caps: &[VarId]) -> bool {
        b.iter().any(|s| {
            let direct = match &s.kind {
                StmtKind::Assign(t, _) => {
                    let mut vs = vec![];
                    t.vars(&mut vs);
                    vs.iter().any(|v| caps.contains(v))
                }
                StmtKind::AugVar(v, _, _) => caps.contains(v),
                StmtKind::AugAssign(e, _, _) => matches!(e.kind, ExprKind::Var(v) if caps.contains(&v)),
                StmtKind::For(t, ..) => {
                    let mut vs = vec![];
                    t.vars(&mut vs);
                    vs.iter().any(|v| caps.contains(v))
                }
                _ => false,
            };
            direct || s.blocks().iter().any(|b| walk(b, caps))
        })
    }
    walk(&f.body, &caps)
}

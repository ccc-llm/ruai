//! Builtin functions and methods of builtin types.

use super::enc::{Block, Heap, Val};
use super::{unsupported, Lab, Res, K, W};
use crate::py::ir::*;

/// Host math operations of one argument.
const MATH_OPS: &[&str] = &["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "log2", "log10", "expm1", "log1p", "cbrt", "degrees", "radians", "gamma", "lgamma", "erf", "erfc"];

impl<'a> W<'a> {
    pub fn builtin(&mut self, name: &str, a: &[Expr], e: &Expr) -> Res<()> {
        match name {
            "print" => {
                let sbn = self.rt_sb_new()?;
                let sba = self.rt_sb_add()?;
                let sbs = self.rt_sb_str()?;
                let out = self.rt_out()?;
                let sb = self.fx.local(Val::Ref(Heap::Idx(self.t.sb)));
                // Python evaluates every argument before printing any
                let mut vals = vec![];
                for x in &a[2..] {
                    self.expr(x)?;
                    let l = match self.val(&x.ty)? {
                        Some(v) => {
                            let l = self.fx.local(v);
                            self.c().local_set(l);
                            Some(l)
                        }
                        None => None,
                    };
                    vals.push(l);
                }
                self.c().i32_const(0).call(sbn).local_set(sb);
                for (i, (x, l)) in a[2..].iter().zip(vals).enumerate() {
                    if i > 0 {
                        self.c().local_get(sb);
                        self.expr(&a[0])?;
                        self.c().call(sba);
                    }
                    self.c().local_get(sb);
                    if let Some(l) = l {
                        self.c().local_get(l);
                    }
                    self.emit_str(&x.ty)?;
                    self.c().call(sba);
                }
                self.c().local_get(sb);
                self.expr(&a[1])?;
                self.c().call(sba).local_get(sb).call(sbs).i32_const(1).call(out);
            }
            "len" => {
                self.expr(&a[0])?;
                match &a[0].ty {
                    Ty::Str => {
                        let s = self.t.str;
                        self.c().struct_get(s, 1);
                    }
                    Ty::List(t) => {
                        let k = self.kind(t)?;
                        let l = self.t.list[k as usize];
                        self.c().struct_get(l, 1);
                    }
                    Ty::Dict(..) | Ty::Set(_) => {
                        let d = self.t.dict;
                        self.c().struct_get(d, 5);
                    }
                    Ty::Tuple(ts) => {
                        let n = ts.len();
                        self.c().drop().i32_const(n as i32);
                    }
                    Ty::Class(c) | Ty::Obj(c) => {
                        let Some(f) = self.p.classes[*c].method("__len__") else { return unsupported("len() of this object") };
                        let fi = self.user[f];
                        self.c().call(fi);
                        return Ok(());
                    }
                    t => return unsupported(format!("len() of {:?}", t)),
                }
                self.c().i64_extend_i32_s();
            }
            "truth" => self.cond(&a[0])?,
            "str" => self.str_of(&a[0])?,
            "repr" => {
                self.expr(&a[0])?;
                self.emit_repr(&a[0].ty)?;
            }
            "abs" => {
                self.expr(&a[0])?;
                match &a[0].ty {
                    Ty::Int => {
                        let (x, f) = (self.fx.local(Val::I64), self.rt_int_sub()?);
                        self.c().local_tee(x).i64_const(0).i64_lt_s().if_(Block::Val(Val::I64)).i64_const(0).local_get(x).call(f).else_().local_get(x).end();
                    }
                    Ty::Float => {
                        self.c().f64_abs();
                    }
                    t => return unsupported(format!("abs() of {:?}", t)),
                }
            }
            "min" | "max" => {
                let v = self.val(&e.ty)?.unwrap();
                let acc = self.fx.local(v);
                let x = self.fx.local(v);
                self.expr(&a[0])?;
                self.c().local_set(acc);
                for y in &a[1..] {
                    self.expr(y)?;
                    self.c().local_set(x);
                    // the first of equal values wins
                    self.c().local_get(x).local_get(acc).local_get(x).local_get(acc);
                    match (&e.ty, name) {
                        (Ty::Int, "min") => self.c().i64_lt_s(),
                        (Ty::Int, _) => self.c().i64_gt_s(),
                        (Ty::Float, "min") => self.c().f64_lt(),
                        (Ty::Float, _) => self.c().f64_gt(),
                        _ => return unsupported("min()/max() of these values"),
                    };
                    self.c().select().local_set(acc);
                }
                self.c().local_get(acc);
            }
            "min_iter" | "max_iter" | "min_or" | "max_or" => {
                let ExprKind::Iter(it) = &a[0].kind else { return unsupported(name) };
                let et = self.iter_ty(it)?;
                let key = if name.ends_with("iter") && matches!(a[1].kind, ExprKind::Lambda(..)) { Some(&a[1]) } else { None };
                self.collect(it, &et)?;
                let lv = self.val(&Ty::list(et.clone()))?.unwrap();
                let l = self.fx.local(lv);
                self.c().local_set(l);
                let k = self.kind(&et)?;
                let lt = self.t.list[k as usize];
                if name.ends_with("_or") {
                    let rv = self.val(&e.ty)?.unwrap();
                    self.c().local_get(l).struct_get(lt, 1).i32_eqz().if_(Block::Val(rv));
                    self.labeled(Lab::Plain, |w| {
                        w.expr(&a[1])?;
                        w.c().else_();
                        w.min_max_list(l, &et, None, name.starts_with("max"))
                    })?;
                    self.c().end();
                } else {
                    self.c().local_get(l).struct_get(lt, 1).i32_eqz().if_(Block::Empty);
                    self.labeled(Lab::Plain, |w| w.raise_msg("ValueError", &format!("{}() iterable argument is empty", &name[..3])))?;
                    self.c().end();
                    self.min_max_list(l, &et, key, name.starts_with("max"))?;
                }
            }
            "sum" => {
                let ExprKind::Iter(it) = &a[0].kind else { return unsupported("sum") };
                let et = self.iter_ty(it)?;
                self.collect(it, &et)?;
                let k = self.kind(&et)?;
                let lt = self.t.list[k as usize];
                let (l, i) = (self.fx.local(Val::Ref(Heap::Idx(lt))), self.fx.local(Val::I32));
                self.c().local_set(l).i32_const(0).local_set(i);
                let rv = self.val(&e.ty)?.unwrap();
                let acc = self.fx.local(rv);
                self.zero(rv);
                self.c().local_set(acc);
                let add = if e.ty == Ty::Int { Some(self.rt_int_add()?) } else { None };
                // floats: Neumaier's compensated sum, as CPython 3.12+
                let comp = self.fx.local(Val::F64);
                let (t, x) = (self.fx.local(Val::F64), self.fx.local(Val::F64));
                if add.is_none() {
                    self.c().f64_const(0.0).local_set(comp);
                }
                self.c().block(Block::Empty).loop_(Block::Empty);
                self.c().local_get(i).local_get(l).struct_get(lt, 1).i32_ge_s().br_if(1);
                self.c().local_get(acc).local_get(l).local_get(i).i64_extend_i32_s();
                self.list_get(&et)?;
                match (&et, &e.ty) {
                    (Ty::Bool, Ty::Int) => {
                        self.c().i64_extend_i32_u();
                    }
                    (Ty::Int, Ty::Float) => {
                        self.c().f64_convert_i64_s();
                    }
                    _ => {}
                }
                match add {
                    Some(f) => {
                        self.c().call(f).local_set(acc);
                    }
                    None => {
                        // t = acc + x; comp += |acc| >= |x| ? (acc - t) + x : (x - t) + acc
                        self.c().local_set(x).local_get(x).f64_add().local_set(t);
                        self.c().local_get(acc).local_get(t).f64_sub().local_get(x).f64_add();
                        self.c().local_get(x).local_get(t).f64_sub().local_get(acc).f64_add();
                        self.c().local_get(acc).f64_abs().local_get(x).f64_abs().f64_ge().select();
                        self.c().local_get(comp).f64_add().local_set(comp).local_get(t).local_set(acc);
                    }
                };
                self.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                self.c().end().end().local_get(acc);
                if add.is_none() {
                    // add the compensation when it is finite and not zero
                    self.c().local_get(comp).f64_add().local_get(acc);
                    self.c().local_get(comp).local_get(comp).f64_sub().f64_const(0.0).f64_eq().local_get(comp).f64_const(0.0).f64_ne().i32_and().select();
                }
            }
            "any" | "all" => {
                let ExprKind::Iter(it) = &a[0].kind else { return unsupported(name) };
                let et = self.iter_ty(it)?;
                let r = self.fx.local(Val::I32);
                let any = name == "any";
                self.c().i32_const(!any as i32).local_set(r);
                let cur = self.iter_init(it)?;
                self.c().block(Block::Empty).loop_(Block::Empty);
                self.fx.labels.extend([Lab::Plain, Lab::Plain]);
                let res = (|| -> Res<()> {
                    self.iter_next(&cur, 1)?;
                    self.truth(&et)?;
                    if !any {
                        self.c().i32_eqz();
                    }
                    self.c().if_(Block::Empty).i32_const(any as i32).local_set(r).br(2).end().br(0);
                    Ok(())
                })();
                self.fx.labels.truncate(self.fx.labels.len() - 2);
                res?;
                self.c().end().end().local_get(r);
            }
            "sorted" => {
                let ExprKind::Iter(it) = &a[0].kind else { return unsupported("sorted") };
                let et = self.iter_ty(it)?;
                self.collect(it, &et)?;
                // sorted gets a fresh list (collect may return the original)
                let k = self.kind(&et)?;
                let lt = self.t.list[k as usize];
                let l = self.fx.local(Val::Ref(Heap::Idx(lt)));
                let copy = self.rt_list_slice(k)?;
                self.c().i64_const(0).i32_const(0).i64_const(0).i32_const(0).i64_const(0).i32_const(0).call(copy).local_set(l);
                let rev = self.const_bool(&a[2])?;
                self.sort_list(l, &et, &a[1], rev)?;
                self.c().local_get(l);
            }
            "list" => {
                let ExprKind::Iter(it) = &a[0].kind else { return unsupported("list") };
                let et = self.iter_ty(it)?;
                self.collect(it, &et)?;
                if let Iter::Over(x) = it.as_ref() {
                    if matches!(x.ty, Ty::List(_)) {
                        // list(xs) is a copy
                        let k = self.kind(&et)?;
                        let copy = self.rt_list_slice(k)?;
                        self.c().i64_const(0).i32_const(0).i64_const(0).i32_const(0).i64_const(0).i32_const(0).call(copy);
                    }
                }
            }
            "set" => {
                let ExprKind::Iter(it) = &a[0].kind else { return unsupported("set") };
                let et = self.iter_ty(it)?;
                self.collect(it, &et)?;
                let k = self.kind(&et)?;
                let lt = self.t.list[k as usize];
                let (l, i) = (self.fx.local(Val::Ref(Heap::Idx(lt))), self.fx.local(Val::I32));
                self.c().local_set(l).i32_const(0).local_set(i);
                self.dict_new(&et)?;
                let d = self.fx.local(Val::Ref(Heap::Idx(self.t.dict)));
                self.c().local_set(d);
                let set = self.rt_dict_set(&et, &Ty::None)?;
                self.c().block(Block::Empty).loop_(Block::Empty);
                self.c().local_get(i).local_get(l).struct_get(lt, 1).i32_ge_s().br_if(1);
                self.c().local_get(d).local_get(l).local_get(i).i64_extend_i32_s();
                self.list_get(&et)?;
                self.box_eq(&et)?;
                self.c().ref_null(Heap::Eq).call(set).local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                self.c().end().end().local_get(d);
            }
            "int" => {
                self.expr(&a[0])?;
                match &a[0].ty {
                    Ty::Str => {
                        let f = self.rt_parse_int()?;
                        self.c().i32_const(10).call(f);
                    }
                    Ty::Float => {
                        let f = self.rt_float_int()?;
                        self.c().call(f);
                    }
                    Ty::Bool => {
                        self.c().i64_extend_i32_u();
                    }
                    Ty::Int => {}
                    t => return unsupported(format!("int() of {:?}", t)),
                }
            }
            "int_base" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                self.c().i32_wrap_i64();
                let f = self.rt_parse_int()?;
                self.c().call(f);
            }
            "float" => {
                self.expr(&a[0])?;
                match &a[0].ty {
                    Ty::Str => {
                        let f = self.rt_parse_float()?;
                        self.c().call(f);
                    }
                    Ty::Int => {
                        self.c().f64_convert_i64_s();
                    }
                    Ty::Float => {}
                    t => return unsupported(format!("float() of {:?}", t)),
                }
            }
            "round" => {
                // round(x) → int, half to even
                self.expr(&a[0])?;
                let f = self.rt_float_int()?;
                self.c().f64_nearest().call(f);
            }
            "round2" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                // via the host: correctly rounded like Python
                self.c().f64_convert_i64_s();
                self.math("round")?;
            }
            "round_int" => {
                // round(n, ndigits) of an int: to a multiple of 10^-ndigits, half to even
                self.expr(&a[0])?;
                if let Some(nd) = a.get(1) {
                    self.expr(nd)?;
                    let f = self.rt_round_int()?;
                    self.c().call(f);
                }
            }
            "ord" => {
                self.expr(&a[0])?;
                let f = self.rt_ord()?;
                self.c().call(f);
            }
            "chr" => {
                self.expr(&a[0])?;
                let f = self.rt_chr()?;
                self.c().call(f);
            }
            "divmod" => {
                let Ty::Tuple(ts) = &e.ty else { unreachable!() };
                let tt = self.tuple_ty(ts)?;
                let v = self.val(&a[0].ty)?.unwrap();
                let (x, y) = (self.fx.local(v), self.fx.local(v));
                self.expr(&a[0])?;
                self.c().local_set(x);
                self.expr(&a[1])?;
                self.c().local_set(y);
                let (fd, md) = if a[0].ty == Ty::Int { (self.rt_int_floordiv()?, self.rt_int_mod()?) } else { (self.rt_float_floordiv()?, self.rt_float_mod()?) };
                self.c().local_get(x).local_get(y).call(fd).local_get(x).local_get(y).call(md).struct_new(tt);
            }
            "powmod" => {
                for x in a {
                    self.expr(x)?;
                }
                let f = self.rt_powmod()?;
                self.c().call(f);
            }
            "sys.setrecursionlimit" => {
                // wasm engines have their own stack limit
                self.expr(&a[0])?;
                self.c().drop();
            }
            "is" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                self.c().ref_eq();
            }
            "index" => return unsupported("numpy-style indexing"),
            "unpack" => return unsupported("unpacking this value"),
            "input" => {
                if !a.is_empty() {
                    let out = self.rt_out()?;
                    self.expr(&a[0])?;
                    self.c().i32_const(1).call(out);
                }
                let f = self.rt_read_line()?;
                self.c().call(f);
            }
            n if n.starts_with("math.") => self.math_fn(&n[5..], a)?,
            n if n.starts_with("list.") => self.list_method(&n[5..], a, e)?,
            n if n.starts_with("dict.") => self.dict_method(&n[5..], a, e)?,
            n if n.starts_with("set.") => self.set_method(&n[4..], a)?,
            n if n.starts_with("str.") => self.str_method(&n[4..], a, e)?,
            "sys.exit" => {
                if let Some(x) = a.first() {
                    self.expr(x)?;
                    self.c().i32_wrap_i64();
                } else {
                    self.c().i32_const(0);
                }
                let t = self.exit_tag;
                self.c().throw(t);
            }
            n => return unsupported(format!("`{}`", n)),
        }
        Ok(())
    }

    fn const_bool(&mut self, e: &Expr) -> Res<bool> {
        match e.kind {
            ExprKind::Bool(b) => Ok(b),
            _ => unsupported("a reverse= that is not a constant"),
        }
    }

    /// Evaluate a one-parameter lambda's body for the value on the stack.
    pub fn apply_lambda(&mut self, f: &Expr, arg_ty: &Ty) -> Res<()> {
        let ExprKind::Lambda(ps, body) = &f.kind else { return unsupported("a key that is not a lambda") };
        if ps.len() != 1 {
            return unsupported("a key lambda of several parameters");
        }
        match self.fx.vars.get(&ps[0]) {
            Some(&l) => {
                self.c().local_set(l);
            }
            None => {
                if self.val(arg_ty)?.is_some() {
                    self.c().drop();
                }
            }
        }
        self.expr(body)
    }

    /// min / max of the non-empty list in local `l` (by `key`)
    fn min_max_list(&mut self, l: u32, et: &Ty, key: Option<&Expr>, max: bool) -> Res<()> {
        let k = self.kind(et)?;
        let lt = self.t.list[k as usize];
        let ev = self.val(et)?.unwrap();
        let kt = match key {
            Some(Expr { kind: ExprKind::Lambda(_, b), .. }) => b.ty.clone(),
            _ => et.clone(),
        };
        let kv = self.val(&kt)?.unwrap();
        let cmp = self.cmp_fn(&kt)?;
        let (best, bk, x, xk, i) = (self.fx.local(ev), self.fx.local(kv), self.fx.local(ev), self.fx.local(kv), self.fx.local(Val::I32));
        let key_of = |w: &mut W<'a>, src: u32| -> Res<()> {
            w.c().local_get(src);
            if let Some(f) = key {
                w.apply_lambda(f, et)?;
            }
            Ok(())
        };
        self.c().local_get(l).i64_const(0);
        self.list_get(et)?;
        self.c().local_set(best);
        key_of(self, best)?;
        self.c().local_set(bk).i32_const(1).local_set(i);
        self.c().block(Block::Empty).loop_(Block::Empty);
        self.fx.labels.extend([Lab::Plain, Lab::Plain]);
        let r = (|| -> Res<()> {
            self.c().local_get(i).local_get(l).struct_get(lt, 1).i32_ge_s().br_if(1);
            self.c().local_get(l).local_get(i).i64_extend_i32_s();
            self.list_get(et)?;
            self.c().local_set(x);
            key_of(self, x)?;
            self.c().local_tee(xk).local_get(bk).call(cmp).i32_const(0);
            if max {
                self.c().i32_gt_s();
            } else {
                self.c().i32_lt_s();
            }
            self.c().if_(Block::Empty).local_get(x).local_set(best).local_get(xk).local_set(bk).end();
            self.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            Ok(())
        })();
        self.fx.labels.truncate(self.fx.labels.len() - 2);
        r?;
        self.c().end().end().local_get(best);
        Ok(())
    }

    /// Sort the list in local `l` in place (stable; by `key` unless it is None).
    fn sort_list(&mut self, l: u32, et: &Ty, key: &Expr, rev: bool) -> Res<()> {
        let k = self.kind(et)?;
        let lt = self.t.list[k as usize];
        let kt = match &key.kind {
            ExprKind::Lambda(_, b) => b.ty.clone(),
            _ => et.clone(),
        };
        let kk = self.kind(&kt)?;
        // the keys, as a list
        let new = self.rt_list_new(kk)?;
        let app = self.rt_list_append(kk)?;
        let (keys, i) = (self.fx.local(Val::Ref(Heap::Idx(self.t.list[kk as usize]))), self.fx.local(Val::I32));
        let lambda = matches!(key.kind, ExprKind::Lambda(..));
        self.c().i32_const(0).local_set(i);
        if lambda {
            self.c().local_get(l).struct_get(lt, 1).call(new).local_set(keys);
            self.c().block(Block::Empty).loop_(Block::Empty);
            self.fx.labels.extend([Lab::Plain, Lab::Plain]);
            let r = (|| -> Res<()> {
                self.c().local_get(i).local_get(l).struct_get(lt, 1).i32_ge_s().br_if(1);
                self.c().local_get(keys).local_get(l).local_get(i).i64_extend_i32_s();
                self.list_get(et)?;
                self.apply_lambda(key, et)?;
                self.elem_in(&kt)?;
                self.c().call(app).local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                Ok(())
            })();
            self.fx.labels.truncate(self.fx.labels.len() - 2);
            r?;
            self.c().end().end();
        } else {
            self.c().local_get(l).local_set(keys);
        }
        let perm = self.rt_sort_perm(&kt, rev)?;
        let apply = self.rt_apply_perm(k)?;
        self.c().local_get(l).local_get(keys).call(perm).call(apply);
        Ok(())
    }

    // ------------------------------------------------------------ methods

    fn list_method(&mut self, m: &str, a: &[Expr], e: &Expr) -> Res<()> {
        let Ty::List(et) = &a[0].ty else { return unsupported("this list method") };
        let et = et.as_ref().clone();
        let k = self.kind(&et)?;
        let lt = self.t.list[k as usize];
        match m {
            "append" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                self.elem_in(&et)?;
                let f = self.rt_list_append(k)?;
                self.c().call(f);
            }
            "extend" => {
                self.expr(&a[0])?;
                match &a[1].kind {
                    ExprKind::Iter(it) => self.collect(it, &et)?,
                    _ => self.expr(&a[1])?,
                }
                let f = self.rt_list_extend(&et)?;
                self.c().call(f);
            }
            "insert" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                self.expr(&a[2])?;
                self.elem_in(&et)?;
                let f = self.rt_list_insert(k)?;
                self.c().call(f);
            }
            "pop" | "pop_at" => {
                self.expr(&a[0])?;
                if m == "pop" {
                    self.c().i64_const(-1).i32_const(1);
                } else {
                    self.expr(&a[1])?;
                    self.c().i32_const(0);
                }
                let f = self.rt_list_pop(k)?;
                self.c().call(f);
                if let Some(Val::Ref(h)) = self.val(&et)? {
                    self.c().ref_cast(h);
                }
            }
            "index" | "count" | "remove" => {
                let l = self.fx.local(Val::Ref(Heap::Idx(lt)));
                self.expr(&a[0])?;
                self.c().local_set(l);
                let xv = self.val(&et)?.unwrap_or(Val::I32);
                let x = self.fx.local(xv);
                self.expr(&a[1])?;
                self.c().local_set(x);
                let find = self.rt_list_index(&et)?;
                match m {
                    "index" => {
                        let r = self.fx.local(Val::I64);
                        self.c().local_get(l).local_get(x).i32_const(0).call(find).local_tee(r).i64_const(0).i64_lt_s().if_(Block::Empty);
                        self.labeled(Lab::Plain, |w| {
                            w.c().local_get(x);
                            w.emit_repr(&et)?;
                            w.str_const(" is not in list");
                            let cat = w.rt_str_concat()?;
                            w.c().call(cat);
                            w.throw_kind("ValueError");
                            Ok(())
                        })?;
                        self.c().end().local_get(r);
                    }
                    "count" => {
                        let (n, i) = (self.fx.local(Val::I64), self.fx.local(Val::I64));
                        self.c().i64_const(0).local_set(n).i64_const(0).local_set(i);
                        self.c().block(Block::Empty).loop_(Block::Empty);
                        self.c().local_get(l).local_get(x).local_get(i).i32_wrap_i64().call(find).local_tee(i).i64_const(0).i64_lt_s().br_if(1);
                        self.c().local_get(n).i64_const(1).i64_add().local_set(n).local_get(i).i64_const(1).i64_add().local_set(i).br(0);
                        self.c().end().end().local_get(n);
                    }
                    _ => {
                        let r = self.fx.local(Val::I64);
                        self.c().local_get(l).local_get(x).i32_const(0).call(find).local_tee(r).i64_const(0).i64_lt_s().if_(Block::Empty);
                        self.labeled(Lab::Plain, |w| {
                            w.throw_lit("ValueError", "list.remove(x): x not in list");
                            Ok(())
                        })?;
                        self.c().end();
                        let del = self.rt_list_del(&et)?;
                        self.c().local_get(l).local_get(r).call(del);
                    }
                }
            }
            "reverse" => {
                self.expr(&a[0])?;
                let f = self.rt_list_reverse(k)?;
                self.c().call(f);
            }
            "clear" => {
                self.expr(&a[0])?;
                self.c().i32_const(0).struct_set(lt, 1);
            }
            "copy" => {
                self.expr(&a[0])?;
                let copy = self.rt_list_slice(k)?;
                self.c().i64_const(0).i32_const(0).i64_const(0).i32_const(0).i64_const(0).i32_const(0).call(copy);
            }
            "swap" => {
                // xs[i], xs[j] = xs[j], xs[i]
                let ev = self.val(&et)?.unwrap_or(Val::I32);
                let (l, i, j, t) = (self.fx.local(Val::Ref(Heap::Idx(lt))), self.fx.local(Val::I64), self.fx.local(Val::I64), self.fx.local(ev));
                self.expr(&a[0])?;
                self.c().local_set(l);
                self.expr(&a[1])?;
                self.c().local_set(i);
                self.expr(&a[2])?;
                self.c().local_set(j);
                let set = self.rt_list_set(&et)?;
                self.c().local_get(l).local_get(i);
                self.list_get(&et)?;
                self.c().local_set(t).local_get(l).local_get(i).local_get(l).local_get(j);
                self.list_get(&et)?;
                self.c().call(set).local_get(l).local_get(j).local_get(t).call(set);
            }
            "sort" => {
                let l = self.fx.local(Val::Ref(Heap::Idx(lt)));
                self.expr(&a[0])?;
                self.c().local_set(l);
                let rev = self.const_bool(&a[2])?;
                self.sort_list(l, &et, &a[1], rev)?;
            }
            _ => return unsupported(format!("list.{}", m)),
        }
        let _ = e;
        Ok(())
    }

    fn dict_method(&mut self, m: &str, a: &[Expr], e: &Expr) -> Res<()> {
        let Ty::Dict(kt, vt, _) = &a[0].ty else { return unsupported("this dict method") };
        let (kt, vt) = (kt.as_ref().clone(), vt.as_ref().clone());
        let d = self.t.dict;
        let ae = self.t.arr[3];
        let dl = self.fx.local(Val::Ref(Heap::Idx(d)));
        self.expr(&a[0])?;
        self.c().local_set(dl);
        match m {
            "get" | "get_opt" | "pop" | "pop_default" | "setdefault" => {
                let find = self.rt_dict_find()?;
                let ent = self.fx.local(Val::I32);
                let key = self.fx.local(Val::Ref(Heap::Eq));
                self.expr(&a[1])?;
                self.box_eq(&kt)?;
                self.c().local_set(key).local_get(dl).local_get(key).call(find).local_set(ent);
                let rv = self.val(&e.ty)?;
                self.c().local_get(ent).i32_const(0).i32_ge_s().if_(rv.map(Block::Val).unwrap_or(Block::Empty));
                self.labeled(Lab::Plain, |w| {
                    // present
                    w.c().local_get(dl).struct_get(d, 1).local_get(ent).array_get(ae);
                    w.unbox_eq(&vt)?;
                    if m == "get_opt" {
                        w.box_opt(&vt)?;
                    }
                    if m.starts_with("pop") {
                        let del = w.rt_dict_del(&kt, &vt)?;
                        w.c().local_get(dl).local_get(key).call(del);
                    }
                    w.c().else_();
                    match m {
                        "get" | "pop_default" => w.expr(&a[2])?,
                        "get_opt" => {
                            let v = w.val(&e.ty)?.unwrap();
                            w.zero(v);
                        }
                        "setdefault" => {
                            let set = w.rt_dict_set(&kt, &vt)?;
                            let vv = w.val(&vt)?.unwrap_or(Val::I32);
                            let v = w.fx.local(vv);
                            w.expr(&a[2])?;
                            w.c().local_tee(v);
                            w.box_eq(&vt)?;
                            let t = w.fx.local(Val::Ref(Heap::Eq));
                            w.c().local_set(t).local_get(dl).local_get(key).local_get(t).call(set).local_get(v);
                        }
                        _ => {
                            w.c().local_get(key);
                            w.unbox_eq(&kt)?;
                            w.emit_repr(&kt)?;
                            w.throw_kind("KeyError");
                            if let Some(v) = rv {
                                w.zero(v);
                            }
                        }
                    }
                    Ok(())
                })?;
                self.c().end();
            }
            "copy" => {
                // a new dict with the same entries
                self.dict_new(&kt)?;
                let nd = self.fx.local(Val::Ref(Heap::Idx(d)));
                self.c().local_set(nd);
                self.dict_copy_into(nd, dl, &kt, &vt)?;
                self.c().local_get(nd);
            }
            "update" => {
                let o = self.fx.local(Val::Ref(Heap::Idx(d)));
                self.expr(&a[1])?;
                self.c().local_set(o);
                self.dict_copy_into(dl, o, &kt, &vt)?;
            }
            "clear" => {
                let f = self.rt_dict_new()?;
                let _ = f;
                let a32 = self.t.arr[2];
                self.c().local_get(dl).i32_const(0).struct_set(d, 4).local_get(dl).i32_const(0).struct_set(d, 5);
                self.c().local_get(dl).local_get(dl).struct_get(d, 3).array_len().array_new_default(a32).struct_set(d, 3);
            }
            _ => return unsupported(format!("dict.{}", m)),
        }
        Ok(())
    }

    /// Copy the entries of the dict in local `src` into the dict in local `dst`.
    fn dict_copy_into(&mut self, dst: u32, src: u32, kt: &Ty, vt: &Ty) -> Res<()> {
        let (d, ae) = (self.t.dict, self.t.arr[3]);
        let next = self.rt_dict_next()?;
        let set = self.rt_dict_set(kt, vt)?;
        let i = self.fx.local(Val::I32);
        self.c().i32_const(0).local_set(i);
        self.c().block(Block::Empty).loop_(Block::Empty);
        self.c().local_get(src).local_get(i).call(next).local_tee(i).i32_const(0).i32_lt_s().br_if(1);
        self.c().local_get(dst).local_get(src).struct_get(d, 0).local_get(i).array_get(ae).local_get(src).struct_get(d, 1).local_get(i).array_get(ae).call(set);
        self.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
        self.c().end().end();
        Ok(())
    }

    fn set_method(&mut self, m: &str, a: &[Expr]) -> Res<()> {
        let Ty::Set(et) = &a[0].ty else { return unsupported("this set method") };
        let et = et.as_ref().clone();
        let d = self.t.dict;
        match m {
            "add" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                self.box_eq(&et)?;
                self.c().ref_null(Heap::Eq);
                let f = self.rt_dict_set(&et, &Ty::None)?;
                self.c().call(f);
            }
            "remove" => {
                self.expr(&a[0])?;
                self.expr(&a[1])?;
                self.box_eq(&et)?;
                let f = self.rt_dict_del(&et, &Ty::None)?;
                self.c().call(f);
            }
            "discard" => {
                let (s, x) = (self.fx.local(Val::Ref(Heap::Idx(d))), self.fx.local(Val::Ref(Heap::Eq)));
                self.expr(&a[0])?;
                self.c().local_set(s);
                self.expr(&a[1])?;
                self.box_eq(&et)?;
                self.c().local_set(x);
                let find = self.rt_dict_find()?;
                let del = self.rt_dict_del(&et, &Ty::None)?;
                self.c().local_get(s).local_get(x).call(find).i32_const(0).i32_ge_s().if_(Block::Empty).local_get(s).local_get(x).call(del).end();
            }
            "copy" => {
                let s = self.fx.local(Val::Ref(Heap::Idx(d)));
                self.expr(&a[0])?;
                self.c().local_set(s);
                self.dict_new(&et)?;
                let nd = self.fx.local(Val::Ref(Heap::Idx(d)));
                self.c().local_set(nd);
                self.dict_copy_into(nd, s, &et, &Ty::None)?;
                self.c().local_get(nd);
            }
            "clear" => {
                let a32 = self.t.arr[2];
                let s = self.fx.local(Val::Ref(Heap::Idx(d)));
                self.expr(&a[0])?;
                self.c().local_tee(s).i32_const(0).struct_set(d, 4).local_get(s).i32_const(0).struct_set(d, 5);
                self.c().local_get(s).local_get(s).struct_get(d, 3).array_len().array_new_default(a32).struct_set(d, 3);
            }
            _ => return unsupported(format!("set.{}", m)),
        }
        Ok(())
    }

    fn str_method(&mut self, m: &str, a: &[Expr], e: &Expr) -> Res<()> {
        match m {
            "join" => {
                // sep.join(strs)
                let lt = self.t.list[K::Ref as usize];
                let l = self.fx.local(Val::Ref(Heap::Idx(lt)));
                match &a[1].kind {
                    ExprKind::Iter(it) => self.collect(it, &Ty::Str)?,
                    _ => self.expr(&a[1])?,
                }
                self.c().local_set(l);
                self.expr(&a[0])?;
                let f = self.rt_str_join()?;
                self.c().local_get(l).call(f);
            }
            "find" | "index" if a.len() == 2 => {
                for x in a {
                    self.expr(x)?;
                }
                let f = self.rt_str_find()?;
                self.c().i64_const(0).call(f);
                if m == "index" {
                    let r = self.fx.local(Val::I64);
                    self.c().local_tee(r).i64_const(0).i64_lt_s().if_(Block::Empty);
                    self.labeled(Lab::Plain, |w| {
                        w.throw_lit("ValueError", "substring not found");
                        Ok(())
                    })?;
                    self.c().end().local_get(r);
                }
            }
            _ => {
                // the host's implementation of the method: arguments are all
                // computed first (they may call the host themselves)
                let mut ls = vec![];
                for x in a {
                    self.expr(x)?;
                    let v = self.val(&x.ty)?.unwrap();
                    let l = self.fx.local(v);
                    self.c().local_set(l);
                    ls.push(l);
                }
                for (x, l) in a.iter().zip(ls) {
                    self.c().local_get(l);
                    self.host_arg(&x.ty)?;
                }
                self.host_call(&format!("str.{}", m), &e.ty)?;
            }
        }
        let _ = e;
        Ok(())
    }

    fn math_fn(&mut self, n: &str, a: &[Expr]) -> Res<()> {
        if matches!(n, "gcd" | "isqrt" | "lcm") {
            for x in a {
                self.expr(x)?;
            }
            let f = match n {
                "gcd" => self.rt_gcd()?,
                "lcm" => self.rt_lcm()?,
                _ => self.rt_isqrt()?,
            };
            self.c().call(f);
            return Ok(());
        }
        for x in a {
            self.expr(x)?;
            if x.ty == Ty::Int {
                self.c().f64_convert_i64_s();
            }
        }
        match n {
            "pi" => {
                self.c().f64_const(std::f64::consts::PI);
            }
            "e" => {
                self.c().f64_const(std::f64::consts::E);
            }
            "tau" => {
                self.c().f64_const(std::f64::consts::TAU);
            }
            "inf" => {
                self.c().f64_const(f64::INFINITY);
            }
            "nan" => {
                self.c().f64_const(f64::NAN);
            }
            "fabs" => {
                self.c().f64_abs();
            }
            "sqrt" => {
                let x = self.fx.local(Val::F64);
                self.c().local_tee(x).f64_const(0.0).f64_lt().if_(Block::Empty);
                self.labeled(Lab::Plain, |w| w.raise_msg("ValueError", "math domain error"))?;
                self.c().end().local_get(x).f64_sqrt();
            }
            "floor" | "ceil" | "trunc" => {
                match n {
                    "floor" => self.c().f64_floor(),
                    "ceil" => self.c().f64_ceil(),
                    _ => self.c().f64_trunc(),
                };
                let f = self.rt_float_int()?;
                self.c().call(f);
            }
            "isnan" => {
                let x = self.fx.local(Val::F64);
                self.c().local_tee(x).local_get(x).f64_ne();
            }
            "isinf" => {
                self.c().f64_abs().f64_const(f64::INFINITY).f64_eq();
            }
            "isfinite" => {
                let x = self.fx.local(Val::F64);
                self.c().local_tee(x).local_get(x).f64_sub().f64_const(0.0).f64_eq();
            }
            "copysign" => {
                self.c().f64_copysign();
            }
            "log2args" => self.math("log_base")?,
            "atan2" | "hypot" | "pow" => self.math(n)?,
            _ if MATH_OPS.contains(&n) => {
                // the host raises ValueError("math domain error") as Python
                self.c().f64_const(0.0);
                self.math(n)?;
            }
            _ => return unsupported(format!("math.{}", n)),
        }
        Ok(())
    }
}

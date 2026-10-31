//! Functions per type: `str` / `repr`, `==`, ordering and hashing, and
//! f-strings.

use super::enc::{Block, Heap, Val};
use super::{unsupported, Res, K, W};
use crate::py::ir::*;

impl<'a> W<'a> {
    fn param_of(&mut self, t: &Ty) -> Res<Val> {
        Ok(self.val(t)?.unwrap_or(Val::I32))
    }

    /// The value on the stack (of type `t`) → `str(value)`.
    pub fn emit_str(&mut self, t: &Ty) -> Res<()> {
        match t {
            Ty::Str => Ok(()),
            Ty::None => {
                self.str_const("None");
                Ok(())
            }
            _ => {
                let f = self.str_fn(t)?;
                self.c().call(f);
                Ok(())
            }
        }
    }

    /// The value on the stack → `repr(value)`.
    pub fn emit_repr(&mut self, t: &Ty) -> Res<()> {
        if *t == Ty::None {
            self.str_const("None");
            return Ok(());
        }
        let f = self.repr_fn(t)?;
        self.c().call(f);
        Ok(())
    }

    /// Push `str(e)`.
    pub fn str_of(&mut self, e: &Expr) -> Res<()> {
        self.expr(e)?;
        self.emit_str(&e.ty)
    }

    pub fn str_fn(&mut self, t: &Ty) -> Res<u32> {
        self.show_fn(t, false)
    }

    pub fn repr_fn(&mut self, t: &Ty) -> Res<u32> {
        if *t == Ty::None {
            return self.rt_fn("repr_none", |w| {
                w.fx.params = vec![Val::I32];
                w.str_const("None");
                Ok(vec![Val::Ref(Heap::Idx(w.t.str))])
            });
        }
        self.show_fn(t, true)
    }

    fn show_fn(&mut self, t: &Ty, repr: bool) -> Res<u32> {
        // shared work first (functions this one calls)
        match t {
            Ty::Int => return self.rt_int_str(),
            Ty::Float => return self.rt_float_str(),
            Ty::Str if repr => return self.rt_str_repr(),
            _ => {}
        }
        let sref = Val::Ref(Heap::Idx(self.t.str));
        let pv = self.param_of(t)?;
        let key = format!("{}:{:?}", if repr { "repr" } else { "str" }, t);
        let t = t.clone();
        let sbn = self.rt_sb_new()?;
        let sba = self.rt_sb_add()?;
        let sbs = self.rt_sb_str()?;
        self.rt_fn(&key, move |w| {
            w.fx.params = vec![pv];
            let sb_t = w.t.sb;
            let lit = |w: &mut W, sb: u32, s: &str| {
                w.c().local_get(sb);
                w.str_const(s);
                w.c().call(sba);
            };
            match &t {
                Ty::Bool => {
                    let s = w.t.str;
                    w.c().local_get(0).if_(Block::Val(Val::Ref(Heap::Idx(s))));
                    w.str_const("True");
                    w.c().else_();
                    w.str_const("False");
                    w.c().end();
                }
                Ty::Str => {
                    w.c().local_get(0);
                }
                Ty::List(e) => {
                    let e = e.as_ref().clone();
                    let k = w.kind(&e)?;
                    let l = w.t.list[k as usize];
                    let (sb, i) = (w.fx.local(Val::Ref(Heap::Idx(sb_t))), w.fx.local(Val::I32));
                    w.c().i32_const(0).call(sbn).local_set(sb);
                    lit(w, sb, "[");
                    w.c().block(Block::Empty).loop_(Block::Empty);
                    w.c().local_get(i).local_get(0).struct_get(l, 1).i32_ge_s().br_if(1);
                    w.c().local_get(i).if_(Block::Empty);
                    lit(w, sb, ", ");
                    w.c().end();
                    w.c().local_get(sb).local_get(0).local_get(i).i64_extend_i32_s();
                    w.list_get(&e)?;
                    w.emit_repr(&e)?;
                    w.c().call(sba).local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                    w.c().end().end();
                    lit(w, sb, "]");
                    w.c().local_get(sb).call(sbs);
                }
                Ty::Tuple(ts) => {
                    let tt = w.tuple_ty(ts)?;
                    let sb = w.fx.local(Val::Ref(Heap::Idx(sb_t)));
                    w.c().i32_const(0).call(sbn).local_set(sb);
                    lit(w, sb, "(");
                    for (i, et) in ts.iter().enumerate() {
                        if i > 0 {
                            lit(w, sb, ", ");
                        }
                        w.c().local_get(sb).local_get(0).struct_get(tt, i as u32);
                        if w.val(et)?.is_none() {
                            w.c().drop();
                        }
                        w.emit_repr(et)?;
                        w.c().call(sba);
                    }
                    lit(w, sb, if ts.len() == 1 { ",)" } else { ")" });
                    w.c().local_get(sb).call(sbs);
                }
                Ty::Dict(kt, _, _) | Ty::Set(kt) if !matches!(t, Ty::Dict(_, _, DK::Counter)) => {
                    let set = matches!(t, Ty::Set(_));
                    let (kt, vt) = (kt.as_ref().clone(), if set { Ty::None } else { match &t { Ty::Dict(_, v, _) => v.as_ref().clone(), _ => unreachable!() } });
                    let default = matches!(t, Ty::Dict(_, _, DK::Default));
                    let (d, ae) = (w.t.dict, w.t.arr[3]);
                    let (sb, i, first) = (w.fx.local(Val::Ref(Heap::Idx(sb_t))), w.fx.local(Val::I32), w.fx.local(Val::I32));
                    if set {
                        w.c().local_get(0).struct_get(d, 5).i32_eqz().if_(Block::Empty);
                        w.str_const("set()");
                        w.c().return_().end();
                    }
                    w.c().i32_const(0).call(sbn).local_set(sb);
                    if default {
                        let factory = match &vt {
                            Ty::Int => "int",
                            Ty::Float => "float",
                            Ty::Str => "str",
                            Ty::List(_) => "list",
                            Ty::Dict(..) => "dict",
                            Ty::Set(_) => "set",
                            _ => return unsupported("printing this defaultdict"),
                        };
                        lit(w, sb, &format!("defaultdict(<class '{}'>, ", factory));
                    }
                    lit(w, sb, "{");
                    let next = w.rt_dict_next()?;
                    w.c().i32_const(1).local_set(first);
                    w.c().block(Block::Empty).loop_(Block::Empty);
                    w.c().local_get(0).local_get(i).call(next).local_tee(i).i32_const(0).i32_lt_s().br_if(1);
                    w.c().local_get(first).i32_eqz().if_(Block::Empty);
                    lit(w, sb, ", ");
                    w.c().end().i32_const(0).local_set(first);
                    w.c().local_get(sb).local_get(0).struct_get(d, 0).local_get(i).array_get(ae);
                    w.unbox_eq(&kt)?;
                    w.emit_repr(&kt)?;
                    w.c().call(sba);
                    if !set {
                        lit(w, sb, ": ");
                        w.c().local_get(sb).local_get(0).struct_get(d, 1).local_get(i).array_get(ae);
                        w.unbox_eq(&vt)?;
                        w.emit_repr(&vt)?;
                        w.c().call(sba);
                    }
                    w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                    w.c().end().end();
                    lit(w, sb, if default { "})" } else { "}" });
                    w.c().local_get(sb).call(sbs);
                }
                Ty::Optional(inner) => {
                    let inner = inner.as_ref().clone();
                    let s = w.t.str;
                    w.c().local_get(0).ref_is_null().if_(Block::Val(Val::Ref(Heap::Idx(s))));
                    w.str_const("None");
                    w.c().else_().local_get(0);
                    w.unbox_opt(&inner)?;
                    if repr {
                        w.emit_repr(&inner)?;
                    } else {
                        w.emit_str(&inner)?;
                    }
                    w.c().end();
                }
                Ty::Exc => {
                    let exc = w.t.exc;
                    w.c().local_get(0).struct_get(exc, 2);
                }
                // a value that may be a subclass: show it as its own class
                Ty::Class(c) if !w.p.classes[*c].subs.is_empty() => {
                    let base = w.class_ty[*c];
                    for s in w.p.subtree(*c) {
                        let f = w.show_fn(&Ty::Obj(s), repr)?;
                        let st = w.class_ty[s];
                        w.c().local_get(0).struct_get(base, 0).i32_const(s as i32).i32_eq().if_(Block::Empty);
                        w.c().local_get(0).ref_cast(Heap::Idx(st)).call(f).return_().end();
                    }
                    w.c().unreachable();
                }
                Ty::Class(c) | Ty::Obj(c) => {
                    let c = *c;
                    let cl = &w.p.classes[c];
                    let m = if repr { w.find_method(c, "__repr__") } else { w.find_method(c, "__str__").or(w.find_method(c, "__repr__")) };
                    match m {
                        Some(f) => {
                            let fi = w.user[f];
                            w.c().local_get(0).call(fi);
                        }
                        None if cl.dataclass => {
                            let ct = w.class_ty[c];
                            let fields: Vec<(String, Ty)> = cl.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
                            let name = cl.name.clone();
                            let sb = w.fx.local(Val::Ref(Heap::Idx(sb_t)));
                            w.c().i32_const(0).call(sbn).local_set(sb);
                            lit(w, sb, &format!("{}(", name));
                            for (i, (n, ft)) in fields.iter().enumerate() {
                                lit(w, sb, &format!("{}{}=", if i > 0 { ", " } else { "" }, n));
                                w.c().local_get(sb).local_get(0).struct_get(ct, i as u32 + 1);
                                if w.val(ft)?.is_none() {
                                    w.c().drop();
                                }
                                w.emit_repr(ft)?;
                                w.c().call(sba);
                            }
                            lit(w, sb, ")");
                            w.c().local_get(sb).call(sbs);
                        }
                        None => return unsupported(format!("printing a `{}` without __str__ / __repr__", cl.name)),
                    }
                }
                t => return unsupported(format!("str() of {:?}", t)),
            }
            Ok(vec![sref])
        })
    }

    /// A method of class `c` or the nearest base class that has it.
    pub fn find_method(&self, c: ClassId, name: &str) -> Option<FuncId> {
        let mut k = Some(c);
        while let Some(x) = k {
            if let Some(f) = self.p.classes[x].method(name) {
                return Some(f);
            }
            k = self.p.classes[x].base;
        }
        None
    }

    /// repr of a str: quotes and escapes as Python's
    pub fn rt_str_repr(&mut self) -> Res<u32> {
        let sbn = self.rt_sb_new()?;
        let sbs = self.rt_sb_str()?;
        let byte = self.rt_sb_byte()?;
        let hex = self.rt_sb_hex()?;
        self.rt_fn("str_repr", |w| {
            let (st, bytes, sbt) = (w.t.str, w.t.bytes, w.t.sb);
            w.fx.params = vec![Val::Ref(Heap::Idx(st))];
            let (b, i, n, c, q, sb, sq, dq) = (
                w.fx.local(Val::Ref(Heap::Idx(bytes))),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::Ref(Heap::Idx(sbt))),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
            );
            w.c().local_get(0).struct_get(st, 0).local_tee(b).array_len().local_set(n);
            // the quote: ' unless the string has ' and no "
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(b).local_get(i).array_get_u(bytes).local_tee(c).i32_const(39).i32_eq().local_get(sq).i32_or().local_set(sq);
            w.c().local_get(c).i32_const(34).i32_eq().local_get(dq).i32_or().local_set(dq);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            w.c().i32_const(34).i32_const(39).local_get(sq).local_get(dq).i32_eqz().i32_and().select().local_set(q);
            w.c().local_get(n).i32_const(2).i32_add().call(sbn).local_set(sb);
            w.c().local_get(sb).local_get(q).call(byte);
            w.c().i32_const(0).local_set(i);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(b).local_get(i).array_get_u(bytes).local_set(c);
            // \\ and the quote
            w.c().local_get(c).i32_const(92).i32_eq().local_get(c).local_get(q).i32_eq().i32_or().if_(Block::Empty);
            w.c().local_get(sb).i32_const(92).call(byte).local_get(sb).local_get(c).call(byte);
            w.c().else_().local_get(c).i32_const(10).i32_eq().if_(Block::Empty);
            w.c().local_get(sb).i32_const(92).call(byte).local_get(sb).i32_const(110).call(byte);
            w.c().else_().local_get(c).i32_const(13).i32_eq().if_(Block::Empty);
            w.c().local_get(sb).i32_const(92).call(byte).local_get(sb).i32_const(114).call(byte);
            w.c().else_().local_get(c).i32_const(9).i32_eq().if_(Block::Empty);
            w.c().local_get(sb).i32_const(92).call(byte).local_get(sb).i32_const(116).call(byte);
            w.c().else_().local_get(c).i32_const(32).i32_lt_u().local_get(c).i32_const(127).i32_eq().i32_or().if_(Block::Empty);
            w.c().local_get(sb).local_get(c).call(hex);
            w.c().else_().local_get(sb).local_get(c).call(byte);
            w.c().end().end().end().end().end();
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            w.c().local_get(sb).local_get(q).call(byte).local_get(sb).call(sbs);
            Ok(vec![Val::Ref(Heap::Idx(st))])
        })
    }

    /// sb_byte(sb, b): append one byte (a code point when < 0x80 or a lead byte)
    pub fn rt_sb_byte(&mut self) -> Res<u32> {
        self.rt_fn("sb_byte", |w| {
            let (sb, bytes) = (w.t.sb, w.t.bytes);
            w.fx.params = vec![Val::Ref(Heap::Idx(sb)), Val::I32];
            let nb = w.fx.local(Val::Ref(Heap::Idx(bytes)));
            w.c().local_get(0).struct_get(sb, 1).local_get(0).struct_get(sb, 0).array_len().i32_ge_u().if_(Block::Empty);
            w.c().local_get(0).struct_get(sb, 1).i32_const(1).i32_shl().i32_const(8).i32_add().array_new_default(bytes).local_tee(nb).i32_const(0).local_get(0).struct_get(sb, 0).i32_const(0).local_get(0).struct_get(sb, 1).array_copy(bytes, bytes);
            w.c().local_get(0).local_get(nb).ref_as_non_null().struct_set(sb, 0).end();
            w.c().local_get(0).struct_get(sb, 0).local_get(0).struct_get(sb, 1).local_get(1).array_set(bytes);
            w.c().local_get(0).local_get(0).struct_get(sb, 1).i32_const(1).i32_add().struct_set(sb, 1);
            // continuation bytes are not characters
            w.c().local_get(0).local_get(0).struct_get(sb, 2).local_get(1).i32_const(0xc0).i32_and().i32_const(0x80).i32_ne().i32_add().struct_set(sb, 2);
            Ok(vec![])
        })
    }

    /// sb_hex(sb, b): `\xNN`
    fn rt_sb_hex(&mut self) -> Res<u32> {
        let byte = self.rt_sb_byte()?;
        self.rt_fn("sb_hex", |w| {
            let sb = w.t.sb;
            w.fx.params = vec![Val::Ref(Heap::Idx(sb)), Val::I32];
            w.c().local_get(0).i32_const(92).call(byte).local_get(0).i32_const(120).call(byte);
            for shift in [4, 0] {
                let d = w.fx.local(Val::I32);
                w.c().local_get(1).i32_const(shift).i32_shr_u().i32_const(15).i32_and().local_tee(d);
                w.c().i32_const(48).i32_add().local_get(d).i32_const(87).i32_add().local_get(d).i32_const(10).i32_lt_u().select().local_set(d);
                w.c().local_get(0).local_get(d).call(byte);
            }
            Ok(vec![])
        })
    }

    // ------------------------------------------------------------ equality, order, hash

    /// eq(a, b) → i32 for values of type `t`
    pub fn eq_fn(&mut self, t: &Ty) -> Res<u32> {
        match t {
            Ty::Str => return self.rt_str_eq(),
            Ty::None => {
                return self.rt_fn("eq_none", |w| {
                    w.fx.params = vec![Val::I32, Val::I32];
                    w.c().i32_const(1);
                    Ok(vec![Val::I32])
                })
            }
            _ => {}
        }
        let pv = self.param_of(t)?;
        let key = format!("eq:{:?}", t);
        let t = t.clone();
        self.rt_fn(&key, move |w| {
            w.fx.params = vec![pv, pv];
            match &t {
                Ty::Int => {
                    w.c().local_get(0).local_get(1).i64_eq();
                }
                Ty::Float => {
                    w.c().local_get(0).local_get(1).f64_eq();
                }
                Ty::Bool => {
                    w.c().local_get(0).local_get(1).i32_eq();
                }
                Ty::List(e) => {
                    let e = e.as_ref().clone();
                    let k = w.kind(&e)?;
                    let l = w.t.list[k as usize];
                    let i = w.fx.local(Val::I32);
                    let eq = w.eq_fn(&e)?;
                    w.c().local_get(0).local_get(1).ref_eq().if_(Block::Empty).i32_const(1).return_().end();
                    w.c().local_get(0).struct_get(l, 1).local_get(1).struct_get(l, 1).i32_ne().if_(Block::Empty).i32_const(0).return_().end();
                    w.c().block(Block::Empty).loop_(Block::Empty);
                    w.c().local_get(i).local_get(0).struct_get(l, 1).i32_ge_s().br_if(1);
                    for side in [0, 1] {
                        w.c().local_get(side).local_get(i).i64_extend_i32_s();
                        w.list_get(&e)?;
                        if w.val(&e)?.is_none() {
                            w.c().i32_const(0);
                        }
                    }
                    w.c().call(eq).i32_eqz().if_(Block::Empty).i32_const(0).return_().end();
                    w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                    w.c().end().end().i32_const(1);
                }
                Ty::Tuple(ts) => {
                    let tt = w.tuple_ty(ts)?;
                    w.c().i32_const(1);
                    for (i, et) in ts.iter().enumerate() {
                        let eq = w.eq_fn(et)?;
                        w.c().local_get(0).struct_get(tt, i as u32).local_get(1).struct_get(tt, i as u32).call(eq).i32_and();
                    }
                }
                Ty::Optional(inner) => {
                    let inner = inner.as_ref().clone();
                    let eq = w.eq_fn(&inner)?;
                    // both None: equal; one None: not
                    w.c().local_get(0).ref_is_null().local_get(1).ref_is_null().i32_or().if_(Block::Empty);
                    w.c().local_get(0).ref_is_null().local_get(1).ref_is_null().i32_and().return_().end();
                    w.c().local_get(0);
                    w.unbox_opt(&inner)?;
                    w.c().local_get(1);
                    w.unbox_opt(&inner)?;
                    w.c().call(eq);
                }
                Ty::Dict(kt, _, _) | Ty::Set(kt) => {
                    let set = matches!(t, Ty::Set(_));
                    let kt = kt.as_ref().clone();
                    let vt = if set { Ty::None } else { match &t { Ty::Dict(_, v, _) => v.as_ref().clone(), _ => unreachable!() } };
                    let (d, ae) = (w.t.dict, w.t.arr[3]);
                    let (i, j) = (w.fx.local(Val::I32), w.fx.local(Val::I32));
                    let next = w.rt_dict_next()?;
                    let find = w.rt_dict_find()?;
                    let veq = w.eq_fn(&vt)?;
                    w.c().local_get(0).struct_get(d, 5).local_get(1).struct_get(d, 5).i32_ne().if_(Block::Empty).i32_const(0).return_().end();
                    w.c().block(Block::Empty).loop_(Block::Empty);
                    w.c().local_get(0).local_get(i).call(next).local_tee(i).i32_const(0).i32_lt_s().br_if(1);
                    w.c().local_get(1).local_get(0).struct_get(d, 0).local_get(i).array_get(ae).call(find).local_tee(j).i32_const(0).i32_lt_s().if_(Block::Empty).i32_const(0).return_().end();
                    if !set {
                        w.c().local_get(0).struct_get(d, 1).local_get(i).array_get(ae);
                        w.unbox_eq(&vt)?;
                        if w.val(&vt)?.is_none() {
                            w.c().i32_const(0);
                        }
                        w.c().local_get(1).struct_get(d, 1).local_get(j).array_get(ae);
                        w.unbox_eq(&vt)?;
                        if w.val(&vt)?.is_none() {
                            w.c().i32_const(0);
                        }
                        w.c().call(veq).i32_eqz().if_(Block::Empty).i32_const(0).return_().end();
                    }
                    let _ = kt;
                    w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                    w.c().end().end().i32_const(1);
                }
                Ty::Class(c) | Ty::Obj(c) => {
                    let c = *c;
                    let cl = &w.p.classes[c];
                    if let Some(f) = cl.method("__eq__") {
                        let fi = w.user[f];
                        w.c().local_get(0).local_get(1).call(fi);
                    } else if cl.dataclass {
                        let ct = w.class_ty[c];
                        let fields: Vec<Ty> = cl.fields.iter().map(|f| f.ty.clone()).collect();
                        // the same class, then the fields
                        w.c().local_get(0).struct_get(ct, 0).local_get(1).struct_get(ct, 0).i32_eq();
                        for (i, ft) in fields.iter().enumerate() {
                            let eq = w.eq_fn(ft)?;
                            w.c().local_get(0).struct_get(ct, i as u32 + 1).local_get(1).struct_get(ct, i as u32 + 1).call(eq).i32_and();
                        }
                    } else {
                        w.c().local_get(0).local_get(1).ref_eq();
                    }
                }
                Ty::Exc => {
                    w.c().local_get(0).local_get(1).ref_eq();
                }
                t => return unsupported(format!("== on {:?}", t)),
            }
            Ok(vec![Val::I32])
        })
    }

    /// cmp(a, b) → -1 / 0 / 1 for ordering values of type `t`
    pub fn cmp_fn(&mut self, t: &Ty) -> Res<u32> {
        if *t == Ty::Str {
            return self.rt_str_cmp();
        }
        let pv = self.param_of(t)?;
        let key = format!("cmp:{:?}", t);
        let t = t.clone();
        self.rt_fn(&key, move |w| {
            w.fx.params = vec![pv, pv];
            match &t {
                Ty::Int => {
                    w.c().local_get(0).local_get(1).i64_gt_s().local_get(0).local_get(1).i64_lt_s().i32_sub();
                }
                Ty::Float => {
                    w.c().local_get(0).local_get(1).f64_gt().local_get(0).local_get(1).f64_lt().i32_sub();
                }
                Ty::Bool => {
                    w.c().local_get(0).local_get(1).i32_gt_s().local_get(0).local_get(1).i32_lt_s().i32_sub();
                }
                Ty::Tuple(ts) => {
                    let tt = w.tuple_ty(ts)?;
                    let r = w.fx.local(Val::I32);
                    for (i, et) in ts.iter().enumerate() {
                        let cmp = w.cmp_fn(et)?;
                        w.c().local_get(0).struct_get(tt, i as u32).local_get(1).struct_get(tt, i as u32).call(cmp).local_tee(r).if_(Block::Empty).local_get(r).return_().end();
                    }
                    w.c().i32_const(0);
                }
                Ty::List(e) => {
                    let e = e.as_ref().clone();
                    let k = w.kind(&e)?;
                    let l = w.t.list[k as usize];
                    let (i, r) = (w.fx.local(Val::I32), w.fx.local(Val::I32));
                    let cmp = w.cmp_fn(&e)?;
                    w.c().block(Block::Empty).loop_(Block::Empty);
                    w.c().local_get(i).local_get(0).struct_get(l, 1).i32_ge_s().local_get(i).local_get(1).struct_get(l, 1).i32_ge_s().i32_or().br_if(1);
                    for side in [0, 1] {
                        w.c().local_get(side).local_get(i).i64_extend_i32_s();
                        w.list_get(&e)?;
                    }
                    w.c().call(cmp).local_tee(r).if_(Block::Empty).local_get(r).return_().end();
                    w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                    w.c().end().end();
                    w.c().local_get(0).struct_get(l, 1).local_get(1).struct_get(l, 1).i32_gt_s().local_get(0).struct_get(l, 1).local_get(1).struct_get(l, 1).i32_lt_s().i32_sub();
                }
                Ty::Class(c) | Ty::Obj(c) => {
                    let Some(lt) = w.p.classes[*c].method("__lt__") else {
                        return unsupported(format!("ordering `{}` objects (no __lt__)", w.p.classes[*c].name));
                    };
                    let fi = w.user[lt];
                    w.c().local_get(0).local_get(1).call(fi).if_(Block::Val(Val::I32)).i32_const(-1);
                    w.c().else_().local_get(1).local_get(0).call(fi).end();
                }
                t => return unsupported(format!("ordering {:?}", t)),
            }
            Ok(vec![Val::I32])
        })
    }

    /// hash(value) → i64 for values of type `t`
    pub fn hash_fn(&mut self, t: &Ty) -> Res<u32> {
        if *t == Ty::Str {
            return self.rt_str_hash();
        }
        let pv = self.param_of(t)?;
        let key = format!("hash:{:?}", t);
        let t = t.clone();
        self.rt_fn(&key, move |w| {
            w.fx.params = vec![pv];
            match &t {
                Ty::Int => {
                    w.c().local_get(0);
                }
                Ty::Float => {
                    // -0.0 hashes as 0.0
                    w.c().local_get(0).f64_const(0.0).f64_add().i64_reinterpret_f64();
                }
                Ty::Bool => {
                    w.c().local_get(0).i64_extend_i32_u();
                }
                Ty::None => {
                    w.c().i64_const(0);
                }
                Ty::Tuple(ts) => {
                    let tt = w.tuple_ty(ts)?;
                    w.c().i64_const(0x345678);
                    for (i, et) in ts.iter().enumerate() {
                        let h = w.hash_fn(et)?;
                        w.c().i64_const(1000003).i64_mul().local_get(0).struct_get(tt, i as u32).call(h).i64_xor();
                    }
                }
                Ty::Optional(inner) => {
                    let inner = inner.as_ref().clone();
                    let h = w.hash_fn(&inner)?;
                    w.c().local_get(0).ref_is_null().if_(Block::Val(Val::I64)).i64_const(0).else_().local_get(0);
                    w.unbox_opt(&inner)?;
                    w.c().call(h).end();
                }
                Ty::Class(c) | Ty::Obj(c) if w.p.classes[*c].dataclass && w.p.classes[*c].method("__eq__").is_none() => {
                    let ct = w.class_ty[*c];
                    let fields: Vec<Ty> = w.p.classes[*c].fields.iter().map(|f| f.ty.clone()).collect();
                    w.c().i64_const(0x345678);
                    for (i, ft) in fields.iter().enumerate() {
                        let h = w.hash_fn(ft)?;
                        w.c().i64_const(1000003).i64_mul().local_get(0).struct_get(ct, i as u32 + 1).call(h).i64_xor();
                    }
                }
                t => return unsupported(format!("hashing {:?}", t)),
            }
            Ok(vec![Val::I64])
        })
    }

    /// A dict's hash function: (eqref) → i64
    pub fn hash_eq_fn(&mut self, t: &Ty) -> Res<u32> {
        let h = self.hash_fn(t)?;
        let key = format!("hasheq:{:?}", t);
        let t = t.clone();
        self.rt_fn(&key, move |w| {
            w.fx.params = vec![Val::Ref(Heap::Eq)];
            w.c().local_get(0);
            w.unbox_eq(&t)?;
            if w.val(&t)?.is_none() {
                w.c().i32_const(0);
            }
            w.c().call(h);
            Ok(vec![Val::I64])
        })
    }

    /// A dict's equality: (eqref, eqref) → i32
    pub fn eq_eq_fn(&mut self, t: &Ty) -> Res<u32> {
        let eq = self.eq_fn(t)?;
        let key = format!("eqeq:{:?}", t);
        let t = t.clone();
        self.rt_fn(&key, move |w| {
            w.fx.params = vec![Val::Ref(Heap::Eq), Val::Ref(Heap::Eq)];
            for i in [0, 1] {
                w.c().local_get(i);
                w.unbox_eq(&t)?;
                if w.val(&t)?.is_none() {
                    w.c().i32_const(0);
                }
            }
            w.c().call(eq);
            Ok(vec![Val::I32])
        })
    }

    // ------------------------------------------------------------ formatting

    /// `format(value, spec)` for the value on the stack.
    pub fn format_spec(&mut self, t: &Ty, spec: &str) -> Res<()> {
        let numeric = spec.chars().last().is_some_and(|c| "bcdoxXneEfFgG%".contains(c));
        match t {
            Ty::Int | Ty::Float => self.host_arg(t)?,
            Ty::Bool if numeric => {
                self.c().i64_extend_i32_u();
                self.host_arg(&Ty::Int)?;
            }
            _ => {
                self.emit_str(t)?;
                self.host_arg(&Ty::Str)?;
            }
        }
        self.str_const(spec);
        self.host_arg(&Ty::Str)?;
        self.host_call("format", &Ty::Str)
    }

    pub fn fstring(&mut self, parts: &[FPart]) -> Res<()> {
        let sbn = self.rt_sb_new()?;
        let sba = self.rt_sb_add()?;
        let sbs = self.rt_sb_str()?;
        let sb = self.fx.local(Val::Ref(Heap::Idx(self.t.sb)));
        self.c().i32_const(0).call(sbn).local_set(sb);
        for p in parts {
            self.c().local_get(sb);
            match p {
                FPart::Lit(s) => self.str_const(s),
                FPart::Val(e, conv, spec) => {
                    self.expr(e)?;
                    let mut t = e.ty.clone();
                    if matches!(conv, Some('r') | Some('a')) {
                        self.emit_repr(&t)?;
                        t = Ty::Str;
                    }
                    if spec.is_empty() {
                        self.emit_str(&t)?;
                    } else {
                        self.format_spec(&t, spec)?;
                    }
                }
            }
            self.c().call(sba);
        }
        self.c().local_get(sb).call(sbs);
        Ok(())
    }
}

#[allow(dead_code)]
fn _k(_: K) {}

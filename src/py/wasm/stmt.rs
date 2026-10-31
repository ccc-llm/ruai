//! Statements and loops.

use super::enc::{Block, Heap, Val};
use super::{unsupported, Lab, Res, K, W};
use crate::py::ir::*;

/// The state of a loop over an iterable (see [`W::iter_init`]).
pub(crate) enum Cursor {
    Range { i: u32, stop: u32, step: u32, up: Option<bool> },
    /// a list from its start (or its end, reversed); `ty` is the element type
    List { xs: u32, i: u32, k: K, ty: Ty, rev: bool },
    Str { s: u32, pos: u32 },
    /// a dict or set: 0 keys, 1 values, 2 items (a tuple)
    Dict { d: u32, i: u32, mode: u8, k: Ty, v: Ty },
    Enumerate { inner: Box<Cursor>, n: u32, ty: Ty },
    Zip { inner: Vec<Cursor>, ty: Ty },
}

impl<'a> W<'a> {
    pub fn block(&mut self, b: &[Stmt]) -> Res<()> {
        for s in b {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn new_label(&mut self) -> u32 {
        self.fx.next_label += 1;
        self.fx.next_label
    }

    /// Run `f` inside a wasm block / loop / if that is a label.
    pub fn labeled(&mut self, l: Lab, f: impl FnOnce(&mut W<'a>) -> Res<()>) -> Res<()> {
        self.fx.labels.push(l);
        let r = f(self);
        self.fx.labels.pop();
        r
    }

    fn stmt(&mut self, s: &Stmt) -> Res<()> {
        match &s.kind {
            StmtKind::Expr(e) => {
                self.expr(e)?;
                if self.val(&e.ty)?.is_some() {
                    self.c().drop();
                }
            }
            StmtKind::Pass => {}
            StmtKind::Assign(t, e) => {
                self.expr(e)?;
                self.assign(t, &e.ty)?;
            }
            StmtKind::AugVar(v, op, e) => {
                let ty = self.p.funcs[self.fx.fid.unwrap()].vars[*v].ty.clone();
                let l = self.fx.vars[v];
                self.c().local_get(l);
                self.expr(e)?;
                self.binop(*op, &ty, &e.ty, s.line)?;
                self.c().local_set(l);
            }
            StmtKind::AugAssign(place, op, e) => self.aug_assign(place, *op, e, s.line)?,
            StmtKind::If(c, a, b) => {
                self.cond(c)?;
                self.c().if_(Block::Empty);
                self.labeled(Lab::Plain, |w| {
                    w.block(a)?;
                    if !b.is_empty() {
                        w.c().else_();
                        w.block(b)?;
                    }
                    Ok(())
                })?;
                self.c().end();
            }
            StmtKind::While(c, body, orelse) => {
                let id = self.new_label();
                self.c().block(Block::Empty);
                self.labeled(Lab::Break(id), |w| {
                    w.c().block(Block::Empty);
                    w.labeled(Lab::Plain, |w| {
                        w.c().loop_(Block::Empty);
                        w.labeled(Lab::Continue(id), |w| {
                            w.cond(c)?;
                            w.c().i32_eqz().br_if(1);
                            w.loop_body(id, body)?;
                            w.c().br(0);
                            Ok(())
                        })?;
                        w.c().end();
                        Ok(())
                    })?;
                    w.c().end();
                    w.block(orelse)
                })?;
                self.c().end();
            }
            StmtKind::For(t, it, body, orelse) => {
                let cur = self.iter_init(it)?;
                let elem = self.iter_ty(it)?;
                let id = self.new_label();
                self.c().block(Block::Empty);
                self.labeled(Lab::Break(id), |w| {
                    w.c().block(Block::Empty);
                    w.labeled(Lab::Plain, |w| {
                        w.c().loop_(Block::Empty);
                        w.labeled(Lab::Continue(id), |w| {
                            // exhausted: leave to the `else` code (depth 1)
                            w.iter_next(&cur, 1)?;
                            w.assign(t, &elem)?;
                            w.loop_body(id, body)?;
                            w.c().br(0);
                            Ok(())
                        })?;
                        w.c().end();
                        Ok(())
                    })?;
                    w.c().end();
                    w.block(orelse)
                })?;
                self.c().end();
            }
            StmtKind::Break | StmtKind::Continue => {
                let (id, nfin) = *self.fx.loops.last().expect("loop");
                self.run_finals(nfin)?;
                let l = if matches!(s.kind, StmtKind::Break) { Lab::Break(id) } else { Lab::Continue(id) };
                let d = self.fx.depth(l);
                self.c().br(d);
            }
            StmtKind::Return(e) => {
                let mut tmp = None;
                if let Some(e) = e {
                    self.expr(e)?;
                    if !self.fx.finals.is_empty() {
                        if let Some(v) = self.val(&e.ty)? {
                            let l = self.fx.local(v);
                            self.c().local_set(l);
                            tmp = Some(l);
                        }
                    }
                }
                self.run_finals(0)?;
                if let Some(l) = tmp {
                    self.c().local_get(l);
                }
                if self.fx.ret.is_some() && e.as_ref().is_some_and(|e| matches!(e.ty, Ty::None)) {
                    let r = self.fx.ret.unwrap();
                    self.zero(r);
                }
                self.c().return_();
            }
            StmtKind::Raise(name, msg) => self.raise(name, msg.as_ref())?,
            StmtKind::Reraise(v) => {
                let l = self.fx.vars[v];
                let tag = self.tag;
                self.c().local_get(l).throw(tag);
            }
            StmtKind::Assert(c, m) => {
                self.cond(c)?;
                self.c().i32_eqz().if_(Block::Empty);
                self.labeled(Lab::Plain, |w| w.raise("AssertionError", m.as_ref()))?;
                self.c().end();
            }
            StmtKind::Del(e) => {
                let ExprKind::Index(b, k) = &e.kind else { return unsupported("this `del`") };
                self.expr(b)?;
                self.expr(k)?;
                match &b.ty {
                    Ty::List(t) => {
                        let f = self.rt_list_del(t)?;
                        self.c().call(f);
                    }
                    Ty::Dict(kt, vt, _) => {
                        let (kt, vt) = (kt.as_ref().clone(), vt.as_ref().clone());
                        self.box_eq(&kt)?;
                        let f = self.rt_dict_del(&kt, &vt)?;
                        self.c().call(f);
                    }
                    t => return unsupported(format!("`del` on {:?}", t)),
                }
            }
            StmtKind::LetSome(v, place, body) => {
                self.expr(place)?;
                let opt = self.val(&place.ty)?.unwrap();
                let tmp = self.fx.local(opt);
                self.c().local_tee(tmp).ref_is_null().if_(Block::Empty);
                self.labeled(Lab::Plain, |w| w.block(body))?;
                self.c().end();
                self.c().local_get(tmp);
                let inner = match &place.ty {
                    Ty::Optional(t) => t.as_ref().clone(),
                    t => t.clone(),
                };
                self.unbox_opt(&inner)?;
                let l = self.fx.vars[v];
                self.c().local_set(l);
            }
            StmtKind::Try(body, hs, orelse, fin) => self.try_stmt(body, hs, orelse, fin)?,
        }
        Ok(())
    }

    fn loop_body(&mut self, id: u32, body: &[Stmt]) -> Res<()> {
        let nfin = self.fx.finals.len();
        self.fx.loops.push((id, nfin));
        self.c().block(Block::Empty);
        let r = self.labeled(Lab::Continue(id), |w| w.block(body));
        self.fx.loops.pop();
        r?;
        self.c().end();
        Ok(())
    }

    /// Run the `finally` code of the `try`s above `keep` (outside each of them).
    fn run_finals(&mut self, keep: usize) -> Res<()> {
        let all = self.fx.finals.clone();
        for i in (keep..all.len()).rev() {
            let saved = self.fx.finals.split_off(i);
            let r = self.block(&all[i]);
            self.fx.finals.extend(saved);
            r?;
        }
        Ok(())
    }

    fn try_stmt(&mut self, body: &[Stmt], hs: &[Handler], orelse: &[Stmt], fin: &[Stmt]) -> Res<()> {
        if fin.is_empty() {
            return self.try_except(body, hs, orelse);
        }
        // try { body } catch_all { finally; rethrow }; finally
        self.c().try_(Block::Empty);
        self.fx.finals.push(fin.to_vec());
        let r = self.labeled(Lab::Plain, |w| {
            w.try_except(body, hs, orelse)?;
            w.fx.finals.pop();
            w.c().0.push(0x19); // catch_all
            w.block(fin)?;
            w.c().rethrow(0);
            Ok(())
        });
        r?;
        self.c().end();
        self.block(fin)
    }

    fn try_except(&mut self, body: &[Stmt], hs: &[Handler], orelse: &[Stmt]) -> Res<()> {
        if hs.is_empty() {
            self.block(body)?;
            return self.block(orelse);
        }
        let tag = self.tag;
        let e = self.fx.local(Val::Ref(Heap::Idx(self.t.exc)));
        // block { try { body } catch { handlers (br 2 leaves the block); rethrow }; else }
        self.c().block(Block::Empty);
        self.labeled(Lab::Plain, |w| {
            w.c().try_(Block::Empty);
            w.labeled(Lab::Plain, |w| {
                w.block(body)?;
                w.c().catch(tag).local_set(e);
                for h in hs {
                    let ids: Vec<i32> = h.kinds.iter().map(|k| w.exc_id(k)).collect();
                    let m = w.rt_exc_match()?;
                    w.c().local_get(e);
                    let arr = w.t.arr_i32;
                    for id in &ids {
                        w.c().i32_const(*id);
                    }
                    w.c().array_new_fixed(arr, ids.len() as u32).call(m).if_(Block::Empty);
                    w.labeled(Lab::Plain, |w| {
                        if let Some(&l) = w.fx.vars.get(&h.var) {
                            w.c().local_get(e).local_set(l);
                        }
                        w.block(&h.body)?;
                        w.c().br(2);
                        Ok(())
                    })?;
                    w.c().end();
                }
                w.c().rethrow(0);
                Ok(())
            })?;
            w.c().end();
            w.block(orelse)
        })?;
        self.c().end();
        Ok(())
    }

    /// `raise K(msg)` / `raise obj`
    pub fn raise(&mut self, kind: &str, msg: Option<&Expr>) -> Res<()> {
        let chain = crate::py::exc::chain(self.p, kind);
        let ids: Vec<i32> = chain.iter().map(|k| self.exc_id(k)).collect();
        self.str_const(kind);
        let arr = self.t.arr_i32;
        for id in &ids {
            self.c().i32_const(*id);
        }
        self.c().array_new_fixed(arr, ids.len() as u32);
        match msg {
            Some(m @ Expr { ty: Ty::Class(c), .. }) => {
                // an exception object: its message is the field msg_
                let c = *c;
                self.expr(m)?;
                let o = self.fx.local(Val::Ref(Heap::Idx(self.class_ty[c])));
                self.c().local_set(o);
                match self.p.classes[c].fields.iter().position(|f| f.name == "msg_") {
                    Some(i) => {
                        let ct = self.class_ty[c];
                        self.c().local_get(o).struct_get(ct, i as u32 + 1);
                    }
                    None => self.str_const(""),
                }
                self.c().local_get(o);
            }
            Some(m) => {
                self.str_of(m)?;
                self.c().ref_null(Heap::Eq);
            }
            None => {
                self.str_const("");
                self.c().ref_null(Heap::Eq);
            }
        }
        let (exc, tag) = (self.t.exc, self.tag);
        self.c().struct_new(exc).throw(tag);
        Ok(())
    }

    /// Store the value on the stack (of type `ty`) into a target.
    pub fn assign(&mut self, t: &Target, ty: &Ty) -> Res<()> {
        match t {
            Target::Var(v) => match self.fx.vars.get(v) {
                Some(&l) => {
                    self.c().local_set(l);
                }
                None => {
                    if self.val(ty)?.is_some() {
                        self.c().drop();
                    }
                }
            },
            Target::Place(e) => {
                let v = self.val(ty)?.unwrap_or(Val::I32);
                let tmp = self.fx.local(v);
                self.c().local_set(tmp);
                self.store(e, |w| {
                    w.c().local_get(tmp);
                    Ok(())
                })?;
            }
            Target::Tuple(ts) => {
                let Ty::Tuple(tys) = ty else {
                    // a list unpacked: `a, b = xs`
                    if let Ty::List(et) = ty {
                        let v = self.val(ty)?.unwrap();
                        let tmp = self.fx.local(v);
                        self.c().local_set(tmp);
                        let f = self.rt_list_unpack(et, ts.len())?;
                        self.c().local_get(tmp).call(f);
                        for (i, t) in ts.iter().enumerate() {
                            self.c().local_get(tmp).i64_const(i as i64);
                            self.list_get(et)?;
                            self.assign(t, et)?;
                        }
                        return Ok(());
                    }
                    return unsupported(format!("unpacking a {:?}", ty));
                };
                let tt = self.tuple_ty(tys)?;
                let tmp = self.fx.local(Val::Ref(Heap::Idx(tt)));
                self.c().local_set(tmp);
                for (i, (t, et)) in ts.iter().zip(tys).enumerate() {
                    self.c().local_get(tmp).struct_get(tt, i as u32);
                    self.assign(t, et)?;
                }
            }
        }
        Ok(())
    }

    /// Store a value into a place (`xs[i]`, `d[k]`, `obj.f`); `val` pushes it.
    pub fn store(&mut self, place: &Expr, val: impl FnOnce(&mut W<'a>) -> Res<()>) -> Res<()> {
        match &place.kind {
            ExprKind::Field(o, f) => {
                let c = match &o.ty {
                    Ty::Class(c) | Ty::Obj(c) => *c,
                    t => return unsupported(format!("setting a field of {:?}", t)),
                };
                self.expr(o)?;
                val(self)?;
                let (ct, i) = (self.class_ty[c], self.field_index(c, f)?);
                self.c().struct_set(ct, i);
            }
            ExprKind::Index(b, k) => {
                self.expr(b)?;
                self.expr(k)?;
                match &b.ty {
                    Ty::List(t) => {
                        let t = t.as_ref().clone();
                        val(self)?;
                        let f = self.rt_list_set(&t)?;
                        self.c().call(f);
                    }
                    Ty::Dict(kt, vt, _) => {
                        let (kt, vt) = (kt.as_ref().clone(), vt.as_ref().clone());
                        self.box_eq(&kt)?;
                        val(self)?;
                        self.box_eq(&vt)?;
                        let f = self.rt_dict_set(&kt, &vt)?;
                        self.c().call(f);
                    }
                    t => return unsupported(format!("item assignment on {:?}", t)),
                }
            }
            _ => return unsupported("this assignment target"),
        }
        Ok(())
    }

    fn aug_assign(&mut self, place: &Expr, op: BinOp, e: &Expr, line: u32) -> Res<()> {
        // `xs += ys` on a list extends it in place
        if let (Ty::List(t), BinOp::Add) = (&place.ty, op) {
            let t = t.as_ref().clone();
            self.expr(place)?;
            self.expr(e)?;
            let f = self.rt_list_extend(&t)?;
            self.c().call(f);
            return Ok(());
        }
        match &place.kind {
            ExprKind::Var(v) => {
                let l = self.fx.vars[v];
                self.c().local_get(l);
                self.expr(e)?;
                self.binop(op, &place.ty, &e.ty, line)?;
                self.c().local_set(l);
            }
            ExprKind::Field(o, f) => {
                let c = match &o.ty {
                    Ty::Class(c) | Ty::Obj(c) => *c,
                    _ => return unsupported("this augmented assignment"),
                };
                let (ct, i) = (self.class_ty[c], self.field_index(c, f)?);
                self.expr(o)?;
                let ov = self.fx.local(Val::Ref(Heap::Idx(ct)));
                self.c().local_tee(ov).local_get(ov).struct_get(ct, i);
                self.expr(e)?;
                self.binop(op, &place.ty, &e.ty, line)?;
                self.c().struct_set(ct, i);
            }
            ExprKind::Index(b, k) => {
                let bv = self.val(&b.ty)?.unwrap();
                let kv = self.val(&k.ty)?.unwrap();
                let (bl, kl) = (self.fx.local(bv), self.fx.local(kv));
                self.expr(b)?;
                self.c().local_set(bl);
                self.expr(k)?;
                self.c().local_set(kl);
                let get = ExprKind::Index(Box::new(local_expr(bl, &b.ty)), Box::new(local_expr(kl, &k.ty)));
                let ge = Expr { id: 0, line, ty: place.ty.clone(), kind: get };
                let target = ge.clone();
                self.store(&target, |w| {
                    w.expr(&ge)?;
                    w.expr(e)?;
                    w.binop(op, &ge.ty, &e.ty, line)
                })?;
            }
            _ => return unsupported("this augmented assignment"),
        }
        Ok(())
    }
}

/// An expression reading a local of the function being compiled (a
/// pseudo-variable: `Const(usize::MAX - local)`).
pub(crate) fn local_expr(l: u32, ty: &Ty) -> Expr {
    Expr { id: 0, line: 0, ty: ty.clone(), kind: ExprKind::Const(usize::MAX - l as usize) }
}

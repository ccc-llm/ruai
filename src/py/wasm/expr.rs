//! Expressions.

use super::enc::{Block, Heap, Val};
use super::stmt::Cursor;
use super::{unsupported, Lab, Res, K, W};
use crate::py::ir::*;

impl<'a> W<'a> {
    /// Push the value of `e` (nothing for the type None).
    pub fn expr(&mut self, e: &Expr) -> Res<()> {
        use ExprKind as X;
        match &e.kind {
            X::Int(v) => {
                self.c().i64_const(*v);
            }
            X::Float(v) => {
                self.c().f64_const(*v);
            }
            X::Bool(b) => {
                self.c().i32_const(*b as i32);
            }
            X::Str(s) => self.str_const(s),
            X::NoneLit => {
                if let Some(v) = self.val(&e.ty)? {
                    self.zero(v);
                }
            }
            X::NoneVal => {
                let v = self.val(&e.ty)?.unwrap();
                self.zero(v);
            }
            X::SomeOf(x) => {
                self.expr(x)?;
                self.box_opt(&x.ty)?;
            }
            X::Upcast(x) => self.expr(x)?,
            X::Var(v) => {
                if let Some(&l) = self.fx.vars.get(v) {
                    self.c().local_get(l);
                }
            }
            X::Const(i) if *i > usize::MAX / 2 => {
                let l = (usize::MAX - *i) as u32;
                self.c().local_get(l);
            }
            X::Const(i) => {
                if let Some(g) = self.consts[*i] {
                    self.c().global_get(g);
                }
            }
            X::IsInst(x, c, var) => {
                let cty = self.val(&x.ty)?.unwrap();
                self.expr(x)?;
                let tmp = self.fx.local(cty);
                self.c().local_set(tmp);
                let ids = self.p.subtree(*c);
                let base = match &x.ty {
                    Ty::Class(b) | Ty::Obj(b) => self.class_ty[*b],
                    _ => return unsupported("isinstance on this type"),
                };
                for (i, id) in ids.iter().enumerate() {
                    self.c().local_get(tmp).struct_get(base, 0).i32_const(*id as i32).i32_eq();
                    if i > 0 {
                        self.c().i32_or();
                    }
                }
                if let Some(v) = var {
                    let l = self.fx.vars[v];
                    let t = self.class_ty[*c];
                    let r = self.fx.local(Val::I32);
                    self.c().local_tee(r).if_(Block::Empty).local_get(tmp).ref_cast(Heap::Idx(t)).local_set(l).end().local_get(r);
                }
            }
            X::IsSome(x, positive, var) => {
                let v = self.val(&x.ty)?.unwrap();
                self.expr(x)?;
                let tmp = self.fx.local(v);
                self.c().local_tee(tmp).ref_is_null();
                if *positive {
                    self.c().i32_eqz();
                }
                if let Some(var) = var {
                    if let Some(&l) = self.fx.vars.get(var) {
                        let inner = match &x.ty {
                            Ty::Optional(t) => t.as_ref().clone(),
                            t => t.clone(),
                        };
                        self.c().local_get(tmp).ref_is_null().i32_eqz().if_(Block::Empty).local_get(tmp);
                        self.unbox_opt(&inner)?;
                        self.c().local_set(l).end();
                    }
                }
            }
            X::SuperInit(f, args) => {
                self.c().local_get(0);
                let base = self.p.funcs[*f].class.unwrap();
                let t = self.class_ty[base];
                self.c().ref_cast(Heap::Idx(t));
                self.args(*f, args)?;
                let fi = self.user[*f];
                self.c().call(fi);
            }
            X::Bin(op, a, b) => {
                self.expr(a)?;
                self.expr(b)?;
                self.binop(*op, &a.ty, &b.ty, e.line)?;
            }
            X::Unary(op, x) => {
                match op {
                    UnOp::Not => {
                        self.cond(x)?;
                        self.c().i32_eqz();
                    }
                    UnOp::Pos => self.expr(x)?,
                    UnOp::Neg => match &x.ty {
                        Ty::Int => {
                            self.c().i64_const(0);
                            self.expr(x)?;
                            let f = self.rt_int_sub()?;
                            self.c().call(f);
                        }
                        Ty::Float => {
                            self.expr(x)?;
                            self.c().f64_neg();
                        }
                        Ty::Bool => {
                            self.c().i64_const(0);
                            self.expr(x)?;
                            self.c().i64_extend_i32_u().i64_sub();
                        }
                        t => return unsupported(format!("unary - on {:?}", t)),
                    },
                    UnOp::Invert => {
                        self.expr(x)?;
                        self.c().i64_const(-1).i64_xor();
                    }
                };
            }
            X::Cmp(first, rest) => self.compare(first, rest)?,
            X::And(xs) | X::Or(xs) => {
                let and = matches!(e.kind, X::And(_));
                if e.ty == Ty::Bool {
                    // short circuit on bools
                    self.cond(&xs[0])?;
                    for x in &xs[1..] {
                        self.c().if_(Block::Val(Val::I32));
                        self.labeled(Lab::Plain, |w| {
                            if and {
                                w.cond(x)?;
                                w.c().else_().i32_const(0);
                            } else {
                                w.c().i32_const(1).else_();
                                w.cond(x)?;
                            }
                            Ok(())
                        })?;
                        self.c().end();
                    }
                } else {
                    // the deciding operand's value
                    let v = self.val(&e.ty)?.unwrap();
                    let tmp = self.fx.local(v);
                    self.expr(&xs[0])?;
                    self.c().local_set(tmp);
                    for x in &xs[1..] {
                        let t = local_expr_ty(tmp, &e.ty);
                        self.cond(&t)?;
                        if !and {
                            self.c().i32_eqz();
                        }
                        self.c().if_(Block::Empty);
                        self.labeled(Lab::Plain, |w| {
                            w.expr(x)?;
                            w.c().local_set(tmp);
                            Ok(())
                        })?;
                        self.c().end();
                    }
                    self.c().local_get(tmp);
                }
            }
            X::IfExp(c, a, b) => {
                let v = self.val(&e.ty)?;
                self.cond(c)?;
                self.c().if_(match v {
                    Some(v) => Block::Val(v),
                    None => Block::Empty,
                });
                self.labeled(Lab::Plain, |w| {
                    w.expr(a)?;
                    w.c().else_();
                    w.expr(b)
                })?;
                self.c().end();
            }
            X::ToFloat(x) => {
                self.expr(x)?;
                match &x.ty {
                    Ty::Bool => self.c().f64_convert_i32_s(),
                    _ => self.c().f64_convert_i64_s(),
                };
            }
            X::Call(f, args) => {
                self.args(*f, args)?;
                let fi = self.user[*f];
                self.c().call(fi);
                self.void_result(*f, &e.ty)?;
            }
            X::Method(recv, f, args) => {
                self.expr(recv)?;
                self.args(*f, args)?;
                let fi = self.user[*f];
                self.c().call(fi);
                self.void_result(*f, &e.ty)?;
            }
            X::New(c, args) => self.new_obj(*c, args)?,
            X::Builtin(name, args) => self.builtin(name, args, e)?,
            X::Field(o, f) => {
                if matches!(o.ty, Ty::Exc) {
                    return unsupported("exception attributes");
                }
                let c = match &o.ty {
                    Ty::Class(c) | Ty::Obj(c) => *c,
                    t => return unsupported(format!("the attribute `{}` of {:?}", f, t)),
                };
                // a property
                if let Some(m) = self.p.classes[c].method(f) {
                    if self.p.funcs[m].property {
                        self.expr(o)?;
                        let fi = self.user[m];
                        self.c().call(fi);
                        return Ok(());
                    }
                }
                self.expr(o)?;
                let (ct, i) = (self.class_ty[c], self.field_index(c, f)?);
                self.c().struct_get(ct, i);
            }
            X::TupleField(o, i) => {
                let Ty::Tuple(ts) = &o.ty else { return unsupported("this tuple") };
                let tt = self.tuple_ty(ts)?;
                self.expr(o)?;
                self.c().struct_get(tt, *i as u32);
            }
            X::Index(b, k) => self.index(b, k, e)?,
            X::Slice(b, start, stop, step) => {
                self.expr(b)?;
                for x in [start, stop, step] {
                    match x {
                        Some(x) => {
                            self.expr(x)?;
                            self.c().i32_const(1);
                        }
                        None => {
                            self.c().i64_const(0).i32_const(0);
                        }
                    }
                }
                let f = match &b.ty {
                    Ty::List(t) => {
                        let k = self.kind(t)?;
                        self.rt_list_slice(k)?
                    }
                    Ty::Str => self.rt_str_slice()?,
                    t => return unsupported(format!("slicing {:?}", t)),
                };
                self.c().call(f);
            }
            X::List(xs) => {
                let Ty::List(t) = &e.ty else { unreachable!() };
                let t = t.as_ref().clone();
                self.list_lit(&t, xs)?;
            }
            X::Tuple(xs) => {
                let Ty::Tuple(ts) = &e.ty else { unreachable!() };
                let tt = self.tuple_ty(ts)?;
                for x in xs {
                    self.expr(x)?;
                    if self.val(&x.ty)?.is_none() {
                        self.c().i32_const(0);
                    }
                }
                self.c().struct_new(tt);
            }
            X::Set(xs) => {
                let Ty::Set(t) = &e.ty else { unreachable!() };
                let t = t.as_ref().clone();
                self.dict_new(&t)?;
                let d = self.fx.local(Val::Ref(Heap::Idx(self.t.dict)));
                self.c().local_set(d);
                for x in xs {
                    self.c().local_get(d);
                    self.expr(x)?;
                    self.box_eq(&t)?;
                    self.c().ref_null(Heap::Eq);
                    let f = self.rt_dict_set(&t, &Ty::None)?;
                    self.c().call(f);
                }
                self.c().local_get(d);
            }
            X::Dict(kvs) => {
                let Ty::Dict(kt, vt, _) = &e.ty else { unreachable!() };
                let (kt, vt) = (kt.as_ref().clone(), vt.as_ref().clone());
                self.dict_new(&kt)?;
                let d = self.fx.local(Val::Ref(Heap::Idx(self.t.dict)));
                self.c().local_set(d);
                for (k, v) in kvs {
                    self.c().local_get(d);
                    self.expr(k)?;
                    self.box_eq(&kt)?;
                    self.expr(v)?;
                    self.box_eq(&vt)?;
                    let f = self.rt_dict_set(&kt, &vt)?;
                    self.c().call(f);
                }
                self.c().local_get(d);
            }
            X::Comp(c) => self.comprehension(c, &e.ty)?,
            X::FStr(parts) => self.fstring(parts)?,
            X::Lambda(..) => return unsupported("lambda"),
            X::Iter(it) => {
                let et = self.iter_ty(it)?;
                self.collect(it, &et)?;
            }
        }
        Ok(())
    }

    /// A call's result of type None: the callee returns nothing.
    fn void_result(&mut self, _f: FuncId, _t: &Ty) -> Res<()> {
        Ok(())
    }

    /// Push the arguments of a call to `f` (skipping values of type None).
    pub fn args(&mut self, _f: FuncId, args: &[Expr]) -> Res<()> {
        for a in args {
            self.expr(a)?;
        }
        Ok(())
    }

    fn new_obj(&mut self, c: ClassId, args: &[Expr]) -> Res<()> {
        let cl = &self.p.classes[c];
        if cl.ext.is_some() {
            return unsupported(format!("`{}` (implemented in Rust)", cl.name));
        }
        let ct = self.class_ty[c];
        let fields: Vec<Ty> = cl.fields.iter().map(|f| f.ty.clone()).collect();
        self.c().i32_const(c as i32);
        match cl.init {
            Some(init) => {
                for t in &fields {
                    let v = self.val(t)?.unwrap_or(Val::I32);
                    self.zero(v);
                }
                self.c().struct_new(ct);
                let o = self.fx.local(Val::Ref(Heap::Idx(ct)));
                self.c().local_tee(o);
                self.args(init, args)?;
                let fi = self.user[init];
                self.c().call(fi).local_get(o);
            }
            None => {
                for (a, t) in args.iter().zip(&fields) {
                    self.expr(a)?;
                    if self.val(t)?.is_none() {
                        self.c().i32_const(0);
                    }
                }
                for t in &fields[args.len().min(fields.len())..] {
                    let v = self.val(t)?.unwrap_or(Val::I32);
                    self.zero(v);
                }
                self.c().struct_new(ct);
            }
        }
        Ok(())
    }

    /// Push the truth value (i32) of `e`.
    pub fn cond(&mut self, e: &Expr) -> Res<()> {
        self.expr(e)?;
        self.truth(&e.ty)
    }

    /// The value on the stack → its truth value.
    pub fn truth(&mut self, t: &Ty) -> Res<()> {
        match t {
            Ty::Bool => {}
            Ty::Int => {
                self.c().i64_const(0).i64_ne();
            }
            Ty::Float => {
                self.c().f64_const(0.0).f64_ne();
            }
            Ty::Str => {
                let s = self.t.str;
                self.c().struct_get(s, 1).i32_const(0).i32_ne();
            }
            Ty::List(e) => {
                let k = self.kind(e)?;
                let l = self.t.list[k as usize];
                self.c().struct_get(l, 1).i32_const(0).i32_ne();
            }
            Ty::Dict(..) | Ty::Set(_) => {
                let d = self.t.dict;
                self.c().struct_get(d, 5).i32_const(0).i32_ne();
            }
            Ty::Optional(_) => {
                self.c().ref_is_null().i32_eqz();
            }
            Ty::None => {
                self.c().i32_const(0);
            }
            Ty::Tuple(ts) => {
                let n = ts.len();
                self.c().drop().i32_const((n > 0) as i32);
            }
            Ty::Class(_) | Ty::Obj(_) | Ty::Exc => {
                self.c().drop().i32_const(1);
            }
            t => return unsupported(format!("the truth value of {:?}", t)),
        }
        Ok(())
    }

    /// The value on the stack (of type `t`, not None) → `T | None`'s representation.
    pub fn box_opt(&mut self, t: &Ty) -> Res<()> {
        match self.val(t)? {
            Some(Val::I64) => {
                let b = self.t.box_i64;
                self.c().struct_new(b);
            }
            Some(Val::F64) => {
                let b = self.t.box_f64;
                self.c().struct_new(b);
            }
            Some(Val::I32) => {
                self.c().ref_i31();
            }
            _ => {}
        }
        Ok(())
    }

    /// A non-null `T | None` on the stack → the `T`.
    pub fn unbox_opt(&mut self, t: &Ty) -> Res<()> {
        match self.val(t)? {
            Some(Val::I64) => {
                let b = self.t.box_i64;
                self.c().struct_get(b, 0);
            }
            Some(Val::F64) => {
                let b = self.t.box_f64;
                self.c().struct_get(b, 0);
            }
            Some(Val::I32) => {
                self.c().i31_get_s();
            }
            _ => {}
        }
        Ok(())
    }

    /// The value on the stack → an `eqref` (numbers boxed).
    pub fn box_eq(&mut self, t: &Ty) -> Res<()> {
        match self.val(t)? {
            Some(Val::I64) => {
                let b = self.t.box_i64;
                self.c().struct_new(b);
            }
            Some(Val::F64) => {
                let b = self.t.box_f64;
                self.c().struct_new(b);
            }
            Some(Val::I32) => {
                self.c().ref_i31();
            }
            Some(_) => {}
            None => {
                self.c().ref_null(Heap::Eq);
            }
        }
        Ok(())
    }

    /// An `eqref` on the stack → a value of type `t`.
    pub fn unbox_eq(&mut self, t: &Ty) -> Res<()> {
        match self.val(t)? {
            Some(Val::I64) => {
                let b = self.t.box_i64;
                self.c().ref_cast(Heap::Idx(b)).struct_get(b, 0);
            }
            Some(Val::F64) => {
                let b = self.t.box_f64;
                self.c().ref_cast(Heap::Idx(b)).struct_get(b, 0);
            }
            Some(Val::I32) => {
                self.c().ref_cast(Heap::I31).i31_get_s();
            }
            Some(Val::Ref(h)) | Some(Val::NonNull(h)) => {
                self.c().ref_cast(h);
            }
            None => {
                self.c().drop();
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ operators

    /// Apply `op` to the two values on the stack.
    pub fn binop(&mut self, op: BinOp, a: &Ty, b: &Ty, _line: u32) -> Res<()> {
        use BinOp::*;
        match (a, b) {
            (Ty::Int, Ty::Int) => {
                let f = match op {
                    Add => self.rt_int_add()?,
                    Sub => self.rt_int_sub()?,
                    Mul => self.rt_int_mul()?,
                    FloorDiv => self.rt_int_floordiv()?,
                    Mod => self.rt_int_mod()?,
                    Pow => self.rt_int_pow()?,
                    Div => {
                        // true division of ints
                        let f = self.rt_int_truediv()?;
                        self.c().call(f);
                        return Ok(());
                    }
                    Shl => self.rt_int_shl()?,
                    Shr => {
                        let f = self.rt_int_shr()?;
                        self.c().call(f);
                        return Ok(());
                    }
                    BitAnd => {
                        self.c().i64_and();
                        return Ok(());
                    }
                    BitOr => {
                        self.c().i64_or();
                        return Ok(());
                    }
                    BitXor => {
                        self.c().i64_xor();
                        return Ok(());
                    }
                };
                self.c().call(f);
            }
            (Ty::Float, Ty::Float) => match op {
                Add => {
                    self.c().f64_add();
                }
                Sub => {
                    self.c().f64_sub();
                }
                Mul => {
                    self.c().f64_mul();
                }
                Div => {
                    let f = self.rt_float_div()?;
                    self.c().call(f);
                }
                FloorDiv => {
                    let f = self.rt_float_floordiv()?;
                    self.c().call(f);
                }
                Mod => {
                    let f = self.rt_float_mod()?;
                    self.c().call(f);
                }
                Pow => {
                    let f = self.rt_float_pow()?;
                    self.c().call(f);
                }
                _ => return unsupported(format!("{:?} on floats", op)),
            },
            (Ty::Bool, Ty::Bool) if matches!(op, BitAnd | BitOr | BitXor) => {
                match op {
                    BitAnd => self.c().i32_and(),
                    BitOr => self.c().i32_or(),
                    _ => self.c().i32_xor(),
                };
            }
            (Ty::Str, Ty::Str) if op == Add => {
                let f = self.rt_str_concat()?;
                self.c().call(f);
            }
            (Ty::Str, Ty::Int) if op == Mul => {
                let f = self.rt_str_mul()?;
                self.c().call(f);
            }
            (Ty::Str, _) if op == Mod => return unsupported("%-formatting"),
            (Ty::Set(t), Ty::Set(_)) if matches!(op, BitOr | BitAnd | Sub | BitXor) => {
                let t = t.as_ref().clone();
                let f = self.rt_set_op(&t, op)?;
                self.c().call(f);
            }
            (Ty::List(t), Ty::List(_)) if op == Add => {
                let t = t.as_ref().clone();
                let f = self.rt_list_concat(&t)?;
                self.c().call(f);
            }
            (Ty::List(t), Ty::Int) if op == Mul => {
                let k = self.kind(t)?;
                let f = self.rt_list_mul(k)?;
                self.c().call(f);
            }
            (Ty::Int, Ty::Float) | (Ty::Float, Ty::Int) => {
                // the front converts mixed operands; keep this for safety
                return unsupported("mixed int / float operands");
            }
            _ => return unsupported(format!("{:?} on {:?} and {:?}", op, a, b)),
        }
        Ok(())
    }

    fn compare(&mut self, first: &Expr, rest: &[(CmpOp, Expr)]) -> Res<()> {
        // a < b < c: b evaluated once, short circuit
        self.expr(first)?;
        let mut prev_ty = first.ty.clone();
        let n = rest.len();
        let mut opened = 0;
        for (i, (op, x)) in rest.iter().enumerate() {
            self.expr(x)?;
            if i + 1 < n {
                let v = self.val(&x.ty)?.unwrap();
                let tmp = self.fx.local(v);
                self.c().local_tee(tmp);
                let pv = self.val(&prev_ty)?.unwrap();
                let a = self.fx.local(v);
                let pa = self.fx.local(pv);
                self.c().local_set(a).local_set(pa).local_get(pa).local_get(a);
                self.cmp_op(*op, &prev_ty, &x.ty)?;
                self.c().if_(Block::Val(Val::I32));
                self.fx.labels.push(Lab::Plain);
                self.c().local_get(tmp);
                opened += 1;
            } else {
                self.cmp_op(*op, &prev_ty, &x.ty)?;
            }
            prev_ty = x.ty.clone();
        }
        for _ in 0..opened {
            self.fx.labels.pop();
            self.c().else_().i32_const(0).end();
        }
        Ok(())
    }

    /// Compare the two values on the stack.
    pub fn cmp_op(&mut self, op: CmpOp, a: &Ty, b: &Ty) -> Res<()> {
        use CmpOp::*;
        if matches!(op, In | NotIn) {
            // `x in coll`: the collection is on top
            self.contains(a, b)?;
            if op == NotIn {
                self.c().i32_eqz();
            }
            return Ok(());
        }
        match (a, b) {
            (Ty::Int, Ty::Int) => {
                match op {
                    Eq => self.c().i64_eq(),
                    Ne => self.c().i64_ne(),
                    Lt => self.c().i64_lt_s(),
                    Le => self.c().i64_le_s(),
                    Gt => self.c().i64_gt_s(),
                    Ge => self.c().i64_ge_s(),
                    _ => unreachable!(),
                };
            }
            (Ty::Float, Ty::Float) => {
                match op {
                    Eq => self.c().f64_eq(),
                    Ne => self.c().f64_ne(),
                    Lt => self.c().f64_lt(),
                    Le => self.c().f64_le(),
                    Gt => self.c().f64_gt(),
                    Ge => self.c().f64_ge(),
                    _ => unreachable!(),
                };
            }
            (Ty::Bool, Ty::Bool) => {
                match op {
                    Eq => self.c().i32_eq(),
                    Ne => self.c().i32_ne(),
                    Lt => self.c().i32_lt_s(),
                    Le => self.c().i32_le_s(),
                    Gt => self.c().i32_gt_s(),
                    Ge => self.c().i32_ge_s(),
                    _ => unreachable!(),
                };
            }
            _ if matches!(op, Eq | Ne) => {
                let f = self.eq_fn(a)?;
                self.c().call(f);
                if op == Ne {
                    self.c().i32_eqz();
                }
            }
            _ => {
                // ordering: cmp → -1 / 0 / 1
                let f = self.cmp_fn(a)?;
                self.c().call(f).i32_const(0);
                match op {
                    Lt => self.c().i32_lt_s(),
                    Le => self.c().i32_le_s(),
                    Gt => self.c().i32_gt_s(),
                    _ => self.c().i32_ge_s(),
                };
            }
        }
        Ok(())
    }

    /// `x in coll` with x and coll on the stack.
    fn contains(&mut self, x: &Ty, coll: &Ty) -> Res<()> {
        match coll {
            Ty::Str => {
                let f = self.rt_str_find()?;
                // find(haystack, needle) — swap: the stack is needle, haystack
                let (a, b) = (self.fx.local(Val::Ref(Heap::Idx(self.t.str))), self.fx.local(Val::Ref(Heap::Idx(self.t.str))));
                self.c().local_set(b).local_set(a).local_get(b).local_get(a).i64_const(0).call(f).i64_const(0).i64_ge_s();
            }
            Ty::List(t) => {
                let t = t.as_ref().clone();
                let v = self.val(x)?.unwrap();
                let lv = self.val(coll)?.unwrap();
                let (xl, ll) = (self.fx.local(v), self.fx.local(lv));
                self.c().local_set(ll).local_set(xl).local_get(ll).local_get(xl);
                let f = self.rt_list_index(&t)?;
                self.c().i32_const(0).call(f).i64_const(0).i64_ge_s();
            }
            Ty::Dict(k, _, _) | Ty::Set(k) => {
                let k = k.as_ref().clone();
                let v = self.val(x)?.unwrap();
                let lv = self.val(coll)?.unwrap();
                let (xl, ll) = (self.fx.local(v), self.fx.local(lv));
                self.c().local_set(ll).local_set(xl).local_get(ll).local_get(xl);
                self.box_eq(&k)?;
                let f = self.rt_dict_find()?;
                self.c().call(f).i32_const(0).i32_ge_s();
            }
            Ty::Tuple(ts) => {
                // compare with each element
                let tt = self.tuple_ty(ts)?;
                let v = self.val(x)?.unwrap();
                let (xl, tl) = (self.fx.local(v), self.fx.local(Val::Ref(Heap::Idx(tt))));
                self.c().local_set(tl).local_set(xl).i32_const(0);
                for (i, t) in ts.iter().enumerate() {
                    self.c().local_get(xl).local_get(tl).struct_get(tt, i as u32);
                    self.cmp_op(CmpOp::Eq, t, t)?;
                    self.c().i32_or();
                }
            }
            t => return unsupported(format!("`in` on {:?}", t)),
        }
        Ok(())
    }

    // ------------------------------------------------------------ containers

    /// A list literal.
    pub fn list_lit(&mut self, t: &Ty, xs: &[Expr]) -> Res<()> {
        let k = self.kind(t)?;
        let new = self.rt_list_new(k)?;
        self.c().i32_const(xs.len() as i32).call(new);
        if xs.is_empty() {
            return Ok(());
        }
        let l = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
        self.c().local_set(l);
        for x in xs {
            self.c().local_get(l);
            self.expr(x)?;
            self.elem_in(t)?;
            let f = self.rt_list_append(k)?;
            self.c().call(f);
        }
        self.c().local_get(l);
        Ok(())
    }

    /// A value of type `t` on the stack → a list element.
    pub fn elem_in(&mut self, t: &Ty) -> Res<()> {
        if self.kind(t)? == K::Ref {
            if self.val(t)?.is_none() {
                self.c().ref_null(Heap::Eq);
            }
        } else if self.val(t)?.is_none() {
            self.c().i32_const(0);
        }
        Ok(())
    }

    /// `list[i64]` on the stack → the element (IndexError when out of range).
    pub fn list_get(&mut self, t: &Ty) -> Res<()> {
        let k = self.kind(t)?;
        let f = self.rt_list_get(k)?;
        self.c().call(f);
        if k == K::Ref {
            if let Some(v) = self.val(t)? {
                let h = match v {
                    Val::Ref(h) | Val::NonNull(h) => h,
                    _ => unreachable!(),
                };
                self.c().ref_cast(h);
            } else {
                self.c().drop();
            }
        } else if self.val(t)?.is_none() {
            self.c().drop();
        }
        Ok(())
    }

    fn index(&mut self, b: &Expr, k: &Expr, e: &Expr) -> Res<()> {
        match &b.ty {
            Ty::List(t) => {
                let t = t.as_ref().clone();
                self.expr(b)?;
                self.expr(k)?;
                self.list_get(&t)?;
            }
            Ty::Str => {
                self.expr(b)?;
                self.expr(k)?;
                let f = self.rt_str_index()?;
                self.c().call(f);
            }
            Ty::Dict(kt, vt, dk) => {
                let (kt, vt) = (kt.as_ref().clone(), vt.as_ref().clone());
                if *dk != DK::Plain && !self.p.present.contains(&e.id) {
                    // defaultdict / Counter: a missing key gets the default value
                    self.expr(b)?;
                    self.expr(k)?;
                    self.box_eq(&kt)?;
                    let f = self.rt_dict_get_default(&kt, &vt, *dk)?;
                    self.c().call(f);
                } else {
                    self.expr(b)?;
                    self.expr(k)?;
                    self.box_eq(&kt)?;
                    let f = self.rt_dict_get(&kt)?;
                    self.c().call(f);
                }
                self.unbox_eq(&vt)?;
            }
            Ty::Tuple(ts) => {
                let ExprKind::Int(i) = k.kind else { return unsupported("a tuple index that is not a constant") };
                let n = ts.len() as i64;
                let i = if i < 0 { i + n } else { i };
                let tt = self.tuple_ty(ts)?;
                self.expr(b)?;
                self.c().struct_get(tt, i as u32);
            }
            t => return unsupported(format!("indexing {:?}", t)),
        }
        Ok(())
    }

    pub fn dict_new(&mut self, kt: &Ty) -> Res<()> {
        let h = self.hash_eq_fn(kt)?;
        let e = self.eq_eq_fn(kt)?;
        let f = self.rt_dict_new()?;
        self.m.declared.extend([h, e]);
        self.c().ref_func(h).ref_func(e).call(f);
        Ok(())
    }

    // ------------------------------------------------------------ iteration

    /// The element type of an iterator.
    pub fn iter_ty(&mut self, it: &Iter) -> Res<Ty> {
        Ok(match it {
            Iter::Range(..) => Ty::Int,
            Iter::Over(e) => match &e.ty {
                Ty::List(t) | Ty::Set(t) => t.as_ref().clone(),
                Ty::Dict(k, _, _) => k.as_ref().clone(),
                Ty::Str => Ty::Str,
                Ty::Tuple(ts) => ts.first().cloned().unwrap_or(Ty::Int),
                t => return unsupported(format!("iterating over {:?}", t)),
            },
            Iter::Keys(e) => match &e.ty {
                Ty::Dict(k, _, _) => k.as_ref().clone(),
                _ => unreachable!(),
            },
            Iter::Values(e) => match &e.ty {
                Ty::Dict(_, v, _) => v.as_ref().clone(),
                _ => unreachable!(),
            },
            Iter::Items(e) => match &e.ty {
                Ty::Dict(k, v, _) => Ty::Tuple(vec![k.as_ref().clone(), v.as_ref().clone()]),
                _ => unreachable!(),
            },
            Iter::Enumerate(i, _) => Ty::Tuple(vec![Ty::Int, self.iter_ty(i)?]),
            Iter::Zip(is) => {
                let mut ts = vec![];
                for i in is {
                    ts.push(self.iter_ty(i)?);
                }
                Ty::Tuple(ts)
            }
            Iter::Reversed(i) => self.iter_ty(i)?,
            // a generator expression: its type, or its element type
            Iter::Gen(e) => match (&e.kind, &e.ty) {
                (_, Ty::Gen(t)) => t.as_ref().clone(),
                (ExprKind::Comp(_), t) => t.clone(),
                (_, Ty::List(t)) => t.as_ref().clone(),
                (_, t) => return unsupported(format!("iterating over {:?}", t)),
            },
        })
    }

    /// Set up a loop over `it`.
    pub fn iter_init(&mut self, it: &Iter) -> Res<Cursor> {
        Ok(match it {
            Iter::Range(a, b, step) => {
                let (i, stop, st) = (self.fx.local(Val::I64), self.fx.local(Val::I64), self.fx.local(Val::I64));
                self.expr(a)?;
                self.c().local_set(i);
                self.expr(b)?;
                self.c().local_set(stop);
                let up = match step {
                    None => {
                        self.c().i64_const(1).local_set(st);
                        Some(true)
                    }
                    Some(s) => {
                        let known = match s.kind {
                            ExprKind::Int(v) if v != 0 => Some(v > 0),
                            _ => None,
                        };
                        self.expr(s)?;
                        self.c().local_tee(st).i64_eqz().if_(Block::Empty);
                        self.labeled(Lab::Plain, |w| w.raise_msg("ValueError", "range() arg 3 must not be zero"))?;
                        self.c().end();
                        known
                    }
                };
                Cursor::Range { i, stop, step: st, up }
            }
            Iter::Reversed(inner) => match inner.as_ref() {
                Iter::Range(a, b, None) => {
                    // reversed(range(a, b)): b - 1 down to a
                    let (i, stop, st) = (self.fx.local(Val::I64), self.fx.local(Val::I64), self.fx.local(Val::I64));
                    self.expr(b)?;
                    self.c().i64_const(1).i64_sub().local_set(i);
                    self.expr(a)?;
                    self.c().i64_const(1).i64_sub().local_set(stop);
                    self.c().i64_const(-1).local_set(st);
                    Cursor::Range { i, stop, step: st, up: Some(false) }
                }
                Iter::Over(e) if matches!(e.ty, Ty::List(_)) => {
                    let Ty::List(t) = &e.ty else { unreachable!() };
                    let t = t.as_ref().clone();
                    let k = self.kind(&t)?;
                    let xs = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
                    let i = self.fx.local(Val::I32);
                    self.expr(e)?;
                    let lt = self.t.list[k as usize];
                    self.c().local_tee(xs).struct_get(lt, 1).local_set(i);
                    Cursor::List { xs, i, k, ty: t, rev: true }
                }
                _ => {
                    // materialize, then walk backwards
                    let t = self.iter_ty(inner)?;
                    self.collect(inner, &t)?;
                    let k = self.kind(&t)?;
                    let xs = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
                    let i = self.fx.local(Val::I32);
                    let lt = self.t.list[k as usize];
                    self.c().local_tee(xs).struct_get(lt, 1).local_set(i);
                    Cursor::List { xs, i, k, ty: t, rev: true }
                }
            },
            Iter::Gen(e) if matches!(e.kind, ExprKind::Comp(_)) => {
                let ExprKind::Comp(c) = &e.kind else { unreachable!() };
                let t = self.iter_ty(it)?;
                let k = self.kind(&t)?;
                let xs = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
                let i = self.fx.local(Val::I32);
                self.comprehension_as_list(c, &t)?;
                self.c().local_set(xs).i32_const(0).local_set(i);
                Cursor::List { xs, i, k, ty: t, rev: false }
            }
            Iter::Over(e) | Iter::Gen(e) => match &e.ty {
                Ty::List(t) | Ty::Gen(t) => {
                    let t = t.as_ref().clone();
                    let k = self.kind(&t)?;
                    let xs = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
                    let i = self.fx.local(Val::I32);
                    if let (ExprKind::Comp(c), Ty::Gen(_)) = (&e.kind, &e.ty) {
                        // a generator expression, computed at once
                        self.comprehension_as_list(c, &t)?;
                    } else {
                        self.expr(e)?;
                    }
                    self.c().local_set(xs).i32_const(0).local_set(i);
                    Cursor::List { xs, i, k, ty: t, rev: false }
                }
                Ty::Str => {
                    let (s, pos) = (self.fx.local(Val::Ref(Heap::Idx(self.t.str))), self.fx.local(Val::I32));
                    self.expr(e)?;
                    self.c().local_set(s).i32_const(0).local_set(pos);
                    Cursor::Str { s, pos }
                }
                Ty::Dict(k, v, _) => self.dict_cursor(e, k, v, 0)?,
                Ty::Set(k) => self.dict_cursor(e, k, &Ty::None, 0)?,
                Ty::Tuple(ts) => {
                    // a homogeneous tuple: as a list
                    let t = ts.first().cloned().unwrap_or(Ty::Int);
                    let tt = self.tuple_ty(ts)?;
                    let tl = self.fx.local(Val::Ref(Heap::Idx(tt)));
                    self.expr(e)?;
                    self.c().local_set(tl);
                    let xs: Vec<Expr> = (0..ts.len()).map(|i| Expr { id: 0, line: 0, ty: t.clone(), kind: ExprKind::TupleField(Box::new(local_expr_ty(tl, &e.ty)), i) }).collect();
                    self.list_lit(&t, &xs)?;
                    let k = self.kind(&t)?;
                    let xl = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
                    let i = self.fx.local(Val::I32);
                    self.c().local_set(xl).i32_const(0).local_set(i);
                    Cursor::List { xs: xl, i, k, ty: t, rev: false }
                }
                t => return unsupported(format!("iterating over {:?}", t)),
            },
            Iter::Keys(e) => {
                let Ty::Dict(k, v, _) = &e.ty else { unreachable!() };
                self.dict_cursor(e, k, v, 0)?
            }
            Iter::Values(e) => {
                let Ty::Dict(k, v, _) = &e.ty else { unreachable!() };
                self.dict_cursor(e, k, v, 1)?
            }
            Iter::Items(e) => {
                let Ty::Dict(k, v, _) = &e.ty else { unreachable!() };
                self.dict_cursor(e, k, v, 2)?
            }
            Iter::Enumerate(inner, start) => {
                let ty = self.iter_ty(it)?;
                let cur = self.iter_init(inner)?;
                let n = self.fx.local(Val::I64);
                match start {
                    Some(s) => self.expr(s)?,
                    None => {
                        self.c().i64_const(0);
                    }
                }
                self.c().local_set(n);
                Cursor::Enumerate { inner: Box::new(cur), n, ty }
            }
            Iter::Zip(is) => {
                let ty = self.iter_ty(it)?;
                let mut inner = vec![];
                for i in is {
                    inner.push(self.iter_init(i)?);
                }
                Cursor::Zip { inner, ty }
            }
        })
    }

    fn dict_cursor(&mut self, e: &Expr, k: &Ty, v: &Ty, mode: u8) -> Res<Cursor> {
        let (d, i) = (self.fx.local(Val::Ref(Heap::Idx(self.t.dict))), self.fx.local(Val::I32));
        self.expr(e)?;
        self.c().local_set(d).i32_const(0).local_set(i);
        Ok(Cursor::Dict { d, i, mode, k: k.clone(), v: v.clone() })
    }

    /// Push the next element, or `br` to `exit` (a depth counted from the
    /// code's position) when the iterator is exhausted.
    pub fn iter_next(&mut self, cur: &Cursor, exit: u32) -> Res<()> {
        match cur {
            Cursor::Range { i, stop, step, up } => {
                match up {
                    Some(true) => {
                        self.c().local_get(*i).local_get(*stop).i64_ge_s().br_if(exit);
                    }
                    Some(false) => {
                        self.c().local_get(*i).local_get(*stop).i64_le_s().br_if(exit);
                    }
                    None => {
                        // step > 0 ? i >= stop : i <= stop
                        self.c().local_get(*i).local_get(*stop).i64_ge_s();
                        self.c().local_get(*i).local_get(*stop).i64_le_s();
                        self.c().local_get(*step).i64_const(0).i64_gt_s().select().br_if(exit);
                    }
                }
                self.c().local_get(*i).local_get(*i).local_get(*step).i64_add().local_set(*i);
            }
            Cursor::List { xs, i, k, ty, rev } => {
                let lt = self.t.list[*k as usize];
                if *rev {
                    self.c().local_get(*i).i32_eqz().br_if(exit);
                    self.c().local_get(*i).i32_const(1).i32_sub().local_set(*i);
                    // the list may have shrunk
                    self.c().local_get(*xs).local_get(*i).i64_extend_i32_s();
                } else {
                    self.c().local_get(*i).local_get(*xs).struct_get(lt, 1).i32_ge_s().br_if(exit);
                    self.c().local_get(*xs).local_get(*i).i64_extend_i32_s();
                    self.c().local_get(*i).i32_const(1).i32_add().local_set(*i);
                }
                let ty = ty.clone();
                self.list_get(&ty)?;
            }
            Cursor::Str { s, pos } => {
                let st = self.t.str;
                let bytes = self.t.bytes;
                self.c().local_get(*pos).local_get(*s).struct_get(st, 0).array_len().i32_ge_u().br_if(exit);
                let f = self.rt_str_char_at()?;
                // char_at(s, pos) → (char, next pos)
                self.c().local_get(*s).local_get(*pos).call(f).local_set(*pos);
                let _ = bytes;
            }
            Cursor::Dict { d, i, mode, k, v } => {
                // skip deleted entries
                let f = self.rt_dict_next()?;
                self.c().local_get(*d).local_get(*i).call(f).local_tee(*i).i32_const(0).i32_lt_s().br_if(exit);
                let (dt, ae) = (self.t.dict, self.t.arr[3]);
                let (k, v) = (k.clone(), v.clone());
                match mode {
                    0 => {
                        self.c().local_get(*d).struct_get(dt, 0).local_get(*i).array_get(ae);
                        self.unbox_eq(&k)?;
                    }
                    1 => {
                        self.c().local_get(*d).struct_get(dt, 1).local_get(*i).array_get(ae);
                        self.unbox_eq(&v)?;
                    }
                    _ => {
                        self.c().local_get(*d).struct_get(dt, 0).local_get(*i).array_get(ae);
                        self.unbox_eq(&k)?;
                        self.c().local_get(*d).struct_get(dt, 1).local_get(*i).array_get(ae);
                        self.unbox_eq(&v)?;
                        if self.val(&v)?.is_none() {
                            self.c().i32_const(0);
                        }
                        let tt = self.tuple_ty(&[k, v])?;
                        self.c().struct_new(tt);
                    }
                }
                self.c().local_get(*i).i32_const(1).i32_add().local_set(*i);
            }
            Cursor::Enumerate { inner, n, ty } => {
                let Ty::Tuple(ts) = ty else { unreachable!() };
                let tt = self.tuple_ty(ts)?;
                let ev = self.val(&ts[1])?;
                // the element first (it may leave), then the count
                self.iter_next(inner, exit)?;
                let tmp = ev.map(|v| self.fx.local(v));
                if let Some(t) = tmp {
                    self.c().local_set(t);
                }
                self.c().local_get(*n);
                match tmp {
                    Some(t) => self.c().local_get(t),
                    None => self.c().i32_const(0),
                };
                self.c().struct_new(tt).local_get(*n).i64_const(1).i64_add().local_set(*n);
            }
            Cursor::Zip { inner, ty } => {
                let Ty::Tuple(ts) = ty else { unreachable!() };
                let tt = self.tuple_ty(ts)?;
                let mut tmps = vec![];
                for (c, t) in inner.iter().zip(ts) {
                    self.iter_next(c, exit)?;
                    let v = self.val(t)?.unwrap_or(Val::I32);
                    let l = self.fx.local(v);
                    self.c().local_set(l);
                    tmps.push(l);
                }
                for l in tmps {
                    self.c().local_get(l);
                }
                self.c().struct_new(tt);
            }
        }
        Ok(())
    }

    /// Materialize an iterator as a list of `t`.
    pub fn collect(&mut self, it: &Iter, t: &Ty) -> Res<()> {
        if let Iter::Gen(e) = it {
            if let ExprKind::Comp(c) = &e.kind {
                return self.comprehension_as_list(c, t);
            }
        }
        let k = self.kind(t)?;
        let new = self.rt_list_new(k)?;
        self.c().i32_const(0).call(new);
        let l = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
        self.c().local_set(l);
        let cur = self.iter_init(it)?;
        let app = self.rt_list_append(k)?;
        self.c().block(Block::Empty).loop_(Block::Empty);
        self.fx.labels.extend([Lab::Plain, Lab::Plain]);
        let t2 = t.clone();
        let r = (|| -> Res<()> {
            let v = self.val(&t2)?;
            let tmp = v.map(|v| self.fx.local(v));
            self.iter_next(&cur, 1)?;
            if let Some(tmp) = tmp {
                self.c().local_set(tmp).local_get(l).local_get(tmp);
            } else {
                self.c().local_get(l);
            }
            self.elem_in(&t2)?;
            self.c().call(app).br(0);
            Ok(())
        })();
        self.fx.labels.truncate(self.fx.labels.len() - 2);
        r?;
        self.c().end().end().local_get(l);
        Ok(())
    }

    fn comprehension(&mut self, c: &Comp, t: &Ty) -> Res<()> {
        match (c.kind, t) {
            (CompKind::List, Ty::List(et)) | (CompKind::Gen, Ty::Gen(et)) => {
                let et = et.as_ref().clone();
                self.comprehension_as_list(c, &et)
            }
            // a generator expression typed by its elements: computed as a list
            (CompKind::Gen, et) => {
                let et = et.clone();
                self.comprehension_as_list(c, &et)
            }
            (CompKind::Set, Ty::Set(et)) => {
                let et = et.as_ref().clone();
                self.dict_new(&et)?;
                let d = self.fx.local(Val::Ref(Heap::Idx(self.t.dict)));
                self.c().local_set(d);
                let set = self.rt_dict_set(&et, &Ty::None)?;
                self.comp_loops(c, 0, &mut |w| {
                    w.c().local_get(d);
                    w.expr(&c.elt)?;
                    w.box_eq(&et)?;
                    w.c().ref_null(Heap::Eq).call(set);
                    Ok(())
                })?;
                self.c().local_get(d);
                Ok(())
            }
            (CompKind::Dict, Ty::Dict(kt, vt, _)) => {
                let (kt, vt) = (kt.as_ref().clone(), vt.as_ref().clone());
                self.dict_new(&kt)?;
                let d = self.fx.local(Val::Ref(Heap::Idx(self.t.dict)));
                self.c().local_set(d);
                let set = self.rt_dict_set(&kt, &vt)?;
                let val = c.val.clone().unwrap();
                self.comp_loops(c, 0, &mut |w| {
                    w.c().local_get(d);
                    w.expr(&c.elt)?;
                    w.box_eq(&kt)?;
                    w.expr(&val)?;
                    w.box_eq(&vt)?;
                    w.c().call(set);
                    Ok(())
                })?;
                self.c().local_get(d);
                Ok(())
            }
            _ => unsupported("this comprehension"),
        }
    }

    pub fn comprehension_as_list(&mut self, c: &Comp, et: &Ty) -> Res<()> {
        let k = self.kind(et)?;
        let new = self.rt_list_new(k)?;
        self.c().i32_const(0).call(new);
        let l = self.fx.local(Val::Ref(Heap::Idx(self.t.list[k as usize])));
        self.c().local_set(l);
        let app = self.rt_list_append(k)?;
        let et = et.clone();
        self.comp_loops(c, 0, &mut |w| {
            w.c().local_get(l);
            w.expr(&c.elt)?;
            w.elem_in(&et)?;
            w.c().call(app);
            Ok(())
        })?;
        self.c().local_get(l);
        Ok(())
    }

    fn comp_loops(&mut self, c: &Comp, g: usize, body: &mut dyn FnMut(&mut W<'a>) -> Res<()>) -> Res<()> {
        if g == c.gens.len() {
            return body(self);
        }
        let gen = &c.gens[g];
        let cur = self.iter_init(&gen.iter)?;
        let et = self.iter_ty(&gen.iter)?;
        self.c().block(Block::Empty).loop_(Block::Empty);
        self.fx.labels.extend([Lab::Plain, Lab::Plain]);
        let r = (|| -> Res<()> {
            self.iter_next(&cur, 1)?;
            self.assign(&gen.target, &et)?;
            // conditions: skip to the next element
            self.c().block(Block::Empty);
            self.fx.labels.push(Lab::Plain);
            let r = (|| -> Res<()> {
                for cnd in &gen.conds {
                    self.cond(cnd)?;
                    self.c().i32_eqz().br_if(0);
                }
                self.comp_loops(c, g + 1, body)
            })();
            self.fx.labels.pop();
            r?;
            self.c().end().br(0);
            Ok(())
        })();
        self.fx.labels.truncate(self.fx.labels.len() - 2);
        r?;
        self.c().end().end();
        Ok(())
    }

    /// `raise K(literal message)`
    pub fn raise_msg(&mut self, kind: &str, msg: &str) -> Res<()> {
        let m = Expr { id: 0, line: 0, ty: Ty::Str, kind: ExprKind::Str(msg.to_string()) };
        self.raise(kind, Some(&m))
    }
}

/// An expression reading a local (see `stmt::local_expr`).
pub(crate) fn local_expr_ty(l: u32, ty: &Ty) -> Expr {
    super::stmt::local_expr(l, ty)
}

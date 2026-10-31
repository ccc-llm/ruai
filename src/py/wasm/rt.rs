//! The runtime: functions generated into the module on first use (numbers
//! with Python's semantics, strings, lists, dicts, exceptions, host I/O).

use super::enc::{Block, Heap, Val};
use super::{Res, H, K, W};
use crate::py::ir::*;

/// Locals of a runtime function: parameters first.
macro_rules! params {
    ($w:expr, $($v:expr),*) => {{
        $w.fx.params = vec![$($v),*];
    }};
}

const INT_OVERFLOW: &str = "integer overflow (py2rs: int is a 64-bit integer)";

impl<'a> W<'a> {
    fn sref(&self) -> Val {
        Val::Ref(Heap::Idx(self.t.str))
    }
    fn lref(&self, k: K) -> Val {
        Val::Ref(Heap::Idx(self.t.list[k as usize]))
    }
    fn kval(&self, k: K) -> Val {
        match k {
            K::I64 => Val::I64,
            K::F64 => Val::F64,
            K::I32 => Val::I32,
            K::Ref => Val::Ref(Heap::Eq),
        }
    }

    /// The message (a str) on the stack → raise `kind`.
    pub fn throw_kind(&mut self, kind: &str) {
        let m = self.fx.local(self.sref());
        self.c().local_set(m);
        let chain = crate::py::exc::chain(self.p, kind);
        let ids: Vec<i32> = chain.iter().map(|k| self.exc_id(k)).collect();
        self.str_const(kind);
        let arr = self.t.arr_i32;
        for id in &ids {
            self.c().i32_const(*id);
        }
        let (exc, tag) = (self.t.exc, self.tag);
        self.c().array_new_fixed(arr, ids.len() as u32).local_get(m).ref_null(Heap::Eq).struct_new(exc).throw(tag);
    }

    /// raise `kind` with a literal message.
    pub fn throw_lit(&mut self, kind: &str, msg: &str) {
        self.str_const(msg);
        self.throw_kind(kind);
    }

    // ------------------------------------------------------------ memory and host

    /// mem_ensure(n): the memory holds at least n bytes.
    pub fn rt_mem_ensure(&mut self) -> Res<u32> {
        self.rt_fn("mem_ensure", |w| {
            params!(w, Val::I32);
            // if size*64K < n: grow by (n - size*64K + 64K-1) / 64K pages
            w.c().memory_size().i32_const(16).i32_shl().local_get(0).i32_lt_u().if_(Block::Empty);
            w.c().local_get(0).memory_size().i32_const(16).i32_shl().i32_sub().i32_const(65535).i32_add().i32_const(16).i32_shr_u().memory_grow().drop();
            w.c().end();
            Ok(vec![])
        })
    }

    /// str_arg(s): pass a string argument to the host.
    pub fn rt_str_arg(&mut self) -> Res<u32> {
        let ens = self.rt_mem_ensure()?;
        self.rt_fn("str_arg", |w| {
            params!(w, w.sref());
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (b, n, i) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(st, 0).local_tee(b).array_len().local_tee(n).call(ens);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(i).local_get(b).local_get(i).array_get_u(bytes).i32_store8(0);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            let arg = w.host(H::Arg);
            w.c().i32_const(0).local_get(n).call(arg);
            Ok(vec![])
        })
    }

    /// take_str(len): the host's pending result as a str.
    pub fn rt_take_str(&mut self) -> Res<u32> {
        let ens = self.rt_mem_ensure()?;
        self.rt_fn("take_str", |w| {
            params!(w, Val::I32);
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (b, i, n) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I32));
            let take = w.host(H::Take);
            w.c().local_get(0).call(ens).i32_const(0).call(take);
            w.c().local_get(0).array_new_default(bytes).local_set(b);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(0).i32_ge_u().br_if(1);
            w.c().local_get(b).local_get(i).local_get(i).i32_load8_u(0).array_set(bytes);
            // count code points: bytes that are not 10xxxxxx
            w.c().local_get(n).local_get(i).i32_load8_u(0).i32_const(0xc0).i32_and().i32_const(0x80).i32_ne().i32_add().local_set(n);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            w.c().local_get(b).ref_as_non_null().local_get(n).struct_new(st);
            Ok(vec![w.sref()])
        })
    }

    /// out(s, fd): write a string.
    pub fn rt_out(&mut self) -> Res<u32> {
        let arg = self.rt_str_arg()?;
        self.rt_fn("out", |w| {
            params!(w, w.sref(), Val::I32);
            let write = w.host(H::Write);
            w.c().local_get(0).call(arg).local_get(1).call(write);
            Ok(vec![])
        })
    }

    // ------------------------------------------------------------ exceptions

    /// exc_match(e, ids): does e's class chain contain one of ids?
    pub fn rt_exc_match(&mut self) -> Res<u32> {
        self.rt_fn("exc_match", |w| {
            let (exc, arr) = (w.t.exc, w.t.arr_i32);
            params!(w, Val::Ref(Heap::Idx(exc)), Val::Ref(Heap::Idx(arr)));
            let (i, j, ch) = (w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::Ref(Heap::Idx(arr))));
            w.c().local_get(0).struct_get(exc, 1).local_set(ch);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(1).array_len().i32_ge_u().br_if(1);
            w.c().i32_const(0).local_set(j);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(j).local_get(ch).array_len().i32_ge_u().br_if(1);
            w.c().local_get(ch).local_get(j).array_get(arr).local_get(1).local_get(i).array_get(arr).i32_eq().if_(Block::Empty).i32_const(1).return_().end();
            w.c().local_get(j).i32_const(1).i32_add().local_set(j).br(0);
            w.c().end().end();
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().i32_const(0);
            Ok(vec![Val::I32])
        })
    }

    /// report(e): the traceback of an uncaught exception, on stderr.
    pub fn rt_report(&mut self) -> Res<u32> {
        let out = self.rt_out()?;
        let cat = self.rt_str_concat()?;
        self.rt_fn("report", |w| {
            let (exc, st) = (w.t.exc, w.t.str);
            params!(w, Val::Ref(Heap::Idx(exc)));
            w.str_const("Traceback (most recent call last):\n");
            w.c().i32_const(2).call(out);
            w.c().local_get(0).struct_get(exc, 0).i32_const(2).call(out);
            w.c().local_get(0).struct_get(exc, 2).struct_get(st, 1).if_(Block::Empty);
            w.str_const(": ");
            w.c().local_get(0).struct_get(exc, 2).call(cat).i32_const(2).call(out).end();
            w.str_const("\n");
            w.c().i32_const(2).call(out);
            Ok(vec![])
        })
    }

    // ------------------------------------------------------------ ints

    fn overflow_if(&mut self) {
        self.c().if_(Block::Empty);
        self.throw_lit("OverflowError", INT_OVERFLOW);
        self.c().end();
    }

    pub fn rt_int_add(&mut self) -> Res<u32> {
        self.rt_fn("int_add", |w| {
            params!(w, Val::I64, Val::I64);
            let r = w.fx.local(Val::I64);
            w.c().local_get(0).local_get(1).i64_add().local_set(r);
            // overflow: both operands' signs differ from the result's
            w.c().local_get(0).local_get(r).i64_xor().local_get(1).local_get(r).i64_xor().i64_and().i64_const(0).i64_lt_s();
            w.overflow_if();
            w.c().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_int_sub(&mut self) -> Res<u32> {
        self.rt_fn("int_sub", |w| {
            params!(w, Val::I64, Val::I64);
            let r = w.fx.local(Val::I64);
            w.c().local_get(0).local_get(1).i64_sub().local_set(r);
            w.c().local_get(0).local_get(1).i64_xor().local_get(0).local_get(r).i64_xor().i64_and().i64_const(0).i64_lt_s();
            w.overflow_if();
            w.c().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_int_mul(&mut self) -> Res<u32> {
        self.rt_fn("int_mul", |w| {
            params!(w, Val::I64, Val::I64);
            let r = w.fx.local(Val::I64);
            w.c().local_get(0).local_get(1).i64_mul().local_set(r);
            // a == -1: overflow iff b == MIN; a != 0: overflow iff r / a != b
            w.c().local_get(0).i64_const(-1).i64_eq().if_(Block::Val(Val::I32));
            w.c().local_get(1).i64_const(i64::MIN).i64_eq();
            w.c().else_().local_get(0).i64_eqz().if_(Block::Val(Val::I32)).i32_const(0);
            w.c().else_().local_get(r).local_get(0).i64_div_s().local_get(1).i64_ne().end();
            w.c().end();
            w.overflow_if();
            w.c().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    fn zero_div_if(&mut self, local: u32, val: Val) {
        match val {
            Val::I64 => self.c().local_get(local).i64_eqz(),
            _ => self.c().local_get(local).f64_const(0.0).f64_eq(),
        };
        self.c().if_(Block::Empty);
        self.throw_lit("ZeroDivisionError", "division by zero");
        self.c().end();
    }

    pub fn rt_int_floordiv(&mut self) -> Res<u32> {
        self.rt_fn("int_floordiv", |w| {
            params!(w, Val::I64, Val::I64);
            w.zero_div_if(1, Val::I64);
            w.c().local_get(0).i64_const(i64::MIN).i64_eq().local_get(1).i64_const(-1).i64_eq().i32_and();
            w.overflow_if();
            let q = w.fx.local(Val::I64);
            w.c().local_get(0).local_get(1).i64_div_s().local_set(q);
            // round toward -inf: a remainder and different signs
            w.c().local_get(0).local_get(1).i64_rem_s().i64_const(0).i64_ne();
            w.c().local_get(0).i64_const(0).i64_lt_s().local_get(1).i64_const(0).i64_lt_s().i32_ne().i32_and();
            w.c().if_(Block::Empty).local_get(q).i64_const(1).i64_sub().local_set(q).end();
            w.c().local_get(q);
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_int_mod(&mut self) -> Res<u32> {
        self.rt_fn("int_mod", |w| {
            params!(w, Val::I64, Val::I64);
            w.zero_div_if(1, Val::I64);
            let r = w.fx.local(Val::I64);
            w.c().local_get(0).local_get(1).i64_rem_s().local_tee(r).i64_const(0).i64_ne();
            w.c().local_get(r).i64_const(0).i64_lt_s().local_get(1).i64_const(0).i64_lt_s().i32_ne().i32_and();
            w.c().if_(Block::Empty).local_get(r).local_get(1).i64_add().local_set(r).end();
            w.c().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_int_pow(&mut self) -> Res<u32> {
        let mul = self.rt_int_mul()?;
        self.rt_fn("int_pow", |w| {
            params!(w, Val::I64, Val::I64);
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty);
            w.throw_lit("ValueError", "py2rs: int ** negative int gives a float; use a float base");
            w.c().end();
            // square and multiply, checked
            let (r, b, e) = (w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64));
            w.c().i64_const(1).local_set(r).local_get(0).local_set(b).local_get(1).local_set(e);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(e).i64_eqz().br_if(1);
            w.c().local_get(e).i64_const(1).i64_and().i64_eqz().i32_eqz().if_(Block::Empty).local_get(r).local_get(b).call(mul).local_set(r).end();
            w.c().local_get(e).i64_const(1).i64_shr_u().local_tee(e).i64_eqz().br_if(1);
            w.c().local_get(b).local_get(b).call(mul).local_set(b).br(0);
            w.c().end().end().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    /// round(x, n) of ints: n ≥ 0 keeps x; n < 0 rounds to a multiple of 10^-n (half to even)
    pub fn rt_round_int(&mut self) -> Res<u32> {
        let fd = self.rt_int_floordiv()?;
        let mul = self.rt_int_mul()?;
        self.rt_fn("round_int", |w| {
            params!(w, Val::I64, Val::I64);
            let (p, q, r) = (w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64));
            w.c().local_get(1).i64_const(0).i64_ge_s().if_(Block::Empty).local_get(0).return_().end();
            w.c().local_get(1).i64_const(-19).i64_lt_s().if_(Block::Empty).i64_const(0).return_().end();
            // p = 10^-n
            w.c().i64_const(1).local_set(p);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(1).i64_eqz().br_if(1);
            w.c().local_get(p).i64_const(10).i64_mul().local_set(p).local_get(1).i64_const(1).i64_add().local_set(1).br(0);
            w.c().end().end();
            w.c().local_get(0).local_get(p).call(fd).local_set(q);
            w.c().local_get(0).local_get(q).local_get(p).i64_mul().i64_sub().local_set(r);
            w.c().local_get(r).i64_const(2).i64_mul().local_get(p).i64_gt_s();
            w.c().local_get(r).i64_const(2).i64_mul().local_get(p).i64_eq().local_get(q).i64_const(1).i64_and().i64_eqz().i32_eqz().i32_and().i32_or();
            w.c().if_(Block::Empty).local_get(q).i64_const(1).i64_add().local_set(q).end();
            w.c().local_get(q).local_get(p).call(mul);
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_int_truediv(&mut self) -> Res<u32> {
        self.rt_fn("int_truediv", |w| {
            params!(w, Val::I64, Val::I64);
            w.zero_div_if(1, Val::I64);
            w.c().local_get(0).f64_convert_i64_s().local_get(1).f64_convert_i64_s().f64_div();
            Ok(vec![Val::F64])
        })
    }

    pub fn rt_int_shl(&mut self) -> Res<u32> {
        self.rt_fn("int_shl", |w| {
            params!(w, Val::I64, Val::I64);
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty);
            w.throw_lit("ValueError", "negative shift count");
            w.c().end();
            w.c().local_get(0).i64_eqz().if_(Block::Empty).i64_const(0).return_().end();
            // overflow: n >= 64, or shifting back does not give a
            w.c().local_get(1).i64_const(64).i64_ge_s();
            w.c().local_get(0).local_get(1).i64_shl().local_get(1).i64_shr_s().local_get(0).i64_ne().i32_or();
            w.overflow_if();
            w.c().local_get(0).local_get(1).i64_shl();
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_int_shr(&mut self) -> Res<u32> {
        self.rt_fn("int_shr", |w| {
            params!(w, Val::I64, Val::I64);
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty);
            w.throw_lit("ValueError", "negative shift count");
            w.c().end();
            w.c().local_get(1).i64_const(63).i64_gt_s().if_(Block::Empty).i64_const(63).local_set(1).end();
            w.c().local_get(0).local_get(1).i64_shr_s();
            Ok(vec![Val::I64])
        })
    }

    // ------------------------------------------------------------ floats

    /// Two f64 on the stack → the host's math operation `op`.
    pub fn math(&mut self, op: &str) -> Res<()> {
        let (x, y) = (self.fx.local(Val::F64), self.fx.local(Val::F64));
        self.c().local_set(y).local_set(x).local_get(x);
        self.host_arg(&Ty::Float)?;
        self.c().local_get(y);
        self.host_arg(&Ty::Float)?;
        self.host_call(op, &Ty::Float)
    }

    pub fn rt_float_div(&mut self) -> Res<u32> {
        self.rt_fn("float_div", |w| {
            params!(w, Val::F64, Val::F64);
            w.zero_div_if(1, Val::F64);
            w.c().local_get(0).local_get(1).f64_div();
            Ok(vec![Val::F64])
        })
    }

    pub fn rt_float_mod(&mut self) -> Res<u32> {
        self.rt_fn("float_mod", |w| {
            params!(w, Val::F64, Val::F64);
            w.zero_div_if(1, Val::F64);
            let m = w.fx.local(Val::F64);
            w.c().local_get(0).local_get(1);
            w.math("fmod")?;
            w.c().local_tee(m).f64_const(0.0).f64_ne().if_(Block::Val(Val::F64));
            w.c().local_get(1).f64_const(0.0).f64_lt().local_get(m).f64_const(0.0).f64_lt().i32_ne().if_(Block::Val(Val::F64));
            w.c().local_get(m).local_get(1).f64_add().else_().local_get(m).end();
            w.c().else_().f64_const(0.0).local_get(1).f64_copysign().end();
            Ok(vec![Val::F64])
        })
    }

    pub fn rt_float_floordiv(&mut self) -> Res<u32> {
        self.rt_fn("float_floordiv", |w| {
            params!(w, Val::F64, Val::F64);
            w.zero_div_if(1, Val::F64);
            let (m, d, f) = (w.fx.local(Val::F64), w.fx.local(Val::F64), w.fx.local(Val::F64));
            w.c().local_get(0).local_get(1);
            w.math("fmod")?;
            w.c().local_set(m);
            w.c().local_get(0).local_get(m).f64_sub().local_get(1).f64_div().local_set(d);
            w.c().local_get(m).f64_const(0.0).f64_ne().local_get(1).f64_const(0.0).f64_lt().local_get(m).f64_const(0.0).f64_lt().i32_ne().i32_and();
            w.c().if_(Block::Empty).local_get(d).f64_const(1.0).f64_sub().local_set(d).end();
            w.c().local_get(d).f64_const(0.0).f64_ne().if_(Block::Val(Val::F64));
            w.c().local_get(d).f64_floor().local_tee(f).f64_const(1.0).f64_add().local_get(f).local_get(d).local_get(f).f64_sub().f64_const(0.5).f64_gt().select();
            w.c().else_().f64_const(0.0).local_get(0).local_get(1).f64_div().f64_copysign().end();
            Ok(vec![Val::F64])
        })
    }

    pub fn rt_float_pow(&mut self) -> Res<u32> {
        self.rt_fn("float_pow", |w| {
            params!(w, Val::F64, Val::F64);
            w.c().local_get(0).f64_const(0.0).f64_eq().local_get(1).f64_const(0.0).f64_lt().i32_and().if_(Block::Empty);
            w.throw_lit("ZeroDivisionError", "zero to a negative power");
            w.c().end();
            // a negative base to a finite fractional power
            w.c().local_get(0).f64_const(0.0).f64_lt();
            w.c().local_get(1).local_get(1).f64_sub().f64_const(0.0).f64_eq().i32_and(); // finite
            w.c().local_get(1).f64_trunc().local_get(1).f64_ne().i32_and().if_(Block::Empty);
            w.throw_lit("ValueError", "py2rs: a negative number to a fractional power is complex in Python");
            w.c().end();
            let r = w.fx.local(Val::F64);
            w.c().local_get(0).local_get(1);
            w.math("pow")?;
            w.c().local_set(r);
            // overflow: an infinite result of finite operands
            w.c().local_get(r).local_get(r).f64_sub().f64_const(0.0).f64_ne().local_get(r).local_get(r).f64_eq().i32_and();
            w.c().local_get(0).local_get(0).f64_sub().f64_const(0.0).f64_eq().i32_and();
            w.c().local_get(1).local_get(1).f64_sub().f64_const(0.0).f64_eq().i32_and().if_(Block::Empty);
            w.throw_lit("OverflowError", "(34, 'Result too large')");
            w.c().end().local_get(r);
            Ok(vec![Val::F64])
        })
    }

    // ------------------------------------------------------------ strings

    pub fn rt_str_concat(&mut self) -> Res<u32> {
        self.rt_fn("str_concat", |w| {
            params!(w, w.sref(), w.sref());
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (a, b, r) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::Ref(Heap::Idx(bytes))));
            w.c().local_get(0).struct_get(st, 0).local_set(a).local_get(1).struct_get(st, 0).local_set(b);
            w.c().local_get(a).array_len().local_get(b).array_len().i32_add().array_new_default(bytes).local_set(r);
            w.c().local_get(r).i32_const(0).local_get(a).i32_const(0).local_get(a).array_len().array_copy(bytes, bytes);
            w.c().local_get(r).local_get(a).array_len().local_get(b).i32_const(0).local_get(b).array_len().array_copy(bytes, bytes);
            w.c().local_get(r).ref_as_non_null().local_get(0).struct_get(st, 1).local_get(1).struct_get(st, 1).i32_add().struct_new(st);
            Ok(vec![w.sref()])
        })
    }

    pub fn rt_str_eq(&mut self) -> Res<u32> {
        self.rt_fn("str_eq", |w| {
            params!(w, w.sref(), w.sref());
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (a, b, i, n) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(0).local_get(1).ref_eq().if_(Block::Empty).i32_const(1).return_().end();
            w.c().local_get(0).struct_get(st, 0).local_set(a).local_get(1).struct_get(st, 0).local_set(b);
            w.c().local_get(a).array_len().local_tee(n).local_get(b).array_len().i32_ne().if_(Block::Empty).i32_const(0).return_().end();
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(a).local_get(i).array_get_u(bytes).local_get(b).local_get(i).array_get_u(bytes).i32_ne().if_(Block::Empty).i32_const(0).return_().end();
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().i32_const(1);
            Ok(vec![Val::I32])
        })
    }

    /// str_cmp(a, b) → -1 / 0 / 1 (code point order = UTF-8 byte order)
    pub fn rt_str_cmp(&mut self) -> Res<u32> {
        self.rt_fn("str_cmp", |w| {
            params!(w, w.sref(), w.sref());
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (a, b, i, n, x, y) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(st, 0).local_set(a).local_get(1).struct_get(st, 0).local_set(b);
            // n = min(len a, len b)
            w.c().local_get(a).array_len().local_get(b).array_len().local_get(a).array_len().local_get(b).array_len().i32_lt_u().select().local_set(n);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(a).local_get(i).array_get_u(bytes).local_tee(x).local_get(b).local_get(i).array_get_u(bytes).local_tee(y).i32_ne().if_(Block::Empty);
            w.c().i32_const(-1).i32_const(1).local_get(x).local_get(y).i32_lt_u().select().return_().end();
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            w.c().local_get(a).array_len().local_get(b).array_len().i32_gt_u().local_get(a).array_len().local_get(b).array_len().i32_lt_u().i32_sub();
            Ok(vec![Val::I32])
        })
    }

    /// FNV-1a over the bytes
    pub fn rt_str_hash(&mut self) -> Res<u32> {
        self.rt_fn("str_hash", |w| {
            params!(w, w.sref());
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (b, i, h) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I64));
            w.c().local_get(0).struct_get(st, 0).local_set(b).i64_const(0xcbf29ce484222325u64 as i64).local_set(h);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(b).array_len().i32_ge_u().br_if(1);
            w.c().local_get(h).local_get(b).local_get(i).array_get_u(bytes).i64_extend_i32_u().i64_xor().i64_const(0x100000001b3).i64_mul().local_set(h);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().local_get(h);
            Ok(vec![Val::I64])
        })
    }

    /// The length in bytes of the UTF-8 character starting with byte `b` (on the stack).
    fn utf8_len(&mut self) {
        let b = self.fx.local(Val::I32);
        self.c().local_tee(b).i32_const(0x80).i32_lt_u().if_(Block::Val(Val::I32)).i32_const(1);
        self.c().else_().local_get(b).i32_const(0xe0).i32_lt_u().if_(Block::Val(Val::I32)).i32_const(2);
        self.c().else_().i32_const(3).i32_const(4).local_get(b).i32_const(0xf0).i32_lt_u().select().end().end();
    }

    /// char_at(s, pos) → (the character at byte pos, the next pos)
    pub fn rt_str_char_at(&mut self) -> Res<u32> {
        self.rt_fn("str_char_at", |w| {
            params!(w, w.sref(), Val::I32);
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (b, n, r) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::Ref(Heap::Idx(bytes))));
            w.c().local_get(0).struct_get(st, 0).local_tee(b).local_get(1).array_get_u(bytes);
            w.utf8_len();
            w.c().local_tee(n).array_new_default(bytes).local_tee(r).i32_const(0).local_get(b).local_get(1).local_get(n).array_copy(bytes, bytes);
            w.c().local_get(r).ref_as_non_null().i32_const(1).struct_new(st).local_get(1).local_get(n).i32_add();
            Ok(vec![w.sref(), Val::I32])
        })
    }

    /// The byte offset of code point `i` (on the stack, an i32) of the str in
    /// local `s` → i32.
    fn byte_offset(&mut self, s: u32) {
        let (st, bytes) = (self.t.str, self.t.bytes);
        let (i, pos, b) = (self.fx.local(Val::I32), self.fx.local(Val::I32), self.fx.local(Val::Ref(Heap::Idx(bytes))));
        self.c().local_set(i).i32_const(0).local_set(pos);
        // ASCII: the same
        self.c().local_get(s).struct_get(st, 0).local_tee(b).array_len().local_get(s).struct_get(st, 1).i32_eq().if_(Block::Val(Val::I32)).local_get(i);
        self.c().else_();
        self.c().block(Block::Empty).loop_(Block::Empty);
        self.c().local_get(i).i32_eqz().br_if(1);
        self.c().local_get(pos).local_get(b).local_get(pos).array_get_u(bytes);
        self.utf8_len();
        self.c().i32_add().local_set(pos).local_get(i).i32_const(1).i32_sub().local_set(i).br(0);
        self.c().end().end().local_get(pos).end();
    }

    /// s[i]
    pub fn rt_str_index(&mut self) -> Res<u32> {
        let ch = self.rt_str_char_at()?;
        self.rt_fn("str_index", |w| {
            params!(w, w.sref(), Val::I64);
            let st = w.t.str;
            let n = w.fx.local(Val::I64);
            w.c().local_get(0).struct_get(st, 1).i64_extend_i32_s().local_set(n);
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty).local_get(1).local_get(n).i64_add().local_set(1).end();
            w.c().local_get(1).i64_const(0).i64_lt_s().local_get(1).local_get(n).i64_ge_s().i32_or().if_(Block::Empty);
            w.throw_lit("IndexError", "string index out of range");
            w.c().end();
            w.c().local_get(0).local_get(1).i32_wrap_i64();
            w.byte_offset(0);
            w.c().call(ch).drop();
            Ok(vec![w.sref()])
        })
    }

    /// Python's slice.indices: (start, has, stop, has, step, has, n) → (start, stop, step, count)
    pub fn rt_slice_indices(&mut self) -> Res<u32> {
        self.rt_fn("slice_indices", |w| {
            params!(w, Val::I64, Val::I32, Val::I64, Val::I32, Val::I64, Val::I32, Val::I64);
            let (start, stop, step, n) = (0u32, 2u32, 4u32, 6u32);
            w.c().local_get(5).i32_eqz().if_(Block::Empty).i64_const(1).local_set(step).end();
            w.c().local_get(step).i64_eqz().if_(Block::Empty);
            w.throw_lit("ValueError", "slice step cannot be zero");
            w.c().end();
            // missing bounds: the extremes for the step's direction
            for (v, has, lo_first) in [(start, 1u32, true), (stop, 3u32, false)] {
                w.c().local_get(has).i32_eqz().if_(Block::Empty);
                // start: step < 0 ? MAX : MIN... as CPython: start defaults to (step<0 ? MAX : 0), stop to (step<0 ? MIN : MAX)
                if lo_first {
                    w.c().i64_const(i64::MAX).i64_const(0).local_get(step).i64_const(0).i64_lt_s().select().local_set(v);
                } else {
                    w.c().i64_const(i64::MIN).i64_const(i64::MAX).local_get(step).i64_const(0).i64_lt_s().select().local_set(v);
                }
                w.c().end();
                // adjust: negative → + n, then clamp
                w.c().local_get(v).i64_const(0).i64_lt_s().if_(Block::Empty);
                w.c().local_get(v).i64_const(i64::MIN).i64_eq().if_(Block::Val(Val::I64)).i64_const(-1).else_().local_get(v).local_get(n).i64_add().end().local_set(v);
                w.c().local_get(v).i64_const(0).i64_lt_s().if_(Block::Empty);
                w.c().i64_const(-1).i64_const(0).local_get(step).i64_const(0).i64_lt_s().select().local_set(v).end();
                w.c().else_().local_get(v).local_get(n).i64_ge_s().if_(Block::Empty);
                w.c().local_get(n).i64_const(1).i64_sub().local_get(n).local_get(step).i64_const(0).i64_lt_s().select().local_set(v).end();
                w.c().end();
            }
            // count
            let cnt = w.fx.local(Val::I64);
            w.c().local_get(step).i64_const(0).i64_lt_s().if_(Block::Empty);
            w.c().local_get(stop).local_get(start).i64_lt_s().if_(Block::Empty);
            w.c().local_get(start).local_get(stop).i64_sub().i64_const(1).i64_sub().i64_const(0).local_get(step).i64_sub().i64_div_s().i64_const(1).i64_add().local_set(cnt).end();
            w.c().else_();
            w.c().local_get(start).local_get(stop).i64_lt_s().if_(Block::Empty);
            w.c().local_get(stop).local_get(start).i64_sub().i64_const(1).i64_sub().local_get(step).i64_div_s().i64_const(1).i64_add().local_set(cnt).end();
            w.c().end();
            w.c().local_get(start).local_get(stop).local_get(step).local_get(cnt);
            Ok(vec![Val::I64, Val::I64, Val::I64, Val::I64])
        })
    }

    /// s[a:b:c] (each bound with a has-flag)
    pub fn rt_str_slice(&mut self) -> Res<u32> {
        let ind = self.rt_slice_indices()?;
        let ch = self.rt_str_char_at()?;
        let sbn = self.rt_sb_new()?;
        let sba = self.rt_sb_add()?;
        let sbs = self.rt_sb_str()?;
        self.rt_fn("str_slice", |w| {
            params!(w, w.sref(), Val::I64, Val::I32, Val::I64, Val::I32, Val::I64, Val::I32);
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (start, step, cnt, sb, j) = (w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::Ref(Heap::Idx(w.t.sb))), w.fx.local(Val::I64));
            let stop = w.fx.local(Val::I64);
            for i in 1..7 {
                w.c().local_get(i);
            }
            w.c().local_get(0).struct_get(st, 1).i64_extend_i32_s().call(ind).local_set(cnt).local_set(step).local_set(stop).local_set(start);
            // ASCII and step 1: copy the bytes
            w.c().local_get(0).struct_get(st, 0).array_len().local_get(0).struct_get(st, 1).i32_eq().local_get(step).i64_const(1).i64_eq().i32_and().if_(Block::Empty);
            let r = w.fx.local(Val::Ref(Heap::Idx(bytes)));
            w.c().local_get(cnt).i32_wrap_i64().array_new_default(bytes).local_tee(r).i32_const(0).local_get(0).struct_get(st, 0).local_get(start).i32_wrap_i64().local_get(cnt).i32_wrap_i64().array_copy(bytes, bytes);
            w.c().local_get(r).ref_as_non_null().local_get(cnt).i32_wrap_i64().struct_new(st).return_().end();
            w.c().local_get(cnt).i32_wrap_i64().call(sbn).local_set(sb);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(j).local_get(cnt).i64_ge_s().br_if(1);
            w.c().local_get(sb).local_get(0).local_get(start).local_get(j).local_get(step).i64_mul().i64_add().i32_wrap_i64();
            w.byte_offset(0);
            w.c().call(ch).drop().call(sba);
            w.c().local_get(j).i64_const(1).i64_add().local_set(j).br(0);
            w.c().end().end().local_get(sb).call(sbs);
            Ok(vec![w.sref()])
        })
    }

    /// s * n
    pub fn rt_str_mul(&mut self) -> Res<u32> {
        let sbn = self.rt_sb_new()?;
        let sba = self.rt_sb_add()?;
        let sbs = self.rt_sb_str()?;
        self.rt_fn("str_mul", |w| {
            params!(w, w.sref(), Val::I64);
            let sb = w.fx.local(Val::Ref(Heap::Idx(w.t.sb)));
            w.c().i32_const(0).call(sbn).local_set(sb);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(1).i64_const(0).i64_le_s().br_if(1);
            w.c().local_get(sb).local_get(0).call(sba).local_get(1).i64_const(1).i64_sub().local_set(1).br(0);
            w.c().end().end().local_get(sb).call(sbs);
            Ok(vec![w.sref()])
        })
    }

    /// find(haystack, needle, start) → code point index or -1
    pub fn rt_str_find(&mut self) -> Res<u32> {
        self.rt_fn("str_find", |w| {
            params!(w, w.sref(), w.sref(), Val::I64);
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (h, nd, i, j, hn, nn) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(st, 0).local_tee(h).array_len().local_set(hn);
            w.c().local_get(1).struct_get(st, 0).local_tee(nd).array_len().local_set(nn);
            w.c().local_get(2).i64_const(0).i64_lt_s().if_(Block::Empty).local_get(2).local_get(0).struct_get(st, 1).i64_extend_i32_s().i64_add().local_set(2).end();
            w.c().local_get(2).i64_const(0).i64_lt_s().if_(Block::Empty).i64_const(0).local_set(2).end();
            w.c().local_get(2).local_get(0).struct_get(st, 1).i64_extend_i32_s().i64_gt_s().if_(Block::Empty).i64_const(-1).return_().end();
            w.c().local_get(2).i32_wrap_i64();
            w.byte_offset(0);
            w.c().local_set(i);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(nn).i32_add().local_get(hn).i32_gt_u().br_if(1);
            w.c().i32_const(0).local_set(j);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(j).local_get(nn).i32_ge_u().br_if(1);
            w.c().local_get(h).local_get(i).local_get(j).i32_add().array_get_u(bytes).local_get(nd).local_get(j).array_get_u(bytes).i32_ne().br_if(1);
            w.c().local_get(j).i32_const(1).i32_add().local_set(j).br(0);
            w.c().end().end();
            w.c().local_get(j).local_get(nn).i32_eq().if_(Block::Empty);
            // found at byte i: count the code points before it
            let (k, cp) = (w.fx.local(Val::I32), w.fx.local(Val::I64));
            w.c().i32_const(0).local_set(k).i64_const(0).local_set(cp);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(k).local_get(i).i32_ge_u().br_if(1);
            w.c().local_get(cp).local_get(h).local_get(k).array_get_u(bytes).i32_const(0xc0).i32_and().i32_const(0x80).i32_ne().i64_extend_i32_u().i64_add().local_set(cp);
            w.c().local_get(k).i32_const(1).i32_add().local_set(k).br(0);
            w.c().end().end().local_get(cp).return_().end();
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().i64_const(-1);
            Ok(vec![Val::I64])
        })
    }

    /// int → str
    pub fn rt_int_str(&mut self) -> Res<u32> {
        self.rt_fn("int_str", |w| {
            params!(w, Val::I64);
            let (st, bytes) = (w.t.str, w.t.bytes);
            let (buf, i, u, r) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::I64), w.fx.local(Val::Ref(Heap::Idx(bytes))));
            w.c().i32_const(20).array_new_default(bytes).local_set(buf).i32_const(20).local_set(i);
            // the magnitude as unsigned (MIN works)
            w.c().i64_const(0).local_get(0).i64_sub().local_get(0).local_get(0).i64_const(0).i64_lt_s().select().local_set(u);
            w.c().loop_(Block::Empty);
            w.c().local_get(i).i32_const(1).i32_sub().local_set(i);
            w.c().local_get(buf).local_get(i).local_get(u).i64_const(10).i64_rem_u().i32_wrap_i64().i32_const(48).i32_add().array_set(bytes);
            w.c().local_get(u).i64_const(10).i64_div_u().local_tee(u).i64_eqz().i32_eqz().br_if(0);
            w.c().end();
            w.c().local_get(0).i64_const(0).i64_lt_s().if_(Block::Empty).local_get(i).i32_const(1).i32_sub().local_tee(i).local_get(buf).local_get(i).i32_const(45).array_set(bytes).drop().end();
            w.c().i32_const(20).local_get(i).i32_sub().array_new_default(bytes).local_tee(r).i32_const(0).local_get(buf).local_get(i).i32_const(20).local_get(i).i32_sub().array_copy(bytes, bytes);
            w.c().local_get(r).ref_as_non_null().i32_const(20).local_get(i).i32_sub().struct_new(st);
            Ok(vec![w.sref()])
        })
    }

    /// float → str (repr)
    pub fn rt_float_str(&mut self) -> Res<u32> {
        self.rt_fn("float_str", |w| {
            params!(w, Val::F64);
            w.c().local_get(0);
            w.host_arg(&Ty::Float)?;
            w.host_call("float_repr", &Ty::Str)?;
            Ok(vec![w.sref()])
        })
    }

    // ------------------------------------------------------------ string builder

    pub fn rt_sb_new(&mut self) -> Res<u32> {
        self.rt_fn("sb_new", |w| {
            params!(w, Val::I32);
            let (sb, bytes) = (w.t.sb, w.t.bytes);
            w.c().local_get(0).i32_const(16).i32_add().array_new_default(bytes).i32_const(0).i32_const(0).struct_new(sb);
            Ok(vec![Val::Ref(Heap::Idx(sb))])
        })
    }

    /// sb_add(sb, s)
    pub fn rt_sb_add(&mut self) -> Res<u32> {
        self.rt_fn("sb_add", |w| {
            let (sb, bytes, st) = (w.t.sb, w.t.bytes, w.t.str);
            params!(w, Val::Ref(Heap::Idx(sb)), w.sref());
            let (b, need, nb) = (w.fx.local(Val::Ref(Heap::Idx(bytes))), w.fx.local(Val::I32), w.fx.local(Val::Ref(Heap::Idx(bytes))));
            w.c().local_get(1).struct_get(st, 0).local_set(b);
            w.c().local_get(0).struct_get(sb, 1).local_get(b).array_len().i32_add().local_tee(need);
            w.c().local_get(0).struct_get(sb, 0).array_len().i32_gt_u().if_(Block::Empty);
            w.c().local_get(need).i32_const(1).i32_shl().array_new_default(bytes).local_tee(nb).i32_const(0).local_get(0).struct_get(sb, 0).i32_const(0).local_get(0).struct_get(sb, 1).array_copy(bytes, bytes);
            w.c().local_get(0).local_get(nb).ref_as_non_null().struct_set(sb, 0).end();
            w.c().local_get(0).struct_get(sb, 0).local_get(0).struct_get(sb, 1).local_get(b).i32_const(0).local_get(b).array_len().array_copy(bytes, bytes);
            w.c().local_get(0).local_get(need).struct_set(sb, 1);
            w.c().local_get(0).local_get(0).struct_get(sb, 2).local_get(1).struct_get(st, 1).i32_add().struct_set(sb, 2);
            Ok(vec![])
        })
    }

    /// sb_str(sb) → str
    pub fn rt_sb_str(&mut self) -> Res<u32> {
        self.rt_fn("sb_str", |w| {
            let (sb, bytes, st) = (w.t.sb, w.t.bytes, w.t.str);
            params!(w, Val::Ref(Heap::Idx(sb)));
            let r = w.fx.local(Val::Ref(Heap::Idx(bytes)));
            w.c().local_get(0).struct_get(sb, 1).array_new_default(bytes).local_tee(r).i32_const(0).local_get(0).struct_get(sb, 0).i32_const(0).local_get(0).struct_get(sb, 1).array_copy(bytes, bytes);
            w.c().local_get(r).ref_as_non_null().local_get(0).struct_get(sb, 2).struct_new(st);
            Ok(vec![w.sref()])
        })
    }

    // ------------------------------------------------------------ lists

    pub fn rt_list_new(&mut self, k: K) -> Res<u32> {
        self.rt_fn(&format!("list_new_{:?}", k), |w| {
            params!(w, Val::I32);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            w.c().local_get(0).i32_const(4).local_get(0).i32_const(4).i32_gt_u().select().array_new_default(arr).i32_const(0).struct_new(l);
            Ok(vec![w.lref(k)])
        })
    }

    pub fn rt_list_append(&mut self, k: K) -> Res<u32> {
        self.rt_fn(&format!("list_append_{:?}", k), |w| {
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            params!(w, w.lref(k), w.kval(k));
            let (n, nb) = (w.fx.local(Val::I32), w.fx.local(Val::Ref(Heap::Idx(arr))));
            w.c().local_get(0).struct_get(l, 1).local_tee(n).local_get(0).struct_get(l, 0).array_len().i32_eq().if_(Block::Empty);
            w.c().local_get(n).i32_const(1).i32_shl().i32_const(4).i32_add().array_new_default(arr).local_tee(nb).i32_const(0).local_get(0).struct_get(l, 0).i32_const(0).local_get(n).array_copy(arr, arr);
            w.c().local_get(0).local_get(nb).ref_as_non_null().struct_set(l, 0).end();
            w.c().local_get(0).struct_get(l, 0).local_get(n).local_get(1).array_set(arr);
            w.c().local_get(0).local_get(n).i32_const(1).i32_add().struct_set(l, 1);
            Ok(vec![])
        })
    }

    /// The index on the stack (i64) for a list in local 0 → a checked i32 index.
    fn list_index_check(&mut self, k: K, msg: &str) {
        let l = self.t.list[k as usize];
        let i = self.fx.local(Val::I64);
        self.c().local_tee(i).i64_const(0).i64_lt_s().if_(Block::Empty).local_get(i).local_get(0).struct_get(l, 1).i64_extend_i32_s().i64_add().local_set(i).end();
        self.c().local_get(i).i64_const(0).i64_lt_s().local_get(i).local_get(0).struct_get(l, 1).i64_extend_i32_s().i64_ge_s().i32_or().if_(Block::Empty);
        self.throw_lit("IndexError", msg);
        self.c().end().local_get(i).i32_wrap_i64();
    }

    pub fn rt_list_get(&mut self, k: K) -> Res<u32> {
        self.rt_fn(&format!("list_get_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let i = w.fx.local(Val::I32);
            w.c().local_get(1);
            w.list_index_check(k, "list index out of range");
            w.c().local_set(i).local_get(0).struct_get(l, 0).local_get(i).array_get(arr);
            Ok(vec![w.kval(k)])
        })
    }

    pub fn rt_list_set(&mut self, t: &Ty) -> Res<u32> {
        let k = self.kind(t)?;
        self.rt_fn(&format!("list_set_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64, w.kval(k));
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let i = w.fx.local(Val::I32);
            w.c().local_get(1);
            w.list_index_check(k, "list assignment index out of range");
            w.c().local_set(i).local_get(0).struct_get(l, 0).local_get(i).local_get(2).array_set(arr);
            Ok(vec![])
        })
    }

    pub fn rt_list_del(&mut self, t: &Ty) -> Res<u32> {
        let k = self.kind(t)?;
        self.rt_fn(&format!("list_del_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let i = w.fx.local(Val::I32);
            w.c().local_get(1);
            w.list_index_check(k, "list assignment index out of range");
            w.c().local_set(i);
            // shift the rest left
            w.c().local_get(0).struct_get(l, 0).local_get(i).local_get(0).struct_get(l, 0).local_get(i).i32_const(1).i32_add();
            w.c().local_get(0).struct_get(l, 1).local_get(i).i32_sub().i32_const(1).i32_sub().array_copy(arr, arr);
            w.c().local_get(0).local_get(0).struct_get(l, 1).i32_const(1).i32_sub().struct_set(l, 1);
            if k == K::Ref {
                w.c().local_get(0).struct_get(l, 0).local_get(0).struct_get(l, 1).ref_null(Heap::Eq).array_set(arr);
            }
            Ok(vec![])
        })
    }

    /// pop(xs, i) (i: -1 for the last)
    pub fn rt_list_pop(&mut self, k: K) -> Res<u32> {
        let get = self.rt_list_get(k)?;
        self.rt_fn(&format!("list_pop_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64, Val::I32);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let (i, x) = (w.fx.local(Val::I32), w.fx.local(w.kval(k)));
            w.c().local_get(0).struct_get(l, 1).i32_eqz().local_get(2).i32_and().if_(Block::Empty);
            w.throw_lit("IndexError", "pop from empty list");
            w.c().end();
            w.c().local_get(0).struct_get(l, 1).i32_eqz().if_(Block::Empty);
            w.throw_lit("IndexError", "pop index out of range");
            w.c().end();
            w.c().local_get(1);
            w.list_index_check(k, "pop index out of range");
            w.c().local_set(i).local_get(0).local_get(i).i64_extend_i32_s().call(get).local_set(x);
            w.c().local_get(0).struct_get(l, 0).local_get(i).local_get(0).struct_get(l, 0).local_get(i).i32_const(1).i32_add();
            w.c().local_get(0).struct_get(l, 1).local_get(i).i32_sub().i32_const(1).i32_sub().array_copy(arr, arr);
            w.c().local_get(0).local_get(0).struct_get(l, 1).i32_const(1).i32_sub().struct_set(l, 1).local_get(x);
            Ok(vec![w.kval(k)])
        })
    }

    /// insert(xs, i, x): i clamped like Python
    pub fn rt_list_insert(&mut self, k: K) -> Res<u32> {
        let app = self.rt_list_append(k)?;
        self.rt_fn(&format!("list_insert_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64, w.kval(k));
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let (n, i) = (w.fx.local(Val::I64), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(l, 1).i64_extend_i32_s().local_set(n);
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty).local_get(1).local_get(n).i64_add().local_set(1).end();
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty).i64_const(0).local_set(1).end();
            w.c().local_get(1).local_get(n).i64_gt_s().if_(Block::Empty).local_get(n).local_set(1).end();
            w.c().local_get(1).i32_wrap_i64().local_set(i);
            // append (grows), then shift right and put x at i
            w.c().local_get(0).local_get(2).call(app);
            w.c().local_get(0).struct_get(l, 0).local_get(i).i32_const(1).i32_add().local_get(0).struct_get(l, 0).local_get(i);
            w.c().local_get(n).i32_wrap_i64().local_get(i).i32_sub().array_copy(arr, arr);
            w.c().local_get(0).struct_get(l, 0).local_get(i).local_get(2).array_set(arr);
            Ok(vec![])
        })
    }

    /// extend(xs, ys)
    pub fn rt_list_extend(&mut self, t: &Ty) -> Res<u32> {
        let k = self.kind(t)?;
        let app = self.rt_list_append(k)?;
        self.rt_fn(&format!("list_extend_{:?}", k), |w| {
            params!(w, w.lref(k), w.lref(k));
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let (i, n) = (w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(1).struct_get(l, 1).local_set(n);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(0).local_get(1).struct_get(l, 0).local_get(i).array_get(arr).call(app);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            Ok(vec![])
        })
    }

    /// a + b
    pub fn rt_list_concat(&mut self, t: &Ty) -> Res<u32> {
        let k = self.kind(t)?;
        let new = self.rt_list_new(k)?;
        let ext = self.rt_list_extend(t)?;
        self.rt_fn(&format!("list_concat_{:?}", k), |w| {
            params!(w, w.lref(k), w.lref(k));
            let l = w.t.list[k as usize];
            let r = w.fx.local(w.lref(k));
            w.c().local_get(0).struct_get(l, 1).local_get(1).struct_get(l, 1).i32_add().call(new).local_tee(r).local_get(0).call(ext);
            w.c().local_get(r).local_get(1).call(ext).local_get(r);
            Ok(vec![w.lref(k)])
        })
    }

    /// xs * n
    pub fn rt_list_mul(&mut self, k: K) -> Res<u32> {
        let new = self.rt_list_new(k)?;
        let app = self.rt_list_append(k)?;
        self.rt_fn(&format!("list_mul_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let (r, i, n) = (w.fx.local(w.lref(k)), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().i32_const(0).call(new).local_set(r).local_get(0).struct_get(l, 1).local_set(n);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(1).i64_const(0).i64_le_s().br_if(1);
            w.c().i32_const(0).local_set(i);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_u().br_if(1);
            w.c().local_get(r).local_get(0).struct_get(l, 0).local_get(i).array_get(arr).call(app);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            w.c().local_get(1).i64_const(1).i64_sub().local_set(1).br(0);
            w.c().end().end().local_get(r);
            Ok(vec![w.lref(k)])
        })
    }

    /// xs[a:b:c]
    pub fn rt_list_slice(&mut self, k: K) -> Res<u32> {
        let ind = self.rt_slice_indices()?;
        let new = self.rt_list_new(k)?;
        let app = self.rt_list_append(k)?;
        self.rt_fn(&format!("list_slice_{:?}", k), |w| {
            params!(w, w.lref(k), Val::I64, Val::I32, Val::I64, Val::I32, Val::I64, Val::I32);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let (start, stop, step, cnt, j, r) = (w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(w.lref(k)));
            for i in 1..7 {
                w.c().local_get(i);
            }
            w.c().local_get(0).struct_get(l, 1).i64_extend_i32_s().call(ind).local_set(cnt).local_set(step).local_set(stop).local_set(start);
            w.c().local_get(cnt).i32_wrap_i64().call(new).local_set(r);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(j).local_get(cnt).i64_ge_s().br_if(1);
            w.c().local_get(r).local_get(0).struct_get(l, 0).local_get(start).local_get(j).local_get(step).i64_mul().i64_add().i32_wrap_i64().array_get(arr).call(app);
            w.c().local_get(j).i64_const(1).i64_add().local_set(j).br(0);
            w.c().end().end().local_get(r);
            Ok(vec![w.lref(k)])
        })
    }

    /// index(xs, x, start) → the first index ≥ start where xs[i] == x, or -1
    pub fn rt_list_index(&mut self, t: &Ty) -> Res<u32> {
        let k = self.kind(t)?;
        let eq = self.eq_fn(t)?;
        let et = self.val(t)?;
        let key = format!("list_index:{:?}", t);
        let t = t.clone();
        self.rt_fn(&key, |w| {
            params!(w, w.lref(k), et.unwrap_or(Val::I32), Val::I32);
            let (arr, l) = (w.t.arr[k as usize], w.t.list[k as usize]);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(2).local_get(0).struct_get(l, 1).i32_ge_s().br_if(1);
            w.c().local_get(0).struct_get(l, 0).local_get(2).array_get(arr);
            if k == K::Ref {
                if let Some(Val::Ref(h)) = w.val(&t)? {
                    w.c().ref_cast(h);
                }
            }
            w.c().local_get(1).call(eq).if_(Block::Empty).local_get(2).i64_extend_i32_s().return_().end();
            w.c().local_get(2).i32_const(1).i32_add().local_set(2).br(0);
            w.c().end().end().i64_const(-1);
            Ok(vec![Val::I64])
        })
    }

    /// The length check of `a, b = xs`.
    pub fn rt_list_unpack(&mut self, t: &Ty, n: usize) -> Res<u32> {
        let k = self.kind(t)?;
        let istr = self.rt_int_str()?;
        let cat = self.rt_str_concat()?;
        self.rt_fn(&format!("list_unpack_{:?}_{}", k, n), |w| {
            params!(w, w.lref(k));
            let l = w.t.list[k as usize];
            w.c().local_get(0).struct_get(l, 1).i32_const(n as i32).i32_lt_s().if_(Block::Empty);
            w.str_const(&format!("not enough values to unpack (expected {}, got ", n));
            w.c().local_get(0).struct_get(l, 1).i64_extend_i32_s().call(istr).call(cat);
            w.str_const(")");
            w.c().call(cat);
            w.throw_kind("ValueError");
            w.c().end();
            w.c().local_get(0).struct_get(l, 1).i32_const(n as i32).i32_gt_s().if_(Block::Empty);
            w.str_const(&format!("too many values to unpack (expected {})", n));
            w.throw_kind("ValueError");
            w.c().end();
            Ok(vec![])
        })
    }

    // ------------------------------------------------------------ dicts and sets

    fn deleted(&mut self) -> u32 {
        if let Some(&g) = self.rts.get("global:deleted") {
            return g;
        }
        let b = self.t.box_i64;
        let mut init = super::enc::Code::default();
        init.i64_const(0).struct_new(b);
        self.m.globals.push((Val::NonNull(Heap::Idx(b)), false, init));
        let g = (self.m.globals.len() - 1) as u32;
        self.rts.insert("global:deleted".into(), g);
        g
    }

    pub fn rt_dict_new(&mut self) -> Res<u32> {
        self.rt_fn("dict_new", |w| {
            let (d, ae, a64, a32) = (w.t.dict, w.t.arr[3], w.t.arr[0], w.t.arr[2]);
            params!(w, Val::NonNull(Heap::Idx(w.t.hashfn)), Val::NonNull(Heap::Idx(w.t.eqfn)));
            w.c().i32_const(8).array_new_default(ae).i32_const(8).array_new_default(ae).i32_const(8).array_new_default(a64).i32_const(16).array_new_default(a32);
            w.c().i32_const(0).i32_const(0).local_get(0).local_get(1).struct_new(d);
            Ok(vec![Val::Ref(Heap::Idx(d))])
        })
    }

    /// The slot of `key` (hash h) in the index: its entry or a free slot.
    pub fn rt_dict_slot(&mut self) -> Res<u32> {
        let del = self.deleted();
        self.rt_fn("dict_slot", |w| {
            let (d, ae, a64, a32, eqfn) = (w.t.dict, w.t.arr[3], w.t.arr[0], w.t.arr[2], w.t.eqfn);
            params!(w, Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Eq), Val::I64);
            let (mask, pos, e) = (w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(d, 3).array_len().i32_const(1).i32_sub().local_set(mask);
            // mix the hash
            w.c().local_get(2).i64_const(0x9e3779b97f4a7c15u64 as i64).i64_mul().i64_const(32).i64_rotl().i32_wrap_i64().local_get(mask).i32_and().local_set(pos);
            w.c().loop_(Block::Empty);
            w.c().local_get(0).struct_get(d, 3).local_get(pos).array_get(a32).local_tee(e).i32_eqz().if_(Block::Empty).local_get(pos).return_().end();
            w.c().local_get(e).i32_const(1).i32_sub().local_set(e);
            w.c().local_get(0).struct_get(d, 2).local_get(e).array_get(a64).local_get(2).i64_eq();
            w.c().if_(Block::Empty);
            w.c().local_get(0).struct_get(d, 0).local_get(e).array_get(ae).global_get(del).ref_eq().i32_eqz().if_(Block::Empty);
            w.c().local_get(0).struct_get(d, 0).local_get(e).array_get(ae).local_get(1).local_get(0).struct_get(d, 7).call_ref(eqfn).if_(Block::Empty).local_get(pos).return_().end();
            w.c().end().end();
            w.c().local_get(pos).i32_const(1).i32_add().local_get(mask).i32_and().local_set(pos).br(0);
            w.c().end().unreachable();
            Ok(vec![Val::I32])
        })
    }

    /// find(d, key) → entry index or -1
    pub fn rt_dict_find(&mut self) -> Res<u32> {
        let slot = self.rt_dict_slot()?;
        self.rt_fn("dict_find", |w| {
            let (d, a32, hashfn) = (w.t.dict, w.t.arr[2], w.t.hashfn);
            params!(w, Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Eq));
            w.c().local_get(0).struct_get(d, 3).local_get(0).local_get(1).local_get(1).local_get(0).struct_get(d, 6).call_ref(hashfn).call(slot).array_get(a32).i32_const(1).i32_sub();
            Ok(vec![Val::I32])
        })
    }

    /// Grow / compact the entries and rebuild the index.
    fn rt_dict_resize(&mut self) -> Res<u32> {
        let del = self.deleted();
        self.rt_fn("dict_resize", |w| {
            let (d, ae, a64, a32) = (w.t.dict, w.t.arr[3], w.t.arr[0], w.t.arr[2]);
            params!(w, Val::Ref(Heap::Idx(d)));
            let (cap, ks, vs, hs, i, j) = (w.fx.local(Val::I32), w.fx.local(Val::Ref(Heap::Idx(ae))), w.fx.local(Val::Ref(Heap::Idx(ae))), w.fx.local(Val::Ref(Heap::Idx(a64))), w.fx.local(Val::I32), w.fx.local(Val::I32));
            let (mask, pos) = (w.fx.local(Val::I32), w.fx.local(Val::I32));
            // capacity: 2 × live entries, at least 8, a power of two
            w.c().i32_const(8).local_set(cap);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(cap).local_get(0).struct_get(d, 5).i32_const(1).i32_shl().i32_ge_u().br_if(1);
            w.c().local_get(cap).i32_const(1).i32_shl().local_set(cap).br(0);
            w.c().end().end();
            w.c().local_get(0).struct_get(d, 0).local_set(ks).local_get(0).struct_get(d, 1).local_set(vs).local_get(0).struct_get(d, 2).local_set(hs);
            w.c().local_get(0).local_get(cap).array_new_default(ae).struct_set(d, 0);
            w.c().local_get(0).local_get(cap).array_new_default(ae).struct_set(d, 1);
            w.c().local_get(0).local_get(cap).array_new_default(a64).struct_set(d, 2);
            w.c().local_get(0).local_get(cap).i32_const(1).i32_shl().array_new_default(a32).struct_set(d, 3);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(0).struct_get(d, 4).i32_ge_u().br_if(1);
            w.c().local_get(ks).local_get(i).array_get(ae).global_get(del).ref_eq().i32_eqz().if_(Block::Empty);
            w.c().local_get(0).struct_get(d, 0).local_get(j).local_get(ks).local_get(i).array_get(ae).array_set(ae);
            w.c().local_get(0).struct_get(d, 1).local_get(j).local_get(vs).local_get(i).array_get(ae).array_set(ae);
            w.c().local_get(0).struct_get(d, 2).local_get(j).local_get(hs).local_get(i).array_get(a64).array_set(a64);
            // a free slot for it (keys are distinct): probe as dict_slot does
            w.c().local_get(0).struct_get(d, 3).array_len().i32_const(1).i32_sub().local_set(mask);
            w.c().local_get(hs).local_get(i).array_get(a64).i64_const(0x9e3779b97f4a7c15u64 as i64).i64_mul().i64_const(32).i64_rotl().i32_wrap_i64().local_get(mask).i32_and().local_set(pos);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(0).struct_get(d, 3).local_get(pos).array_get(a32).i32_eqz().br_if(1);
            w.c().local_get(pos).i32_const(1).i32_add().local_get(mask).i32_and().local_set(pos).br(0);
            w.c().end().end();
            w.c().local_get(0).struct_get(d, 3).local_get(pos).local_get(j).i32_const(1).i32_add().array_set(a32);
            w.c().local_get(j).i32_const(1).i32_add().local_set(j).end();
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            w.c().local_get(0).local_get(j).struct_set(d, 4);
            Ok(vec![])
        })
    }

    /// set(d, key, value)
    pub fn rt_dict_set(&mut self, _k: &Ty, _v: &Ty) -> Res<u32> {
        let slot = self.rt_dict_slot()?;
        let resize = self.rt_dict_resize()?;
        self.rt_fn("dict_set", |w| {
            let (d, ae, a64, a32, hashfn) = (w.t.dict, w.t.arr[3], w.t.arr[0], w.t.arr[2], w.t.hashfn);
            params!(w, Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Eq), Val::Ref(Heap::Eq));
            let (h, pos, e, u) = (w.fx.local(Val::I64), w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(Val::I32));
            w.c().local_get(1).local_get(0).struct_get(d, 6).call_ref(hashfn).local_set(h);
            w.c().local_get(0).local_get(1).local_get(h).call(slot).local_set(pos);
            w.c().local_get(0).struct_get(d, 3).local_get(pos).array_get(a32).local_tee(e).if_(Block::Empty);
            w.c().local_get(0).struct_get(d, 1).local_get(e).i32_const(1).i32_sub().local_get(2).array_set(ae).return_().end();
            // a new entry: room first
            w.c().local_get(0).struct_get(d, 4).local_get(0).struct_get(d, 0).array_len().i32_eq().if_(Block::Empty);
            w.c().local_get(0).call(resize).local_get(0).local_get(1).local_get(h).call(slot).local_set(pos).end();
            w.c().local_get(0).struct_get(d, 4).local_set(u);
            w.c().local_get(0).struct_get(d, 0).local_get(u).local_get(1).array_set(ae);
            w.c().local_get(0).struct_get(d, 1).local_get(u).local_get(2).array_set(ae);
            w.c().local_get(0).struct_get(d, 2).local_get(u).local_get(h).array_set(a64);
            w.c().local_get(0).struct_get(d, 3).local_get(pos).local_get(u).i32_const(1).i32_add().array_set(a32);
            w.c().local_get(0).local_get(u).i32_const(1).i32_add().struct_set(d, 4);
            w.c().local_get(0).local_get(0).struct_get(d, 5).i32_const(1).i32_add().struct_set(d, 5);
            Ok(vec![])
        })
    }

    /// d[k] → value (KeyError with the key's repr)
    pub fn rt_dict_get(&mut self, kt: &Ty) -> Res<u32> {
        let find = self.rt_dict_find()?;
        let repr = self.repr_fn(kt)?;
        let kv = self.val(kt)?;
        let key = format!("dict_get:{:?}", kt);
        let kt = kt.clone();
        self.rt_fn(&key, |w| {
            let (d, ae) = (w.t.dict, w.t.arr[3]);
            params!(w, Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Eq));
            let e = w.fx.local(Val::I32);
            w.c().local_get(0).local_get(1).call(find).local_tee(e).i32_const(0).i32_lt_s().if_(Block::Empty);
            w.c().local_get(1);
            w.unbox_eq(&kt)?;
            if kv.is_none() {
                w.str_const("None");
            } else {
                w.c().call(repr);
            }
            w.throw_kind("KeyError");
            w.c().end().local_get(0).struct_get(d, 1).local_get(e).array_get(ae);
            Ok(vec![Val::Ref(Heap::Eq)])
        })
    }

    /// d[k] of a defaultdict / Counter: inserts the default value
    pub fn rt_dict_get_default(&mut self, kt: &Ty, vt: &Ty, dk: DK) -> Res<u32> {
        let find = self.rt_dict_find()?;
        let set = self.rt_dict_set(kt, vt)?;
        let key = format!("dict_get_default:{:?}:{:?}", kt, vt);
        let vt = vt.clone();
        let _ = dk;
        self.rt_fn(&key, |w| {
            let (d, ae) = (w.t.dict, w.t.arr[3]);
            params!(w, Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Eq));
            let (e, v) = (w.fx.local(Val::I32), w.fx.local(Val::Ref(Heap::Eq)));
            w.c().local_get(0).local_get(1).call(find).local_tee(e).i32_const(0).i32_ge_s().if_(Block::Empty);
            w.c().local_get(0).struct_get(d, 1).local_get(e).array_get(ae).return_().end();
            w.default_value(&vt)?;
            w.box_eq(&vt)?;
            w.c().local_set(v).local_get(0).local_get(1).local_get(v).call(set).local_get(v);
            Ok(vec![Val::Ref(Heap::Eq)])
        })
    }

    /// The value of a type's no-argument constructor (`int()`, `list()`, ...).
    pub fn default_value(&mut self, t: &Ty) -> Res<()> {
        match t {
            Ty::Int => {
                self.c().i64_const(0);
            }
            Ty::Float => {
                self.c().f64_const(0.0);
            }
            Ty::Bool => {
                self.c().i32_const(0);
            }
            Ty::Str => self.str_const(""),
            Ty::List(e) => {
                let k = self.kind(e)?;
                let f = self.rt_list_new(k)?;
                self.c().i32_const(0).call(f);
            }
            Ty::Dict(k, _, _) | Ty::Set(k) => {
                let k = k.as_ref().clone();
                self.dict_new(&k)?;
            }
            t => return super::unsupported(format!("a default value of {:?}", t)),
        }
        Ok(())
    }

    /// del d[k]
    pub fn rt_dict_del(&mut self, kt: &Ty, _vt: &Ty) -> Res<u32> {
        let find = self.rt_dict_find()?;
        let repr = self.repr_fn(kt)?;
        let del = self.deleted();
        let key = format!("dict_del:{:?}", kt);
        let kt = kt.clone();
        self.rt_fn(&key, |w| {
            let (d, ae) = (w.t.dict, w.t.arr[3]);
            params!(w, Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Eq));
            let e = w.fx.local(Val::I32);
            w.c().local_get(0).local_get(1).call(find).local_tee(e).i32_const(0).i32_lt_s().if_(Block::Empty);
            w.c().local_get(1);
            w.unbox_eq(&kt)?;
            w.c().call(repr);
            w.throw_kind("KeyError");
            w.c().end();
            w.c().local_get(0).struct_get(d, 0).local_get(e).global_get(del).array_set(ae);
            w.c().local_get(0).struct_get(d, 1).local_get(e).ref_null(Heap::Eq).array_set(ae);
            w.c().local_get(0).local_get(0).struct_get(d, 5).i32_const(1).i32_sub().struct_set(d, 5);
            Ok(vec![])
        })
    }

    /// next(d, i) → the first live entry ≥ i, or -1
    pub fn rt_dict_next(&mut self) -> Res<u32> {
        let del = self.deleted();
        self.rt_fn("dict_next", |w| {
            let (d, ae) = (w.t.dict, w.t.arr[3]);
            params!(w, Val::Ref(Heap::Idx(d)), Val::I32);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(1).local_get(0).struct_get(d, 4).i32_ge_u().br_if(1);
            w.c().local_get(0).struct_get(d, 0).local_get(1).array_get(ae).global_get(del).ref_eq().i32_eqz().if_(Block::Empty).local_get(1).return_().end();
            w.c().local_get(1).i32_const(1).i32_add().local_set(1).br(0);
            w.c().end().end().i32_const(-1);
            Ok(vec![Val::I32])
        })
    }
}


//! Calls into the JavaScript host, and runtime functions built on them;
//! sorting, `str.join`, `list.reverse`.

use super::enc::{Block, Heap, Val};
use super::{unsupported, Res, HOST_ERRORS, H, K, W};
use crate::py::ir::*;

impl<'a> W<'a> {
    /// The value on the stack (of type `t`) → a host argument.
    pub fn host_arg(&mut self, t: &Ty) -> Res<()> {
        match t {
            Ty::Str => {
                let f = self.rt_str_arg()?;
                self.c().call(f);
            }
            Ty::Int => {
                let f = self.host(H::IntArg);
                self.c().call(f);
            }
            Ty::Bool => {
                let f = self.host(H::IntArg);
                self.c().i64_extend_i32_u().call(f);
            }
            Ty::Float => {
                let f = self.host(H::FloatArg);
                self.c().call(f);
            }
            Ty::List(e) if **e == Ty::Str => {
                // the strings, then their number
                let f = self.rt_list_arg()?;
                self.c().call(f);
            }
            t => return unsupported(format!("passing {:?} to the host", t)),
        }
        Ok(())
    }

    /// Run host operation `op` (arguments pushed) → its result as a `ret`.
    pub fn host_call(&mut self, op: &str, ret: &Ty) -> Res<()> {
        self.str_const(op);
        self.host_arg(&Ty::Str)?;
        let check = self.rt_host_check()?;
        match ret {
            Ty::Float => {
                let f = self.host(H::FCall);
                self.c().call(f).call(check);
            }
            Ty::Int => {
                let f = self.host(H::Call);
                self.c().call(f).call(check);
            }
            Ty::Bool => {
                let f = self.host(H::Call);
                self.c().call(f).call(check).i32_wrap_i64();
            }
            Ty::None => {
                let f = self.host(H::Call);
                self.c().call(f).call(check).drop();
            }
            Ty::Str => {
                let (f, p) = (self.host(H::Call), self.host(H::Pending));
                let take = self.rt_take_str()?;
                self.c().call(f).call(check).drop().call(p).call(take);
            }
            Ty::List(e) if **e == Ty::Str => {
                let f = self.host(H::Call);
                let items = self.rt_take_list()?;
                self.c().call(f).call(check).call(items);
            }
            Ty::Tuple(ts) if ts.iter().all(|t| *t == Ty::Str) => {
                // a list of strings → a tuple
                let f = self.host(H::Call);
                let items = self.rt_take_list()?;
                let tt = self.tuple_ty(ts)?;
                let l = self.fx.local(Val::Ref(Heap::Idx(self.t.list[K::Ref as usize])));
                self.c().call(f).call(check).call(items).local_set(l);
                for i in 0..ts.len() {
                    self.c().local_get(l).i64_const(i as i64);
                    self.list_get(&Ty::Str)?;
                }
                self.c().struct_new(tt);
            }
            t => return unsupported(format!("a host result of {:?}", t)),
        }
        Ok(())
    }

    /// host_check(): raise the error of the last host operation, if any.
    fn rt_host_check(&mut self) -> Res<u32> {
        let take = self.rt_take_str()?;
        self.rt_fn("host_check", |w| {
            let (failed, pending) = (w.host(H::Failed), w.host(H::Pending));
            let k = w.fx.local(Val::I32);
            w.c().call(failed).local_tee(k).if_(Block::Empty);
            for (i, kind) in HOST_ERRORS.iter().enumerate().skip(1) {
                w.c().local_get(k).i32_const(i as i32).i32_eq().if_(Block::Empty);
                w.c().call(pending).call(take);
                w.throw_kind(kind);
                w.c().end();
            }
            w.c().end();
            Ok(vec![])
        })
    }

    /// take_list(n) → list[str] of the host's last list result
    fn rt_take_list(&mut self) -> Res<u32> {
        let take = self.rt_take_str()?;
        let new = self.rt_list_new(K::Ref)?;
        let app = self.rt_list_append(K::Ref)?;
        self.rt_fn("take_list", |w| {
            w.fx.params = vec![Val::I64];
            let item = w.host(H::Item);
            let lt = w.t.list[K::Ref as usize];
            let (l, i) = (w.fx.local(Val::Ref(Heap::Idx(lt))), w.fx.local(Val::I32));
            w.c().local_get(0).i32_wrap_i64().call(new).local_set(l);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).i64_extend_i32_s().local_get(0).i64_ge_s().br_if(1);
            w.c().local_get(l).local_get(i).call(item).call(take).call(app);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().local_get(l);
            Ok(vec![Val::Ref(Heap::Idx(lt))])
        })
    }

    /// list_arg(xs): each string, then the count
    fn rt_list_arg(&mut self) -> Res<u32> {
        let arg = self.rt_str_arg()?;
        self.rt_fn("list_arg", |w| {
            let lt = w.t.list[K::Ref as usize];
            w.fx.params = vec![Val::Ref(Heap::Idx(lt))];
            let i = w.fx.local(Val::I32);
            let ia = w.host(H::IntArg);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(0).struct_get(lt, 1).i32_ge_s().br_if(1);
            w.c().local_get(0).local_get(i).i64_extend_i32_s();
            w.list_get(&Ty::Str)?;
            w.c().call(arg).local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().local_get(0).struct_get(lt, 1).i64_extend_i32_s().call(ia);
            Ok(vec![])
        })
    }

    /// int(s, base)
    pub fn rt_parse_int(&mut self) -> Res<u32> {
        self.rt_fn("parse_int", |w| {
            w.fx.params = vec![Val::Ref(Heap::Idx(w.t.str)), Val::I32];
            w.c().local_get(0);
            w.host_arg(&Ty::Str)?;
            w.c().local_get(1).i64_extend_i32_s();
            w.host_arg(&Ty::Int)?;
            w.host_call("int", &Ty::Int)?;
            Ok(vec![Val::I64])
        })
    }

    /// float(s)
    pub fn rt_parse_float(&mut self) -> Res<u32> {
        self.rt_fn("parse_float", |w| {
            w.fx.params = vec![Val::Ref(Heap::Idx(w.t.str))];
            w.c().local_get(0);
            w.host_arg(&Ty::Str)?;
            w.host_call("float", &Ty::Float)?;
            Ok(vec![Val::F64])
        })
    }

    /// int(x) of a float: truncated; NaN and infinities raise
    pub fn rt_float_int(&mut self) -> Res<u32> {
        self.rt_fn("float_int", |w| {
            w.fx.params = vec![Val::F64];
            w.c().local_get(0).local_get(0).f64_ne().if_(Block::Empty);
            w.throw_lit("ValueError", "cannot convert float NaN to integer");
            w.c().end();
            w.c().local_get(0).f64_abs().f64_const(9.223372036854775807e18).f64_ge().if_(Block::Empty);
            w.throw_lit("OverflowError", "cannot convert float infinity to integer");
            w.c().end();
            w.c().local_get(0).i64_trunc_f64_s();
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_ord(&mut self) -> Res<u32> {
        self.rt_fn("ord", |w| {
            w.fx.params = vec![Val::Ref(Heap::Idx(w.t.str))];
            w.c().local_get(0);
            w.host_arg(&Ty::Str)?;
            w.host_call("ord", &Ty::Int)?;
            Ok(vec![Val::I64])
        })
    }

    pub fn rt_chr(&mut self) -> Res<u32> {
        self.rt_fn("chr", |w| {
            w.fx.params = vec![Val::I64];
            w.c().local_get(0);
            w.host_arg(&Ty::Int)?;
            w.host_call("chr", &Ty::Str)?;
            Ok(vec![Val::Ref(Heap::Idx(w.t.str))])
        })
    }

    /// input(): a line without its newline (EOFError at the end)
    pub fn rt_read_line(&mut self) -> Res<u32> {
        self.rt_fn("read_line", |w| {
            w.host_call("input", &Ty::Str)?;
            Ok(vec![Val::Ref(Heap::Idx(w.t.str))])
        })
    }

    /// sep.join(strs)
    pub fn rt_str_join(&mut self) -> Res<u32> {
        let sbn = self.rt_sb_new()?;
        let sba = self.rt_sb_add()?;
        let sbs = self.rt_sb_str()?;
        self.rt_fn("str_join", |w| {
            let lt = w.t.list[K::Ref as usize];
            w.fx.params = vec![Val::Ref(Heap::Idx(w.t.str)), Val::Ref(Heap::Idx(lt))];
            let (sb, i) = (w.fx.local(Val::Ref(Heap::Idx(w.t.sb))), w.fx.local(Val::I32));
            w.c().i32_const(0).call(sbn).local_set(sb);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(1).struct_get(lt, 1).i32_ge_s().br_if(1);
            w.c().local_get(i).if_(Block::Empty).local_get(sb).local_get(0).call(sba).end();
            w.c().local_get(sb).local_get(1).local_get(i).i64_extend_i32_s();
            w.list_get(&Ty::Str)?;
            w.c().call(sba).local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end().local_get(sb).call(sbs);
            Ok(vec![Val::Ref(Heap::Idx(w.t.str))])
        })
    }

    pub fn rt_list_reverse(&mut self, k: K) -> Res<u32> {
        self.rt_fn(&format!("list_reverse_{:?}", k), |w| {
            let (arr, lt) = (w.t.arr[k as usize], w.t.list[k as usize]);
            let ev = match k {
                K::I64 => Val::I64,
                K::F64 => Val::F64,
                K::I32 => Val::I32,
                K::Ref => Val::Ref(Heap::Eq),
            };
            w.fx.params = vec![Val::Ref(Heap::Idx(lt))];
            let (i, j, t, a) = (w.fx.local(Val::I32), w.fx.local(Val::I32), w.fx.local(ev), w.fx.local(Val::Ref(Heap::Idx(arr))));
            w.c().local_get(0).struct_get(lt, 0).local_set(a).local_get(0).struct_get(lt, 1).i32_const(1).i32_sub().local_set(j);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(j).i32_ge_s().br_if(1);
            w.c().local_get(a).local_get(i).array_get(arr).local_set(t);
            w.c().local_get(a).local_get(i).local_get(a).local_get(j).array_get(arr).array_set(arr);
            w.c().local_get(a).local_get(j).local_get(t).array_set(arr);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).local_get(j).i32_const(1).i32_sub().local_set(j).br(0);
            w.c().end().end();
            Ok(vec![])
        })
    }

    /// sort_perm(keys) → the stable sorting permutation (merge sort), as an i32 array
    pub fn rt_sort_perm(&mut self, kt: &Ty, rev: bool) -> Res<u32> {
        let k = self.kind(kt)?;
        let cmp = self.cmp_fn(kt)?;
        let key = format!("sort_perm:{:?}:{}", kt, rev);
        let kt = kt.clone();
        self.rt_fn(&key, move |w| {
            let (lt, a32) = (w.t.list[k as usize], w.t.arr_i32);
            w.fx.params = vec![Val::Ref(Heap::Idx(lt))];
            let n = w.fx.local(Val::I32);
            let (p, q, width, lo, mid, hi, i, j, o) = (
                w.fx.local(Val::Ref(Heap::Idx(a32))),
                w.fx.local(Val::Ref(Heap::Idx(a32))),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
                w.fx.local(Val::I32),
            );
            let tmp = w.fx.local(Val::Ref(Heap::Idx(a32)));
            w.c().local_get(0).struct_get(lt, 1).local_tee(n).array_new_default(a32).local_set(p);
            w.c().local_get(n).array_new_default(a32).local_set(q);
            // identity
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(n).i32_ge_s().br_if(1);
            w.c().local_get(p).local_get(i).local_get(i).array_set(a32).local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            // bottom-up merge sort: runs of `width`
            w.c().i32_const(1).local_set(width);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(width).local_get(n).i32_ge_s().br_if(1);
            w.c().i32_const(0).local_set(lo);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(lo).local_get(n).i32_ge_s().br_if(1);
            // mid = min(lo + width, n), hi = min(lo + 2 width, n)
            w.c().local_get(lo).local_get(width).i32_add().local_tee(mid).local_get(n).local_get(mid).local_get(n).i32_lt_s().select().local_set(mid);
            w.c().local_get(mid).local_get(width).i32_add().local_tee(hi).local_get(n).local_get(hi).local_get(n).i32_lt_s().select().local_set(hi);
            w.c().local_get(lo).local_tee(i).local_set(o).local_get(mid).local_set(j);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(o).local_get(hi).i32_ge_s().br_if(1);
            // take from the right only when it is strictly before the left (stable)
            w.c().local_get(j).local_get(hi).i32_lt_s().if_(Block::Val(Val::I32));
            w.c().local_get(i).local_get(mid).i32_ge_s().if_(Block::Val(Val::I32)).i32_const(1).else_();
            for side in [j, i] {
                w.c().local_get(0).local_get(p).local_get(side).array_get(a32).i64_extend_i32_s();
                w.list_get(&kt)?;
            }
            w.c().call(cmp);
            if rev {
                w.c().i32_const(0).i32_gt_s();
            } else {
                w.c().i32_const(0).i32_lt_s();
            }
            w.c().end().else_().i32_const(0).end();
            w.c().if_(Block::Empty);
            w.c().local_get(q).local_get(o).local_get(p).local_get(j).array_get(a32).array_set(a32).local_get(j).i32_const(1).i32_add().local_set(j);
            w.c().else_();
            w.c().local_get(q).local_get(o).local_get(p).local_get(i).array_get(a32).array_set(a32).local_get(i).i32_const(1).i32_add().local_set(i);
            w.c().end();
            w.c().local_get(o).i32_const(1).i32_add().local_set(o).br(0);
            w.c().end().end();
            w.c().local_get(lo).local_get(width).i32_const(1).i32_shl().i32_add().local_set(lo).br(0);
            w.c().end().end();
            w.c().local_get(p).local_set(tmp).local_get(q).local_set(p).local_get(tmp).local_set(q);
            w.c().local_get(width).i32_const(1).i32_shl().local_set(width).br(0);
            w.c().end().end().local_get(p);
            Ok(vec![Val::Ref(Heap::Idx(a32))])
        })
    }

    /// apply_perm(xs, perm): xs reordered in place
    pub fn rt_apply_perm(&mut self, k: K) -> Res<u32> {
        self.rt_fn(&format!("apply_perm_{:?}", k), |w| {
            let (arr, lt, a32) = (w.t.arr[k as usize], w.t.list[k as usize], w.t.arr_i32);
            w.fx.params = vec![Val::Ref(Heap::Idx(lt)), Val::Ref(Heap::Idx(a32))];
            let (old, i) = (w.fx.local(Val::Ref(Heap::Idx(arr))), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(lt, 0).local_set(old);
            w.c().local_get(0).local_get(old).array_len().array_new_default(arr).struct_set(lt, 0);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(i).local_get(1).array_len().i32_ge_s().br_if(1);
            w.c().local_get(0).struct_get(lt, 0).local_get(i).local_get(old).local_get(1).local_get(i).array_get(a32).array_get(arr).array_set(arr);
            w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
            w.c().end().end();
            Ok(vec![])
        })
    }

    /// math.gcd(a, b)
    pub fn rt_gcd(&mut self) -> Res<u32> {
        self.rt_fn("gcd", |w| {
            w.fx.params = vec![Val::I64, Val::I64];
            let t = w.fx.local(Val::I64);
            for i in [0, 1] {
                w.c().local_get(i).i64_const(0).i64_lt_s().if_(Block::Empty).i64_const(0).local_get(i).i64_sub().local_set(i).end();
            }
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(1).i64_eqz().br_if(1);
            w.c().local_get(0).local_get(1).i64_rem_u().local_set(t).local_get(1).local_set(0).local_get(t).local_set(1).br(0);
            w.c().end().end().local_get(0);
            Ok(vec![Val::I64])
        })
    }

    /// math.lcm(a, b)
    pub fn rt_lcm(&mut self) -> Res<u32> {
        let gcd = self.rt_gcd()?;
        let mul = self.rt_int_mul()?;
        self.rt_fn("lcm", |w| {
            w.fx.params = vec![Val::I64, Val::I64];
            let g = w.fx.local(Val::I64);
            w.c().local_get(0).local_get(1).call(gcd).local_tee(g).i64_eqz().if_(Block::Empty).i64_const(0).return_().end();
            w.c().local_get(0).local_get(g).i64_div_s().local_get(1).call(mul).local_tee(g).i64_const(0).i64_lt_s().if_(Block::Empty).i64_const(0).local_get(g).i64_sub().local_set(g).end().local_get(g);
            Ok(vec![Val::I64])
        })
    }

    /// math.isqrt(n)
    pub fn rt_isqrt(&mut self) -> Res<u32> {
        self.rt_fn("isqrt", |w| {
            w.fx.params = vec![Val::I64];
            let r = w.fx.local(Val::I64);
            w.c().local_get(0).i64_const(0).i64_lt_s().if_(Block::Empty);
            w.throw_lit("ValueError", "isqrt() argument must be nonnegative");
            w.c().end();
            // the float estimate, then corrected
            w.c().local_get(0).f64_convert_i64_s().f64_sqrt().i64_trunc_f64_s().local_set(r);
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(r).local_get(r).i64_mul().local_get(0).i64_le_s().local_get(r).i64_const(3037000499).i64_le_s().i32_and().br_if(1);
            w.c().local_get(r).i64_const(1).i64_sub().local_set(r).br(0);
            w.c().end().end();
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(r).i64_const(3037000499).i64_ge_s().br_if(1);
            w.c().local_get(r).i64_const(1).i64_add().local_tee(r).local_get(r).i64_mul().local_get(0).i64_gt_s().if_(Block::Empty).local_get(r).i64_const(1).i64_sub().local_set(r).br(2).end();
            w.c().br(0);
            w.c().end().end().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    /// pow(a, b, m)
    pub fn rt_powmod(&mut self) -> Res<u32> {
        let md = self.rt_int_mod()?;
        self.rt_fn("powmod", |w| {
            w.fx.params = vec![Val::I64, Val::I64, Val::I64];
            let (m, r, x, e) = (w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64));
            w.c().local_get(2).i64_eqz().if_(Block::Empty);
            w.throw_lit("ValueError", "pow() 3rd argument cannot be 0");
            w.c().end();
            w.c().local_get(1).i64_const(0).i64_lt_s().if_(Block::Empty);
            w.throw_lit("ValueError", "py2rs: negative exponent in pow(a, b, m) is not supported");
            w.c().end();
            // |m| (unsigned), the base reduced into [0, |m|)
            w.c().local_get(2).i64_const(0).i64_lt_s().if_(Block::Val(Val::I64)).i64_const(0).local_get(2).i64_sub().else_().local_get(2).end().local_set(m);
            w.c().local_get(0).local_get(m).call(md).local_set(x);
            w.c().i64_const(1).local_get(m).i64_rem_u().local_set(r).local_get(1).local_set(e);
            // (p + q) mod m without overflow, unsigned (p, q < m)
            let addmod = |w: &mut W<'a>, p: u32, q: u32, dst: u32| {
                w.c().local_get(p).local_get(q).i64_add().local_set(dst);
                w.c().local_get(dst).local_get(p).i64_lt_u().local_get(dst).local_get(m).i64_ge_u().i32_or().if_(Block::Empty).local_get(dst).local_get(m).i64_sub().local_set(dst).end();
            };
            let (acc, y, t) = (w.fx.local(Val::I64), w.fx.local(Val::I64), w.fx.local(Val::I64));
            // mulmod(a, b) into dst by doubling
            let mulmod = |w: &mut W<'a>, a: u32, b: u32, dst: u32| {
                w.c().i64_const(0).local_set(acc).local_get(a).local_set(t).local_get(b).local_set(y);
                w.c().block(Block::Empty).loop_(Block::Empty);
                w.c().local_get(y).i64_eqz().br_if(1);
                w.c().local_get(y).i64_const(1).i64_and().i64_eqz().i32_eqz().if_(Block::Empty);
                addmod(w, acc, t, acc);
                w.c().end();
                addmod(w, t, t, t);
                w.c().local_get(y).i64_const(1).i64_shr_u().local_set(y).br(0);
                w.c().end().end().local_get(acc).local_set(dst);
            };
            w.c().block(Block::Empty).loop_(Block::Empty);
            w.c().local_get(e).i64_eqz().br_if(1);
            w.c().local_get(e).i64_const(1).i64_and().i64_eqz().i32_eqz().if_(Block::Empty);
            mulmod(w, r, x, r);
            w.c().end();
            mulmod(w, x, x, x);
            w.c().local_get(e).i64_const(1).i64_shr_u().local_set(e).br(0);
            w.c().end().end();
            // the sign of m
            w.c().local_get(2).i64_const(0).i64_lt_s().local_get(r).i64_eqz().i32_eqz().i32_and().if_(Block::Empty).local_get(r).local_get(m).i64_sub().local_set(r).end();
            w.c().local_get(r);
            Ok(vec![Val::I64])
        })
    }

    /// a | b, a & b, a - b, a ^ b on sets
    pub fn rt_set_op(&mut self, t: &Ty, op: BinOp) -> Res<u32> {
        let new = self.rt_dict_new()?;
        let next = self.rt_dict_next()?;
        let find = self.rt_dict_find()?;
        let set = self.rt_dict_set(t, &Ty::None)?;
        self.rt_fn(&format!("set_op_{:?}", op), move |w| {
            let (d, ae) = (w.t.dict, w.t.arr[3]);
            w.fx.params = vec![Val::Ref(Heap::Idx(d)), Val::Ref(Heap::Idx(d))];
            let (r, i) = (w.fx.local(Val::Ref(Heap::Idx(d))), w.fx.local(Val::I32));
            w.c().local_get(0).struct_get(d, 6).local_get(0).struct_get(d, 7).call(new).local_set(r);
            // keys of `src` kept when (found in `other`) == keep_found; None: all
            let pass = |w: &mut W<'a>, src: u32, other: u32, keep_found: Option<bool>| {
                w.c().i32_const(0).local_set(i);
                w.c().block(Block::Empty).loop_(Block::Empty);
                w.c().local_get(src).local_get(i).call(next).local_tee(i).i32_const(0).i32_lt_s().br_if(1);
                match keep_found {
                    None => w.c().i32_const(1),
                    Some(k) => {
                        w.c().local_get(other).local_get(src).struct_get(d, 0).local_get(i).array_get(ae).call(find).i32_const(0).i32_ge_s();
                        if k {
                            w.c()
                        } else {
                            w.c().i32_eqz()
                        }
                    }
                };
                w.c().if_(Block::Empty).local_get(r).local_get(src).struct_get(d, 0).local_get(i).array_get(ae).ref_null(Heap::Eq).call(set).end();
                w.c().local_get(i).i32_const(1).i32_add().local_set(i).br(0);
                w.c().end().end();
            };
            match op {
                BinOp::BitOr => {
                    pass(w, 0, 1, None);
                    pass(w, 1, 0, None);
                }
                BinOp::BitAnd => pass(w, 0, 1, Some(true)),
                BinOp::Sub => pass(w, 0, 1, Some(false)),
                _ => {
                    pass(w, 0, 1, Some(false));
                    pass(w, 1, 0, Some(false));
                }
            }
            w.c().local_get(r);
            Ok(vec![Val::Ref(Heap::Idx(d))])
        })
    }
}

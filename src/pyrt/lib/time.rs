//! More of `time` than py0 builds in: declared by `src/py/lib/time.pyi`.

/// The local time now: (year, month, day, hour, minute, second, weekday Mon=0, yearday 1..).
fn localtime() -> [i64; 8] {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    #[cfg(unix)]
    {
        let t = now as libc::time_t;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        if !unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
            return [
                tm.tm_year as i64 + 1900,
                tm.tm_mon as i64 + 1,
                tm.tm_mday as i64,
                tm.tm_hour as i64,
                tm.tm_min as i64,
                tm.tm_sec as i64,
                (tm.tm_wday as i64 + 6) % 7,
                tm.tm_yday as i64 + 1,
            ];
        }
    }
    // UTC (civil from days, Howard Hinnant's algorithm)
    let days = now.div_euclid(86400);
    let secs = now.rem_euclid(86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][(m - 1) as usize] + if leap && m > 2 { 1 } else { 0 };
    [y, m, d, secs / 3600, secs / 60 % 60, secs % 60, (days + 3).rem_euclid(7), before + d]
}

const DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTHS: [&str; 12] =
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/// `time.strftime(format)` for the local time now (the C locale's names).
pub fn strftime(fmt: &str) -> String {
    let [y, mo, d, h, mi, s, wd, yd] = localtime();
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y % 100)),
            Some('m') => out.push_str(&format!("{:02}", mo)),
            Some('d') => out.push_str(&format!("{:02}", d)),
            Some('H') => out.push_str(&format!("{:02}", h)),
            Some('I') => out.push_str(&format!("{:02}", if h % 12 == 0 { 12 } else { h % 12 })),
            Some('M') => out.push_str(&format!("{:02}", mi)),
            Some('S') => out.push_str(&format!("{:02}", s)),
            Some('p') => out.push_str(if h < 12 { "AM" } else { "PM" }),
            Some('j') => out.push_str(&format!("{:03}", yd)),
            Some('A') => out.push_str(DAYS[wd as usize]),
            Some('a') => out.push_str(&DAYS[wd as usize][..3]),
            Some('B') => out.push_str(MONTHS[(mo - 1) as usize]),
            Some('b') => out.push_str(&MONTHS[(mo - 1) as usize][..3]),
            Some('F') => out.push_str(&format!("{}-{:02}-{:02}", y, mo, d)),
            Some('T') => out.push_str(&format!("{:02}:{:02}:{:02}", h, mi, s)),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

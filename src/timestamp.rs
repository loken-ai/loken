//! Wall-clock timestamps as the API writes and reads them.
//!
//! A written timestamp is RFC 3339 in UTC with the offset spelled `+00:00`.
//! Its fraction holds 0, 3, 6 or 9 digits: the fewest of those that keep every nonzero digit.

use jiff::fmt::temporal::DateTimePrinter;
use jiff::tz::Offset;
use jiff::{SignedDuration, Timestamp};
use std::time::SystemTime;

/// The current time in whole seconds since the Unix epoch.
pub fn now_secs() -> i64 {
    Timestamp::now().as_second()
}

/// The current time in RFC 3339.
pub fn now_rfc3339() -> String {
    rfc3339(Timestamp::now())
}

/// The time `secs` seconds from now in RFC 3339.
///
/// A sum past the range of a timestamp is held at the bound it passes.
pub fn rfc3339_in_secs(secs: i64) -> String {
    let now = Timestamp::now();
    let at = now
        .checked_add(SignedDuration::from_secs(secs))
        .unwrap_or(if secs < 0 {
            Timestamp::MIN
        } else {
            Timestamp::MAX
        });
    rfc3339(at)
}

/// A system time in RFC 3339, or `None` past the range of a timestamp.
pub fn system_time_rfc3339(t: SystemTime) -> Option<String> {
    Timestamp::try_from(t).ok().map(rfc3339)
}

/// The whole seconds since the Unix epoch of an RFC 3339 timestamp, or `None` when it does not parse.
pub fn parse_rfc3339_secs(s: &str) -> Option<i64> {
    s.parse::<Timestamp>().ok().map(Timestamp::as_second)
}

/// `ts` in RFC 3339, in UTC.
pub fn rfc3339(ts: Timestamp) -> String {
    let nanos = ts.subsec_nanosecond().unsigned_abs();
    let digits = if nanos == 0 {
        0
    } else if nanos.is_multiple_of(1_000_000) {
        3
    } else if nanos.is_multiple_of(1_000) {
        6
    } else {
        9
    };
    DateTimePrinter::new()
        .precision(Some(digits))
        .timestamp_with_offset_to_string(&ts, Offset::UTC)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: i64, nanos: i32) -> Timestamp {
        Timestamp::new(secs, nanos).unwrap()
    }

    #[test]
    fn the_fraction_takes_the_fewest_groups_of_3_digits() {
        assert_eq!(rfc3339(at(1_700_000_000, 0)), "2023-11-14T22:13:20+00:00");
        assert_eq!(
            rfc3339(at(1_700_000_000, 120_000_000)),
            "2023-11-14T22:13:20.120+00:00"
        );
        assert_eq!(
            rfc3339(at(1_700_000_000, 123_400_000)),
            "2023-11-14T22:13:20.123400+00:00"
        );
        assert_eq!(
            rfc3339(at(1_700_000_000, 123_456_780)),
            "2023-11-14T22:13:20.123456780+00:00"
        );
        assert_eq!(
            rfc3339(at(1_700_000_000, 1)),
            "2023-11-14T22:13:20.000000001+00:00"
        );
    }

    #[test]
    fn the_current_time_has_the_written_shape() {
        let s = now_rfc3339();
        assert!(s.ends_with("+00:00"), "{s}");
        assert_eq!(s.as_bytes()[10], b'T', "{s}");
        let fraction = s
            .strip_suffix("+00:00")
            .unwrap()
            .split_once('.')
            .map_or(0, |(_, f)| f.len());
        assert!([0, 3, 6, 9].contains(&fraction), "{s}");
    }

    #[test]
    fn a_written_timestamp_reads_back_to_its_second() {
        let ts = at(1_700_000_000, 987_654_321);
        assert_eq!(parse_rfc3339_secs(&rfc3339(ts)), Some(1_700_000_000));
    }

    #[test]
    fn rfc3339_offsets_and_suffixes_parse() {
        assert_eq!(
            parse_rfc3339_secs("2023-11-14T22:13:20Z"),
            Some(1_700_000_000)
        );
        assert_eq!(
            parse_rfc3339_secs("2023-11-14T22:13:20.5z"),
            Some(1_700_000_000)
        );
        assert_eq!(
            parse_rfc3339_secs("2023-11-15T00:13:20+02:00"),
            Some(1_700_000_000)
        );
        assert_eq!(
            parse_rfc3339_secs("2023-11-14T21:13:20-01:00"),
            Some(1_700_000_000)
        );
        assert_eq!(parse_rfc3339_secs("2023-11-14"), None);
        assert_eq!(parse_rfc3339_secs("2023-11-14T22:13:20"), None);
        assert_eq!(parse_rfc3339_secs("never"), None);
    }

    #[test]
    fn a_time_ahead_is_that_many_seconds_away() {
        let before = now_secs();
        let ahead = parse_rfc3339_secs(&rfc3339_in_secs(30 * 60)).unwrap();
        let after = now_secs();
        assert!((before + 30 * 60..=after + 30 * 60).contains(&ahead));
    }

    #[test]
    fn a_time_ahead_past_the_range_is_held_at_the_bound() {
        assert_eq!(rfc3339_in_secs(i64::MAX), rfc3339(Timestamp::MAX));
        assert_eq!(rfc3339_in_secs(i64::MIN), rfc3339(Timestamp::MIN));
    }

    #[test]
    fn a_system_time_writes_as_its_timestamp() {
        let t = SystemTime::UNIX_EPOCH + std::time::Duration::new(1_700_000_000, 250_000_000);
        assert_eq!(
            system_time_rfc3339(t).as_deref(),
            Some("2023-11-14T22:13:20.250+00:00")
        );
    }
}

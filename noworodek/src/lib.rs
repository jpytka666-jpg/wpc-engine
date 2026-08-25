pub const VERSION: &str = "0.0.0";

#[cfg(test)]
mod tests {
    #[test]
    fn noworodek_crate_is_reachable() {
        assert_eq!(crate::VERSION, "0.1.0");
    }
}

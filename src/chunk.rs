pub fn chunk_text(
    text: &str,
    chunk_size: usize,
    chars_per_token: f64,
    overlap_ratio: f64,
) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let chunk_size = (chunk_size as f64 * chars_per_token) as usize;
    let overlap = (chunk_size as f64 * overlap_ratio) as usize;

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end == chars.len() {
            break;
        }
        start = end - overlap;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_chunk_when_text_fits() {
        let chunks = chunk_text("hola mundo", 10, 3.5, 0.15);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hola mundo");
    }

    #[test]
    fn splits_into_multiple_chunks_with_overlap() {
        let text = "0123456789".repeat(10);
        let chunks = chunk_text(&text, 10, 3.0, 0.5);

        assert!(chunks.len() > 1);

        let first_tail = &chunks[0][chunks[0].len() - 15..];
        let second_head = &chunks[1][..15];
        assert_eq!(first_tail, second_head);
    }

    #[test]
    fn does_not_panic_on_accented_characters() {
        let text = "café niño mañana ñoño áéíóú".repeat(20);
        let chunks = chunk_text(&text, 5, 3.5, 0.1);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() > 0);
        }
    }
}

use wpc_runtime::forward_batch::{KvLayer, MmapF32};

#[test]
fn mmap_buffer_reuses_capacity_without_reallocation_until_needed() {
    let mut buf = MmapF32::new(8).expect("initial mmap allocation");
    let initial_capacity = buf.capacity();
    assert_eq!(initial_capacity, 8);

    {
        let slice = buf.as_mut_slice();
        for (i, value) in slice.iter_mut().enumerate() {
            *value = i as f32;
        }
    }
    buf.mark_used(8).expect("mark used");

    buf.ensure_capacity(8).expect("existing capacity must be reused");
    assert_eq!(buf.capacity(), initial_capacity);
    assert_eq!(buf.as_slice()[7], 7.0);
}

#[test]
fn kv_layer_growth_preserves_existing_rows_and_reuses_capacity() {
    let mut kv = KvLayer::with_capacity(4, 2).expect("kv allocation");
    let first_capacity = kv.keys.capacity();

    kv.append_batch(&[1.0; 8], &[2.0; 8], 2)
        .expect("first KV append");
    assert_eq!(kv.seq_len, 2);
    assert_eq!(kv.get_key_row(1).expect("key row")[0], 1.0);

    kv.append_batch(&[3.0; 4], &[4.0; 4], 1)
        .expect("second KV append");
    assert!(kv.keys.capacity() >= first_capacity);
    assert_eq!(kv.seq_len, 3);
    assert_eq!(kv.get_key_row(1).expect("preserved key row")[0], 1.0);
    assert_eq!(kv.get_value_row(2).expect("new value row")[0], 4.0);

    let grown_capacity = kv.keys.capacity();
    kv.append_batch(&[5.0; 4], &[6.0; 4], 1)
        .expect("third KV append");
    assert_eq!(kv.keys.capacity(), grown_capacity);
    assert_eq!(kv.seq_len, 4);
}

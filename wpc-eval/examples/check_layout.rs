use wpc_format::CompressedBlock;
fn main() {
    println!("sizeof = {}", std::mem::size_of::<CompressedBlock>());
    let b = CompressedBlock {
        pattern_id: 0x25,
        residual_id: 0x676D,
        base_value: half::f16::from_f32(-0.0118),
        scale: 5,
    };
    let bytes = b.to_le_bytes();
    print!("bytes = ");
    for x in bytes {
        print!("{:02X} ", x);
    }
    println!();
}

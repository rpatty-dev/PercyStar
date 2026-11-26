use rusty_opus::{Application, OpusEncoder};
use std::{
    fs,
    process::{self, Command},
    time::{SystemTime, UNIX_EPOCH},
};

fn synthetic_bank(stream_count: u32, packet: &[u8]) -> Vec<u8> {
    let data_size = 2 + packet.len();
    let format_size = 32;
    let header_size = 16 + stream_count as usize * (4 + 2 * 8);
    let mut bank = b"ADAT".to_vec();
    bank.extend_from_slice(&0xFF02_0001u32.to_le_bytes());
    bank.extend_from_slice(&stream_count.to_le_bytes());
    bank.extend_from_slice(&(header_size as u32).to_le_bytes());
    for id in 1..=stream_count {
        bank.extend_from_slice(&((2u32 << 29) | id).to_le_bytes());
    }
    for index in 0..stream_count as usize {
        let offset = header_size + index * (data_size + format_size);
        for (tag, size, offset) in [
            (0x55u64, data_size, offset),
            (0x76u64, format_size, offset + data_size),
        ] {
            let chunk = (tag << 56) | ((size as u64) << 28) | offset as u64;
            bank.extend_from_slice(&chunk.to_le_bytes());
        }
    }
    let mut format = [0; 32];
    format[..4].copy_from_slice(&960u32.to_le_bytes());
    format[8..10].copy_from_slice(&48_000u16.to_le_bytes());
    format[0x1C..0x1E].copy_from_slice(&[11, 1]);
    for _ in 0..stream_count {
        bank.extend_from_slice(&(packet.len() as u16).to_le_bytes());
        bank.extend_from_slice(packet);
        bank.extend_from_slice(&format);
    }
    bank
}

#[test]
fn decode_audio_exports_the_first_bank_above_256_streams() {
    let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
    let mut opus = [0; 256];
    let packet_size = encoder.encode(&[0.0; 960], 960, &mut opus).unwrap();
    let packet = &opus[..packet_size];

    let mut archive = vec![0; 0x1000];
    archive.extend_from_slice(&synthetic_bank(257, packet));
    // A second bank keeps the scan bounded even if the first bank is skipped.
    archive.resize(0x1_0000, 0);
    archive.extend_from_slice(&synthetic_bank(1, packet));

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = std::env::temp_dir().join(format!("percystar-adat-{}-{stamp}", process::id()));
    fs::create_dir(&tmp).unwrap();
    let input = tmp.join("synthetic.rpf");
    let out_dir = tmp.join("wav");
    fs::write(&input, archive).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_PercyStar"))
        .arg("--decode-audio")
        .arg(&input)
        .arg(&out_dir)
        .arg("1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_dir(&out_dir).unwrap().count(),
        257,
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    for index in 0..257 {
        let name = format!("bank_000_off_00001000_257st_s{index:02}.wav");
        let wav = fs::read(out_dir.join(name)).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[24..28], &48_000u32.to_le_bytes());
        assert_eq!(wav.len(), 44 + 960 * 2);
    }
    fs::remove_dir_all(tmp).unwrap();
}

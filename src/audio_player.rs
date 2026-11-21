use crate::awc;
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct LoadedWavTrack {
    pub stream_idx: u32,
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub total_frames: usize,
    pub duration_secs: f32,
    pub coff: usize,
    pub csz: usize,
    pub sub_stream: Option<(usize, usize)>,
    pub waveform_peaks: Vec<f32>,
    pub waveform_rms: Vec<f32>,
    pub overall_peak: f32,
}

impl LoadedWavTrack {
    // compute waveform summary and duration from decoded pcm without keeping samples in ram
    pub fn from_pcm_summary(
        stream_idx: u32,
        name: String,
        sample_rate: u32,
        channels: u16,
        samples: &[i16],
        coff: usize,
        csz: usize,
        sub_stream: Option<(usize, usize)>,
    ) -> Self {
        let ch = channels.max(1) as usize;
        let sr = sample_rate.max(1);
        let total_frames = samples.len() / ch;
        let duration_secs = total_frames as f32 / sr as f32;

        const BINS: usize = 200;
        let mut waveform_peaks = vec![0.0f32; BINS];
        let mut waveform_rms = vec![0.0f32; BINS];
        let mut overall_peak = 0.0f32;

        if total_frames > 0 {
            for b in 0..BINS {
                let f0 = b * total_frames / BINS;
                let f1 = ((b + 1) * total_frames / BINS).max(f0 + 1).min(total_frames);
                let stride = ((f1 - f0) / 128).max(1);
                let mut peak = 0.0f32;
                let mut sum_sq = 0.0f64;
                let mut cnt = 0usize;
                let mut f = f0;
                while f < f1 {
                    let base = f * ch;
                    let mut frame_abs = 0.0f32;
                    for c in 0..ch {
                        let s = (samples[base + c] as f32 / 32768.0).abs();
                        if s > frame_abs {
                            frame_abs = s;
                        }
                        sum_sq += (s as f64) * (s as f64);
                        cnt += 1;
                    }
                    if frame_abs > peak {
                        peak = frame_abs;
                    }
                    f += stride;
                }
                waveform_peaks[b] = peak;
                waveform_rms[b] = if cnt > 0 { (sum_sq / cnt as f64).sqrt() as f32 } else { 0.0 };
                if peak > overall_peak {
                    overall_peak = peak;
                }
            }
        }

        Self {
            stream_idx,
            name,
            sample_rate: sr,
            channels: channels.max(1),
            total_frames,
            duration_secs,
            coff,
            csz,
            sub_stream,
            waveform_peaks,
            waveform_rms,
            overall_peak,
        }
    }

    // instantaneous peak and rms loudness around frame within ~40ms window
    pub fn loudness_at_frame(samples: &[i16], channels: u16, sample_rate: u32, total_frames: usize, frame: usize) -> (f32, f32) {
        if total_frames == 0 || samples.is_empty() {
            return (0.0, 0.0);
        }
        let ch = channels.max(1) as usize;
        let win = (sample_rate as usize / 25).max(64);
        let f0 = frame.saturating_sub(win / 2);
        let f1 = (frame + win / 2).min(samples.len() / ch);
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut cnt = 0usize;
        for f in f0..f1 {
            let base = f * ch;
            for c in 0..ch {
                let s = (samples[base + c] as f32 / 32768.0).abs();
                if s > peak {
                    peak = s;
                }
                sum_sq += (s as f64) * (s as f64);
                cnt += 1;
            }
        }
        let rms = if cnt > 0 { (sum_sq / cnt as f64).sqrt() as f32 } else { 0.0 };
        (peak, rms)
    }
}

#[cfg(windows)]
mod backend {
    use std::mem::size_of;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    const WAVE_MAPPER: u32 = 0xFFFF_FFFF;
    const TIME_BYTES: u32 = 0x0004;
    const WHDR_DONE: u32 = 0x0000_0001;
    const NUM_BUFS: usize = 4;

    #[repr(C)]
    struct WAVEFORMATEX {
        w_format_tag: u16,
        n_channels: u16,
        n_samples_per_sec: u32,
        n_avg_bytes_per_sec: u32,
        n_block_align: u16,
        w_bits_per_sample: u16,
        cb_size: u16,
    }

    #[repr(C)]
    struct WAVEHDR {
        lp_data: *mut u8,
        dw_buffer_length: u32,
        dw_bytes_recorded: u32,
        dw_user: usize,
        dw_flags: u32,
        dw_loops: u32,
        lp_next: usize,
        reserved: usize,
    }

    #[repr(C)]
    struct MMTIME {
        w_type: u32,
        u: [u32; 2],
    }

    #[link(name = "winmm")]
    extern "system" {
        fn waveOutOpen(phwo: *mut isize, u_device_id: u32, pwfx: *const WAVEFORMATEX, dw_callback: usize, dw_instance: usize, fdw_open: u32) -> u32;
        fn waveOutClose(hwo: isize) -> u32;
        fn waveOutPrepareHeader(hwo: isize, pwh: *mut WAVEHDR, cbwh: u32) -> u32;
        fn waveOutUnprepareHeader(hwo: isize, pwh: *mut WAVEHDR, cbwh: u32) -> u32;
        fn waveOutWrite(hwo: isize, pwh: *mut WAVEHDR, cbwh: u32) -> u32;
        fn waveOutPause(hwo: isize) -> u32;
        fn waveOutRestart(hwo: isize) -> u32;
        fn waveOutReset(hwo: isize) -> u32;
        fn waveOutGetPosition(hwo: isize, pmmt: *mut MMTIME, cbmmt: u32) -> u32;
        fn waveOutSetVolume(hwo: isize, dw_volume: u32) -> u32;
    }

    struct RingSlot {
        hdr: Box<WAVEHDR>,
        buf: Vec<u8>,
        prepared: bool,
    }

    unsafe impl Send for RingSlot {}

    pub struct WaveOutHandle {
        hwo: isize,
        block_align: usize,
        pub start_frame: usize,
        pub total_frames: usize,
        stop_flag: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
        feeder: Option<JoinHandle<()>>,
        slots: Arc<std::sync::Mutex<Vec<RingSlot>>>,
    }

    unsafe impl Send for WaveOutHandle {}

    fn fill_slot(slot: &mut RingSlot, samples: &[i16], ch: usize, cursor: &mut usize, total_frames: usize, chunk_frames: usize) -> bool {
        if *cursor >= total_frames {
            return false;
        }
        let end = (*cursor + chunk_frames).min(total_frames);
        let slice = &samples[*cursor * ch..end * ch];
        slot.buf.clear();
        #[cfg(target_endian = "little")]
        {
            // zero overhead bulk copy on little endian targets
            let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 2) };
            slot.buf.extend_from_slice(bytes);
        }
        #[cfg(not(target_endian = "little"))]
        {
            slot.buf.reserve(slice.len() * 2);
            for &s in slice {
                slot.buf.extend_from_slice(&s.to_le_bytes());
            }
        }
        slot.hdr.lp_data = slot.buf.as_mut_ptr();
        slot.hdr.dw_buffer_length = slot.buf.len() as u32;
        *cursor = end;
        true
    }

    impl WaveOutHandle {
        pub fn play_from(samples: Arc<Vec<i16>>, sample_rate: u32, channels: u16, start_frame: usize, volume: f32) -> Result<Self, String> {
            let ch = channels.max(1) as usize;
            let block_align = ch * 2;
            let total_frames = samples.len() / ch;
            let start = start_frame.min(total_frames);
            let chunk_frames = (sample_rate as usize / 5).max(1024);

            let wfx = WAVEFORMATEX {
                w_format_tag: 1,
                n_channels: channels.max(1),
                n_samples_per_sec: sample_rate.max(1),
                n_avg_bytes_per_sec: sample_rate.max(1) * block_align as u32,
                n_block_align: block_align as u16,
                w_bits_per_sample: 16,
                cb_size: 0,
            };

            let mut hwo: isize = 0;
            let rc = unsafe { waveOutOpen(&mut hwo, WAVE_MAPPER, &wfx, 0, 0, 0) };
            if rc != 0 || hwo == 0 {
                return Err(format!("waveOutOpen failed (code {rc})"));
            }

            let mut slots = Vec::with_capacity(NUM_BUFS);
            let mut cursor = start;
            for _ in 0..NUM_BUFS {
                let mut buf = Vec::with_capacity(chunk_frames * block_align);
                let mut slot = RingSlot {
                    hdr: Box::new(WAVEHDR {
                        lp_data: buf.as_mut_ptr(),
                        dw_buffer_length: 0,
                        dw_bytes_recorded: 0,
                        dw_user: 0,
                        dw_flags: WHDR_DONE,
                        dw_loops: 0,
                        lp_next: 0,
                        reserved: 0,
                    }),
                    buf,
                    prepared: false,
                };
                if fill_slot(&mut slot, &samples, ch, &mut cursor, total_frames, chunk_frames) {
                    let rc_p = unsafe { waveOutPrepareHeader(hwo, &mut *slot.hdr, size_of::<WAVEHDR>() as u32) };
                    if rc_p == 0 {
                        slot.prepared = true;
                        unsafe { waveOutWrite(hwo, &mut *slot.hdr, size_of::<WAVEHDR>() as u32); }
                    }
                }
                slots.push(slot);
            }

            let stop_flag = Arc::new(AtomicBool::new(false));
            let finished = Arc::new(AtomicBool::new(cursor >= total_frames && slots.iter().all(|s| !s.prepared)));
            let slots_arc = Arc::new(std::sync::Mutex::new(slots));

            let v = (volume.clamp(0.0, 1.0) * 65535.0) as u32;
            unsafe { waveOutSetVolume(hwo, (v << 16) | v); }

            let sf = Arc::clone(&stop_flag);
            let fin = Arc::clone(&finished);
            let sl = Arc::clone(&slots_arc);
            let feeder = thread::spawn(move || {
                while !sf.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(25));
                    if sf.load(Ordering::Relaxed) {
                        break;
                    }
                    let mut guard = sl.lock().unwrap();
                    let mut any_active = false;
                    for slot in guard.iter_mut() {
                        if slot.prepared && (slot.hdr.dw_flags & WHDR_DONE) == 0 {
                            any_active = true;
                            continue;
                        }
                        if slot.prepared {
                            unsafe { waveOutUnprepareHeader(hwo, &mut *slot.hdr, size_of::<WAVEHDR>() as u32); }
                            slot.prepared = false;
                        }
                        if fill_slot(slot, &samples, ch, &mut cursor, total_frames, chunk_frames) {
                            let rc_p = unsafe { waveOutPrepareHeader(hwo, &mut *slot.hdr, size_of::<WAVEHDR>() as u32) };
                            if rc_p == 0 {
                                slot.prepared = true;
                                unsafe { waveOutWrite(hwo, &mut *slot.hdr, size_of::<WAVEHDR>() as u32); }
                                any_active = true;
                            }
                        }
                    }
                    if !any_active && cursor >= total_frames {
                        fin.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            });

            Ok(Self {
                hwo,
                block_align,
                start_frame: start,
                total_frames,
                stop_flag,
                finished,
                feeder: Some(feeder),
                slots: slots_arc,
            })
        }

        pub fn set_volume(&mut self, volume: f32) {
            let v = (volume.clamp(0.0, 1.0) * 65535.0) as u32;
            let dword = (v << 16) | v;
            unsafe { waveOutSetVolume(self.hwo, dword); }
        }

        pub fn pause(&mut self) {
            unsafe { waveOutPause(self.hwo); }
        }

        pub fn resume(&mut self) {
            unsafe { waveOutRestart(self.hwo); }
        }

        pub fn current_frame(&mut self) -> (usize, bool) {
            let mut mmt = MMTIME { w_type: TIME_BYTES, u: [0, 0] };
            let rc = unsafe { waveOutGetPosition(self.hwo, &mut mmt, size_of::<MMTIME>() as u32) };
            let played_frames = if rc == 0 {
                (mmt.u[0] as usize) / self.block_align
            } else {
                0
            };
            let cur = (self.start_frame + played_frames).min(self.total_frames);
            let done = self.finished.load(Ordering::Relaxed) || cur >= self.total_frames;
            (cur, done)
        }
    }

    impl Drop for WaveOutHandle {
        fn drop(&mut self) {
            self.stop_flag.store(true, Ordering::Relaxed);
            if let Some(j) = self.feeder.take() {
                let _ = j.join();
            }
            unsafe { waveOutReset(self.hwo); }
            if let Ok(mut guard) = self.slots.lock() {
                for slot in guard.iter_mut() {
                    if slot.prepared {
                        unsafe { waveOutUnprepareHeader(self.hwo, &mut *slot.hdr, size_of::<WAVEHDR>() as u32); }
                        slot.prepared = false;
                    }
                }
            }
            unsafe { waveOutClose(self.hwo); }
        }
    }
}

#[cfg(not(windows))]
mod backend {
    use std::time::Instant;

    pub struct WaveOutHandle {
        start_frame: usize,
        total_frames: usize,
        sample_rate: u32,
        started_at: Instant,
        paused_frame: Option<usize>,
    }

    impl WaveOutHandle {
        pub fn play_from(samples: std::sync::Arc<Vec<i16>>, sample_rate: u32, channels: u16, start_frame: usize, _volume: f32) -> Result<Self, String> {
            let ch = channels.max(1) as usize;
            let total_frames = samples.len() / ch;
            Ok(Self {
                start_frame: start_frame.min(total_frames),
                total_frames,
                sample_rate: sample_rate.max(1),
                started_at: Instant::now(),
                paused_frame: None,
            })
        }

        pub fn set_volume(&mut self, _volume: f32) {}
        pub fn pause(&mut self) {
            let (cur, _) = self.current_frame();
            self.paused_frame = Some(cur);
        }
        pub fn resume(&mut self) {
            if let Some(f) = self.paused_frame.take() {
                self.start_frame = f;
                self.started_at = Instant::now();
            }
        }
        pub fn current_frame(&mut self) -> (usize, bool) {
            if let Some(f) = self.paused_frame {
                return (f, f >= self.total_frames);
            }
            let elapsed = self.started_at.elapsed().as_secs_f64();
            let cur = (self.start_frame + (elapsed * self.sample_rate as f64) as usize).min(self.total_frames);
            (cur, cur >= self.total_frames)
        }
    }
}

pub struct AudioPlayerState {
    pub tracks: Vec<LoadedWavTrack>,
    pub awc_bytes: Option<Arc<Vec<u8>>>,
    pub active_idx: Option<usize>,
    pub active_samples: Arc<Vec<i16>>,
    pub playing: bool,
    pub paused: bool,
    pub cursor_frame: usize,
    pub volume: f32,
    pub auto_advance: bool,
    pub loop_track: bool,
    pub filter: String,
    pub status_msg: String,
    pub error: Option<String>,
    pub decoding_idx: Option<usize>,
    pending_start_frame: usize,
    scrub_frame: Option<usize>,
    pending_decode: Arc<Mutex<Option<(u64, usize, Result<Vec<i16>, String>)>>>,
    decode_gen: u64,
    handle: Option<backend::WaveOutHandle>,
}

impl Default for AudioPlayerState {
    fn default() -> Self {
        Self {
            tracks: Vec::new(),
            awc_bytes: None,
            active_idx: None,
            active_samples: Arc::new(Vec::new()),
            playing: false,
            paused: false,
            cursor_frame: 0,
            volume: 0.85,
            auto_advance: true,
            loop_track: false,
            filter: String::new(),
            status_msg: String::new(),
            error: None,
            decoding_idx: None,
            pending_start_frame: 0,
            scrub_frame: None,
            pending_decode: Arc::new(Mutex::new(None)),
            decode_gen: 0,
            handle: None,
        }
    }
}

impl AudioPlayerState {
    pub fn set_tracks(&mut self, tracks: Vec<LoadedWavTrack>, awc_bytes: Arc<Vec<u8>>) {
        self.clear_all();
        self.tracks = tracks;
        self.awc_bytes = Some(awc_bytes);
    }

    pub fn clear_all(&mut self) {
        self.stop();
        self.tracks.clear();
        self.awc_bytes = None;
        self.active_idx = None;
        self.active_samples = Arc::new(Vec::new());
        self.filter.clear();
        self.status_msg.clear();
        self.error = None;
        self.decoding_idx = None;
        self.pending_start_frame = 0;
        self.scrub_frame = None;
        self.decode_gen = self.decode_gen.wrapping_add(1);
        *self.pending_decode.lock().unwrap() = None;
    }

    pub fn poll_decode(&mut self) {
        let ready = self.pending_decode.lock().unwrap().take();
        let Some((gen, idx, res)) = ready else { return };
        if gen != self.decode_gen || self.decoding_idx != Some(idx) {
            return;
        }
        self.decoding_idx = None;
        let start_frame = self.pending_start_frame;
        self.pending_start_frame = 0;
        match res {
            Ok(pcm) => {
                self.active_samples = Arc::new(pcm);
                self.active_idx = Some(idx);
                self.start_playback_at(idx, start_frame);
            }
            Err(e) => {
                self.error = Some(e);
                self.playing = false;
                self.paused = false;
            }
        }
    }

    fn start_playback_at(&mut self, idx: usize, start_frame: usize) {
        let Some((sample_rate, channels, total_frames)) = self
            .tracks
            .get(idx)
            .map(|t| (t.sample_rate, t.channels, t.total_frames))
        else {
            return;
        };
        let clamped = start_frame.min(total_frames);
        match backend::WaveOutHandle::play_from(
            Arc::clone(&self.active_samples),
            sample_rate,
            channels,
            clamped,
            self.volume,
        ) {
            Ok(h) => {
                self.handle = Some(h);
                self.active_idx = Some(idx);
                self.cursor_frame = clamped;
                self.playing = true;
                self.paused = false;
            }
            Err(e) => {
                self.error = Some(e);
                self.playing = false;
                self.paused = false;
            }
        }
    }

    pub fn play_track(&mut self, idx: usize, start_frame: usize) {
        let Some((coff, csz, sub_stream)) = self.tracks.get(idx).map(|t| (t.coff, t.csz, t.sub_stream)) else {
            return;
        };
        self.handle = None;
        self.error = None;
        self.status_msg.clear();
        self.playing = false;
        self.paused = false;
        self.scrub_frame = None;

        if self.active_idx == Some(idx) && !self.active_samples.is_empty() {
            self.start_playback_at(idx, start_frame);
            return;
        }

        // decode pcm asynchronously on worker thread so large streams never freeze ui
        self.active_samples = Arc::new(Vec::new());
        self.active_idx = Some(idx);
        self.cursor_frame = start_frame;
        self.pending_start_frame = start_frame;
        let Some(raw) = self.awc_bytes.clone() else {
            self.error = Some("No AWC buffer loaded".into());
            return;
        };
        self.decode_gen = self.decode_gen.wrapping_add(1);
        let gen = self.decode_gen;
        self.decoding_idx = Some(idx);
        let pending = Arc::clone(&self.pending_decode);
        *pending.lock().unwrap() = None;
        std::thread::spawn(move || {
            let res = awc::decode_single_awc_chunk(&raw, coff, csz, sub_stream).map(|(_, pcm)| pcm);
            *pending.lock().unwrap() = Some((gen, idx, res));
        });
    }

    pub fn toggle_play_pause(&mut self) {
        if self.playing && !self.paused {
            if let Some(h) = self.handle.as_mut() {
                let (cur, _) = h.current_frame();
                self.cursor_frame = cur;
                h.pause();
                self.paused = true;
            }
        } else if self.paused {
            if let Some(h) = self.handle.as_mut() {
                h.resume();
                self.paused = false;
            }
        } else if let Some(idx) = self.active_idx.or_else(|| (!self.tracks.is_empty()).then_some(0)) {
            let start = if let Some(t) = self.tracks.get(idx) {
                if self.cursor_frame >= t.total_frames { 0 } else { self.cursor_frame }
            } else {
                0
            };
            self.play_track(idx, start);
        }
    }

    pub fn seek_to_frame(&mut self, frame: usize) {
        let Some(idx) = self.active_idx else { return };
        let Some(track) = self.tracks.get(idx) else { return };
        let target = frame.min(track.total_frames);
        self.cursor_frame = target;
        self.scrub_frame = None;
        if self.playing && !self.paused {
            self.play_track(idx, target);
        } else {
            self.handle = None;
            self.playing = false;
            self.paused = false;
        }
    }

    pub fn seek_by_secs(&mut self, delta_secs: f32) {
        let Some(idx) = self.active_idx else { return };
        let Some((sample_rate, total_frames)) = self.tracks.get(idx).map(|t| (t.sample_rate, t.total_frames)) else { return };
        let cur = self.current_frame();
        let delta_frames = (delta_secs * sample_rate as f32).round() as isize;
        let next = (cur as isize + delta_frames).clamp(0, total_frames as isize) as usize;
        self.seek_to_frame(next);
    }

    pub fn stop(&mut self) {
        self.handle = None;
        self.playing = false;
        self.paused = false;
        self.cursor_frame = 0;
        self.scrub_frame = None;
    }

    pub fn current_frame(&mut self) -> usize {
        if let Some(sf) = self.scrub_frame {
            return sf;
        }
        if self.playing && !self.paused {
            let done_idx = if let Some(h) = self.handle.as_mut() {
                let (cur, done) = h.current_frame();
                self.cursor_frame = cur;
                done.then_some(self.active_idx).flatten()
            } else {
                None
            };
            if let Some(idx) = done_idx {
                self.handle = None;
                self.playing = false;
                self.paused = false;
                if self.loop_track {
                    self.play_track(idx, 0);
                } else if self.auto_advance && idx + 1 < self.tracks.len() {
                    self.play_track(idx + 1, 0);
                }
            }
        }
        self.cursor_frame
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(h) = self.handle.as_mut() {
            h.set_volume(self.volume);
        }
    }
}

fn fmt_time(secs: f32) -> String {
    let s = secs.max(0.0);
    let mins = (s / 60.0).floor() as u32;
    let rem = s - (mins as f32) * 60.0;
    format!("{mins:02}:{rem:04.1}")
}

pub fn render_awc_audio_view(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    player: &mut AudioPlayerState,
    status_headline: &str,
) {
    player.poll_decode();
    let cur_frame = player.current_frame();
    if (player.playing && !player.paused) || player.decoding_idx.is_some() {
        ctx.request_repaint();
    }

    if !ctx.wants_keyboard_input() && !player.tracks.is_empty() {
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            player.toggle_play_pause();
        } else if player.active_idx.is_some() && ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            player.seek_by_secs(-5.0);
        } else if player.active_idx.is_some() && ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            player.seek_by_secs(5.0);
        }
    }

    // Header bar + stream filter
    ui.horizontal(|ui| {
        ui.label(RichText::new(status_headline).strong().color(Color32::WHITE));
        ui.add_space(8.0);
        if player.tracks.len() > 8 {
            ui.add(
                egui::TextEdit::singleline(&mut player.filter)
                    .desired_width(170.0)
                    .hint_text("Filter streams..."),
            );
            if !player.filter.is_empty() && ui.small_button("x").clicked() {
                player.filter.clear();
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{} stream(s)", player.tracks.len()))
                    .size(11.5)
                    .color(Color32::from_rgb(175, 160, 200)),
            );
        });
    });
    ui.add_space(4.0);
    ui.separator();

    let needle = player.filter.trim().to_ascii_lowercase();
    let visible_indices: Vec<usize> = if needle.is_empty() {
        (0..player.tracks.len()).collect()
    } else {
        player.tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.name.to_ascii_lowercase().contains(&needle)
                    || format!("#{}", t.stream_idx).contains(&needle)
            })
            .map(|(i, _)| i)
            .collect()
    };

    let player_height = 136.0;
    let list_height = (ui.available_height() - player_height - 8.0).max(80.0);

    // Virtualized full-width stream list
    let mut clicked_track: Option<usize> = None;
    let row_height = 34.0;
    egui::ScrollArea::vertical()
        .id_salt("awc_stream_list")
        .max_height(list_height)
        .auto_shrink([false, false])
        .show_rows(ui, row_height, visible_indices.len(), |ui, row_range| {
            ui.set_min_width(ui.available_width());
            for vis_pos in row_range {
                let i = visible_indices[vis_pos];
                let t = &player.tracks[i];
                let is_active = player.active_idx == Some(i);
                let is_playing = is_active && player.playing && !player.paused;
                let row_bg = if is_active {
                    Color32::from_rgb(56, 30, 82)
                } else if vis_pos % 2 == 0 {
                    Color32::from_rgb(24, 15, 36)
                } else {
                    Color32::from_rgb(29, 18, 43)
                };
                let stroke = if is_active {
                    Stroke::new(1.0_f32, Color32::from_rgb(255, 168, 28))
                } else {
                    Stroke::NONE
                };

                let frame_resp = egui::Frame::new()
                    .fill(row_bg)
                    .stroke(stroke)
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.horizontal(|ui| {
                            let is_decoding = player.decoding_idx == Some(i);
                            let btn_label = if is_decoding {
                                "Loading..."
                            } else if is_playing {
                                "Pause"
                            } else {
                                "Play"
                            };
                            let btn = egui::Button::new(
                                RichText::new(btn_label)
                                    .strong()
                                    .color(if is_active { Color32::BLACK } else { Color32::WHITE }),
                            )
                            .fill(if is_active {
                                Color32::from_rgb(255, 168, 28)
                            } else {
                                Color32::from_rgb(52, 34, 74)
                            })
                            .min_size(Vec2::new(if is_decoding { 72.0 } else { 48.0 }, 22.0));

                            if ui.add(btn).clicked() {
                                clicked_track = Some(i);
                            }

                            ui.add_space(6.0);
                            ui.label(
                                RichText::new(format!("Stream #{}", t.stream_idx))
                                    .strong()
                                    .color(if is_active {
                                        Color32::from_rgb(255, 168, 28)
                                    } else {
                                        Color32::from_rgb(225, 215, 240)
                                    }),
                            );
                            ui.label(
                                RichText::new(&t.name)
                                    .monospace()
                                    .size(11.5)
                                    .color(Color32::from_rgb(165, 150, 190)),
                            );

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let peak_pct = t.overall_peak.clamp(0.0, 1.0) * 100.0;
                                let peak_db = if t.overall_peak > 1e-5 { 20.0 * t.overall_peak.log10() } else { -96.0 };
                                let tip = format!(
                                    "Max Peak Loudness: {peak_pct:.0}% ({peak_db:.1} dBFS)\nOrange = normal · Pink (>85%) = near clipping"
                                );

                                let (bar_rect, bar_resp) = ui.allocate_exact_size(Vec2::new(64.0, 10.0), Sense::hover());
                                let p = ui.painter();
                                p.rect_filled(bar_rect, CornerRadius::same(2), Color32::from_rgb(40, 26, 58));
                                let fill_w = (bar_rect.width() * t.overall_peak.clamp(0.0, 1.0)).max(2.0);
                                let fill_rect = Rect::from_min_size(bar_rect.min, Vec2::new(fill_w, bar_rect.height()));
                                let peak_col = if t.overall_peak > 0.85 {
                                    Color32::from_rgb(255, 72, 165)
                                } else {
                                    Color32::from_rgb(255, 168, 28)
                                };
                                p.rect_filled(fill_rect, CornerRadius::same(2), peak_col);
                                bar_resp.on_hover_text(&tip);

                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(format!("Peak {peak_pct:.0}%"))
                                        .size(11.0)
                                        .color(Color32::from_rgb(200, 185, 220)),
                                )
                                .on_hover_text(tip);

                                ui.add_space(10.0);
                                let ch_str = if t.channels == 2 { "Stereo" } else { "Mono" };
                                ui.label(
                                    RichText::new(format!("{} · {} kHz · {}", fmt_time(t.duration_secs), t.sample_rate / 1000, ch_str))
                                        .size(11.5)
                                        .color(Color32::from_rgb(175, 200, 225)),
                                );
                            });
                        });
                    });

                if frame_resp.response.interact(Sense::click()).clicked() {
                    clicked_track = Some(i);
                }
                ui.add_space(2.0);
            }
        });

    if let Some(i) = clicked_track {
        if player.active_idx == Some(i) && (player.playing || player.paused) {
            player.toggle_play_pause();
        } else {
            player.play_track(i, 0);
        }
    }

    ui.add_space(6.0);

    // Bottom Audio Player Overlay Bar
    egui::Frame::new()
        .fill(Color32::from_rgb(26, 15, 40))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(105, 62, 145)))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            let active_track = player.active_idx.and_then(|i| player.tracks.get(i)).cloned();

            // Row 1: Track info + transport / seek / loop / auto-next / export / volume
            ui.horizontal(|ui| {
                if let Some(ref t) = active_track {
                    ui.label(
                        RichText::new(format!("Stream #{} · {}", t.stream_idx, t.name))
                            .strong()
                            .color(Color32::from_rgb(255, 168, 28)),
                    );
                } else {
                    ui.label(RichText::new("Select a stream above to play (Space = Play/Pause)").color(Color32::from_rgb(165, 150, 190)));
                }

                ui.add_space(8.0);
                let can_prev = player.active_idx.map(|i| i > 0).unwrap_or(false);
                if ui.add_enabled(can_prev, egui::Button::new("|<")).on_hover_text("Previous stream").clicked() {
                    if let Some(i) = player.active_idx {
                        player.play_track(i - 1, 0);
                    }
                }
                if ui.add_enabled(active_track.is_some(), egui::Button::new("-5s")).on_hover_text("Seek back 5s (Left Arrow)").clicked() {
                    player.seek_by_secs(-5.0);
                }
                let is_decoding_active = player.decoding_idx.is_some();
                let play_lbl = if is_decoding_active {
                    "Decoding..."
                } else if player.playing && !player.paused {
                    "Pause"
                } else {
                    "Play"
                };
                let play_btn = egui::Button::new(RichText::new(play_lbl).strong().color(Color32::BLACK))
                    .fill(Color32::from_rgb(255, 168, 28));
                if ui.add_enabled(!player.tracks.is_empty() && !is_decoding_active, play_btn).clicked() {
                    player.toggle_play_pause();
                }
                if ui.add_enabled(player.playing || player.paused, egui::Button::new("Stop")).clicked() {
                    player.stop();
                }
                if ui.add_enabled(active_track.is_some(), egui::Button::new("+5s")).on_hover_text("Seek forward 5s (Right Arrow)").clicked() {
                    player.seek_by_secs(5.0);
                }
                let can_next = player
                    .active_idx
                    .map(|i| i + 1 < player.tracks.len())
                    .unwrap_or(false);
                if ui.add_enabled(can_next, egui::Button::new(">|")).on_hover_text("Next stream").clicked() {
                    if let Some(i) = player.active_idx {
                        player.play_track(i + 1, 0);
                    }
                }

                ui.checkbox(&mut player.loop_track, "Loop");
                ui.checkbox(&mut player.auto_advance, "Auto");

                if let Some(ref t) = active_track {
                    if ui.button("Export WAV").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .set_file_name(&t.name)
                            .add_filter("WAV Audio", &["wav"])
                            .save_file()
                        {
                            let export_res: Result<(), String> = if !player.active_samples.is_empty() {
                                awc::write_pcm_wav(&p, t.sample_rate, t.channels, &player.active_samples)
                                    .map_err(|e| format!("WAV write error: {e}"))
                            } else if let Some(ref raw) = player.awc_bytes {
                                awc::decode_single_awc_chunk(raw, t.coff, t.csz, t.sub_stream)
                                    .and_then(|(ch, pcm)| {
                                        awc::write_pcm_wav(&p, t.sample_rate, ch, &pcm)
                                            .map_err(|e| format!("WAV write error: {e}"))
                                    })
                            } else {
                                Err("No AWC buffer loaded".into())
                            };
                            match export_res {
                                Ok(()) => {
                                    player.status_msg = format!("Exported {}", p.display());
                                    player.error = None;
                                }
                                Err(e) => player.error = Some(e),
                            }
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut vol = player.volume;
                    if ui.add(egui::Slider::new(&mut vol, 0.0..=1.0).show_value(false).custom_formatter(|v, _| format!("{:.0}%", v * 100.0))).changed() {
                        player.set_volume(vol);
                    }
                    ui.label(RichText::new(format!("Vol {:.0}%", player.volume * 100.0)).size(11.5).color(Color32::from_rgb(200, 190, 220)));
                });
            });

            ui.add_space(6.0);

            // Row 2: Waveform Playbar (smooth scrub while dragging, commit on release/click)
            let bar_w = ui.available_width();
            let (wave_rect, wave_resp) = ui.allocate_exact_size(Vec2::new(bar_w, 48.0), Sense::click_and_drag());
            let painter = ui.painter_at(wave_rect);
            painter.rect_filled(wave_rect, CornerRadius::same(4), Color32::from_rgb(16, 10, 26));
            painter.rect_stroke(wave_rect, CornerRadius::same(4), Stroke::new(1.0_f32, Color32::from_rgb(65, 42, 92)), egui::StrokeKind::Inside);

            if let Some(ref t) = active_track {
                let frac = if t.total_frames > 0 {
                    (cur_frame as f32 / t.total_frames as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let bins = t.waveform_peaks.len().max(1);
                let step = wave_rect.width() / bins as f32;
                let bar_draw_w = (step * 0.72).max(1.5);
                let mid_y = wave_rect.center().y;
                let max_half_h = (wave_rect.height() - 8.0) * 0.5;

                for (b, (&peak, &rms)) in t.waveform_peaks.iter().zip(t.waveform_rms.iter()).enumerate() {
                    let bx = wave_rect.min.x + b as f32 * step + step * 0.14;
                    let bin_frac = (b as f32 + 0.5) / bins as f32;
                    let played = bin_frac <= frac;

                    let h_peak = (peak * max_half_h).max(1.5);
                    let h_rms = (rms * max_half_h).max(1.0);

                    let peak_col = if played {
                        if peak > 0.85 {
                            Color32::from_rgb(255, 72, 165)
                        } else {
                            Color32::from_rgb(255, 168, 28)
                        }
                    } else {
                        Color32::from_rgb(76, 54, 105)
                    };
                    let rms_col = if played {
                        Color32::from_rgb(255, 215, 120)
                    } else {
                        Color32::from_rgb(104, 78, 138)
                    };

                    painter.rect_filled(
                        Rect::from_min_max(Pos2::new(bx, mid_y - h_peak), Pos2::new(bx + bar_draw_w, mid_y + h_peak)),
                        CornerRadius::same(1),
                        peak_col,
                    );
                    painter.rect_filled(
                        Rect::from_min_max(Pos2::new(bx, mid_y - h_rms), Pos2::new(bx + bar_draw_w, mid_y + h_rms)),
                        CornerRadius::same(1),
                        rms_col,
                    );
                }

                let head_x = wave_rect.min.x + wave_rect.width() * frac;
                painter.line_segment(
                    [Pos2::new(head_x, wave_rect.min.y + 1.0), Pos2::new(head_x, wave_rect.max.y - 1.0)],
                    Stroke::new(2.0_f32, Color32::WHITE),
                );
                painter.circle_filled(Pos2::new(head_x, wave_rect.min.y + 5.0), 4.0, Color32::from_rgb(255, 168, 28));

                if t.total_frames > 0 {
                    if wave_resp.dragged() {
                        if let Some(pos) = wave_resp.interact_pointer_pos() {
                            let seek_frac = ((pos.x - wave_rect.min.x) / wave_rect.width()).clamp(0.0, 1.0);
                            player.scrub_frame = Some((seek_frac * t.total_frames as f32) as usize);
                        }
                    }
                    if wave_resp.drag_stopped() {
                        if let Some(sf) = player.scrub_frame.take() {
                            player.seek_to_frame(sf);
                        }
                    } else if wave_resp.clicked() {
                        if let Some(pos) = wave_resp.interact_pointer_pos() {
                            let seek_frac = ((pos.x - wave_rect.min.x) / wave_rect.width()).clamp(0.0, 1.0);
                            player.seek_to_frame((seek_frac * t.total_frames as f32) as usize);
                        }
                    }
                }
                if let Some(hover_pos) = wave_resp.hover_pos() {
                    let hfrac = ((hover_pos.x - wave_rect.min.x) / wave_rect.width()).clamp(0.0, 1.0);
                    let hsecs = hfrac * t.duration_secs;
                    wave_resp.on_hover_text(format!("Seek to {}", fmt_time(hsecs)));
                }
            } else {
                painter.text(
                    wave_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Waveform / Loudness Playbar",
                    egui::FontId::proportional(12.0),
                    Color32::from_rgb(120, 100, 150),
                );
            }

            ui.add_space(4.0);

            // Row 3: Time readout + status + Live Loudness Level Meter
            ui.horizontal(|ui| {
                if let Some(ref t) = active_track {
                    let cur_secs = cur_frame as f32 / t.sample_rate as f32;
                    ui.label(
                        RichText::new(format!("{} / {}", fmt_time(cur_secs), fmt_time(t.duration_secs)))
                            .monospace()
                            .size(11.5)
                            .color(Color32::from_rgb(225, 215, 240)),
                    );
                    if !player.status_msg.is_empty() {
                        ui.add_space(8.0);
                        ui.label(RichText::new(&player.status_msg).size(11.0).color(Color32::from_rgb(100, 220, 140)));
                    }

                    let (live_peak, live_rms) = if player.playing && !player.paused {
                        LoadedWavTrack::loudness_at_frame(
                            &player.active_samples,
                            t.channels,
                            t.sample_rate,
                            t.total_frames,
                            cur_frame,
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let db = if live_peak > 1e-5 { 20.0 * live_peak.log10() } else { -96.0 };

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (meter_rect, _) = ui.allocate_exact_size(Vec2::new(140.0, 10.0), Sense::hover());
                        let mp = ui.painter();
                        mp.rect_filled(meter_rect, CornerRadius::same(2), Color32::from_rgb(18, 12, 28));
                        let mw = (meter_rect.width() * live_peak.clamp(0.0, 1.0)).max(0.0);
                        if mw > 0.5 {
                            let col = if live_peak > 0.85 {
                                Color32::from_rgb(255, 72, 165)
                            } else if live_peak > 0.5 {
                                Color32::from_rgb(255, 168, 28)
                            } else {
                                Color32::from_rgb(100, 220, 140)
                            };
                            mp.rect_filled(
                                Rect::from_min_size(meter_rect.min, Vec2::new(mw, meter_rect.height())),
                                CornerRadius::same(2),
                                col,
                            );
                        }
                        let rw = (meter_rect.width() * live_rms.clamp(0.0, 1.0)).max(0.0);
                        if rw > 0.5 {
                            mp.rect_filled(
                                Rect::from_min_size(meter_rect.min, Vec2::new(rw, meter_rect.height())),
                                CornerRadius::same(2),
                                Color32::from_rgba_unmultiplied(255, 255, 255, 140),
                            );
                        }
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(format!("Level: {:5.1} dB ({:.0}%)", db, live_peak * 100.0))
                                .monospace()
                                .size(11.0)
                                .color(Color32::from_rgb(180, 170, 205)),
                        );
                    });
                } else if let Some(ref err) = player.error {
                    ui.label(RichText::new(err).color(Color32::from_rgb(255, 120, 120)));
                }
            });
        });
}

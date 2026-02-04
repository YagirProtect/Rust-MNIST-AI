use eframe::emath::Pos2;
use tch::{Device, Kind, Tensor};
use tch::nn::Module;
use crate::shared_lib::c_trainer_config::TrainerConfig;
use crate::solver_lib::c_ai_module::AIModule;
use crate::solver_lib::c_painter_module::PainterModule;

pub fn predict_from_canvas(painter: &PainterModule, ai_module: &mut AIModule, config: &TrainerConfig) {
    let rect = match painter.canvas_rect {
        Some(r) => r,
        None => return,
    };

    let hi = rasterize_strokes_to_hi(
        rect,
        &painter.strokes,
        &painter.sizes,
        &painter.current_stroke,
        painter.brush_size,
    );
    let pixels = hi_to_mnist28(&hi);


    let x = Tensor::from_slice(pixels.as_slice())
        .to_kind(Kind::Float)
        .to_device(ai_module.device)
        .view([1i64, config.image_dim]);

    let (pred, probs) = tch::no_grad(|| {
        let logits = ai_module.model.forward(&x);
        let pred = logits.argmax(-1, false).int64_value(&[0]);

        let p = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
        let mut arr = [0.0f32; 10];
        for i in 0..10 {
            arr[i] = p.double_value(&[0, i as i64]) as f32;
        }
        (pred, arr)
    });

    ai_module.predicted = Some(pred);
    ai_module.probs = probs;
    ai_module.last_28_pixels = Some(pixels);
}


fn splat_disk(buf: &mut [f32], w: i32, h: i32, cx: f32, cy: f32, r: f32) {
    let min_x = (cx - r).floor() as i32;
    let max_x = (cx + r).ceil() as i32;
    let min_y = (cy - r).floor() as i32;
    let max_y = (cy + r).ceil() as i32;

    let r2 = r * r;

    for y in min_y..=max_y {
        if y < 0 || y >= h { continue; }
        for x in min_x..=max_x {
            if x < 0 || x >= w { continue; }
            let dx = (x as f32 + 0.5) - cx;
            let dy = (y as f32 + 0.5) - cy;
            if dx*dx + dy*dy <= r2 {
                let idx = (y as usize) * (w as usize) + (x as usize);
                buf[idx] = 1.0;
            }
        }
    }
}
const HI_W: i32 = 280;
const HI_H: i32 = 280;
fn rasterize_strokes_to_hi(
    rect: egui::Rect,
    strokes: &[Vec<egui::Pos2>],
    sizes: &[f32],
    current: &[egui::Pos2],
    current_size: f32,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; (HI_W * HI_H) as usize];
    for (i, s) in strokes.iter().enumerate() {
        let r = sizes.get(i).copied().unwrap_or(8.0);
        draw_one(s, r, &rect, &mut buf);
    }

    if !current.is_empty() {
        draw_one(current, current_size, &rect, &mut buf);
    }

    buf
}

fn to_hi(p: Pos2, rect: &egui::Rect) -> (f32, f32) {
    let lx = (p.x - rect.min.x) / rect.width();  // 0..1
    let ly = (p.y - rect.min.y) / rect.height(); // 0..1
    (lx * HI_W as f32, ly * HI_H as f32)
}

pub fn draw_one(points: &[Pos2], radius_canvas: f32, rect: &egui::Rect, buf: &mut Vec<f32>) {
    if points.is_empty() { return; }

    let r_hi = radius_canvas * (HI_W as f32 / rect.width());

    for pair in points.windows(2) {
        let (ax, ay) = to_hi(pair[0], rect);
        let (bx, by) = to_hi(pair[1], rect);

        let dx = bx - ax;
        let dy = by - ay;
        let dist = (dx * dx + dy * dy).sqrt();

        let step = (r_hi * 0.5).max(1.0);
        let steps = (dist / step).ceil() as i32;

        for i in 0..=steps {
            let t = if steps == 0 { 0.0 } else { i as f32 / steps as f32 };
            let x = ax + dx * t;
            let y = ay + dy * t;
            splat_disk(buf, HI_W, HI_H, x, y, r_hi);
        }
    }
}

fn downsample_hi_to_28(hi: &[f32]) -> Vec<f32> {
    let block = 10;
    let mut out = vec![0.0f32; 28 * 28];

    for y in 0..28 {
        for x in 0..28 {
            let mut sum = 0.0;
            for by in 0..block {
                for bx in 0..block {
                    let hx = x * block + bx;
                    let hy = y * block + by;
                    let idx = hy * (HI_W as usize) + hx;
                    sum += hi[idx];
                }
            }
            out[y * 28 + x] = sum / (block * block) as f32; // 0..1
        }
    }
    out
}


use std::{fs, fs::File, io::{self, Write}, path::Path};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn save_bmp_gray_f32(path: impl AsRef<Path>, w: u32, h: u32, pixels: &[f32], invert: bool) -> io::Result<()> {
    assert_eq!(pixels.len(), (w * h) as usize);

    let path = path.as_ref();
    let mut f = File::create(path)?;

    // BMP 24bpp, rows padded to 4 bytes
    let row_bytes = w * 3;
    let row_stride = ((row_bytes + 3) / 4) * 4;
    let pixel_data_size = row_stride * h;
    let file_size = 54 + pixel_data_size;

    // --- BMP header (14 bytes) ---
    f.write_all(b"BM")?;
    f.write_all(&(file_size as u32).to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?; // reserved1
    f.write_all(&0u16.to_le_bytes())?; // reserved2
    f.write_all(&54u32.to_le_bytes())?; // offset to pixel data

    // --- DIB header BITMAPINFOHEADER (40 bytes) ---
    f.write_all(&40u32.to_le_bytes())?;             // header size
    f.write_all(&(w as i32).to_le_bytes())?;        // width
    f.write_all(&(h as i32).to_le_bytes())?;        // height (positive => bottom-up)
    f.write_all(&1u16.to_le_bytes())?;              // planes
    f.write_all(&24u16.to_le_bytes())?;             // bpp
    f.write_all(&0u32.to_le_bytes())?;              // compression BI_RGB
    f.write_all(&(pixel_data_size as u32).to_le_bytes())?;
    f.write_all(&2835u32.to_le_bytes())?;           // x ppm (72 DPI)
    f.write_all(&2835u32.to_le_bytes())?;           // y ppm
    f.write_all(&0u32.to_le_bytes())?;              // colors used
    f.write_all(&0u32.to_le_bytes())?;              // important colors

    // Pixel data (BGR), bottom-up
    let pad = vec![0u8; (row_stride - row_bytes) as usize];

    for y in 0..h {
        let src_y = h - 1 - y; // bottom-up
        for x in 0..w {
            let idx = (src_y * w + x) as usize;
            let mut v = pixels[idx].clamp(0.0, 1.0);
            if invert { v = 1.0 - v; } // если хочешь инвертировать
            let b = (v * 255.0).round() as u8;
            // BGR
            f.write_all(&[b, b, b])?;
        }
        f.write_all(&pad)?;
    }

    Ok(())
}

fn bbox_of_hi(hi: &[f32], thr: f32) -> Option<(i32,i32,i32,i32)> {
    let mut min_x = HI_W;
    let mut min_y = HI_H;
    let mut max_x = -1;
    let mut max_y = -1;

    for y in 0..HI_H {
        for x in 0..HI_W {
            let v = hi[(y as usize)* (HI_W as usize) + (x as usize)];
            if v > thr {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if max_x >= 0 { Some((min_x, min_y, max_x, max_y)) } else { None }
}

fn resize_bilinear(src: &[f32], sw: i32, sh: i32, dw: i32, dh: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; (dw*dh) as usize];
    for y in 0..dh {
        let fy = (y as f32) * (sh as f32 - 1.0) / (dh as f32 - 1.0);
        let y0 = fy.floor() as i32;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;

        for x in 0..dw {
            let fx = (x as f32) * (sw as f32 - 1.0) / (dw as f32 - 1.0);
            let x0 = fx.floor() as i32;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;

            let a = src[(y0 as usize)* (sw as usize) + (x0 as usize)];
            let b = src[(y0 as usize)* (sw as usize) + (x1 as usize)];
            let c = src[(y1 as usize)* (sw as usize) + (x0 as usize)];
            let d = src[(y1 as usize)* (sw as usize) + (x1 as usize)];

            let ab = a + (b - a) * tx;
            let cd = c + (d - c) * tx;
            out[(y as usize)* (dw as usize) + (x as usize)] = ab + (cd - ab) * ty;
        }
    }
    out
}

fn hi_to_mnist28(hi: &[f32]) -> Vec<f32> {
    let mut out28 = vec![0.0f32; 28 * 28];

    let Some((min_x, min_y, max_x, max_y)) = bbox_of_hi(hi, 0.05) else {
        return out28;
    };

    // делаем квадратный bbox + паддинг
    let bw = max_x - min_x + 1;
    let bh = max_y - min_y + 1;
    let side = bw.max(bh);

    let pad = (side as f32 * 0.20).ceil() as i32; // 20% поля (важно для 6/9)
    let cx = (min_x + max_x) / 2;
    let cy = (min_y + max_y) / 2;

    let half = side / 2 + pad;
    let x0 = (cx - half).clamp(0, HI_W - 1);
    let y0 = (cy - half).clamp(0, HI_H - 1);
    let x1 = (cx + half).clamp(0, HI_W - 1);
    let y1 = (cy + half).clamp(0, HI_H - 1);

    let cw = x1 - x0 + 1;
    let ch = y1 - y0 + 1;

    // crop
    let mut crop = vec![0.0f32; (cw * ch) as usize];
    for y in 0..ch {
        for x in 0..cw {
            crop[(y as usize) * (cw as usize) + (x as usize)] =
                hi[((y0 + y) as usize) * (HI_W as usize) + ((x0 + x) as usize)];
        }
    }

    // resize crop -> 20x20
    let r20 = resize_bilinear(&crop, cw, ch, 20, 20);

    // вставляем в центр 28x28
    for y in 0..20 {
        for x in 0..20 {
            out28[(y + 4) * 28 + (x + 4)] = r20[(y * 20 + x) as usize].clamp(0.0, 1.0);
        }
    }

    out28
}

pub fn save_sample_u8(pixels28: &[f32], label: u8) -> io::Result<PathBuf> {
    assert_eq!(pixels28.len(), 28 * 28);

    let mut dir = PathBuf::from("mydata");
    dir.push(label.to_string());
    fs::create_dir_all(&dir)?;

    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let mut path = dir.clone();
    path.push(format!("{ts}.bin"));

    let mut bytes = Vec::with_capacity(28 * 28);
    for &v in pixels28 {
        let b = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        bytes.push(b);
    }

    fs::write(&path, bytes)?;
    Ok(path)
}
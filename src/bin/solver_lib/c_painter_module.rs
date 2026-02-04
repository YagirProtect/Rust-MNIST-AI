use eframe::egui;
use eframe::egui::{Context, StrokeKind, Ui};

pub struct PainterModule {
    pub strokes: Vec<Vec<egui::Pos2>>,
    pub sizes: Vec<f32>,
    pub current_stroke: Vec<egui::Pos2>,
    pub brush_size: f32,
}

impl PainterModule {
    fn draw_polyline(painter: &egui::Painter, points: &[egui::Pos2], stroke: egui::Stroke) {
        if points.len() < 2 {
            return;
        }
        for w in points.windows(2) {
            painter.line_segment([w[0], w[1]], stroke);
        }
    }

    fn draw_dots(painter: &egui::Painter, points: &[egui::Pos2], radius: f32, color: egui::Color32) {
        for &p in points {
            painter.circle_filled(p, radius, color);
        }
    }
}

impl PainterModule {
    pub fn draw_painter_panel(&mut self, ui: &mut Ui) {
        let canvas_size = egui::vec2(420.0, 420.0);
        let (rect, response) = ui.allocate_exact_size(canvas_size, egui::Sense::drag());

        let painter = ui.painter();

        painter.rect_filled(rect, 0.0, egui::Color32::WHITE);
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.0, egui::Color32::DARK_GRAY),
            StrokeKind::Middle
        );

        if response.drag_started() {
            self.current_stroke.clear();
        }

        if response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                if rect.contains(pos) {
                    self.current_stroke.push(pos);
                }
            }
        }

        if response.drag_stopped() {
            if self.current_stroke.len() >= 2 {
                self.strokes.push(self.current_stroke.clone());
                self.sizes.push(self.brush_size.clone());
            }

            self.current_stroke.clear();
        }


        for i in 0..self.strokes.len() {
            let line = &self.strokes[i];
            let size = self.sizes[i];
            Self::draw_dots(painter, line, size, egui::Color32::BLACK);
        }
        Self::draw_dots(painter, &self.current_stroke, self.brush_size, egui::Color32::BLACK);
    }

    pub fn clear(&mut self) {
        self.strokes.clear();
        self.current_stroke.clear();
        self.sizes.clear();
    }


}

impl Default for PainterModule {
    fn default() -> Self {
        Self{
            strokes: vec![],
            sizes: vec![],
            current_stroke: vec![],
            brush_size: 10.0,
        }
    }
}



use image::imageops::crop_imm;
use image::RgbImage;

use insightface::calculate_embedding;
use insightface::detect_faces;
use insightface::non_maximum_suppression;
use ndarray::Array;
use onnxruntime::environment::Environment;
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::LoggingLevel;

pub fn read_rgb_image(image_path: &str) -> RgbImage {
    let image = image::open(image_path).unwrap().to_rgb8();

    return image;
}

fn tensor_from_image(
    image: &RgbImage,
) -> ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 4]>> {
    let shape = image.dimensions();

    let input = Array::from_shape_fn(
        (1_usize, 3_usize, shape.0 as usize, shape.1 as usize),
        |(_, c, i, j)| ((image[(j as _, i as _)][c] as f32) - 127_f32) / 128_f32,
    );

    return input;
}

fn main() {
    let image = read_rgb_image("t1640.bmp");
    let input = tensor_from_image(&image);

    let image112 = read_rgb_image("112.bmp");
    let input112 = tensor_from_image(&image112);

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()
        .unwrap();

    let mut det_10g = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file("det_10g.onnx")
        .unwrap();

    let faces = detect_faces(&mut det_10g, input, 0.5);
    let faces = non_maximum_suppression(faces, 0.4);

    println!("{:#?}", faces);

    let mut w600k_r50 = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_number_threads(1)
        .unwrap()
        .with_model_from_file("w600k_r50.onnx")
        .unwrap();

    let embedding = calculate_embedding(&mut w600k_r50, input112);

    println!("Emb: {:#?}", embedding);

    let mut index = 1;
    for face in faces {
        let x1 = face.bbox.0.round() as u32;
        let y1 = face.bbox.1.round() as u32;

        let x2 = face.bbox.2.round() as u32;
        let y2 = face.bbox.3.round() as u32;

        let face_crop = crop_imm(&image, x1, y1, x2 - x1, y2 - y1);

        face_crop
            .to_image()
            .save(format!("face-{index:04}.png"))
            .unwrap();
        index += 1;
    }
}

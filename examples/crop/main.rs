use std::time::Instant;

use image::Rgb;
use image::RgbImage;

use image::Rgba32FImage;
use insightface::calculate_embedding;
use insightface::crop_face;
use insightface::detect_faces;
use insightface::non_maximum_suppression;
use insightface::swap_face;
use ndarray::Array;
use ort::GraphOptimizationLevel;
use ort::Session;

pub fn read_rgba32f(image_path: &str) -> Rgba32FImage {
    let image = image::open(image_path).unwrap().to_rgba32f();

    return image;
}

pub fn read_rgb8(image_path: &str) -> RgbImage {
    let image = image::open(image_path).unwrap().to_rgb8();

    return image;
}

fn to_tensor(image: &Rgba32FImage) -> ndarray::Array<f32, ndarray::Dim<[usize; 4]>> {
    let shape = image.dimensions();

    let input = Array::from_shape_fn(
        (1_usize, 3_usize, shape.0 as usize, shape.1 as usize),
        |(_, c, i, j)| ((image[(j as _, i as _)][c] as f32) - 0.5f32) / 0.5f32,
    );

    return input;
}

// fn to_image(
//     tensor: ndarray::Array<f32, ndarray::Dim<[usize; 4]>>,
// ) -> ndarray::Array<f32, ndarray::Dim<[usize; 4]>> {
//     let shape = tensor.dim();

//     let input = Array::from_shape_fn(
//         (1_usize, 3_usize, shape.0 as usize, shape.1 as usize),
//         |(_, c, i, j)| ((tensor[(j as _, i as _)][c] as f32) - 0.5f32) / 0.5f32,
//     );

//     return input;
// }

fn to_rgb8(image: &Rgba32FImage) -> RgbImage {
    let output = RgbImage::from_fn(image.width(), image.height(), |i, j| {
        let px = image.get_pixel(i, j);
        let p = px.0;

        Rgb::<u8>([
            (p[0] * 255.0) as u8,
            (p[1] * 255.0) as u8,
            (p[2] * 255.0) as u8,
        ])
    });

    output
}

fn main() {
    let now = Instant::now();
    let img_org = read_rgba32f("t1.jpg");
    let img_640 = read_rgba32f("t1640.bmp");

    let input = to_tensor(&img_640);

    let det_10g = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .unwrap()
        .with_intra_threads(8)
        .unwrap()
        .commit_from_file("det_10g.onnx")
        .unwrap();

    let faces = detect_faces(&det_10g, input, 0.5);
    let faces = non_maximum_suppression(faces, 0.4);

    let w600k_r50 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .unwrap()
        .with_intra_threads(8)
        .unwrap()
        .commit_from_file("w600k_r50.onnx")
        .unwrap();

    let inswapper_128 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .unwrap()
        .with_intra_threads(8)
        .unwrap()
        .commit_from_file("inswapper_128.onnx")
        .unwrap();

    let mut index = 1;
    for face in faces {
        let f = face * 2;
        let face_crop = crop_face(&img_org, &f.keypoints, 112);
        let target_face = crop_face(&img_org, &f.keypoints, 128);

        let embedding = calculate_embedding(&w600k_r50, to_tensor(&face_crop));

        let fake = swap_face(&inswapper_128, to_tensor(&target_face), &embedding);
        //println!("Emb: {:#?}", embedding);

        println!("Fake: {:#?}", fake.dim());

        to_rgb8(&face_crop)
            .save(format!("face-112-{index:04}.png"))
            .unwrap();
        to_rgb8(&target_face)
            .save(format!("face-128-{index:04}.png"))
            .unwrap();
        index += 1;
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}

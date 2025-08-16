use std::ops::Mul;

use aikit::{umeyama, warp_into};
use image::Rgba32FImage;
use nalgebra::Matrix3;
use ort::{inputs, session::Session, value::Value};

/// Detects faces in the input image.
/// * `session` ONNX Runtime Session. Must be initialized by `det_10g.onnx` file from `buffalo_l`.
/// * `image` An array of size `[n, 3, 640, 640]`.
/// * `threshold` Score threshold. Usually set to `0.5`.
pub fn detect_faces(
    model: &mut Session,
    image: ndarray::Array<f32, ndarray::Dim<[usize; 4]>>,
    threshold: f32,
) -> Vec<Face> {
    let dim = image.dim();
    if dim.1 != 3 || dim.2 != 640 || dim.3 != 640 {
        //
        panic!("Dimension should be [n, 3, 640, 640]");
    }

    let input_value = Value::from_array(image).unwrap();
    let outputs = model.run(inputs![input_value]).unwrap();

    let mut result: Vec<Face> = vec![];

    let (shape08, data08) = outputs[0].try_extract_tensor().unwrap();
    let scores08 =
        ndarray::ArrayView2::from_shape((shape08[0] as usize, shape08[1] as usize), data08)
            .unwrap();

    let (shape16, data16) = outputs[1].try_extract_tensor().unwrap();
    let scores16 =
        ndarray::ArrayView2::from_shape((shape16[0] as usize, shape16[1] as usize), data16)
            .unwrap();

    let (shape32, data32) = outputs[2].try_extract_tensor().unwrap();
    let scores32 =
        ndarray::ArrayView2::from_shape((shape32[0] as usize, shape32[1] as usize), data32)
            .unwrap();

    let (shape08, data08) = &outputs[3].try_extract_tensor().unwrap();
    let bboxes08 =
        ndarray::ArrayView2::from_shape((shape08[0] as usize, shape08[1] as usize), data08)
            .unwrap()
            .into_dyn();

    let (shapebboxes16, databboxes16) = &outputs[4].try_extract_tensor().unwrap();
    let bboxes16 = ndarray::ArrayView2::from_shape(
        (shapebboxes16[0] as usize, shapebboxes16[1] as usize),
        databboxes16,
    )
    .unwrap()
    .into_dyn();

    let (shapebboxes32, databboxes32) = &outputs[5].try_extract_tensor().unwrap();
    let bboxes32 = ndarray::ArrayView2::from_shape(
        (shapebboxes32[0] as usize, shapebboxes32[1] as usize),
        databboxes32,
    )
    .unwrap()
    .into_dyn();

    let (shapekpsses08, datakpsess08) = &outputs[6].try_extract_tensor().unwrap();
    let kpsses08 = ndarray::ArrayView2::from_shape(
        (shapekpsses08[0] as usize, shapekpsses08[1] as usize),
        datakpsess08,
    )
    .unwrap()
    .into_dyn();

    let (shapekpsses16, datakpsess16) = &outputs[7].try_extract_tensor().unwrap();
    let kpsses16 = ndarray::ArrayView2::from_shape(
        (shapekpsses16[0] as usize, shapekpsses16[1] as usize),
        datakpsess16,
    )
    .unwrap()
    .into_dyn();

    let (shapekpsses32, datakpsess32) = &outputs[8].try_extract_tensor().unwrap();
    let kpsses32 = ndarray::ArrayView2::from_shape(
        (shapekpsses32[0] as usize, shapekpsses32[1] as usize),
        datakpsess32,
    )
    .unwrap()
    .into_dyn();

    for index in 0..12800 {
        let score = scores08[[index, 0]];
        if score > threshold {
            let bbox = distance2bbox(index, 8, &bboxes08);
            let keypoints = distance2kps(index, 8, &kpsses08);

            result.push(Face {
                score,
                bbox,
                keypoints,
            });
        }
    }

    for index in 0..3200 {
        let score = scores16[[index, 0]];
        if score > threshold {
            let bbox = distance2bbox(index, 16, &bboxes16);
            let keypoints = distance2kps(index, 16, &kpsses16);

            result.push(Face {
                score,
                bbox,
                keypoints,
            });
        }
    }

    for index in 0..800 {
        let score = scores32[[index, 0]];
        if score > threshold {
            let bbox = distance2bbox(index, 32, &bboxes32);
            let keypoints = distance2kps(index, 32, &kpsses32);

            result.push(Face {
                score,
                bbox,
                keypoints,
            });
        }
    }

    result.sort_by(|a, b| (b.score.partial_cmp(&a.score).unwrap()));

    return result;
}

fn distance2bbox(
    index: usize,
    stride: usize,
    distance: &ndarray::prelude::ArrayBase<
        ndarray::ViewRepr<&f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> (f32, f32, f32, f32) {
    let m = 640 / stride;
    let x = ((index / 2) * stride) % 640;
    let y = (((index / 2) / m) * stride) % 640;

    let x1 = x as f32 - distance[[index, 0]] * stride as f32;
    let y1 = y as f32 - distance[[index, 1]] * stride as f32;

    let x2 = x as f32 + distance[[index, 2]] * stride as f32;
    let y2 = y as f32 + distance[[index, 3]] * stride as f32;

    return (x1, y1, x2, y2);
}

fn distance2kps(
    index: usize,
    stride: usize,
    distance: &ndarray::prelude::ArrayBase<
        ndarray::ViewRepr<&f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> [(f32, f32); 5] {
    let m = 640 / stride;
    let x = ((index / 2) * stride) % 640;
    let y = (((index / 2) / m) * stride) % 640;

    let x1 = x as f32 + distance[[index, 0]] * stride as f32;
    let y1 = y as f32 + distance[[index, 1]] * stride as f32;

    let x2 = x as f32 + distance[[index, 2]] * stride as f32;
    let y2 = y as f32 + distance[[index, 3]] * stride as f32;

    let x3 = x as f32 + distance[[index, 4]] * stride as f32;
    let y3 = y as f32 + distance[[index, 5]] * stride as f32;

    let x4 = x as f32 + distance[[index, 6]] * stride as f32;
    let y4 = y as f32 + distance[[index, 7]] * stride as f32;

    let x5 = x as f32 + distance[[index, 8]] * stride as f32;
    let y5 = y as f32 + distance[[index, 9]] * stride as f32;

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)];
}

/// a post-processing technique used in object detection to eliminate duplicate detections and select
/// the most relevant detected objects. This helps reduce false positives and the computational
/// complexity of a detection algorithm.
///
/// `threshold` usually set to `0.4`
pub fn non_maximum_suppression(mut input: Vec<Face>, threshold: f32) -> Vec<Face> {
    let mut keep: Vec<Face> = vec![];

    while input.len() > 1 {
        let first = input[0];
        keep.push(first);

        let mut i = 1;

        while i < input.len() {
            let ri = input[i];

            let xf1 = first.bbox.0;
            let yf1 = first.bbox.1;

            let xf2 = first.bbox.2;
            let yf2 = first.bbox.3;

            let areaf = (xf2 - xf1 + 1f32) * (yf2 - yf1 + 1f32);

            let xi1 = ri.bbox.0;
            let yi1 = ri.bbox.1;

            let xi2 = ri.bbox.2;
            let yi2 = ri.bbox.3;

            let areai = (xi2 - xi1 + 1f32) * (yi2 - yi1 + 1f32);

            let xx1 = xi1.max(xf1);
            let yy1 = yi1.max(yf1);

            let xx2 = xi2.min(xf2);
            let yy2 = yi2.min(yf2);

            let w = 0f32.max(xx2 - xx1 + 1f32);
            let h = 0f32.max(yy2 - yy1 + 1f32);
            let inter = w * h;

            let ovr = inter / (areaf + areai - inter);

            if ovr > threshold {
                input.remove(i);
            } else {
                i += 1;
            }
        }

        input.remove(0);
    }

    return keep;
}

/// Calulates the embedding of the input image.
/// * `session` ONNX Runtime Session. Must be initialized by `w600k_r50.onnx` file from `buffalo_l`.
/// * `image` An array of size `[n, 3, 112, 112]`.
pub fn calculate_embedding(
    session: &mut Session,
    image: ndarray::Array<f32, ndarray::Dim<[usize; 4]>>,
) -> [f32; 512] {
    let dim = image.dim();
    if dim.1 != 3 || dim.2 != 112 || dim.3 != 112 {
        panic!("Dimension should be [n, 3, 112, 112]");
    }

    let input_value = Value::from_array(image).unwrap();
    let outputs = session.run(inputs![input_value]).unwrap();

    let embedding = &outputs[0];

    let (_, data) = embedding.try_extract_tensor().unwrap();
    let ptr = data.as_ptr();

    let slice = unsafe { std::slice::from_raw_parts::<f32>(ptr, 512) };
    return slice.try_into().unwrap();
}

pub fn swap_face(
    session: &mut ort::session::Session,
    target: ndarray::Array<f32, ndarray::Dim<[usize; 4]>>,
    source: &[f32; 512],
) -> ndarray::Array4<f32> {
    let src = ndarray::Array2::from_shape_vec((1, 512), Vec::<f32>::from(source)).unwrap();

    //println!("SRC: {:#?}", src);
    let input_value = Value::from_array(target).unwrap();
    let src = Value::from_array(src).unwrap();

    let outputs = session.run(inputs![input_value, src]).unwrap();

    let (shaperesult, dataresult) = &outputs[0].try_extract_tensor().unwrap();
    let result = ndarray::ArrayView2::from_shape(
        (shaperesult[0] as usize, shaperesult[1] as usize),
        dataresult,
    )
    .unwrap()
    .into_dyn();

    result.to_shape((dim.0, 3, 128, 128)).unwrap().into_owned()
}
pub fn crop_face(image: &Rgba32FImage, keypoints: &[(f32, f32); 5], size: u32) -> Rgba32FImage {
    let m = umeyama(&keypoints, &ARCFACE_DST);

    let mut output = Rgba32FImage::new(size, size);
    warp_into(image, m, &mut output);

    output
}

pub fn estimate_norm(keypoints: &[(f32, f32); 5]) -> Matrix3<f32> {
    let m = umeyama(&keypoints, &ARCFACE_DST);

    m
}

#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub score: f32,
    pub bbox: (f32, f32, f32, f32),
    pub keypoints: [(f32, f32); 5],
}

impl Mul<f32> for Face {
    type Output = Face;

    fn mul(self, rhs: f32) -> Self::Output {
        let face = Face {
            score: self.score,
            bbox: (
                self.bbox.0 * rhs,
                self.bbox.1 * rhs,
                self.bbox.2 * rhs,
                self.bbox.3 * rhs,
            ),
            keypoints: self.keypoints.map(|v| (v.0 * rhs, v.1 * rhs)),
        };

        face
    }
}

impl Mul<i32> for Face {
    type Output = Face;

    fn mul(self, rhs: i32) -> Self::Output {
        let face = Face {
            score: self.score,
            bbox: (
                self.bbox.0 * rhs as f32,
                self.bbox.1 * rhs as f32,
                self.bbox.2 * rhs as f32,
                self.bbox.3 * rhs as f32,
            ),
            keypoints: self.keypoints.map(|v| (v.0 * rhs as f32, v.1 * rhs as f32)),
        };

        face
    }
}

#[cfg(test)]
mod tests {
    //    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

const ARCFACE_DST: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

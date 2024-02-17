use onnxruntime::{session::Session, tensor::OrtOwnedTensor};

/// Detects faces in the input image.
/// * `session` ONNX Runtime Session. Must be initialized by `det_10g.onnx` file from `buffalo_l`.
/// * `image` An array of size `[1, 3, 640, 640]`.
/// * `score_threshold` Score threshold. Usually set to `0.5`.
/// * `nms_threshold` Non-maximum suppression threshold. Usually set to `0.4`.
pub fn detect_faces<'a>(
    session: &mut Session<'a>,
    image: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 4]>>,
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<Face> {
    // Multiple inputs and outputs are possible
    let input_tensor = vec![image];
    let net_outs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor).unwrap();

    let mut box08: Vec<(f32, f32, f32, f32)> = vec![];
    let mut box16: Vec<(f32, f32, f32, f32)> = vec![];
    let mut box32: Vec<(f32, f32, f32, f32)> = vec![];

    let mut kps08: Vec<[(f32, f32); 5]> = vec![];
    let mut kps16: Vec<[(f32, f32); 5]> = vec![];
    let mut kps32: Vec<[(f32, f32); 5]> = vec![];

    let mut result: Vec<Face> = vec![];

    let scores08 = &net_outs[0];
    let scores16 = &net_outs[1];
    let scores32 = &net_outs[2];
    let bboxes08 = &net_outs[3];
    let bboxes16 = &net_outs[4];
    let bboxes32 = &net_outs[5];
    let kpsses08 = &net_outs[6];
    let kpsses16 = &net_outs[7];
    let kpsses32 = &net_outs[8];

    for index in 0..12800 {
        let score = scores08[[index, 0]];
        if score > score_threshold {
            let bbox = distance2bbox(index, 8, bboxes08);
            box08.push(bbox);

            let keypoints = distance2kps(index, 8, kpsses08);
            kps08.push(keypoints);

            result.push(Face {
                score,
                bbox,
                keypoints,
            });
        }
    }

    for index in 0..3200 {
        let score = scores16[[index, 0]];
        if score > score_threshold {
            let bbox = distance2bbox(index, 16, bboxes16);
            box16.push(bbox);

            let keypoints = distance2kps(index, 16, kpsses16);
            kps16.push(keypoints);

            result.push(Face {
                score,
                bbox,
                keypoints,
            });
        }
    }

    for index in 0..800 {
        let score = scores32[[index, 0]];
        if score > score_threshold {
            let bbox = distance2bbox(index, 32, bboxes32);
            box32.push(bbox);

            let keypoints = distance2kps(index, 32, kpsses32);
            kps32.push(keypoints);

            result.push(Face {
                score,
                bbox,
                keypoints,
            });
        }
    }

    result.sort_by(|a, b| (b.score.partial_cmp(&a.score).unwrap()));

    let faces = nms(result, nms_threshold);

    return faces;
}

fn distance2bbox(
    index: usize,
    stride: usize,
    distance: &OrtOwnedTensor<'_, '_, f32, ndarray::prelude::Dim<ndarray::IxDynImpl>>,
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
    distance: &OrtOwnedTensor<'_, '_, f32, ndarray::prelude::Dim<ndarray::IxDynImpl>>,
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

fn nms(mut input: Vec<Face>, threshold: f32) -> Vec<Face> {
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

#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub score: f32,
    pub bbox: (f32, f32, f32, f32),
    pub keypoints: [(f32, f32); 5],
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

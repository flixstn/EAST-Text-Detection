use clap::{App, Arg};
use opencv::{
    Error, 
    Result, 
    core::{CV_PI, CV_32F, Point2f, RotatedRect, Scalar, Size, Size2f}, 
    dnn::{blob_from_image_to, nms_boxes_rotated, read_net}, 
    highgui::{imshow, wait_key}, 
    imgcodecs::{IMREAD_COLOR, imread}, 
    imgproc::{LINE_8, line},
    prelude::{Mat, MatTrait, NetTrait, RotatedRectTrait}, 
    types::{VectorOfMat, VectorOfPoint2f, VectorOfRotatedRect, VectorOfString, VectorOff32, VectorOfi32}
};

fn main() -> Result<(), Error> {
    // parse arguments
    // file: image file
    // weights: network weights
    let matches = App::new("Knowledge Component Extraction")
        .version("0.1.0")
        .about("Analyzes programming videos and extracts knowledge components")
        .arg(Arg::with_name("file")
                .short("f")
                .long("file")
                .takes_value(true)
                .help("Video or image file as input"))
        .arg(Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Yolo weights"))
        .get_matches();

    // read arguments for network configuration and image processing
    let model_weights = matches.value_of("weights").unwrap();
    let img_file = matches.value_of("file").unwrap();

    // image file and network
    let mut img = imread(img_file, IMREAD_COLOR)?;
    let mut net = read_net(model_weights, "", "")?;

    //image size
    let img_width = img.cols();
    let img_height = img.rows();
    let inp_width = 320;
    let inp_height = 320;

    // confidence and non-maximum suppression threshold
    let conf_threshold= 0.5_f32;
    let nms_threshold = 0.4_f32;

    let mut output = VectorOfMat::new();
    let mut output_layers = VectorOfString::new();
    output_layers.push("feature_fusion/Conv_7/Sigmoid");
    output_layers.push("feature_fusion/concat_3");

    let mut blob = Mat::default()?;

    blob_from_image_to(&img, &mut blob, 1.0, Size::new(inp_width, inp_height), Scalar::new(123.68, 116.78, 103.94, 0.), true, false, CV_32F)?;
    net.set_input(&blob, "", 1.0, Scalar::new(0., 0., 0., 0.))?;
    net.forward(&mut output, &output_layers)?;

    let scores = output.get(0)?;
    let geometry = output.get(1)?;

    // decode predicted bounding boxes
    let mut boxes = VectorOfRotatedRect::new();
    let mut confidences = VectorOff32::new();
    decode(&scores, &geometry, conf_threshold, &mut boxes, &mut confidences)?;

    let mut indices = VectorOfi32::new();
    nms_boxes_rotated(&boxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1., 0)?;

    // render detections
    let ratio = Point2f::new(img_width as f32 / inp_width as f32, img_height as f32 / inp_height as f32);
    for num in indices.iter() {
        let bbox = &boxes.get(num as usize)?;
        let mut vertices = VectorOfPoint2f::from(vec![Point2f::default(); 4]);
        
        bbox.points(&mut vertices.as_mut_slice())?;
        
        for j in 0..4 {
            let p1 = vertices.get(j)?.x * ratio.x;
            let p2 = vertices.get(j)?.y * ratio.y;
            vertices.set(j, Point2f::new(p1, p2))?;
        }   
        
        for j in 0..4 {
            line(&mut img, 
                 vertices.get(j)?.to::<i32>().unwrap(), 
                 vertices.get((j + 1) % 4)?.to::<i32>().unwrap(), 
                 Scalar::new(0., 255., 0., 0.), 
                 2, 
                 LINE_8, 
                 0)?;
        }
    }

    // show image and wait for keyboard input
    imshow("image", &img.clone())?;
    let _ = wait_key(0)?;

    Ok(())
}

fn decode(scores: &Mat, geometry: &Mat, score_thresh: f32, detections: &mut VectorOfRotatedRect, confidences: &mut VectorOff32) -> Result<(), Error> {
    detections.clear();

    let height = scores.mat_size()[2]; 
    let width = scores.mat_size()[3];

    for y in 0..height {
        for x in 0..width {
            let score = scores.at_nd::<f32>(&[0,0,y,x])?;

            if *score < score_thresh {
                continue;
            }
            let offset_x = (x * 4) as f32;
            let offset_y = (y * 4) as f32;
            let angle = geometry.at_nd::<f32>(&[0,4,y,x])?;

            // calculate cos and sin of angle
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let h = *geometry.at_nd::<f32>(&[0,0,y,x])? + *geometry.at_nd::<f32>(&[0,2,y,x])?;
            let w = *geometry.at_nd::<f32>(&[0,1,y,x])? + *geometry.at_nd::<f32>(&[0,3,y,x])?;

            // calculate offset
            let x1 = offset_x + cos_a * *geometry.at_nd::<f32>(&[0,1,y,x])? + sin_a * *geometry.at_nd::<f32>(&[0,2,y,x])?;
            let x2 = offset_y - sin_a * *geometry.at_nd::<f32>(&[0,1,y,x])? + cos_a * *geometry.at_nd::<f32>(&[0,2,y,x])?;
            let offset = Point2f::new(x1, x2);

            // create rectangle
            let p1 = Point2f::new(-sin_a * h, -cos_a * h) + offset;
            let p3 = Point2f::new(-cos_a * w, sin_a * w) + offset;
            let rect = RotatedRect::new((p1+p3) * 0.5f32, Size2f::new(w, h), -angle * 180. / CV_PI as f32)?;

            detections.push(rect);
            confidences.push(*score);
        }
    }

    Ok(())
}

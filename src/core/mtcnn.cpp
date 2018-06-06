// std
#include <vector>
#include <string>
#include <exception>

#include "mtcnn.h"

using namespace std;
using namespace cv;
using namespace face;

#include <limits>
int fix(float f)
{
  f *= 1 + numeric_limits<float>::epsilon();
  return (int)f;
}

Mtcnn::Mtcnn(const string & model_dir, bool precise_landmark) :
  enable_precise_landmark(precise_landmark)
{
  // load models
  Pnet = make_shared<caffe::Net<float>>(model_dir + "/MTCNN_det1.prototxt", caffe::TEST);
  Pnet->CopyTrainedLayersFrom(model_dir + "/MTCNN_det1.caffemodel");
  Rnet = make_shared<caffe::Net<float>>(model_dir + "/MTCNN_det2.prototxt", caffe::TEST);
  Rnet->CopyTrainedLayersFrom(model_dir + "/MTCNN_det2.caffemodel");
  Onet = make_shared<caffe::Net<float>>(model_dir + "/MTCNN_det3.prototxt", caffe::TEST);
  Onet->CopyTrainedLayersFrom(model_dir + "/MTCNN_det3.caffemodel");
  if (enable_precise_landmark)
  {
    Lnet = make_shared<caffe::Net<float>>(model_dir + "/MTCNN_det4.prototxt", caffe::TEST);
    Lnet->CopyTrainedLayersFrom(model_dir + "/MTCNN_det4.caffemodel");
  }

  // init reference facial points
  ref_96_112.reserve(5);
  ref_96_112.emplace_back(29.2946, 50.6963); // as c++ or python index start from 0
  ref_96_112.emplace_back(64.5318, 50.5014); // w.r.t matlab index start from 1
  ref_96_112.emplace_back(47.0252, 70.7366); // this will result a minus one in landmarks
  ref_96_112.emplace_back(32.5493, 91.3655);
  ref_96_112.emplace_back(61.7299, 91.2041);

  ref_112_112.reserve(5);
  for (const auto & pt : ref_96_112)
    ref_112_112.emplace_back(pt.x + 8.f, pt.y);
}

/* #define DEBUG */
#include <iostream>
void imsave(const Mat & sample, const vector<Proposal> & pros, const string & path)
{
  Mat img;
  sample.copyTo(img);
  for (const auto & pro : pros)
    rectangle(img, Rect(pro.bbox.x1, pro.bbox.y1, pro.bbox.x2 - pro.bbox.x1,
      pro.bbox.y2 -  pro.bbox.y1), Scalar(255, 0, 0), 2);
  imwrite(path, img);
}
void imsave(const Mat & sample, const vector<BBox> & bboxes, const string & path)
{
  Mat img;
  sample.copyTo(img);
  for (const auto & bbox : bboxes)
    rectangle(img, Rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1,
      bbox.y2 - bbox.y1), Scalar(255, 0, 0), 2);
  imwrite(path, img);
}
void imsave(const Mat & sample, const vector<FaceInfo> & infos, const string & path)
{
  Mat img;
  sample.copyTo(img);
  for (const auto & info : infos) {
    rectangle(img, Rect(info.bbox.x1, info.bbox.y1, info.bbox.x2 - info.bbox.x1,
      info.bbox.y2 -  info.bbox.y1), Scalar(255, 0, 0), 2);
    for (const auto pt : info.fpts)
      circle(img, cv::Point(fix(pt.x), fix(pt.y)), 2, Scalar(0, 255, 0), -1);
  }

  imwrite(path, img);
}

vector<FaceInfo> Mtcnn::Detect(const Mat & sample, bool precise_landmark)
{
  Mat normed_sample;
  #ifdef NORM_FARST
    sample.convertTo(normed_sample, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
  #else
    sample.convertTo(normed_sample, CV_32FC3);
  #endif // NORM_FARST	

  vector<BBox> bboxes = ProposalNetwork(normed_sample);
  #ifdef DEBUG
  imsave(normed_sample, bboxes, "pnet.png");
  #endif
  bboxes = RefineNetwork(normed_sample, bboxes);

  #ifdef DEBUG
  imsave(normed_sample, bboxes, "rnet.png");
  #endif
  vector<FaceInfo> infos = OutputNetwork(normed_sample, bboxes);

  #ifdef DEBUG
  imsave(normed_sample, infos, "onet.png");
  #endif

  if (enable_precise_landmark && precise_landmark)
    LandmarkNetwork(normed_sample, infos);
  #ifdef DEBUG
  imsave(normed_sample, infos, "lnet.png");
  #endif

  // matlab CS to C++ CS
  for (auto & info : infos)
  {
    info.bbox.x1 -= 1;
    info.bbox.y1 -= 1;
    for (auto & fpt : info.fpts)
    {
      fpt.x -= 1;
      fpt.y -= 1;
    }
  }
    
  return move(infos);
}

vector<FaceInfo> Mtcnn::DetectAgain(const Mat & sample, const BBox & bbox, bool precise_landmark)
{
  Mat normed_sample;
  #ifdef NORM_FARST
    sample.convertTo(normed_sample, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
  #else
    sample.convertTo(normed_sample, CV_32FC3);
  #endif // NORM_FARST	

  vector<BBox> bboxes = {bbox};
  bboxes = RefineNetwork(normed_sample, bboxes);
  vector<FaceInfo> infos = OutputNetwork(normed_sample, bboxes);
  if (enable_precise_landmark && precise_landmark)
    LandmarkNetwork(normed_sample, infos);

  // matlab CS to C++ CS
  for (auto & info : infos)
  {
    info.bbox.x1 -= 1;
    info.bbox.y1 -= 1;
    for (auto & fpt : info.fpts)
    {
      fpt.x -= 1;
      fpt.y -= 1;
    }
  }
    
  return move(infos);
}

Mat Mtcnn::Align(const Mat & sample, const FPoints & fpts, int width)
{
  CHECK(width == 96 || width == 112) << "Face align only support width to be 96 or 112";
  Mat tform, face;
  if (width == 96)
    tform = cp2tform(fpts, ref_96_112);
  else
    tform = cp2tform(fpts, ref_112_112);
  tform = tform.colRange(0, 2).t();
  warpAffine(sample, face, tform, Size(width, 112));
  return face;
}

Mat Mtcnn::DrawInfos(const Mat & sample, const vector<FaceInfo> & infos)
{
  Mat canvas = sample.clone();
  for (const auto & info : infos) {
    rectangle(canvas, Rect(info.bbox.x1, info.bbox.y1, info.bbox.x2 - info.bbox.x1,
      info.bbox.y2 -  info.bbox.y1), Scalar(255, 0, 0), 2);
    for (const auto pt : info.fpts)
      circle(canvas, cv::Point(fix(pt.x), fix(pt.y)), 5, Scalar(0, 255, 0), -1);
    putText(canvas, to_string(info.score), cv::Point((int)(info.bbox.x1), (int)(info.bbox.y1)),
           FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
  }
  return canvas;
}

void Mtcnn::SetBatchSize(shared_ptr<caffe::Net<float>> net, const int batch_size)
{
  caffe::Blob<float>* input_layer = net->input_blobs()[0];
  vector<int> input_shape = input_layer->shape();
  input_shape[0] = batch_size;
  input_layer->Reshape(input_shape);
  net->Reshape();
}

vector<vector<Mat>> Mtcnn::WarpInputLayer(shared_ptr<caffe::Net<float>> net)
{
  vector<vector<Mat>> input_channals;
  caffe::Blob<float>* input_layer = net->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();

  for (int i = 0; i < input_layer->num(); ++i)
  {
    vector<Mat> channals;
    for (int j = 0; j < input_layer->channels(); ++j)
    {
      channals.emplace_back(height, width, CV_32FC1, input_data);
      input_data += width * height;
    }
    input_channals.push_back(move(channals));
  }

  return move(input_channals);
}

vector<float> Mtcnn::ScalePyramid(const int min_len)
{
  vector<float> scales;
  float max_scale = 12.0f / face_min_size;
  float min_scale = 12.0f / std::min(min_len, face_max_size);
  for (float scale = max_scale; scale >= min_scale; scale *= scale_factor)
    scales.push_back(scale);
  return move(scales);
}

vector<Proposal> Mtcnn::GetCandidates(const float scale, const caffe::Blob<float>* scores,
  const caffe::Blob<float>* regs)
{
  int stride = 2;
  int cell_size = 12;
  int output_width = regs->width();
  int output_height = regs->height();
  vector<Proposal> pros;

  for (int i = 0; i < output_height; ++i)
    for (int j = 0; j < output_width; ++j)
      if (scores->data_at(0, 1, i, j) >= thresholds[0])
      {
        // bounding box
        BBox bbox(
          fix((j * stride + 1) / scale),	    // x1
          fix((i * stride + 1) / scale),	    // y1
          fix((j * stride + cell_size) / scale),	// x2
          fix((i * stride + cell_size) / scale));	// y2

        // bbox regression
        Reg reg(
          regs->data_at(0, 0, i, j),	// reg_x1
          regs->data_at(0, 1, i, j),	// reg_y1
          regs->data_at(0, 2, i, j),	// reg_x2
          regs->data_at(0, 3, i, j));	// reg_y2
        // face confidence
        float score = scores->data_at(0, 1, i, j);
        pros.emplace_back(move(bbox), score, move(reg));
      }

  return move(pros);
}

vector<Proposal> Mtcnn::NonMaximumSuppression(vector<Proposal>& pros,
  const float threshold, const NMS_TYPE type)
{
  if (pros.size() <= 1)
    return move(pros);

  sort(pros.begin(), pros.end(),
    // Lambda function: descending order by score.
    [](const Proposal& x, const Proposal& y) -> bool { return x.score > y.score; });


  vector<Proposal> nms_pros;
  while (!pros.empty()) {
    // select maximun candidates.
    Proposal max = move(pros[0]);
    pros.erase(pros.begin());
    float max_area = (max.bbox.x2 - max.bbox.x1 + 1)
      * (max.bbox.y2 - max.bbox.y1 + 1);
    // filter out overlapped candidates in the rest.
    int idx = 0;
    while (idx < pros.size()) {
      // computer intersection.
      float x1 = std::max(max.bbox.x1, pros[idx].bbox.x1);
      float y1 = std::max(max.bbox.y1, pros[idx].bbox.y1);
      float x2 = std::min(max.bbox.x2, pros[idx].bbox.x2);
      float y2 = std::min(max.bbox.y2, pros[idx].bbox.y2);
      float overlap = 0;
      if (x1 <= x2 && y1 <= y2)
      {
        float inter = (x2 - x1 + 1) * (y2 - y1 + 1);
        // computer denominator.
        float outer;
        float area = (pros[idx].bbox.x2 - pros[idx].bbox.x1 + 1)
          * (pros[idx].bbox.y2 - pros[idx].bbox.y1 + 1);
        if (type == IoM)	// Intersection over Minimum
          outer = std::min(max_area, area);
        else	// Intersection over Union
          outer = max_area + area - inter;
        overlap = inter / outer;
      }

      if (overlap > threshold)	// erase overlapped candidate
        pros.erase(pros.begin() + idx);
      else
        idx++;	// check next candidate
    }
    nms_pros.push_back(move(max));
  }

  return move(nms_pros);
}

void Mtcnn::BoxRegression(vector<Proposal>& pros)
{
  for (auto& pro : pros) {
    float width = pro.bbox.x2 - pro.bbox.x1 + 1;
    float height = pro.bbox.y2 - pro.bbox.y1 + 1;
    pro.bbox.x1 += pro.reg.x1 * width;	// x1
    pro.bbox.y1 += pro.reg.y1 * height;	// y1
    pro.bbox.x2 += pro.reg.x2 * width;	// x2
    pro.bbox.y2 += pro.reg.y2 * height;	// y2
  }
}

void Mtcnn::Square(vector<BBox> & bboxes) {
  for (auto& bbox : bboxes)
    Square(bbox);
}

void Mtcnn::Square(BBox & bbox)
{
  float w = bbox.x2 - bbox.x1;
  float h = bbox.y2 - bbox.y1;
  float maxl = max(w, h);
  bbox.x1 += (w - maxl) * 0.5;
  bbox.y1 += (h - maxl) * 0.5;
  bbox.x2 = fix(bbox.x1 + maxl);
  bbox.y2 = fix(bbox.y1 + maxl);
  bbox.x1 = fix(bbox.x1);
  bbox.y1 = fix(bbox.y1);
}

Mat Mtcnn::CropPadding(const Mat& sample, const BBox& bbox)
{
  Rect img_rect(0, 0, sample.cols, sample.rows);
  Rect crop_rect(Point2f(bbox.x1-1, bbox.y1-1), Point2f(bbox.x2, bbox.y2));
  Mat crop(crop_rect.size(), CV_32FC3, Scalar(0.0));
  Rect inter_on_sample = crop_rect & img_rect;
  // shifting inter from image CS (coordinate system) to crop CS.
  if (inter_on_sample.width * inter_on_sample.height)
  {
    Rect inter_on_crop = inter_on_sample - crop_rect.tl();
    sample(inter_on_sample).copyTo(crop(inter_on_crop));
 }
  
  return crop;
}

vector<BBox> Mtcnn::ProposalNetwork(const Mat & sample)
{
  int min_len = std::min(sample.cols, sample.rows);
  vector<float> scales = ScalePyramid(min_len);
  vector<Proposal> total_pros;

  caffe::Blob<float>* input_layer = Pnet->input_blobs()[0];
  for (float scale : scales)
  {
    int height = static_cast<int>(ceil(sample.rows * scale));
    int width = static_cast<int>(ceil(sample.cols * scale));
    Mat img;
    resize(sample, img, Size(width, height));

    #ifndef NORM_FARST
      img.convertTo(img, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
    #endif // !NORM_FARST

    // Reshape Net.
    input_layer->Reshape(1, 3, height, width);
    Pnet->Reshape();

    vector<Mat> channels = WarpInputLayer(Pnet)[0];
    split(img, channels);

    const vector<caffe::Blob<float>*> out = Pnet->Forward();
    vector<Proposal> pros = GetCandidates(scale, out[1], out[0]);
 
    // intra scale nms
    pros = NonMaximumSuppression(pros, 0.5f, IoU);

    if (!pros.empty()) {
      total_pros.insert(total_pros.end(), pros.begin(), pros.end());
    }
  }
  // inter scale nms
  total_pros = NonMaximumSuppression(total_pros, 0.7f, IoU);
  BoxRegression(total_pros);

  #ifdef DEBUG
  cout << "pnet: ";
  for (auto& pro : total_pros)
    cout << pro.score << " ";
  cout << endl;
  #endif


  vector<BBox> bboxes;
  for (auto& pro : total_pros)
    bboxes.push_back(move(pro.bbox));

  return move(bboxes);
}

vector<BBox> Mtcnn::RefineNetwork(const Mat & sample, vector<BBox> & bboxes)
{
  if (bboxes.empty())
    return move(bboxes);

  size_t num = bboxes.size();
  Square(bboxes);	// convert bbox to square

  SetBatchSize(Rnet, num);
  vector<vector<Mat>> input_channels = WarpInputLayer(Rnet);

  for (int i = 0; i < num; ++i)
  {
    Mat crop = CropPadding(sample, bboxes[i]);
    resize(crop, crop, Size(24, 24));
    #ifndef NORM_FARST
      crop.convertTo(crop, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
    #endif // NORM_FARST
    split(crop, input_channels[i]);
  }

  const vector<caffe::Blob<float>*> out = Rnet->Forward();
  caffe::Blob<float>*   regs = out[0];
  caffe::Blob<float>* scores = out[1];

  vector<Proposal> pros;
  for (int i = 0; i < num; ++i)
  {
    if (scores->data_at(i, 1, 0, 0) >= thresholds[1]) {
      Reg reg(
        regs->data_at(i, 0, 0, 0),	// x1
        regs->data_at(i, 1, 0, 0),	// y1
        regs->data_at(i, 2, 0, 0),	// x2
        regs->data_at(i, 3, 0, 0));	// y2
      float score = scores->data_at(i, 1, 0, 0);
      pros.emplace_back(move(bboxes[i]), score, move(reg));
    }
  }

  pros = NonMaximumSuppression(pros, 0.7f, IoU);
  BoxRegression(pros);

  #ifdef DEBUG
  cout << "rnet: ";
  for (auto& pro : pros)
    cout << pro.score << " ";
  cout << endl;
  #endif

  bboxes.clear();
  for (auto& pro : pros)
    bboxes.push_back(move(pro.bbox));

  return move(bboxes);
}

vector<FaceInfo> Mtcnn::OutputNetwork(const Mat & sample, vector<BBox> & bboxes)
{
  vector<FaceInfo> infos;
  if (bboxes.empty())
    return move(infos);

  size_t num = bboxes.size();
  Square(bboxes);	// convert bbox to square

  SetBatchSize(Onet, num);
  vector<vector<Mat> > input_channels = WarpInputLayer(Onet);
  for (int i = 0; i < num; ++i)
  {
    Mat crop = CropPadding(sample, bboxes[i]);
    resize(crop, crop, Size(48, 48));
    #ifndef NORM_FARST
      crop.convertTo(crop, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
    #endif // NORM_FARST
    split(crop, input_channels[i]);
  }

  const vector<caffe::Blob<float>*> out = Onet->Forward();
  caffe::Blob<float>* regs   = out[0];
  caffe::Blob<float>* fpts   = out[1];
  caffe::Blob<float>* scores = out[2];

  vector<Proposal> pros;
  for (int i = 0; i < num; i++)
  {
    BBox& bbox = bboxes[i];
    if (scores->data_at(i, 1, 0, 0) >= thresholds[2]) {
      Reg reg(regs->data_at(i, 0, 0, 0),	// x1
              regs->data_at(i, 1, 0, 0),	// y1
              regs->data_at(i, 2, 0, 0),	// x2
              regs->data_at(i, 3, 0, 0));	// y2
      float score = scores->data_at(i, 1, 0, 0);
      // facial landmarks
      FPoints fpt;
      float width = bbox.x2 - bbox.x1 + 1;
      float height = bbox.y2 - bbox.y1 + 1;
      for (int j = 0; j < 5; j++)
        fpt.emplace_back(fpts->data_at(i, j, 0, 0) * width + bbox.x1 - 1,
                         fpts->data_at(i, j+5, 0, 0) * height + bbox.y1 - 1);
      pros.emplace_back(move(bbox), score, move(fpt), move(reg));
    }
  }

  BoxRegression(pros);
  pros = NonMaximumSuppression(pros, 0.7f, IoM);

  #ifdef DEBUG
  cout << "onet: ";
  for (auto& pro : pros)
    cout << pro.score << " ";
  cout << endl;
  #endif

  for (auto & pro : pros)
    infos.emplace_back(move(pro.bbox), pro.score, move(pro.fpts));

  return move(infos);
}

void Mtcnn::LandmarkNetwork(const Mat & sample, vector<FaceInfo>& infos)
{
  if (infos.empty())
    return;

  size_t num = infos.size();
  SetBatchSize(Lnet, num);
  Rect img_rect(0, 0, sample.cols, sample.rows);
  vector<vector<Mat>> input_channels = WarpInputLayer(Lnet);
  vector<int> patch_sizes;
  for (int i = 0; i < num; ++i)
  {
    FaceInfo & info = infos[i];
    float patchw = std::max(info.bbox.x2 - info.bbox.x1,
      info.bbox.y2 - info.bbox.y1) + 1;
    int patchs = patchw * 0.25f;
    if (patchs % 2 == 1)
      patchs += 1;
    patch_sizes.push_back(patchs);
    for (int j = 0; j < 5; ++j)
    {
      BBox patch;
      patch.x1 = fix(info.fpts[j].x - patchs * 0.5f);
      patch.y1 = fix(info.fpts[j].y - patchs * 0.5f);
      patch.x2 = patch.x1 + patchs;
      patch.y2 = patch.y1 + patchs;
      Mat crop = CropPadding(sample, patch);
      resize(crop, crop, Size(24, 24));
      #ifndef NORM_FARST
        crop.convertTo(crop, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
      #endif // NORM_FARST

      // extract channels of certain patch
      vector<Mat> patch_channels;
      patch_channels.push_back(input_channels[i][3 * j + 0]);	// B
      patch_channels.push_back(input_channels[i][3 * j + 1]);	// G
      patch_channels.push_back(input_channels[i][3 * j + 2]);	// R
      split(crop, patch_channels);
    }
  }

  const vector<caffe::Blob<float>*> out = Lnet->Forward();

  // for every facial landmark
  for (int j = 0; j < 5; ++j)
  {
    caffe::Blob<float>* offs = out[j];
    // for every face
    for (int i = 0; i < num; ++i)
    {
      int patch_size = patch_sizes[i];
      float off_x = offs->data_at(i, 0, 0, 0) - 0.5;
      float off_y = offs->data_at(i, 1, 0, 0) - 0.5;
      // Dot not make large movement with relative offset > 0.35
      if (fabs(off_x) <= 0.35 && fabs(off_y) <= 0.35)
      {
        infos[i].fpts[j].x += off_x * patch_size;
        infos[i].fpts[j].y += off_y * patch_size;
      }
    }
  }
}

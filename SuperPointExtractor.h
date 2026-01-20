#ifndef ORB_SLAM3_SUPERPOINT_EXTRACTOR_H_
#define ORB_SLAM3_SUPERPOINT_EXTRACTOR_H_

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace ORB_SLAM3 {

class SuperPointExtractor : public cv::Feature2D {
 public:
  struct Config {
    int cell = 8;               // Fixed SuperPoint grid size (H/8, W/8).
    int input_multiple = 8;     // Input dimensions must be divisible by 8.
    int nms_dist = 4;           // NMS radius in pixels.
    float conf_thresh = 0.015f; // Confidence threshold.
    float nn_thresh = 0.7f;     // Descriptor match threshold (L2).
    int border_remove = 4;      // Remove keypoints near image border.
    int max_keypoints = 1000;   // Cap for downstream SLAM usage.
    bool normalize_desc = true; // L2 normalize descriptors.
    bool use_cuda = false;      // Placeholder for future backend use.
  };

  explicit SuperPointExtractor(const std::string& model_path,
                               const Config& config = Config());
  ~SuperPointExtractor() override = default;

  void detect(cv::InputArray image,
              std::vector<cv::KeyPoint>& keypoints,
              cv::InputArray mask = cv::noArray()) override;

  void compute(cv::InputArray image,
               std::vector<cv::KeyPoint>& keypoints,
               cv::OutputArray descriptors) override;

  void detectAndCompute(cv::InputArray image,
                        cv::InputArray mask,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false) override;

  int descriptorSize() const override;
  int descriptorType() const override;
  int defaultNorm() const override;
  bool empty() const override;

  const Config& GetConfig() const;
  void SetConfig(const Config& config);
  const std::string& GetModelPath() const;
  void SetModelPath(const std::string& model_path);

 private:
  void PreprocessImage(const cv::Mat& gray, cv::Mat& input_float) const;
  void RunNetwork(const cv::Mat& input_float, cv::Mat& semi, cv::Mat& desc) const;
  void Postprocess(const cv::Mat& semi,
                   const cv::Mat& desc,
                   std::vector<cv::KeyPoint>& keypoints,
                   cv::Mat& descriptors) const;

  std::string model_path_;
  Config config_;
  bool initialized_ = false;
};

}  // namespace ORB_SLAM3

#endif  // ORB_SLAM3_SUPERPOINT_EXTRACTOR_H_

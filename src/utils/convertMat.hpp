#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

using namespace std;

#ifndef CONVERTIMAGE_H_
#define CONVERTIMAGE_H_

namespace utils {

/**
 * Classe que converte as imagens para base64 e virse e versa
 */
class ImageConverter {
public:
  /**
   * Constritor default da classe
   */
  ImageConverter() {}

  /**
   * Método que converte uma imagem base64 em um cv::Mat
   * @param imageBase64, imagem em base64
   * @return imagem em cv::Mat
   */
  cv::Mat str2mat(const string &imageBase64);

  /**
   * Método que converte uma cv::Mat numa imagem em base64
   * @param img, imagem em cv::Mat
   * @return imagem em base64
   */
  string mat2str(const cv::Mat &img);

  virtual ~ImageConverter();

private:
  std::string base64_encode(uchar const *bytesToEncode, unsigned int inLen);

  std::string base64_decode(std::string const &encodedString);
};
} // namespace utils
#endif /* CONVERTIMAGE_H_ */
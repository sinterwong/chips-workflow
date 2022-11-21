/**
 * @file video_record.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <any>
#include <string>

namespace module {
namespace utils {
class VideoRecord {
  
  explicit VideoRecord(std::string const &outfile);

  /**
   * @brief Initialize the stream
   * 
   * @return true 
   * @return false 
   */
  virtual bool init() = 0;

  /**
   * @brief Destory the stream.
   * 
   * @return true 
   * @return false 
   */
  virtual bool destory() = 0;

  /**
   * @brief Whether the stream is working.
   * 
   * @return true 
   * @return false 
   */
  virtual bool check() = 0;

  virtual ~VideoRecord();

protected:
  std::any stream;
};
} // namespace utils
} // namespace module
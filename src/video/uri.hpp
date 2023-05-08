#ifndef __URI_RESOURCE_H_
#define __URI_RESOURCE_H_

#include <string>
namespace video::utils {

struct URI {
public:
  /**
   * Default constructor.
   */
  URI();

  /**
   * Construct a new URI from the given resource string.
   * @see the documentation above for valid string formats.
   */
  URI(std::string const &uri);

  /**
   * Parse the URI from the given resource string.
   * @see the documentation above for valid string formats.
   */
  bool Parse(std::string const &uri);

  /**
   * Log the URI, with an optional prefix label.
   */
  void Print(std::string const &prefix) const;

  /**
   * Cast to std::string
   */
  operator std::string () const { return string; }


  /**
   * Assignment operator (parse URI string)
   */
  inline void operator=(std::string const &uri) { Parse(uri); }

  /**
   * Full resource URI (what was originally parsed)
   */
  std::string string;

  /**
   * Protocol string (e.g. `file`, `csi`, `v4l2`, `rtp`, ect)
   */
  std::string protocol;

  /**
   * Path, IP address, or device name
   */
  std::string location;

  /**
   * File extension (for files only, otherwise empty)
   */
  std::string extension;

  /**
   * IP port, camera port, ect.
   */
  int port;
};
} // namespace module::utils
#endif
//
// Created by xl on 2023/8/29.
//

#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

void extractIPAndPort(const std::string &url, std::string &ip, int &port) {
  size_t start = url.find("://");
  if (start == std::string::npos) {
    std::cout << "Invalid URL format" << std::endl;
    return;
  }
  start += 3; // Skip "://"

  size_t end = url.find("/", start);
  std::string hostAndPort = url.substr(start, end - start);

  // Remove username and password if present
  size_t atSign = hostAndPort.find("@");
  if (atSign != std::string::npos) {
    hostAndPort = hostAndPort.substr(atSign + 1);
  }

  size_t colon = hostAndPort.find(":");
  if (colon == std::string::npos) {
    std::cout << "Port not specified, using default" << std::endl;
    ip = hostAndPort;
    port = 554; // Default RTSP port
  } else {
    ip = hostAndPort.substr(0, colon);
    port = std::stoi(hostAndPort.substr(colon + 1));
  }
}

std::string extractSessionID(const std::string &response) {
  std::string session_id;
  size_t pos = response.find("Session: ");
  if (pos != std::string::npos) {
    pos += 9; // Length of "Session: "
    size_t end_pos = response.find("\r\n", pos);
    if (end_pos != std::string::npos) {
      session_id = response.substr(pos, end_pos - pos);
    }
  }
  return session_id;
}

std::string extractMediaUUID(const std::string &response) {
  std::string media_uuid;
  size_t pos = response.find("mediaUUID=");
  if (pos != std::string::npos) {
    pos += 10; // Length of "mediaUUID="
    size_t end_pos = response.find("\r\n", pos);
    if (end_pos == std::string::npos) {
      // If the mediaUUID is at the end of the string, use the remaining
      // characters
      end_pos = response.length();
    }
    media_uuid = response.substr(pos, end_pos - pos);
  }
  return media_uuid;
}

int main() {
  std::string stream_url =
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101";

  // delete the last ‘/’
  if (!stream_url.empty() && stream_url.back() == '/') {
    stream_url.erase(stream_url.size() - 1);
  }

  std::string server_ip;
  int server_port;
  extractIPAndPort(stream_url, server_ip, server_port);
  std::cout << "server:" << server_ip << ", port:" << server_port << std::endl;
  int sock = socket(AF_INET, SOCK_STREAM, 0);

  sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(server_port);

  //    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);
  server_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
  int connect_status =
      connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr));
  if (connect_status < 0) {
    std::cout << "connect failed" << std::endl;
    return 1;
  }

  std::cout << "OPTIONS" << std::endl;
  // Send OPTIONS request
  //    char* options_request = "OPTIONS rtsp://192.168.0.235:9554/live/test3
  //    RTSP/1.0\r\nCSeq: 1\r\n\r\n";
  std::string options_request =
      "OPTIONS " + stream_url + " RTSP/1.0\r\nCSeq: 1\r\n\r\n";

  send(sock, options_request.c_str(), options_request.length(), 0);

  char buffer[4096];
  recv(sock, buffer, sizeof(buffer), 0);
  std::cout << "Received OPTIONS response:\n" << buffer << std::endl;

  // Send DESCRIBE request
  //    char* describe_request = "DESCRIBE rtsp://192.168.0.235:9554/live/test3
  //    RTSP/1.0\r\nCSeq: 2\r\n\r\n";
  std::string describe_request =
      "DESCRIBE " + stream_url + " RTSP/1.0\r\nCSeq: 2\r\n\r\n";
  send(sock, describe_request.c_str(), describe_request.length(), 0);

  recv(sock, buffer, sizeof(buffer), 0);
  std::cout << "Received DESCRIBE response:\n" << buffer << std::endl;

  // Parse DESCRIBE response to get media UUID
  std::string describe_response(buffer);
  std::string media_uuid = extractMediaUUID(describe_response);

  if (!media_uuid.empty()) {
    std::cout << "Extracted mediaUUID: " << media_uuid << std::endl;
  } else {
    std::cout << "Failed to extract mediaUUID" << std::endl;
  }

  std::cout << "Parsed media UUID: " << media_uuid << std::endl;

  // Send SETUP request with interleaved mode
  std::string setup_request =
      "SETUP " + stream_url + "/mediaUUID=" + media_uuid +
      " RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;interleaved=0-1\r\n\r\n";
  send(sock, setup_request.c_str(), setup_request.length(), 0);
  recv(sock, buffer, sizeof(buffer), 0);
  std::cout << "Received SETUP response:\n" << buffer << std::endl;

  // Assume buffer contains the SETUP response
  std::string setup_response(buffer);
  std::string session_id = extractSessionID(setup_response);

  if (!session_id.empty()) {
    std::cout << "Extracted Session ID: " << session_id << std::endl;
  } else {
    std::cout << "Failed to extract Session ID" << std::endl;
  }

  // Send PLAY request

  std::string play_request = "PLAY " + stream_url +
                             " RTSP/1.0\r\nCSeq: 4\r\nSession: " + session_id +
                             "\r\n\r\n";
  send(sock, play_request.c_str(), play_request.length(), 0);
  recv(sock, buffer, sizeof(buffer), 0);
  std::cout << "Received PLAY response:\n" << buffer << std::endl;

  // Start reading interleaved RTP packets and RTSP messages
  while (true) {
    char ch;
    int bytesRead = recv(sock, &ch, 1, 0);
    if (bytesRead <= 0)
      break;

    if (ch == '$') {
      // This is the start of an interleaved RTP packet
      char channel;
      recv(sock, &channel, 1, 0); // Channel identifier
      std::cout << "Received interleaved RTP packet on channel: "
                << (int)channel << std::endl;

      unsigned short rtpPacketLength;
      recv(sock, &rtpPacketLength, 2, 0);
      rtpPacketLength = ntohs(rtpPacketLength); // Convert to host byte order
      std::cout << "RTP packet length: " << rtpPacketLength << std::endl;

      char rtpPacket[rtpPacketLength];
      recv(sock, rtpPacket, rtpPacketLength, 0);

      // Process the RTP packet (e.g., extract H.264 data)
      // ...

    } else {
      // Handle RTSP messages (if any)
      // For simplicity, we'll just print the character
      std::cout << "Received RTSP character: " << ch << std::endl;
    }
  }

  // Close the connection
  std::string teardown_request =
      "TEARDOWN " + stream_url + " RTSP/1.0\r\nCSeq: 5\r\n\r\n";
  send(sock, teardown_request.c_str(), teardown_request.length(), 0);
  recv(sock, buffer, sizeof(buffer), 0);
  std::cout << "Received TEARDOWN response:\n" << buffer << std::endl;

  close(sock);
  return 0;
}

#include <cstdio>
#include "rclcpp/rclcpp.hpp"


class IngestorNode : public rclcpp::Node
{
public:
  IngestorNode() 
  : Node("Ingestor") 
  {
    subscription_ = this->create_subscription<std_msgs::msg::
  }
private:
};


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IngestorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}

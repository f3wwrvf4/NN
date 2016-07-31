// mnist

#include <windows.h>

#include <NN_net.h>
#include <NN_fw.h>
#include <NN_mnist.h>

int main()
{
  typedef NN::MNIST CONTENT;
  const char* fpath = "mnist.nn";

  int mid_layer = 1000;
  NN::Network::InitParam
    init_param[] =
  {
    {{ CONTENT::DataSize, 1000}, NN::Network::LogisticLayer },
    {{ 1000, CONTENT::LabelSize}, NN::Network::SoftMaxLayer },
  };
  int layer_num = ARRAY_NUM(init_param);

  const int batch_size = 50;
  const int train_count = 10;

  CONTENT content;
  content.LoadTrainData();

  DWORD tick = GetTickCount();
  std::cout << "start.." << tick << std::endl;
  NN::Train(content, fpath, init_param, layer_num, batch_size, train_count);
  std::cout << "tick = " << (GetTickCount() - tick) << std::endl;

  content.LoadTestData();
  NN::Test(content, fpath, init_param, layer_num);

  getchar();
  return 0;
}

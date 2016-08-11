// iris

#include <windows.h>

#include <NN_net.h>
#include <NN_fw.h>
#include <NN_iris.h>

int main()
{
  typedef NN::Iris CONTENT;
  const char* fpath = "iris.nn";

  int mid_layer = 4;
  NN::Network::InitParam
    init_param[] =
  {
    {{ CONTENT::DataSize, mid_layer}, NN::Network::LogisticLayer },
    {{ mid_layer, CONTENT::LabelSize}, NN::Network::SoftMaxLayer },
  };
  int layer_num = ARRAY_NUM(init_param);

  const int batch_size = 5;
  const int train_count = 50;

  CONTENT content;
  content.LoadData();

  DWORD tick = GetTickCount();
  std::cout << "start.." << tick << std::endl;
  NN::Train(content, fpath, init_param, layer_num, batch_size, train_count);
  std::cout << "tick = " << (GetTickCount() - tick) << std::endl;

  NN::Test(content, fpath, init_param, layer_num);

  getchar();
  return 0;
}

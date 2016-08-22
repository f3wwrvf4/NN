// mnist

#include <windows.h>

#include <NN_net.h>
#include <NN_fw.h>
#include <NN_mnist.h>

int main()
{
  NN::MathInit();

  typedef NN::MNIST CONTENT;
  const char* fpath = "mnist.nn";

  int mid_layer = 300;
  NN::Network::InitParam
    init_param[] =
  {
    {{ CONTENT::DataSize, mid_layer}, NN::Network::LogisticLayer },
    {{ mid_layer, CONTENT::LabelSize}, NN::Network::SoftMaxLayer },
  };
  int layer_num = ARRAY_NUM(init_param);

  const int batch_size = 50;
  const int train_count = 10;

  NN::Network net_train(layer_num, init_param, batch_size);
  net_train.load(fpath);
  NN::Network net_test(layer_num, init_param, 1);

  CONTENT::Content trainData, testData;
  CONTENT::LoadTrainData(trainData);
  CONTENT::LoadTestData(testData);

  int count = 5;
  while (--count) {
    DWORD tick = GetTickCount();
    std::cout << "start.." << std::endl;
    NN::Train(net_train, trainData, batch_size, train_count);
    std::cout << "tick = " << (GetTickCount() - tick) << std::endl;

    net_train.save(fpath);

    net_test.load(fpath);
    NN::Test(net_test, testData);
  }

  getchar();
  return 0;
}

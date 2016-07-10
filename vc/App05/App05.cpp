// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <windows.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <new>
#include <string>
#include <list>

#include <NN_math.h>
#include <NN_net.h>
#include <NN_mnist.h>
#include <NN_mnist.h>

#include <stdlib.h> 
#include <iostream> 
#include <new> 

#if 0

static size_t newCount = 0; 
void* _cdecl operator new(size_t size) 
{
   ++newCount; 
   void* addr = malloc(size); 
   std::cout << "(" << newCount << ") " << "new " << size << "bytes addr 0x" << std::hex << addr << std::dec << std::endl; 
   return addr;
} 
void  _cdecl operator delete(void *p) 
{ 
   --newCount; 
   std::cout << "(" << newCount << ") " << "delete " << p << std::endl; 
   free(p); 
} 

#endif


int main()
{
  NN::MNIST mnist;

  NN::Network::InitParam
    init_param[] = 
  {
    {{ NN::MNIST::DataSize, 1000}, NN::Network::LogisticLayer },
    {{ 1000, NN::MNIST::LabelSize}, NN::Network::LogisticLayer },
  };
  int layer_num = ARRAY_NUM(init_param);

  const char* fpath = "mnist.nn";
#if 0
  {
    mnist.LoadTrainData();
    int dataCount = mnist.GetTrainDataCount();
    int batch_size = 100;

    NN::Network net(layer_num, init_param, batch_size);
    net.load(fpath);

    NN::Matrix in(batch_size, NN::MNIST::DataSize + 1);
    NN::Matrix out(batch_size, NN::MNIST::LabelSize);
    const int count = dataCount / batch_size;

    int train = 1000;
    while (--train) {
      mnist.shuffle();
      for (int i = 0; i < count; ++i) {
        int idx = i*batch_size;
        //        std::cout << std::setw(2)<< idx << ":";
        net.train(
          mnist.GetTrainInputData(idx, batch_size, &in),
          mnist.GetTrainOutputData(idx, batch_size, &out));
      }
      net.save(fpath);
    }
  }
#else
  {
    NN::Network net(layer_num, init_param, 1);
    net.load(fpath);

    NN::Matrix in(1, NN::MNIST::DataSize + 1);
    NN::Matrix out(1, NN::MNIST::LabelSize);
    const int LabelSize = NN::MNIST::LabelSize;
    mnist.LoadTestData();
    int count = mnist.GetTestDataCount();

    int answer[LabelSize][LabelSize] = {};

    for (int i = 0; i < count; ++i) {
      mnist.GetTestInputData(i, 1, &in);
      mnist.GetTestOutputData(i, 1, &out);

      const NN::Matrix& res = net.eval(in);


      float x[LabelSize];
      for (int j = 0; j < LabelSize; ++j) {
        x[j] = 1.0f - res(0, j);
        x[j] *= x[j];
      }

      float t = x[0];
      int min = 0;
      for (int j = 1; j < LabelSize; ++j) {
        if (x[j] < t) {
          min = j;
          t = x[j];
        }
      }

      int actual = 0;
      for (int j = 1; j < LabelSize; ++j) {
        if (out(0, j) != 0) {
          actual = j;
          break;
        }
      }

      ++answer[actual][min];
    }

    for (int i = 0; i < LabelSize; ++i) {
      std::cout << "[" << i << "] ";
      for (int j = 0; j < LabelSize; ++j) {
        std::cout << std::right << std::setw(6) << answer[i][j] << " ";
      }
      int a_sam = 0;
      for (int j = 0; j < LabelSize; ++j) {
        a_sam += answer[i][j];
      }
      const int a = answer[i][i];
      const int b = a_sam;
      const float per = a / (float)b;
      std::cout << "(" << 
        std::setw(4)<< a << "/" << 
        std::setw(4)<< b << ")  " <<
        std::setw(4) << per*100 << "%" << 
        std::endl;
    }
    {
      int correct = 0;
      for (int j = 0; j < LabelSize; ++j) {
        correct += answer[j][j];
      }
      const int a = correct;
      const int b = count;
      const float per = a / (float)b;

      std::cout << "all: (" << 
        std::setw(4)<< a << "/" << 
        std::setw(4)<< b << ")  " <<
        std::setw(4) << per*100 << "%" << 
        std::endl;
    }

  }
#endif

  getchar();
  return 0;
}

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


#include <NN_net.h>
#include <NN_math.h>
#include <NN_mnist.h>
#include <NN_iris.h>

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
  NN::Iris iris;
  iris.LoadData();

  int dataCount = iris.GetTrainDataCount();

  NN::Network::InitParam init_param[] = {
    {{NN::Iris::DataSize, 2}, NN::Network::LogisticLayer},
    {{2, NN::Iris::LabelSize}, NN::Network::LogisticLayer},
  };
  int layer_num = ARRAY_NUM(init_param);
  const char* fpath = "iris.nn";
  {
    int batch_size = 3;

    NN::Network net(layer_num, init_param, batch_size);
//    net.load(fpath);

    NN::Matrix in(batch_size, NN::Iris::DataSize+1);
    NN::Matrix out(batch_size, NN::Iris::LabelSize);
    const int count = dataCount / batch_size;

    int train = 1000;
    while (--train) {
      iris.shuffle();
      for (int i = 0; i < count; ++i) {
        int idx = i*batch_size;
//        std::cout << std::setw(2)<< idx << ":";
        net.train(
          iris.GetTrainInputData(idx, batch_size, &in),
          iris.GetTrainOutputData(idx, batch_size, &out));

        if(0)
        for (int j = 0; j < batch_size; ++j) {
          NN::Matrix testin(1, NN::Iris::DataSize + 1);
          NN::Matrix testout(1, NN::Iris::LabelSize);

          iris.GetTrainInputData(idx+j, 1, &testin);
          iris.GetTrainOutputData(idx+j, 1, &testout);

          std::cout << testin;
          std::cout << testout;
          std::cout << net.eval(testin);
        }
      }
    }
    net.save(fpath);
  }

  {
    NN::Network net(layer_num, init_param, 1);
    net.load(fpath);

    NN::Matrix in(1, NN::Iris::DataSize+1);
    NN::Matrix out(1, NN::Iris::LabelSize);

    int hit = 0;

    int count = iris.GetTestDataCount();
    for (int i = 0; i < count; ++i) {

      iris.GetTestInputData(i, 1, &in);
      iris.GetTestOutputData(i, 1, &out);

      const NN::Matrix& res = net.eval(in);

      float x[3];
      x[0] = 1.0f - res(0, 0);
      x[1] = 1.0f - res(0, 1);
      x[2] = 1.0f - res(0, 2);

      x[0] *= x[0];
      x[1] *= x[1];
      x[2] *= x[2];

      float t = x[0];
      int min = 0;
      for (int j = 1; j < 3; ++j) {
        if (x[j] < t) {
          min = j;
          t = x[j];
        }
      }

      if (out(0, min) != 0)
        ++hit;
      else {
        std::cout << in;
        std::cout << out;
        std::cout << res;
      }
    }

    std::cout << hit << "/" << count;
  }


  getchar();
  return 0;
}

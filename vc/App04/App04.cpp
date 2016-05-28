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

  int node_num[] = { NN::Iris::DataSize, 10, 10, NN::Iris::LabelSize };
  int layer_num = ARRAY_NUM(node_num);

  const char* fpath = "iris.nn";
  {
    int batch_size = 5;

    NN::Network net(layer_num, node_num, batch_size);
    net.load(fpath);

    NN::Matrix in(batch_size, NN::Iris::DataSize+1);
    NN::Matrix out(batch_size, NN::Iris::LabelSize);
    const int count = dataCount / batch_size;

    int train = 1000;
    while (--train) {
      iris.shuffle();
      for (int i = 0; i < count; ++i) {
        int idx = i*batch_size;
        std::cout << std::setw(2)<< idx << ":";
        net.train(
          iris.GetTrainInputData(idx, batch_size, &in),
          iris.GetTrainOutputData(idx, batch_size, &out));
      }
    }
    net.save(fpath);
  }

  {
    NN::Network net(layer_num, node_num, 1);
    net.load(fpath);

    NN::Matrix in(1, NN::Iris::DataSize+1);
    NN::Matrix out(1, NN::Iris::LabelSize);

    int count = iris.GetTestDataCount();
    for (int i = 0; i < count; ++i) {
      const NN::Matrix& res = 
        net.eval(iris.GetTestInputData(i, 1, &in));
      iris.GetTestOutputData(i, 1, &out);

      std::cout << out;
      std::cout << res;
    }
  }


  getchar();
  return 0;
}

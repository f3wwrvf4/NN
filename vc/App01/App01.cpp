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

int main()
{
  NN::Network::InitParam init_param[] = {
    {{2, 3}, NN::Network::SoftMaxLayer},
  };
  int layer_num = ARRAY_NUM(init_param);
  {
    int batch_size = 1;
    NN::Network net(layer_num, init_param, batch_size);

    net.load("test.nn");

    NN::Matrix in(batch_size, 3);
    NN::Matrix out(batch_size, 3);

    int hit = 0;

    in(0,0) = 0.0f;
    in(0,1) = 1.0f;
    in(0,2) = 0.0f;

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
}



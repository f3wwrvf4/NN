// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "NN_net.h"
#include "NN_mnist.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <new>
#include <string>
#include <list>

int main()
{
  NN::MNIST mnist;
  mnist.loadTrainData();

#if 1

  if (1) {
    int sz = mnist.imageCol() * mnist.imageRow();
    int L[] = { sz , 1000, 10 };
    NN::Network net(L, ARRAY_NUM(L));

    NN::TrainData* td = new NN::TrainData[mnist.count()];
    for (int i = 0; i < mnist.count(); ++i) {
      new(&td[i]) NN::TrainData(mnist.imageData(i), mnist.labelData(i));
    }
    net.train(td, mnist.count());
    net.save("mnist.nn");

    delete[] td;
  }

  if (0) {
    std::ifstream ifs("iris.txt");

    struct Data
    {
      Data(float in[4], float out[3]) :
        input(4),
        output(3)
      {
        for (int i = 0; i < 4; ++i)
          input[i] = in[i];
        for (int i = 0; i < 3; ++i)
          output[i] = out[i];
      }
      std::vector<float> input;
      std::vector<float> output;
    };

    std::vector<Data> iris;
    while (!ifs.eof()) {
      float in[4];
      char cm;
      std::string str;

      for (int i = 0; i < 4; ++i)
        ifs >> in[i] >> cm;
      ifs >> str;

      float out[3] = {};
      if (str == "setosa")
        out[0] = 1.0;
      if (str == "versicolor")
        out[1] = 1.0;
      if (str == "virginica")
        out[2] = 1.0;
      iris.push_back(Data(in, out));
    }

    int L[] = { 4,100,3 };
    NN::Network net(L, ARRAY_NUM(L));

    NN::TrainData* td = new NN::TrainData[iris.size()];
    for (int i = 0; i < iris.size(); ++i) {
      new(&td[i]) NN::TrainData(iris[i].input, iris[i].output);
    }

    net.train(td, iris.size());
    net.save("iris.nn");
  }



  if (1) {
    int L[] = { 3, 40, 2 };
    NN::Network net(L, ARRAY_NUM(L));

    std::vector<float> in[] = {
      {0, 0, 0}, 
      {1, 0, 1}, 
      {1, 1, 1}, 
      {1, 1, 0}, 
      {1, 0, 0}, 
      {0, 0, 1}, 
      {0, 1, 0},
    };
    std::vector<float> ou[] = {
      {1,1},
      {1,0},
      {0,1},
      {0,1},
      {1,1},
      {1,0},
      {1,1},
    };

    NN::TrainData* td = new NN::TrainData[ARRAY_NUM(in)];
    for (int i = 0; i < ARRAY_NUM(in); ++i) {
      new(&td[i]) NN::TrainData(in[i], ou[i]);
    }
    net.train(td, ARRAY_NUM(in));
    net.save("NN_A7.nn");

    for (int j = 0; j < ARRAY_NUM(in); ++j) {
      std::vector<float> ans = net.eval(in[j]);
      for (int i = 0; i < ans.size(); ++i) {
        std::cout << round(ans[i]) << ",";
      }
      std::cout << std::endl;
    }

  }


#else
  {
    NN::Network* net = NN::Network::CreateFromFile("nand.nn");

    std::vector<float> val;
    val = net->eval({ 0, 0 });
    assert(roundf(val[0]) == 1.0f);
    val = net->eval({ 1, 0 });
    assert(roundf(val[0]) == 1.0f);
    val = net->eval({ 0, 1 });
    assert(roundf(val[0]) == 1.0f);
    val = net->eval({ 1, 1 });
    assert(roundf(val[0]) == 0.0f);

    delete net;
    net = 0;
  }
#endif

  getchar();
  return 0;
}


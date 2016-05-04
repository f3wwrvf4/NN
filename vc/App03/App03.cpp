// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "NN_net.h"
#include "NN_mnist.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <new>
#include <string>
#include <list>


namespace NN2
{
struct Network
{
  void forward()
  {

  }
  void back()
  {

  }


};

}

int main()
{
  struct
  {
    float ti[3];
    float to[2];
  }td[] = {
    {{0, 0, 0}, {1, 1}},
    {{0, 0, 1}, {1, 0}},
    {{0, 1, 0}, {1, 1}},
    {{0, 1, 1}, {1, 0}},
    {{1, 0, 0}, {1, 1}},
    {{1, 0, 1}, {1, 0}},
    {{1, 1, 0}, {0, 1}},
    {{1, 1, 1}, {0, 1}},
  };

  const int bn = 4;
  NN::Matrix input(bn, 4);
  NN::Matrix hidden(bn, 6);
  NN::Matrix out(bn, 2);

  NN::Matrix weigh_h(4, 6);
  NN::Matrix weigh_o(6, 2);

  NN::Matrix err_o(bn, 2);
  NN::Matrix err_h(bn, 6);

  NN::Matrix delta_o(bn, 2);
  NN::Matrix delta_h(bn, 6);
  NN::Matrix delta_i(bn, 4);

  NN::Matrix rdw_o(6, 2);
  NN::Matrix rdw_h(4, 6);

  NN::Matrix differ_o(bn, 2);
  NN::Matrix differ_h(bn, 6);

  weigh_h.random();
  weigh_o.random();

  float error;
  int iTrain = 0;
  while(true) {
    ++iTrain;
    error = 0;

    for (int i = 0; i < bn; ++i) {
      int tid = i +(iTrain & 1 ? 0 : bn);
      for (int j = 0; j < 3; ++j) {
        input(i, j) = td[tid].ti[j];
      }
      input(i, 3) = 1.0f;
    }

    // forward
    NN::Mul(weigh_h, input, hidden);
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < 5; ++j) {
        hidden(i, j) = 1.0f / (1.0f + exp(-hidden(i, j)));

        differ_h(i, j) = hidden(i, j)*(1.0f - hidden(i, j));
      }
      hidden(i, 5) = 1.0f;
      differ_h(i, 5) = 1.0f;
    }
    NN::Mul(weigh_o, hidden, out);
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < 2; ++j) {
        out(i, j) = 1.0f / (1.0f + exp(-out(i, j)));

        differ_o(i, j) = out(i, j)*(1.0f - out(i, j));
      }
    }

    // error
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < 2; ++j) {

        int tid = i +(iTrain & 1 ? 0 : bn);

        const float err = (td[tid].to[j] - out(i, j)) * (td[tid].to[j] - out(i, j));
        error += err;
        delta_o(i, j) = out(i, j) - td[tid].to[j];
        std::cout << "(" << i << "," << j << ")" << td[tid].to[j]
          << " - " << out(i, j) << std::endl;
      }
    }

    // back
    NN::Mul(weigh_o.t(), delta_o, delta_h);
    NN::Hadamard(differ_h, delta_h, delta_h);

    NN::Mul(delta_o, hidden.t(), rdw_o);
    NN::Mul(delta_h, input.t(), rdw_h);

    const float eps = 0.1f;
    NN::Mul(-eps, rdw_h, rdw_h);
    NN::Mul(-eps, rdw_o, rdw_o);
    NN::Add(weigh_h, rdw_h, weigh_h);
    NN::Add(weigh_o, rdw_o, weigh_o);

    std::cout << error << std::endl;
  }

  getchar();
  return 0;
}


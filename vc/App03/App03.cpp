// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <windows.h>


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

  const int InputNum = 3;
  const int MidNum = 5000;
  const int OutNum = 2;

  const int Node1 = InputNum + 1;
  const int Node2 = MidNum + 1;
  const int Node3 = OutNum;

  const int bn = 4;
  NN::Matrix input(bn, Node1);
  NN::Matrix hidden(bn, Node2);
  NN::Matrix out(bn, Node3);

  NN::Matrix weigh_h(Node1, Node2);
  NN::Matrix weigh_o(Node2, Node3);

  NN::Matrix err_o(bn, Node3);
  NN::Matrix err_h(bn, Node2);

  NN::Matrix delta_o(bn, Node3);
  NN::Matrix delta_h(bn, Node2);

  NN::Matrix rdw_h(Node1, Node2);
  NN::Matrix rdw_o(Node2, Node3);

  NN::Matrix differ_h(bn, Node2);

  weigh_h.random();
  weigh_o.random();

  DWORD msec1 = GetTickCount();

  float error;
  int iTrain = 0;
  while(true) {
    ++iTrain;
    error = 0;

    for (int i = 0; i < bn; ++i) {
      int tid = i +(iTrain & 1 ? 0 : bn);
      for (int j = 0; j < InputNum; ++j) {
        input(i, j) = td[tid].ti[j];
      }
      input(i, InputNum) = 1.0f;
    }

    // forward
    NN::Mul(weigh_h, input, hidden);
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < MidNum; ++j) {
        hidden(i, j) = 1.0f / (1.0f + exp(-hidden(i, j)));

        differ_h(i, j) = hidden(i, j)*(1.0f - hidden(i, j));
      }
      hidden(i, MidNum) = 1.0f;
      differ_h(i, MidNum) = 1.0f;
    }

    NN::Mul(weigh_o, hidden, out);
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < OutNum; ++j) {
        out(i, j) = 1.0f / (1.0f + exp(-out(i, j)));
      }
    }

    // error
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < OutNum; ++j) {
        int tid = i +(iTrain & 1 ? 0 : bn);
        const float err = (td[tid].to[j] - out(i, j)) * (td[tid].to[j] - out(i, j));
        error += err;
        delta_o(i, j) = out(i, j) - td[tid].to[j];
      }
    }
    if (error < 0.01f) {
      for (int i = 0; i < bn; ++i) {
        for (int j = 0; j < OutNum; ++j) {
          int tid = i + (iTrain & 1 ? 0 : bn);
          float t = td[tid].to[j];
          float o = out(i, j);

          std::cout << t << " - " << o << std::endl;
        }
      }
      std::cout << iTrain << std::endl;
      break;
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

  }

  DWORD msec2 = GetTickCount();
  std::cout << "sec: " << msec2 - msec1;

  getchar();
  return 0;
}

